"""Read-only estimator-state probes; no new optimization or adopted output."""
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from reference import HERE, FS, B, UNIT, evaluate, project, search, cases, sha, write


def main():
    start=time.perf_counter()
    run=Path(json.loads((HERE/'full_latest.json').read_text(encoding='utf-8'))['path'])
    full=json.loads((run/'result.json').read_text(encoding='utf-8'))
    selected=[r for r in full['rows'] if r['stop']=='numerical_nonstationary']
    lookup={tag:inp for tag,_,inp,_ in cases(False)}
    files=[Path(__file__),HERE/'reference.py',HERE/'停止归因协议.md',run/'result.json']+[lookup[r['tag']] for r in selected]
    hashes={str(p.resolve(strict=True)):sha(p) for p in files}
    out=HERE/('termination_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    (out/'termination_audit.py').write_bytes(Path(__file__).read_bytes())
    write(out/'preflight.json',dict(hashes=hashes,no_new_solver=True,no_truth_in_probes=True))
    rows=[]
    checks={}
    for row in selected:
        with np.load(lookup[row['tag']],allow_pickle=False) as data:
            iq=(data['los']+(data['reflected'] if row['mode']=='MP' else 0)).sum(axis=0)+data['noise']
        n=iq.shape[1]
        f0=np.fft.fftfreq(n,1/FS)
        use=abs(f0)<=B/2
        ft=np.fft.fft(iq,axis=1,norm='ortho')
        f,y=f0[use],ft[:,use].T
        x=np.array(row['final']['x'])
        paths=row['paths']
        r,j=evaluate(x,f,y,paths)
        grad=j.T@r
        k=int(np.argmax(abs(grad)))
        value=float(.5*r@r)
        numeric=[]
        for eps in [1e-4,1e-5,1e-6]:
            v=np.zeros(len(x));v[k]=eps
            a,_=evaluate(x+v,f,y,paths)
            b,_=evaluate(x-v,f,y,paths)
            numeric.append(float((.5*a@a-.5*b@b)/(2*eps)))
        checks[row['tag']+'/analytic_gradient']=abs(numeric[1]-grad[k])<1e-9
        gn=np.linalg.lstsq(j,-r,rcond=1e-10)[0]
        probes=[]
        for name,direction in [('PG',project(x-grad,n)-x),('GN',project(x+gn,n)-x)]:
            for alpha in [1.,.1,.01]:
                xx=x+alpha*direction
                rr,jj= evaluate(xx,f,y,paths)
                probes.append(dict(kind=name,alpha=alpha,cost_change=float(.5*rr@rr-value),
                                   max_position_change_m=float(np.max(np.linalg.norm((xx[:4]-x[:4]).reshape(2,2)*UNIT,axis=1))),
                                   projected_gradient=float(np.max(abs(xx-project(xx-jj.T@rr,n)))),
                                   closure_error=float(np.linalg.norm(xx-project(xx,n)))))
        pick,bank,meta=search(x,paths,f,y,n,use,float(np.mean(abs(ft[:,~use])**2)),len(paths)+1)
        np.savez_compressed(out/(row['tag']+'_next_candidates.npz'),bank=bank)
        checks.update({row['tag']+'/'+k:v for k,v in meta['checks'].items()})
        rows.append(dict(tag=row['tag'],gradient_index=k,analytic=float(grad[k]),numeric=numeric,probes=probes,
                         next_pick=pick,next_threshold=meta['threshold'],stopped_state_not_replaced=True))
    checks['identity_unchanged']=all(sha(p)==h for p,h in hashes.items())
    write(out/'result.json',dict(rows=rows,checks=checks,all_checks_pass=all(checks.values()),seconds=time.perf_counter()-start))
    write(HERE/'termination_latest.json',dict(path=str(out)))
    print(json.dumps(rows,ensure_ascii=False,indent=2))
    print('归因检查',all(checks.values()),out)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
