"""Read-only numerical attribution; no state replaces any estimator output."""
from datetime import datetime
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
from reference import HERE, FS, B, UNIT, evaluate, project, sha, write


def main():
    run=Path(json.loads((HERE/'cross_latest.json').read_text(encoding='utf-8'))['path'])
    data=json.loads((run/'result.json').read_text(encoding='utf-8'))
    pre=json.loads((run/'preflight.json').read_text(encoding='utf-8'))
    assert all(sha(p)==h for p,h in pre['hashes'].items())
    out=HERE/('cross_terminal_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    (out/'cross_terminal_audit.py').write_bytes(Path(__file__).read_bytes())
    rows=[]
    for row in data['rows']:
        if row['stop']=='no_detected_increment' or 'final' not in row:
            continue
        with np.load(run/(row['tag']+'_input.npz'),allow_pickle=False) as inp:
            iq=inp['iq']
        n=iq.shape[1]
        ff=np.fft.fftfreq(n,1/FS);use=abs(ff)<=B/2
        f=ff[use];y=np.fft.fft(iq,axis=1,norm='ortho')[:,use].T
        x=np.array(row['final']['x']);paths=row['paths']
        r,j=evaluate(x,f,y,paths);g=j.T@r
        cost=float(.5*r@r)
        variants={}
        for name,z in [('saved',x),('feasible_projection',project(x,n)),
                       ('approved_pg_trial',project(x-g,n))]:
            rr,jj=evaluate(z,f,y,paths)
            vals=np.sum(z[20:].reshape(-1,3)[:,:2]**2,axis=1)
            variants[name]=dict(cost=float(.5*rr@rr),cost_change=float(.5*rr@rr-cost),
                mapping=float(np.max(abs(z-project(z-jj.T@rr,n)))),
                closure=float(np.linalg.norm(z-project(z,n))),
                max_relative_power=float(vals.max()),
                position_change_m=float(np.max(np.linalg.norm((z[:4]-x[:4]).reshape(2,2)*UNIT,axis=1))))
        rng=np.random.default_rng(2026103031)
        directions=rng.normal(size=(4,len(x)))
        directions/=np.linalg.norm(directions,axis=1)[:,None]
        derivative=[]
        for eps in (1e-4,1e-5,1e-6):
            for v in directions:
                rp,_=evaluate(x+eps*v,f,y,paths);rm,_=evaluate(x-eps*v,f,y,paths)
                fd=float((.5*rp@rp-.5*rm@rm)/(2*eps));exact=float(g@v)
                derivative.append(dict(eps=eps,analytic=exact,finite_difference=fd,absolute_error=abs(fd-exact)))
        rows.append(dict(tag=row['tag'],stop=row['stop'],iterations=row['final']['iterations'],
                         optimizer_message=row['final']['message'],variants=variants,derivative=derivative))
    write(out/'result.json',dict(source=str(run),source_sha=sha(run/'result.json'),
                                unchanged_source_identity=all(sha(p)==h for p,h in pre['hashes'].items()),
                                output_replacement=False,rows=rows))
    write(HERE/'cross_terminal_latest.json',dict(path=str(out)))
    print(json.dumps(rows,ensure_ascii=False,indent=2))
    print('保存',out)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
