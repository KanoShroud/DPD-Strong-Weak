"""Matched frequency-model null diagnostic, not new blind performance."""
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from pilot import HERE, FS, B, cases, evaluate, fit, search, sha, write


def main():
    start=time.perf_counter()
    full=Path(json.loads((HERE/'full_latest.json').read_text(encoding='utf-8'))['path'])
    records=json.loads((full/'result.json').read_text(encoding='utf-8'))['rows']
    chosen=[r for r in records if r['tag'] in ('L20_R200_LOS','L40_R300_LOS')]
    lookup={tag:inp for tag,_,inp,_ in cases(True)}
    paths=[full/'result.json',Path(__file__),HERE/'pilot.py',HERE/'匹配空假设核验协议.md']+[lookup[r['tag']] for r in chosen]
    hashes={str(p.resolve(strict=True)):sha(p) for p in paths}
    out=HERE/('calibration_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    (out/'calibration.py').write_bytes(Path(__file__).read_bytes())
    write(out/'preflight.json',dict(hashes=hashes,seed=2026103014,replicates_per_case=16,
                                  kind='MATCHED_NULL_LOCAL_START_NOT_BLIND_LOCALIZATION', new_threshold_selected=False))
    rng=np.random.default_rng(2026103014)
    rows=[]
    checks={}
    for record in chosen:
        with np.load(lookup[record['tag']],allow_pickle=False) as data:
            iq=data['los'].sum(axis=0)+data['noise']
        n=iq.shape[1]
        freq=np.fft.fftfreq(n,1/FS)
        use=abs(freq)<=B/2
        f=freq[use]
        y=np.fft.fft(iq,axis=1,norm='ortho')[:,use].T
        x=np.asarray(record['los']['x'])
        _,_,(h,inv,s)=evaluate(x,f,y,mean_tangent=True)
        mean=np.einsum('fmq,fq->fm',h,s)
        sigma2=record['sigma2_estimated']
        checks[record['tag']+'/matched_mean']=np.linalg.norm(evaluate(x,f,mean)[0])<1e-10
        for rep in range(16):
            yy=mean+np.sqrt(sigma2/2)*(rng.normal(size=mean.shape)+1j*rng.normal(size=mean.shape))
            # Fit positions/gains starting at generating x: favorable LOCAL null diagnostic.
            # sigma2 is exactly known here only to isolate a different issue from noise estimation.
            xx,meta=fit(x,f,yy,n)
            picks,bank,sm=search(xx,f,yy,n,use,sigma2)
            checks.update({f'{record["tag"]}_{rep}/'+k:v for k,v in sm['checks'].items()})
            rows.append(dict(tag=record['tag'],rep=rep,risk_added=picks['RISK'] is not None,
                             fit_added=picks['FIT'] is not None, picks=picks,
                             stationary=meta['stationary'],projected_gradient=meta['projected_gradient']))
            write(out/'progress.json',rows)
        print(record['tag'],'完成16个匹配模型空假设诊断',flush=True)
    checks['identity_unchanged']=all(sha(p)==h for p,h in hashes.items())
    write(out/'result.json',dict(rows=rows,checks=checks,all_pass=all(checks.values()),
           count=len(rows),risk_added=sum(r['risk_added'] for r in rows),fit_added=sum(r['fit_added'] for r in rows),
           stationary=sum(r['stationary'] for r in rows),seconds=time.perf_counter()-start,
           interpretation='Only 32 matched local null replicates; not calibrated nominal5 guarantee and not a new threshold.'))
    write(HERE/'calibration_latest.json',dict(path=str(out)))
    print('结果',out,'误增',sum(r['risk_added'] for r in rows),'/',len(rows),flush=True)
    assert all(checks.values())


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
