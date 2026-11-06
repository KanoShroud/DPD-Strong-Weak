"""Posthoc all-case attribution, using an already frozen trim-eight estimator."""
from datetime import datetime
import json
import time

import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from geometry_association_gate import ROOT, PAIRS, C, sha, write
from robust_geometry import predicted, refine, select, evaluate

SOURCE=ROOT/'independent_backend_20261029_222111_026721'


def main():
    source=json.loads((SOURCE/'result.json').read_text(encoding='utf-8'))
    files=[ROOT/'independent_attribution.py',ROOT/'独立失败归因协议.md',ROOT/'robust_geometry.py',
           ROOT/'geometry_association_gate.py',SOURCE/'result.json',SOURCE/'protocol.json']
    files+=sorted(SOURCE.glob('*_geometry.npz'))+sorted(SOURCE.glob('*_input.npz'))
    hashes={str(p.resolve(strict=True)):sha(p) for p in files}
    frozen=json.loads((SOURCE/'protocol.json').read_text(encoding='utf-8'))['hashes']
    assert all(h==frozen[p] for p,h in hashes.items() if p in frozen)
    out=ROOT/('independent_attribution_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    (out/'source').mkdir()
    for p in files[:4]:
        (out/'source'/p.name).write_bytes(p.read_bytes())
    write(out/'protocol.json',dict(hashes=hashes,posthoc=True,all_cases=24,trim_keep=8))
    start=time.perf_counter()
    rows=[]
    checks={}
    for row in source['rows']:
        tag=row['tag']
        data=np.load(SOURCE/(tag.rsplit('_',1)[0]+'_input.npz'))
        truth=data['p']
        saved=np.load(SOURCE/(tag+'_geometry.npz'))
        lags=np.array([sorted(np.array(row['peaks'][f'{m}_{n}']['lag_s'][:6])*C) for m,n in PAIRS])
        initial=saved['initial_candidates']
        begin=time.perf_counter()
        p,score,reason,nit=refine(initial,lags,8)
        ids=select(p,score,2)
        seconds=time.perf_counter()-begin
        pred,_=predicted(truth)
        delta=pred[:,:,None]-lags[None]
        nearest=np.argmin(abs(delta),axis=2)
        residual=np.take_along_axis(delta,nearest[:,:,None],axis=2)[:,:,0]
        rec=dict(tag=tag,path=row['path'],layout=row['layout'],rep=row['rep'],
             raw_coverage_m=cdist(initial,truth).min(axis=0),
             all_coverage_m=cdist(saved['p'],truth).min(axis=0),
             trim_coverage_m=cdist(p,truth).min(axis=0),
             all_errors_m=row['geometry_errors_m'],trim=evaluate(p[ids],truth),
             trim_p=p[ids],trim_scores=score[ids],seconds=seconds,
             true_los_peak_residual_m=residual,nearest_peak_index=nearest)
        checks[tag+'/finite']=bool(np.isfinite(p).all() and np.isfinite(score).all())
        checks[tag+'/complete']=len(ids)==2
        rows.append(rec)
        write(out/(tag+'.json'),rec)
        np.savez_compressed(out/(tag+'.npz'),p=p,scores=score,chosen=ids,reason=reason,iterations=nit)
        print(tag,'ALL',np.round(rec['all_errors_m'],1),'TRIM',np.round(rec['trim']['errors_m'],1),flush=True)
    checks.update({p:sha(p)==h for p,h in hashes.items()})
    assert all(checks.values())
    write(out/'result.json',dict(rows=rows,checks=checks,seconds=time.perf_counter()-start))
    print('COMPLETE',out,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
