"""All saved blind MP channels; conditional calibration, no new positions."""
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from discrepancy import HERE, INPUT, SAVED, cfg, ad, stable
from conditional import calibrate, selftest


def main():
    out=HERE/'conditional_v1';out.mkdir(exist_ok=False)
    started=time.perf_counter()
    qa=selftest();cfg.write(out/'correctness.json',qa)
    assert qa['passed'],qa['checks']
    stable.inside.rr.check_hashes(cfg.read(HERE/'discrepancy_v1/identity.json'))
    hashes=ad.source_hashes()
    hashes.update({str(p):cfg.sha(p) for p in [Path(__file__).resolve(),HERE/'conditional.py',HERE/'conditional_protocol.md']})
    cfg.write(out/'identity.json',hashes)
    rows=[]
    for decision in cfg.read(HERE/'discrepancy_v1/blind_decisions.json'):
        tag=decision['tag'];print('START',tag,flush=True)
        with np.load(INPUT/'cases'/tag/'iq.npz',allow_pickle=False) as z:
            iq=z['iq']
        freq,y,n,_=ad.b.fc.cal.spectral(iq)
        result=cfg.read(SAVED/(tag+'_ORTH_SAFE.json'))['result']
        x=np.asarray(result['final']['x']);paths=result['paths'];e=stable.Engine()
        h,inv,s0,residual,*_=e.core(x,freq,y,paths)
        g=np.ones((4,2),complex);g[1:]=(x[4:10]+1j*x[10:16]).reshape(3,2)
        c=np.zeros(2)
        for k,(m,q) in enumerate(paths):
            c[q]+=abs(g[m,q])*np.linalg.norm(x[16+3*k:18+3*k])
        start=time.perf_counter();s,info=calibrate(h,y,c,decision['radius2'])
        row=dict(tag=tag,condition=decision['condition'],seconds=time.perf_counter()-start,
            old_penalty=float(c@np.linalg.norm(s0,axis=0)),new_penalty=info['value'],
            old_rss=float(np.linalg.norm(residual)**2),radius2=decision['radius2'],
            source_norms=np.linalg.norm(s,axis=0),conditional=info,positions_changed=False)
        rows.append(row);cfg.write(out/(tag+'.json'),row);cfg.write(out/'rows.json',rows)
        print('END',tag,info['status'],row['seconds'],flush=True)
    stable.inside.rr.check_hashes(hashes)
    statuses={k:sum(r['conditional']['status']==k for r in rows) for k in sorted({r['conditional']['status'] for r in rows})}
    cfg.write(out/'summary.json',dict(completed=len(rows),statuses=statuses,
        all_feasible_numerical_pass=all(r['conditional']['passed'] for r in rows if r['conditional']['status']!='infeasible_fixed_channel'),
        elapsed_s=time.perf_counter()-started,mean_conditional_s=float(np.mean([r['seconds'] for r in rows])),
        max_conditional_s=max(r['seconds'] for r in rows),new_localization_runs=0,identity_unchanged=True))


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
