"""Observe a zero-regularizer plateau without using truth to make proposals."""
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from discrepancy import HERE, INPUT, SAVED, cfg, ad, stable


def main():
    out=HERE/'zero_plateau_v1';out.mkdir(exist_ok=False)
    hashes=cfg.read(HERE/'discrepancy_v1/identity.json');stable.inside.rr.check_hashes(hashes)
    for p in [Path(__file__).resolve(),HERE/'zero_plateau_protocol.md']:hashes[str(p)]=cfg.sha(p)
    cfg.write(out/'identity.json',hashes);rows=[]
    for d in cfg.read(HERE/'discrepancy_v1/blind_decisions.json'):
        if d['condition']!='L0':continue
        tag=d['tag'];r=cfg.read(SAVED/(tag+'_ORTH_SAFE.json'))['result'];x=np.asarray(r['los']['x'])
        with np.load(INPUT/'cases'/tag/'iq.npz',allow_pickle=False) as z:iq=z['iq']
        f,y,n,_=ad.b.fc.cal.spectral(iq);e=stable.Engine()
        for index in range(4):
            for shift in [-10.,-3.,-1.,-.1,.1,1.,3.,10.]:
                t=x.copy();t[index]+=shift/e.ref.UNIT
                _,_,_,res,*_=e.core(t,f,y,[])
                ratio=float(np.linalg.norm(res)**2/d['radius2'])
                rows.append(dict(tag=tag,index=index,shift_m=shift,rss_ratio=ratio,
                    feasible=bool(ratio<=1 and np.max(abs(t-e.ref.project(t,n)))<1e-7),
                    penalty=0.,p=t[:4].reshape(2,2)*e.ref.UNIT))
    cfg.write(out/'decisions.json',rows)
    evaluated=[]
    for r in rows:
        with np.load(INPUT/'banks'/(r['tag'].split('_')[1]+'_00.npz'),allow_pickle=False) as z:p=z['p']
        evaluated.append(dict(**r,errors=stable.inside.rr.evaluation(np.asarray(r['p']),p)))
    cfg.write(out/'evaluation.json',evaluated)
    stable.inside.rr.check_hashes(hashes)
    feasible=[r for r in evaluated if r['feasible']]
    cfg.write(out/'summary.json',dict(completed=len(rows),feasible_zero_global_optima=len(feasible),
        layouts_with_distinct_optima=len({r['tag'] for r in feasible}),
        max_sampled_feasible_shift_m=max(abs(r['shift_m']) for r in feasible),
        max_sampled_feasible_rmse_m=max(float(np.sqrt(np.mean(np.square(r['errors'])))) for r in feasible),
        identity_unchanged=True,diagnostic_only=True))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
