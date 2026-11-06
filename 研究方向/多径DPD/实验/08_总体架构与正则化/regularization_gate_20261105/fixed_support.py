"""Bounded outer mechanism test, not a complete blind localization method."""
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from discrepancy import HERE, INPUT, SAVED, cfg, ad, stable
from outer_precision import value_gradient


def solve(e,x,f,y,paths,radius,n):
    start=time.perf_counter();initial=x.copy();trace=[];evaluations=0
    value,grad,info=value_gradient(e,x,f,y,paths,radius);evaluations+=1
    initial_value=value;alpha=.1/max(np.max(abs(grad)),1e-12)
    status='iteration_limit'
    for iteration in range(101):
        pg=float(np.max(abs(x-e.ref.project(x-grad,n))))
        trace.append(dict(iteration=iteration,value=value,pg=pg,p=x[:4].reshape(2,2)*e.ref.UNIT,
            rss=info['rss'],multiplier=info['multiplier'],seconds=time.perf_counter()-start))
        if pg<=1e-7:status='projected_stationary';break
        if iteration==100:break
        accepted=False
        for backtrack in range(20):
            trial=e.ref.project(x-alpha*grad,n);step=trial-x
            try:
                vt,gt,it=value_gradient(e,trial,f,y,paths,radius);evaluations+=1
                reason=it['status']
            except ValueError as exc:
                vt,gt=None,None;reason=str(exc)
            if vt is not None and grad@step<0 and vt<=value+1e-4*float(grad@step):
                accepted=True;break
            alpha*=.5
        trace[-1].update(backtracks=backtrack,trial_status=reason)
        if not accepted:status='line_search_unresolved';break
        change=gt-grad;curvature=float(step@change)
        alpha=float(np.clip(step@step/curvature if curvature>0 else .1/max(np.max(abs(gt)),1e-12),1e-12,1e12))
        x,value,grad,info=trial,vt,gt,it
    return dict(x=x,p=x[:4].reshape(2,2)*e.ref.UNIT,initial_p=initial[:4].reshape(2,2)*e.ref.UNIT,
        initial_value=initial_value,value=value,status=status,iterations=iteration,evaluations=evaluations,
        seconds=time.perf_counter()-start,pg=pg,trace=trace,feasible=bool(np.max(abs(x-e.ref.project(x,n)))<1e-7),
        conditional=info)


def main():
    out=HERE/'fixed_support_v1';out.mkdir(exist_ok=False)
    assert cfg.read(HERE/'outer_precision_v1/summary.json')['all_passed']
    hashes=cfg.read(HERE/'outer_precision_v1/identity.json');stable.inside.rr.check_hashes(hashes)
    for p in [Path(__file__).resolve(),HERE/'fixed_support_protocol.md']:hashes[str(p)]=cfg.sha(p)
    cfg.write(out/'identity.json',hashes);rows=[]
    for source in cfg.read(HERE/'conditional_v1/rows.json'):
        if source['conditional']['status']!='interior_discrepancy':continue
        tag=source['tag'];print('START',tag,flush=True)
        with np.load(INPUT/'cases'/tag/'iq.npz',allow_pickle=False) as z:iq=z['iq']
        f,y,n,_=ad.b.fc.cal.spectral(iq)
        old=cfg.read(SAVED/(tag+'_ORTH_SAFE.json'))['result'];e=stable.Engine()
        result=solve(e,np.asarray(old['final']['x']),f,y,old['paths'],source['radius2'],n)
        row=dict(tag=tag,condition=source['condition'],paths=old['paths'],result=result,full_pipeline=False)
        rows.append(row);cfg.write(out/(tag+'.json'),row);cfg.write(out/'decisions.json',rows)
        print('END',tag,result['status'],result['seconds'],flush=True)
    # Truth is loaded only after every decision has been finalized.
    evaluated=[]
    for row in rows:
        layout=row['tag'].split('_')[1]
        with np.load(INPUT/'banks'/(layout+'_00.npz'),allow_pickle=False) as z:p=z['p']
        r=row['result'];before=stable.inside.rr.evaluation(np.asarray(r['initial_p']),p);after=stable.inside.rr.evaluation(np.asarray(r['p']),p)
        evaluated.append(dict(tag=row['tag'],condition=row['condition'],before=before,after=after,
            status=r['status'],seconds=r['seconds'],objective_ratio=r['value']/r['initial_value']))
    cfg.write(out/'evaluation.json',evaluated)
    stable.inside.rr.check_hashes(hashes)
    cfg.write(out/'summary.json',dict(completed=len(rows),stationary=sum(r['result']['status']=='projected_stationary' for r in rows),
        statuses={s:sum(r['result']['status']==s for r in rows) for s in sorted({r['result']['status'] for r in rows})},
        mean_refinement_seconds=float(np.mean([r['seconds'] for r in evaluated])),max_refinement_seconds=max(r['seconds'] for r in evaluated),
        before_rmse=float(np.sqrt(np.mean([np.square(r['before']) for r in evaluated]))),
        after_rmse=float(np.sqrt(np.mean([np.square(r['after']) for r in evaluated]))),identity_unchanged=True,
        scope='fixed_support_feasible_subset_only',not_applicable=12,full_blind_runs=0))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
