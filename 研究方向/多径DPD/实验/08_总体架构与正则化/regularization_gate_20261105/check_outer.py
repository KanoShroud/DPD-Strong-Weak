"""All applicable real saved states: derivatives plus one feasible descent."""
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from discrepancy import HERE, INPUT, SAVED, cfg, ad, stable
from outer import value_gradient


def main():
    out=HERE/'outer_v1';out.mkdir(exist_ok=False)
    stable.inside.rr.check_hashes(cfg.read(HERE/'conditional_v1/identity.json'))
    hashes=ad.source_hashes()
    for p in [Path(__file__).resolve(),HERE/'outer.py',HERE/'conditional.py',HERE/'outer_protocol.md']:
        hashes[str(p)]=cfg.sha(p)
    hashes.update(cfg.read(HERE/'discrepancy_v1/identity.json'))
    cfg.write(out/'identity.json',hashes)
    rng=np.random.default_rng(2026110521);rows=[]
    for source in cfg.read(HERE/'conditional_v1/rows.json'):
        if source['conditional']['status']!='interior_discrepancy':continue
        tag=source['tag'];print('START',tag,flush=True)
        with np.load(INPUT/'cases'/tag/'iq.npz',allow_pickle=False) as z:iq=z['iq']
        f,y,n,_=ad.b.fc.cal.spectral(iq)
        r=cfg.read(SAVED/(tag+'_ORTH_SAFE.json'))['result'];x=np.asarray(r['final']['x']);paths=r['paths']
        e=stable.Engine();row=dict(tag=tag,tests=[])
        try:
            value,grad,info=value_gradient(e,x,f,y,paths,source['radius2'])
            if grad is None:raise ValueError(info['status'])
            groups={'position':np.arange(4),'gain':np.arange(4,16),
                'reflection':np.array([16+3*k+j for k in range(len(paths)) for j in [0,1]]),
                'delay':np.arange(18,len(x),3)}
            for group,indices in groups.items():
                direction=np.zeros_like(x);direction[indices]=rng.normal(size=len(indices));direction/=np.linalg.norm(direction)
                predicted=float(grad@direction)
                for step in [1e-3,3e-4]:
                    vp,_,ip=value_gradient(e,x+step*direction,f,y,paths,source['radius2'],False)
                    vm,_,im=value_gradient(e,x-step*direction,f,y,paths,source['radius2'],False)
                    fd=None if vp is None or vm is None else (vp-vm)/(2*step)
                    passed=fd is not None and abs(fd-predicted)<=1e-5+.02*max(abs(fd),abs(predicted))
                    row['tests'].append(dict(group=group,step=step,prediction=predicted,finite_difference=fd,passed=passed,
                        plus_status=ip['status'],minus_status=im['status']))
            alpha=.1/max(np.max(abs(grad)),1e-12);line=[]
            for i in range(20):
                trial=e.ref.project(x-alpha*grad,n);direction=trial-x
                vt,_,it=value_gradient(e,trial,f,y,paths,source['radius2'],False)
                passed=vt is not None and grad@direction<0 and vt<=value+1e-4*float(grad@direction)
                line.append(dict(alpha=alpha,value=vt,status=it['status'],passed=bool(passed)))
                if passed:break
                alpha*=.5
            row.update(value=value,gradient_max=float(np.max(abs(grad))),line=line,descent_passed=bool(passed),
                derivative_passed=all(t['passed'] for t in row['tests']))
        except ValueError as exc:
            row.update(error=str(exc),descent_passed=False,derivative_passed=False)
        rows.append(row);cfg.write(out/(tag+'.json'),row);cfg.write(out/'rows.json',rows)
        print('END',tag,row['derivative_passed'],row['descent_passed'],flush=True)
    stable.inside.rr.check_hashes(hashes)
    cfg.write(out/'summary.json',dict(completed=len(rows),derivative_passed=sum(r['derivative_passed'] for r in rows),
        descent_passed=sum(r['descent_passed'] for r in rows),all_passed=all(r['descent_passed'] and r['derivative_passed'] for r in rows),
        identity_unchanged=True,complete_localization_runs=0))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
