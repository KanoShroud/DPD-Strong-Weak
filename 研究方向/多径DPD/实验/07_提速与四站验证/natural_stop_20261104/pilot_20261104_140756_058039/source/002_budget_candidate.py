"""Explicit finite-time grouped VP candidate; nonstationary outputs labelled."""
import argparse
from datetime import datetime
import importlib.util
import time

import numpy as np
from scipy.linalg import solve as linear_solve
from threadpoolctl import threadpool_limits
import fast_core as fc
import run_gate as rg

spec=importlib.util.spec_from_file_location('speed_region',fc.BASE/'failure_attribution_20261104/diagnose.py')
regions=importlib.util.module_from_spec(spec);spec.loader.exec_module(regions)
ref,engine=fc.ref,fc.engine


def fit(x0,f,y,paths,n,deadline):
    x=ref.project(x0,n);start=time.perf_counter();damping=1e-3;calls=0;nit=0;reason='time_budget'
    cost,g=fc.objective(x,f,y,paths);calls+=1
    while time.perf_counter()<deadline-.1 and nit<300:
        r,j=fc.evaluate(x,f,y,paths);calls+=1;nit+=1
        cost=float(.5*r@r);g=j.T@r
        if np.max(abs(x-ref.project(x-g,n)))<=1e-7:reason='stationary';break
        scale=np.maximum(np.linalg.norm(j,axis=0),1e-8)
        a=j/scale;gram=a.T@a;gg=g/scale;moved=False
        for _ in range(9):
            if time.perf_counter()>=deadline-.05:break
            dz=linear_solve(gram+damping*np.eye(len(x)),-gg,assume_a='pos',check_finite=False)
            trial=ref.project(x+dz/scale,n)
            value,newg=fc.objective(trial,f,y,paths);calls+=1
            if np.isfinite(value) and value<cost and value<=cost+1e-4*g@(trial-x):
                x,cost,g=trial,value,newg;damping=max(damping/3,1e-8);moved=True;break
            damping=min(damping*10,1e12)
        if not moved:reason='budget_or_step_stall';break
    if nit==300:reason='iteration_budget'
    value,g=fc.objective(x,f,y,paths);calls+=1
    pg=float(np.max(abs(x-ref.project(x-g,n))))
    return x,dict(x=x,p=x[:4].reshape(2,2)*ref.UNIT,cost=value,projected_gradient=pg,stationary=pg<=1e-7,iterations=nit,calls=calls,seconds=time.perf_counter()-start,closure_error=float(np.linalg.norm(x-ref.project(x,n))),reason=reason)


def estimate(iq,budget,two_regions):
    start=time.perf_counter();deadline=start+budget-.15
    geom=regions.frontend(iq)
    selection=regions.alternate_pair(geom['refined'],geom['reward']) if two_regions else None
    ids=selection['first'] if two_regions else engine.joint_select(geom['reward'])[0]
    f,y,n,sigma=fc.cal.spectral(iq);use=abs(np.fft.fftfreq(n,1/ref.FS))<=ref.B/2
    def los_for(pair):
        p=geom['refined'][pair]
        gains=np.column_stack([engine.rank_one(y,f,z)[0] for z in p])
        xx=np.r_[p.ravel()/ref.UNIT,gains[1:].real.ravel(),gains[1:].imag.ravel()]
        return engine.los_fit(xx,f,y,n)
    x,state=los_for(ids);los_states=[state];selected=1
    if two_regions:
        xx,other=los_for(selection['second']);los_states.append(other)
        if other['cost']<state['cost']:x,state=xx,other;selected=2
    frontend_seconds=time.perf_counter()-start
    initial=state;paths=[];history=[];stop='time_budget'
    while len(paths)<64:
        # Do not start an indivisible broad search if the remaining budget is small.
        if time.perf_counter()+.65>=deadline:break
        _,bank,meta=fc.search(x,paths,f,y,n,use,sigma,len(history)+1)
        choices=[]
        for m in range(5):
            for q in range(2):
                idx=np.flatnonzero((bank[:,0]==m)&(bank[:,1]==q)&(bank[:,3]==1)&(bank[:,4]>meta['threshold']))
                if len(idx):choices.append(bank[idx[np.argmax(bank[idx,4])]])
        if not choices:stop='no_detected_increment';break
        choices=sorted(choices,key=lambda a:-a[4])[:64-len(paths)]
        newpaths=paths+[(int(a[0]),int(a[1])) for a in choices]
        xx=ref.project(np.r_[x,np.array([[a[5],a[6],a[2]/ref.FS*ref.B] for a in choices]).ravel()],n)
        value,_=fc.objective(xx,f,y,newpaths)
        if value>state['cost']:xx[len(x):].reshape(-1,3)[:,:2]=0
        xx,new=fit(xx,f,y,newpaths,n,deadline)
        decrease=bool(new['cost']<=state['cost']+1e-10)
        history.append(dict(K=len(newpaths),fit=new,search=meta,decrease=decrease))
        if not decrease:stop='non_decrease';break
        x,paths,state=xx,newpaths,new
        if not new['stationary']:stop=new['reason'];break
    if len(paths)==64:stop='capacity'
    return dict(final=state,los=initial,los_states=los_states,selected_region=selected,selection=selection,paths=paths,history=history,seconds=time.perf_counter()-start,frontend_seconds=frontend_seconds,stop=stop,budget=budget,two_regions=two_regions)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--holdout',action='store_true');args=ap.parse_args()
    tags=[f'{cond}_R{r:02d}' for r in (range(1,8) if args.holdout else [0]) for cond in ['L0','L2','L3','HIGH']]
    out=fc.HERE/(('budget_followup_' if args.holdout else 'budget_pilot_')+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir();print('输出',out,flush=True)
    hashes=rg.preflight(out,tags);contract=fc.load(rg.FORMAL/'contract.json')['contract'];rows=[]
    configs=[('GEO_3',3.,False),('REGION_3',3.,True),('REGION_5',5.,True)]
    try:
        for i,tag in enumerate(tags):
            iq=rg.inputs(tag);order=configs[i%3:]+configs[:i%3]
            for name,budget,two in order:
                print('START',tag,name,flush=True)
                result=estimate(iq,budget,two);fc.write(out/(tag+'_'+name+'.json'),result)
                with np.load(contract['conditions'][tag.split('_')[0]]['bank'],allow_pickle=False) as z:truth=z['p']
                error=np.asarray(ref.error(result['final']['p'],truth))
                row=dict(case=tag,variant=name,seconds=result['seconds'],rmse_m=float(np.sqrt(np.mean(error**2))),errors_m=error,K=len(result['paths']),J=result['final']['cost'],PG=result['final']['projected_gradient'],stationary=result['final']['stationary'],stop=result['stop'],selected_region=result['selected_region'],feasible=result['final'].get('closure_error',0)<1e-7,monotone=all(h['decrease'] for h in result['history']),budget=budget)
                rows.append(row);fc.write(out/'result.json',dict(rows=rows,complete=False));print(row,flush=True)
        fc.write(out/'result.json',dict(rows=rows,complete=True))
    finally:
        unchanged=all(ref.sha(p)==h for p,h in hashes.items());fc.write(out/'postflight.json',dict(unchanged=unchanged));assert unchanged


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
