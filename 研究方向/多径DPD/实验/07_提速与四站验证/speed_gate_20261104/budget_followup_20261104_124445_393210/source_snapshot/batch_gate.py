"""Full blind grouped-candidate diagnostic, no truth inputs to estimator."""
from datetime import datetime
import time
import numpy as np
from threadpoolctl import threadpool_limits
import fast_core as fc
import run_gate as rg
import structured_solver as ss


def solve(iq):
    start=time.perf_counter();engine,ref=fc.engine,fc.ref
    extracted,_,_=engine.extract(iq)
    lags=np.array([sorted(np.array(extracted['CC_ABS'][f'{m}_{nn}']['lag_s'][:6])*ref.C) for m,nn in engine.PAIRS])
    pool=engine.triple_candidates(lags);points,_,reward,_,_=engine.geometry_fit(pool,lags)
    ids,_=engine.joint_select(reward);initial=points[ids]
    f,y,n,sigma2=fc.cal.spectral(iq);use=abs(np.fft.fftfreq(n,1/ref.FS))<=ref.B/2
    gains=np.column_stack([engine.rank_one(y,f,p)[0] for p in initial])
    x0=np.r_[initial.ravel()/ref.UNIT,gains[1:].real.ravel(),gains[1:].imag.ravel()]
    x,state=engine.los_fit(x0,f,y,n);los=state;paths=[];history=[]
    stop='capacity'
    while len(paths)<64:
        step=len(history)+1
        pick,bank,meta=fc.search(x,paths,f,y,n,use,sigma2,step)
        if pick is None:stop='no_detected_increment';break
        selected=[]
        for m in range(5):
            for q in range(2):
                ids=np.flatnonzero((bank[:,0]==m)&(bank[:,1]==q)&(bank[:,3]==1)&(bank[:,4]>meta['threshold']))
                if len(ids):selected.append(bank[ids[np.argmax(bank[ids,4])]])
        selected=sorted(selected,key=lambda a:-a[4])[:64-len(paths)]
        nextpaths=paths+[(int(a[0]),int(a[1])) for a in selected]
        xx=np.r_[x,np.array([[a[5],a[6],a[2]/ref.FS*ref.B] for a in selected]).ravel()]
        # Old feasible model embedded with zero new coefficients is always an option.
        xx=ref.project(xx,n)
        seed_cost,_=fc.objective(xx,f,y,nextpaths)
        if seed_cost>state['cost']:
            xx[len(x):].reshape(-1,3)[:,:2]=0
        xx,new=ss.fit(xx,f,y,nextpaths,n)
        history.append(dict(step=step,added=len(selected),K=len(nextpaths),search=meta,fit=new,seconds_so_far=time.perf_counter()-start))
        if new['cost']>state['cost']+1e-10:stop='numerical_non_decrease';break
        x,paths,state=xx,nextpaths,new
        if not state['stationary']:stop='numerical_nonstationary';break
        if time.perf_counter()-start>30:stop='development_time_gate_30s';break
    return dict(initial=initial,los=los,final=state,paths=paths,history=history,stop=stop,seconds=time.perf_counter()-start)


def main():
    tags=['L0_R00','L2_R00','L3_R00','HIGH_R00']
    out=fc.HERE/('grouped_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir();print('输出',out,flush=True)
    hashes=rg.preflight(out,tags);rows=[]
    contract=fc.load(rg.FORMAL/'contract.json')['contract']
    try:
        for tag in tags:
            iq=rg.inputs(tag)
            # Single inner paired point, from data-selected first increment, not full performance.
            old=fc.load(rg.FORMAL/'cases'/tag/'solve.json')
            if old['history'][0].get('pick'):
                a=old['history'][0]['pick'];paths=[(a['m'],a['q'])]
                x0=np.r_[old['los']['x'],a['rho'],a['delay']]
                f,y,n,_=fc.cal.spectral(iq)
                _,inner=ss.fit(x0,f,y,paths,n)
                fc.write(out/(tag+'_inner.json'),inner)
            print('START',tag,flush=True)
            result=solve(iq);fc.write(out/(tag+'.json'),result)
            with np.load(contract['conditions'][tag.split('_')[0]]['bank'],allow_pickle=False) as z:truth=z['p']
            errors=fc.ref.error(result['final']['p'],truth)
            row=dict(case=tag,seconds=result['seconds'],K=len(result['paths']),stop=result['stop'],J=result['final']['cost'],PG=result['final']['projected_gradient'],rmse_m=float(np.sqrt(np.mean(np.asarray(errors)**2))),errors_m=errors)
            rows.append(row);fc.write(out/'result.json',dict(rows=rows,complete=False));print(row,flush=True)
        fc.write(out/'result.json',dict(rows=rows,complete=True))
    finally:
        unchanged=all(fc.ref.sha(p)==h for p,h in hashes.items());fc.write(out/'postflight.json',dict(unchanged=unchanged));assert unchanged


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
