"""Conventional capped-error geometry and joint coverage diagnostic."""
from datetime import datetime
import json
import time

import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from geometry_association_gate import ROOT, C, B, PAIRS, sha, write
from robust_geometry import predicted, select, evaluate, triple_candidates

H=C/B
SOURCE=ROOT/'independent_backend_20261029_222111_026721'


def quantities(p,lags):
    pred,jac=predicted(p)
    delta=pred[:,:,None]-lags[None]
    nearest=np.argmin(abs(delta),axis=2)
    r=np.take_along_axis(delta,nearest[:,:,None],axis=2)[:,:,0]
    active=abs(r)<H
    cost=np.minimum(r*r,H*H).sum(axis=1)
    return r,jac,nearest,active,cost


def fit(pool,lags):
    p=pool.copy()
    moving=np.ones(len(p),bool)
    status=np.full(len(p),'iteration_cap',dtype='<U24')
    iterations=np.zeros(len(p),int)
    for it in range(100):
        ids=np.flatnonzero(moving)
        if not len(ids):
            break
        r,j,_,active,cost=quantities(p[ids],lags)
        jr=j*active[:,:,None]
        grad=np.einsum('nki,nk->ni',jr,r)
        hess=np.einsum('nki,nkj->nij',jr,jr)+1e-6*np.eye(2)
        step=-np.linalg.solve(hess,grad[:,:,None])[:,:,0]
        stable=np.max(abs(grad),axis=1)<1e-6
        accepted=stable.copy()
        nextp=p[ids].copy()
        for k in range(20):
            todo=~accepted
            if not np.any(todo):
                break
            trial=np.clip(p[ids[todo]]+step[todo]*(.5**k),-3000,3000)
            ok=quantities(trial,lags)[-1]<cost[todo]
            local=np.flatnonzero(todo)[ok]
            nextp[local]=trial[ok]
            accepted[local]=True
        small=np.linalg.norm(nextp-p[ids],axis=1)<1e-4
        status[ids[stable]]='gradient'
        status[ids[~accepted]]='line_search_stall'
        status[ids[accepted & ~stable & small]]='step'
        moving[ids[stable | ~accepted | small]]=False
        assert np.all(quantities(nextp,lags)[-1]<=cost+1e-8)
        p[ids]=nextp
        iterations[ids]=it+1
    r,_,near,active,cost=quantities(p,lags)
    reward=np.zeros((len(p),*lags.shape))
    np.put_along_axis(reward,near[:,:,None],np.maximum(0,1-r*r/(H*H))[:,:,None],axis=2)
    return p,cost,reward.reshape(len(p),-1),status,iterations


def joint_select(reward):
    best=-np.inf
    chosen=None
    for i in range(len(reward)-1):
        value=np.maximum(reward[i],reward[i+1:]).sum(axis=1)
        j=int(np.argmax(value))+i+1
        if value[j-i-1]>best:
            best=float(value[j-i-1])
            chosen=[i,j]
    return chosen,best


def selftest():
    truth=np.array([[700.,500.],[-900.,-650.]])
    lags=predicted(truth)[0].T
    pool=triple_candidates(lags)
    pp,cost,reward,_,_=fit(pool,lags)
    ids,_=joint_select(reward)
    checks={'exact_two':evaluate(pp[ids],truth)['rmse_m']<1e-4}
    example=np.array([[1.,1.,0.],[.9,.9,0.],[0.,0.,.8]])
    checks['no_double_reward']=joint_select(example)[0]==[0,2]
    xx=truth+[[1.,2.],[2.,1.]]
    r,j,_,act,_=quantities(xx,lags)
    grad=2*np.einsum('nki,nk->ni',j*act[:,:,None],r)
    fd=np.column_stack([(quantities(xx+np.eye(2)[i]*1e-3,lags)[-1]-quantities(xx-np.eye(2)[i]*1e-3,lags)[-1])/2e-3 for i in range(2)])
    checks['gradient_away_kinks']=np.max(abs(fd-grad))<1e-5
    checks['bounded_outlier']=np.all(quantities(xx,lags+1e6)[-1]==10*H*H)
    checks['capped_no_outlier_equals_ls']=np.allclose(quantities(xx,lags)[-1],np.sum(quantities(xx,lags)[0]**2,axis=1))
    assert all(checks.values()),checks
    return checks


def main():
    source=json.loads((SOURCE/'result.json').read_text(encoding='utf-8'))
    files=[ROOT/'bounded_consensus.py',ROOT/'有界共识诊断协议.md',ROOT/'robust_geometry.py',
           ROOT/'geometry_association_gate.py',SOURCE/'result.json',SOURCE/'protocol.json']
    files+=sorted(SOURCE.glob('*_geometry.npz'))+sorted(SOURCE.glob('*_input.npz'))
    hashes={str(p.resolve(strict=True)):sha(p) for p in files}
    out=ROOT/('bounded_consensus_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    (out/'source').mkdir()
    for p in files[:4]:
        (out/'source'/p.name).write_bytes(p.read_bytes())
    write(out/'protocol.json',dict(hashes=hashes,posthoc=True,threshold_m=H,threshold_origin='c/B not truth'))
    checks=selftest()
    rows=[]
    start=time.perf_counter()
    for row in source['rows']:
        tag=row['tag']
        truth=np.load(SOURCE/(tag.rsplit('_',1)[0]+'_input.npz'))['p']
        pool=np.load(SOURCE/(tag+'_geometry.npz'))['initial_candidates']
        lags=np.array([sorted(np.array(row['peaks'][f'{m}_{n}']['lag_s'][:6])*C) for m,n in PAIRS])
        begin=time.perf_counter()
        p,cost,reward,status,nit=fit(pool,lags)
        single=select(p,cost,2)
        ids,value=joint_select(reward)
        rec=dict(tag=tag,path=row['path'],layout=row['layout'],rep=row['rep'],
             independent=evaluate(p[single],truth),joint=evaluate(p[ids],truth),
             p=p[ids],independent_p=p[single],joint_reward=value,
             coverage_m=cdist(p,truth).min(axis=0),seconds=time.perf_counter()-begin)
        checks[tag+'/finite']=bool(np.isfinite(p).all() and np.isfinite(reward).all())
        rows.append(rec)
        write(out/(tag+'.json'),rec)
        np.savez_compressed(out/(tag+'.npz'),p=p,cost=cost,reward=reward,chosen=ids,status=status,iterations=nit)
        print(tag,'NMS',np.round(rec['independent']['errors_m'],2),'JOINT',np.round(rec['joint']['errors_m'],2),flush=True)
    checks.update({p:sha(p)==h for p,h in hashes.items()})
    assert all(checks.values())
    write(out/'result.json',dict(rows=rows,checks=checks,seconds=time.perf_counter()-start))
    print('COMPLETE',out,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
