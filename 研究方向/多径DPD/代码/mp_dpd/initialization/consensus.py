# ruff: noqa: F401
# Preserve imported globals for legacy function cloning and reexports.
"""Conventional capped-error geometry and joint coverage diagnostic."""
from datetime import datetime
import json
import time

import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from 研究方向.多径DPD.代码.mp_dpd.initialization.tdoa_candidates import ROOT, C, B, PAIRS, sha, write
from 研究方向.多径DPD.代码.mp_dpd.initialization.robust_geometry import predicted, select, evaluate, triple_candidates

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


# Historical main retained only in the evidence archive.


# Historical batch entry disabled; use the public mp_dpd API.
