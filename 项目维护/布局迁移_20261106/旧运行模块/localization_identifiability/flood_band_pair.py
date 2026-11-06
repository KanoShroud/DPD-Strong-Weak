# ruff: noqa: F401
# Preserve imported globals for legacy function cloning and reexports.
"""Flood Eq.(5), zero-entropy LP reference; not the authors' entropy solver."""
from datetime import datetime
import argparse
import json
import time
import warnings

import numpy as np
from scipy.optimize import linprog, least_squares
from scipy.sparse import coo_matrix, vstack
from threadpoolctl import threadpool_limits

from geometry_association_gate import ROOT, C, B, PAIRS, sha, write, signature
from robust_geometry import predicted, evaluate
from blind_backend_pilot import solve, errors
from flood_lp_check import solve_lp

DEV=ROOT/'consensus_development_20261029_223235_334946'
VAL=ROOT/'consensus_validation_20261029_223520_495993'
INPUT=ROOT/'independent_backend_20261029_222111_026721'


def pruned_lp(cost,trash=1.,eta=1.,capacity=10.,known_k=None):
    n,j=cost.shape
    ii,jj=np.where(cost<trash)
    count=len(ii);total=count+n+j
    eqrow=np.r_[ii,np.arange(n)]
    eqcol=np.r_[np.arange(count),count+np.arange(n)]
    aeq=coo_matrix((np.ones(count+n),(eqrow,eqcol)),shape=(n,total)).tocsr()
    beq=np.ones(n)
    if known_k is not None:
        activation=coo_matrix((np.ones(j),(np.zeros(j,dtype=int),count+n+np.arange(j))),shape=(1,total)).tocsr()
        aeq=vstack([aeq,activation],format='csr')
        beq=np.r_[beq,known_k]
    acol=coo_matrix((np.ones(count),(jj,np.arange(count))),shape=(j,total)).tocsr()
    link=coo_matrix((np.r_[np.ones(count),-np.ones(count)],
         (np.tile(np.arange(count),2),np.r_[np.arange(count),count+n+jj])),shape=(count,total)).tocsr()
    obj=np.r_[cost[ii,jj],np.full(n,trash),np.full(j,eta)]
    begin=time.perf_counter()
    res=linprog(obj,A_ub=vstack([acol,link],format='csr'),b_ub=np.r_[np.full(j,capacity),np.zeros(count)],
                A_eq=aeq,b_eq=beq,bounds=(0,1),method='highs',
                options={'dual_feasibility_tolerance':1e-8,'primal_feasibility_tolerance':1e-8})
    if not res.success:
        raise RuntimeError(res.message)
    m=np.zeros((n,j));m[ii,jj]=res.x[:count]
    t=res.x[count:count+n];q=res.x[count+n:]
    dual=float(res.eqlin.marginals@beq+res.ineqlin.marginals[:j].sum()*capacity+res.upper.marginals.sum())
    checks=dict(row=float(max(abs(m.sum(1)+t-1))),column=float(max(0,max(m.sum(0)-capacity))),
        epigraph=float(max(0,np.max(m-q[None]))),gap=abs(res.fun-dual)/max(1,abs(res.fun)))
    if known_k is not None:
        checks['known_k']=abs(q.sum()-known_k)
    assert max(checks.values())<1e-7,checks
    return m,t,q,dict(objective=float(res.fun),dual=dual,checks=checks,
          iterations=res.nit,seconds=time.perf_counter()-begin,variables=total,full_variables=n*j+n+j)


def reestimate(cost,trash=1.,capacity=10.):
    # No activation penalty after selecting two candidate locations, as in author reestimate routine.
    m,t,q,meta=pruned_lp(cost,trash,0.,capacity)
    return m,t,meta


def estimate(pool,lags,known_k=None):
    begin=time.perf_counter()
    pred=predicted(pool)[0]
    costs=((pred[:,:,None]-lags[None])/(C/B))**2
    cost=costs.reshape(len(pool),-1).T
    m,t,q,meta=pruned_lp(cost,known_k=known_k)
    code_ids=np.argsort(-m.sum(0),kind='stable')[:2]
    paper_ids=np.argsort(-m.max(0),kind='stable')[:2]
    association,trash,assocmeta=reestimate(cost[:,code_ids])
    refined=[];details=[]
    for col,idx in enumerate(code_ids):
        rows=np.flatnonzero(association[:,col]>.7)
        edge=rows//lags.shape[1];obs=lags.ravel()[rows]
        start=pool[idx]
        if not len(rows):
            refined.append(start);details.append(dict(status='no_associated_measurement',rows=[]))
            continue
        def fun(p):
            return (predicted(p[None])[0][0,edge]-obs)/(C/B)
        def jac(p):
            return predicted(p[None])[1][0,edge]/(C/B)
        result=least_squares(fun,start,jac=jac,bounds=(-3000,3000),
               ftol=1e-10,xtol=1e-10,gtol=1e-10,max_nfev=100)
        refined.append(result.x)
        details.append(dict(status=int(result.status),rows=rows.tolist(),nfev=result.nfev,
            initial_cost=float(fun(start)@fun(start)),final_cost=float(result.fun@result.fun)))
    return np.array(refined),dict(cost=cost,M=m,trash=t,q=q,code_ids=code_ids,paper_ids=paper_ids,
        association=association,association_trash=trash,lp=meta,reestimate=assocmeta,
        refinement=details,seconds=time.perf_counter()-begin)


# Historical main retained only in the evidence archive.


# Historical batch entry disabled; use the public mp_dpd API.
