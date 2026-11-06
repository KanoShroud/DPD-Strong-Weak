# ruff: noqa: F401
# Preserve imported globals for legacy function cloning and reexports.
"""Approved feasible-terminal acceptance; frozen solver and budget unchanged."""
import copy
import time
import numpy as np
import reference as ref


def finish(x, result, x0, f, y, paths, n):
    start=time.perf_counter()
    result=copy.deepcopy(result)
    before=copy.deepcopy(result)
    initial=ref.project(x0,n)
    ri,_=ref.evaluate(initial,f,y,paths)
    initial_cost=float(.5*ri@ri)
    z=ref.project(x,n)
    r,j=ref.evaluate(z,f,y,paths)
    cost=float(.5*r@r)
    mapping=float(np.max(abs(z-ref.project(z-j.T@r,n))))
    closure=float(np.linalg.norm(z-ref.project(z,n)))
    acceptable=bool(np.isfinite(z).all() and np.isfinite(cost) and closure<1e-7
                    and cost<=initial_cost+1e-10)
    info=dict(before=before,initial_feasible_cost=initial_cost,projected_cost=cost,
              projected_mapping=mapping,projection_change=float(np.linalg.norm(z-x)),
              projected_closure=closure,pg_attempted=False)
    result['calls']+=2
    if acceptable and mapping>1e-7 and result['iterations']<300:
        trial=ref.project(z-j.T@r,n)
        rr,jj=ref.evaluate(trial,f,y,paths)
        value=float(.5*rr@rr)
        mg=float(np.max(abs(trial-ref.project(trial-jj.T@rr,n))))
        cl=float(np.linalg.norm(trial-ref.project(trial,n)))
        accepted=bool(np.isfinite(trial).all() and np.isfinite(value) and value<cost
                      and mg<=1e-7 and cl<1e-7)
        info.update(pg_attempted=True,pg_accepted=accepted,pg_cost=value,pg_mapping=mg)
        result['iterations']+=1;result['calls']+=1
        if accepted:
            z,cost,mapping,closure=trial,value,mg,cl
    stationary=acceptable and mapping<=1e-7
    info['accepted']=stationary
    result.update(x=z,p=z[:4].reshape(2,2)*ref.UNIT,cost=cost,
                  projected_gradient=mapping,closure_error=closure,stationary=stationary,
                  feasible_acceptance=info,seconds=result['seconds']+time.perf_counter()-start)
    return z,result


def fit(x0,f,y,paths,n):
    start=time.perf_counter()
    x,result=ref.fit(x0,f,y,paths,n)
    x,result=finish(x,result,x0,f,y,paths,n)
    result['seconds']=time.perf_counter()-start
    return x,result
