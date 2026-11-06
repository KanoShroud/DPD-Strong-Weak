"""Projected damped GN of the same VP residual; candidate, not frozen baseline."""
import time
import numpy as np
from scipy.linalg import solve
import fast_core as fc

ref=fc.ref
search=fc.search


def fit(x0,f,y,paths,n):
    start=time.perf_counter();x=ref.project(x0,n);damping=1e-3;calls=0
    accepted=0;history=[]
    for it in range(40):
        r,j=fc.evaluate(x,f,y,paths);calls+=1
        cost=float(.5*r@r);g=j.T@r
        mapping=float(np.max(abs(x-ref.project(x-g,n))))
        if mapping<=1e-7:break
        scale=np.maximum(np.linalg.norm(j,axis=0),1e-8)
        a=j/scale;gram=a.T@a;gg=g/scale
        moved=False
        for _ in range(9):
            dz=solve(gram+damping*np.eye(len(x)),-gg,assume_a='pos',check_finite=False)
            trial=ref.project(x+dz/scale,n)
            value,_=fc.objective(trial,f,y,paths);calls+=1
            if np.isfinite(value) and value<cost and value<=cost+1e-4*g@(trial-x):
                x=trial;damping=max(damping/3,1e-8);accepted+=1;moved=True;break
            damping=min(damping*10,1e12)
        history.append(dict(iteration=it,cost=cost,PG=mapping,damping=damping,accepted=moved))
        if not moved:break
    cost,g=fc.objective(x,f,y,paths);calls+=1
    mapping=float(np.max(abs(x-ref.project(x-g,n))))
    return x,dict(x=x,p=x[:4].reshape(2,2)*ref.UNIT,cost=cost,projected_gradient=mapping,stationary=mapping<=1e-7,iterations=it+1,calls=calls,seconds=time.perf_counter()-start,closure_error=float(np.linalg.norm(x-ref.project(x,n))),accepted_steps=accepted,solver='projected_GN',history=history)
