"""One explicitly approved, verified projected-gradient finishing step."""
import time
import numpy as np
import reference as original


def fit(x0, f, y, paths, n):
    start=time.perf_counter()
    x,result=original.fit(x0,f,y,paths,n)
    if result['stationary'] or result['iterations']>=300:
        return x,result
    before=dict(result)
    r,j=original.evaluate(x,f,y,paths)
    trial=original.project(x-j.T@r,n)
    rr,jj=original.evaluate(trial,f,y,paths)
    value=float(.5*rr@rr)
    mapping=float(np.max(abs(trial-original.project(trial-jj.T@rr,n))))
    feasible=float(np.linalg.norm(trial-original.project(trial,n)))
    accepted=bool(np.isfinite(trial).all() and np.isfinite(value) and feasible<1e-10
                  and value<result['cost'] and mapping<=1e-7)
    info=dict(before=before,accepted=accepted,trial_cost=value,trial_mapping=mapping,
              trial_closure_error=feasible,
              position_change_m=float(np.max(np.linalg.norm((trial[:4]-x[:4]).reshape(2,2)*original.UNIT,axis=1))))
    result['iterations']+=1
    result['calls']+=2
    result['pg_finish']=info
    result['seconds']=time.perf_counter()-start
    if accepted:
        x=trial
        result.update(x=x,p=x[:4].reshape(2,2)*original.UNIT,cost=value,
                      projected_gradient=mapping,stationary=True,closure_error=feasible,
                      message=str(result['message'])+'; verified one-step projected finish')
    return x,result
