"""Interior QN, existing full constrained SLSQP on boundary; no nested QP loop."""
import sys
import time
import types

import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits
import solver as s
import run as runner

ref, fc = s.ref, s.fc
scope = {**fc.fit_base.__globals__, 'evaluate': s.evaluate, 'objective': s.objective}
base_fit = types.FunctionType(fc.fit_base.__code__, scope, argdefs=fc.fit_base.__defaults__)
scope['fit_base'] = base_fit
fast_ref = types.SimpleNamespace(project=ref.project, evaluate=s.evaluate, UNIT=ref.UNIT)
finish = types.FunctionType(fc.finish.__code__, {**fc.finish.__globals__, 'ref': fast_ref})


def fit(x0, f, y, paths, n, method='HYBRID'):
    start = time.perf_counter(); x = ref.project(x0, n)
    value, g = s.objective(x, f, y, paths)
    calls, jc, hist = 1, 0, []
    hessian = None; reason = 'iteration_limit_failure'
    switch = None
    for it in range(300):
        pg = float(np.max(abs(x-ref.project(x-g, n))))
        if pg <= 1e-7:
            reason = 'stationary'; break
        if hessian is None:
            r, j = s.evaluate(x, f, y, paths); jc += 1
            scale = np.maximum(np.linalg.norm(j, axis=0), 1e-8)
            a = j/scale; hessian = a.T@a+1e-3*np.eye(len(x))
        direction = solve(hessian, -g/scale, assume_a='pos')/scale
        if np.linalg.norm(x+direction-ref.project(x+direction, n)) > 1e-10:
            switch = 'constraint_active'; break
        moved = False
        for k in range(20):
            trial = ref.project(x+(.5**k)*direction, n)
            step = trial-x; slope = float(g@step)
            try:
                nv, ng = s.objective(trial, f, y, paths)
            except ValueError as exc:
                if str(exc) != 'station coincidence': raise
                nv, ng = np.inf, g
            calls += 1
            if slope < 0 and nv < value and nv <= value+1e-4*slope:
                dz, dy = step*scale, (ng-g)/scale
                hd = hessian@dz; shs, sy = float(dz@hd), float(dz@dy)
                if shs > 1e-24:
                    w = 1. if sy >= .2*shs else .8*shs/(shs-sy)
                    yy = w*dy+(1-w)*hd
                    hessian += np.outer(yy, yy)/float(dz@yy)-np.outer(hd, hd)/shs
                    hessian = (hessian+hessian.T)/2
                    ev = np.linalg.eigvalsh(hessian)
                    if ev[0] <= ev[-1]*1e-12: hessian = None
                else:
                    hessian = None
                hist.append(dict(iteration=it, cost=value, PG=pg, trials=k+1))
                x, value, g = trial, nv, ng; moved = True; break
        if not moved:
            switch = 'line_search_stall'; break
    if switch is not None:
        x, state = base_fit(x, f, y, paths, n, budget=300-len(hist))
        state['iterations'] += len(hist)
        state['calls'] += calls
        x, state = finish(x, state, x0, f, y, paths, n)
        state.update(reason='stationary' if state['stationary'] else 'constrained_solver_failure',
                     hybrid_switch=switch, qn_history=hist, jacobian_calls=jc,
                     seconds=time.perf_counter()-start, solver='HYBRID')
        return x, state
    value, g = s.objective(x, f, y, paths)
    pg = float(np.max(abs(x-ref.project(x-g, n))))
    return x, dict(x=x, p=x[:4].reshape(2, 2)*ref.UNIT, cost=value,
        projected_gradient=pg, stationary=pg <= 1e-7, reason=reason,
        iterations=len(hist), calls=calls+1, jacobian_calls=jc, history=hist,
        seconds=time.perf_counter()-start, closure_error=float(np.linalg.norm(x-ref.project(x, n))),
        hybrid_switch=None, solver='HYBRID')


if __name__ == '__main__':
    # Run-local substitution only; original and earlier candidate files unchanged.
    s.estimate = types.FunctionType(s.estimate.__code__, {**s.estimate.__globals__, 'fit': fit}, argdefs=s.estimate.__defaults__)
    if '--methods' not in sys.argv:
        sys.argv += ['--methods', 'HYBRID']
    with threadpool_limits(limits=2):
        runner.main()
