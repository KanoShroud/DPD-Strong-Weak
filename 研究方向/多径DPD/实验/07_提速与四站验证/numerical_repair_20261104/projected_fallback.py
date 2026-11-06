"""Single principled fallback: Euclidean projected spectral gradient with Armijo."""
import time
import numpy as np
import stable


class Engine(stable.Engine):
    def __init__(self):
        super().__init__()
        original_fit = self.fit

        def fit(x0, f, y, paths, n, method='GATED'):
            start = time.perf_counter()
            x, state = original_fit(x0, f, y, paths, n, method)
            if state['stationary'] or state['iterations'] >= 300:
                return x, state
            previous = dict(state)
            initial = self.ref.project(x0, n)
            value0, g0 = self.objective(initial, f, y, paths)
            x = self.ref.project(x, n)
            value, g = self.objective(x, f, y, paths)
            extra = 2
            if not np.isfinite(value) or value > value0:
                x, value, g = initial, value0, g0
            assert np.isfinite(value)
            alpha = 1.
            history = []
            reason = 'projected_budget_exhausted'
            used = 0
            for _ in range(300-int(state['iterations'])):
                pg = float(np.max(abs(x-self.ref.project(x-g, n))))
                if pg <= 1e-7:
                    reason = 'projected_stationary'
                    break
                moved = False
                used += 1
                for k in range(20):
                    step_size = alpha*.5**k
                    trial = self.ref.project(x-step_size*g, n)
                    d = trial-x
                    slope = float(g@d)
                    nv, ng = self.objective(trial, f, y, paths)
                    extra += 1
                    if slope < 0 and np.isfinite(nv) and nv < value and nv <= value+1e-4*slope:
                        # The projection inequality is a numerical correctness check.
                        slack = float(slope+float(d@d)/step_size)
                        assert slack <= 1e-8*max(1., abs(slope))
                        history.append(dict(cost_before=value, cost_after=nv, pg=pg,
                                            projection_slack=slack, trials=k+1))
                        dg = ng-g; sy = float(d@dg)
                        alpha = float(np.clip((d@d)/sy, 1e-12, 1e12)) if sy > 0 else 1.
                        x, value, g = trial, nv, ng
                        moved = True
                        break
                if not moved:
                    reason = 'projected_line_search_stall'
                    break
            r, j = self.evaluate(x, f, y, paths)
            extra += 1
            pg = float(np.max(abs(x-self.ref.project(x-j.T@r, n))))
            feasible = np.linalg.norm(x-self.ref.project(x, n)) < 1e-7
            state = dict(state)
            state.update(x=x, p=x[:4].reshape(2, 2)*stable.ad.b.ref.UNIT, cost=float(.5*r@r),
                         projected_gradient=pg, stationary=bool(feasible and pg <= 1e-7 and .5*r@r <= value0+1e-10),
                         reason=reason, closure_error=float(np.linalg.norm(x-self.ref.project(x, n))),
                         iterations=previous['iterations']+used, calls=previous['calls']+extra,
                         seconds=time.perf_counter()-start, projected_fallback_history=history,
                         before_projected_fallback=previous)
            assert state['iterations'] <= 300
            return x, state

        self.fit = fit
        self.estimate_function.__globals__['fit'] = fit
