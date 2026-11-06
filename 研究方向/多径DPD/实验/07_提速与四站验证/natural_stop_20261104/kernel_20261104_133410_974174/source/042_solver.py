"""Same VP objective, guarded two-column inverse, natural numerical termination."""
from pathlib import Path
import sys
import time
import types

import numpy as np
from scipy.linalg import solve as linear_solve

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'speed_gate_20261104'))
import fast_core as fc  # noqa: E402
import run_gate as rg  # noqa: E402
import budget_candidate as bc  # noqa: E402

ref = fc.ref


class GuardedLinalg:
    def pinv(self, h, rcond=1e-12):
        # Only 5x2 full rank matrices; guard condition before normal equations.
        hh = h.conj().transpose(0, 2, 1)
        gram = hh@h
        a, b, d = gram[:, 0, 0].real, gram[:, 0, 1], gram[:, 1, 1].real
        det = a*d-abs(b)**2
        good = det > 1e-4*(a+d)**2
        inv = np.empty_like(hh)
        inv[good, 0] = (d[good, None]*hh[good, 0]-b[good, None]*hh[good, 1])/det[good, None]
        inv[good, 1] = (-b[good, None].conj()*hh[good, 0]+a[good, None]*hh[good, 1])/det[good, None]
        if np.any(~good):
            inv[~good] = np.linalg.pinv(h[~good], rcond=rcond)
        return inv

    def __getattr__(self, name):
        return getattr(np.linalg, name)


class GuardedNumpy:
    linalg = GuardedLinalg()

    def __getattr__(self, name):
        return getattr(np, name)


core = types.FunctionType(fc.core.__code__, {**fc.core.__globals__, 'np': GuardedNumpy()})
objective = types.FunctionType(fc.objective.__code__, {**fc.objective.__globals__, 'core': core})
evaluate = types.FunctionType(fc.evaluate.__code__, {**fc.evaluate.__globals__, 'core': core}, argdefs=fc.evaluate.__defaults__)
search = types.FunctionType(ref.search.__code__, {**ref.search.__globals__, 'evaluate': evaluate})


def fit(x0, f, y, paths, n, method='QN'):
    start = time.perf_counter()
    x = ref.project(x0, n)
    value, g = objective(x, f, y, paths)
    calls, jc, damping, history = 1, 0, 1e-3, []
    inverse = None
    reason = 'iteration_limit_failure'
    for it in range(300):
        pg = float(np.max(abs(x-ref.project(x-g, n))))
        if pg <= 1e-7:
            reason = 'stationary'
            break
        if method == 'GN' or inverse is None:
            r, j = evaluate(x, f, y, paths)
            jc += 1
            scale = np.maximum(np.linalg.norm(j, axis=0), 1e-8)
            a = j/scale
            gram = a.T@a
            if method == 'QN':
                inverse = linear_solve(gram+1e-3*np.eye(len(x)), np.eye(len(x)), assume_a='pos')/scale[:, None]/scale[None, :]
        moved = False
        for trial_index in range(20):
            if method == 'GN':
                direction = linear_solve(gram+damping*np.eye(len(x)), -g/scale, assume_a='pos')/scale
                candidate = ref.project(x+direction, n)
            else:
                candidate = ref.project(x-(.5**trial_index)*(inverse@g), n)
            step = candidate-x
            slope = float(g@step)
            if slope >= 0:
                candidate = ref.project(x-(.5**trial_index)*g, n)
                step = candidate-x
                slope = float(g@step)
            nv, ng = objective(candidate, f, y, paths)
            calls += 1
            if np.isfinite(nv) and slope < 0 and nv < value and nv <= value+1e-4*slope:
                if method == 'QN':
                    delta = ng-g
                    curvature = float(step@delta)
                    if curvature > 1e-12*np.linalg.norm(step)*np.linalg.norm(delta):
                        hd = inverse@delta
                        inverse += ((curvature+delta@hd)/curvature**2)*np.outer(step, step)-(np.outer(hd, step)+np.outer(step, hd))/curvature
                    else:
                        inverse = None
                history.append(dict(iteration=it, cost=value, PG=pg, step_position_m=float(np.max(abs(step[:4]))*ref.UNIT), trials=trial_index+1))
                x, value, g = candidate, nv, ng
                damping = max(damping/3, 1e-8)
                moved = True
                break
            if method == 'GN':
                damping = min(damping*10, 1e12)
        if not moved:
            reason = 'line_search_failure'
            break
    value, g = objective(x, f, y, paths)
    pg = float(np.max(abs(x-ref.project(x-g, n))))
    if pg <= 1e-7:
        reason = 'stationary'
    return x, dict(x=x, p=x[:4].reshape(2, 2)*ref.UNIT, cost=value,
        projected_gradient=pg, stationary=pg <= 1e-7, reason=reason,
        iterations=len(history), calls=calls+1, jacobian_calls=jc, history=history,
        seconds=time.perf_counter()-start, closure_error=float(np.linalg.norm(x-ref.project(x, n))))


def estimate(iq, method='QN', two_regions=False):
    start = time.perf_counter()
    geom = bc.regions.frontend(iq)
    f, y, n, sigma = fc.cal.spectral(iq)
    use = abs(np.fft.fftfreq(n, 1/ref.FS)) <= ref.B/2
    selection = bc.regions.alternate_pair(geom['refined'], geom['reward']) if two_regions else None
    pairs = [selection['first'], selection['second']] if two_regions else [fc.engine.joint_select(geom['reward'])[0]]
    states = []
    for ids in pairs:
        p = geom['refined'][ids]
        gains = np.column_stack([fc.engine.rank_one(y, f, z)[0] for z in p])
        xx = np.r_[p.ravel()/ref.UNIT, gains[1:].real.ravel(), gains[1:].imag.ravel()]
        # Original LOS front retained for attribution; failures explicitly recorded.
        _, state = fc.engine.los_fit(xx, f, y, n)
        states.append(state)
    state = min(states, key=lambda a: a['cost'])
    x = np.array(state['x'])
    los = state
    front_seconds = time.perf_counter()-start
    history, paths = [], []
    stop = 'capacity_failure'
    while len(paths) < 64:
        if not state['stationary']:
            stop = 'numerical_nonstationary'
            break
        _, bank, meta = search(x, paths, f, y, n, use, sigma, len(history)+1)
        if not all(meta['checks'].values()):
            raise ArithmeticError('FFT search self-check failed')
        choices = []
        for m in range(5):
            for q in range(2):
                ids = np.flatnonzero((bank[:, 0] == m)&(bank[:, 1] == q)&(bank[:, 3] == 1)&(bank[:, 4] > meta['threshold']))
                if len(ids):
                    choices.append(bank[ids[np.argmax(bank[ids, 4])]])
        if not choices:
            history.append(dict(search=meta, added=0))
            stop = 'no_detected_increment'
            break
        choices = sorted(choices, key=lambda a: -a[4])[:64-len(paths)]
        pp = paths+[(int(a[0]), int(a[1])) for a in choices]
        xx = ref.project(np.r_[x, np.array([[a[5], a[6], a[2]/ref.FS*ref.B] for a in choices]).ravel()], n)
        if objective(xx, f, y, pp)[0] > state['cost']:
            xx[len(x):].reshape(-1, 3)[:, :2] = 0
        xx, new = fit(xx, f, y, pp, n, method)
        decrease = new['cost'] <= state['cost']+1e-10
        history.append(dict(search=meta, fit=new, added=len(choices), K=len(pp), decrease=decrease))
        if not decrease:
            stop = 'non_decrease_failure'
            break
        x, paths, state = xx, pp, new
    return dict(final=state, los=los, los_states=states, paths=paths, history=history, stop=stop,
        normal=stop == 'no_detected_increment' and state['stationary'],
        seconds=time.perf_counter()-start, frontend_seconds=front_seconds,
        method=method, two_regions=two_regions)
