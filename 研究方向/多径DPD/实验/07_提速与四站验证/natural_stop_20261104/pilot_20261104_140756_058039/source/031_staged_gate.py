"""Inexact working updates; exact terminal stationarity plus renewed search."""
import sys
import time
import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits
import solver as s
import hybrid_gate as h
import run as runner

fc, ref = s.fc, s.ref


def working_fit(x0, f, y, paths, n):
    start = time.perf_counter(); x = ref.project(x0, n)
    cost, g = s.objective(x, f, y, paths)
    first_pg = float(np.max(abs(x-ref.project(x-g, n))))
    target = max(1e-7, .01*first_pg)
    damping = 1e-3; hist = []; calls = 1
    for it in range(50):
        pg = float(np.max(abs(x-ref.project(x-g, n))))
        if pg <= target: break
        r, j = s.evaluate(x, f, y, paths)
        scale = np.maximum(np.linalg.norm(j, axis=0), 1e-8)
        a = j/scale; gram = a.T@a; moved = False
        for _ in range(12):
            direction = solve(gram+damping*np.eye(len(x)), -g/scale, assume_a='pos')/scale
            candidate = ref.project(x+direction, n); step = candidate-x
            try:
                value, ng = s.objective(candidate, f, y, paths)
            except ValueError as exc:
                if str(exc) != 'station coincidence': raise
                value, ng = np.inf, g
            calls += 1
            if value < cost and value <= cost+1e-4*g@step:
                hist.append(dict(iteration=it, cost=cost, PG=pg))
                x, cost, g = candidate, value, ng
                damping = max(damping/3, 1e-8); moved = True; break
            damping = min(damping*10, 1e12)
        if not moved: break
    pg = float(np.max(abs(x-ref.project(x-g, n))))
    return x, dict(x=x, p=x[:4].reshape(2, 2)*ref.UNIT, cost=cost,
        projected_gradient=pg, stationary=pg <= 1e-7, work_ready=pg <= target,
        working_target=target, initial_PG=first_pg, iterations=len(hist), calls=calls,
        history=hist, seconds=time.perf_counter()-start,
        closure_error=float(np.linalg.norm(x-ref.project(x, n))))


def estimate(iq, method='STAGED', two_regions=True, accelerated=True):
    start = time.perf_counter(); geom = s.bc.regions.frontend(iq)
    selection = s.bc.regions.alternate_pair(geom['refined'], geom['reward'])
    f, y, n, sigma = fc.cal.spectral(iq)
    use = abs(np.fft.fftfreq(n, 1/ref.FS)) <= ref.B/2
    states = []
    for ids in [selection['first'], selection['second']]:
        p = geom['refined'][ids]
        gains = np.column_stack([fc.engine.rank_one(y, f, z)[0] for z in p])
        xx = np.r_[p.ravel()/ref.UNIT, gains[1:].real.ravel(), gains[1:].imag.ravel()]
        _, st = s.fast_los(xx, f, y, n); states.append(st)
    state = min(states, key=lambda z: z['cost']); x = np.array(state['x'])
    if not state['stationary']: x, state = h.fit(x, f, y, [], n)
    los = state; front = time.perf_counter()-start
    history, paths = [], []; stop = 'capacity_failure'; scan = 0
    if not state['stationary']: stop = 'los_nonstationary'
    while len(paths) < 64 and stop != 'los_nonstationary':
        scan += 1
        _, bank, meta = s.fast_search(x, paths, f, y, n, use, sigma, scan)
        assert all(meta['checks'].values())
        choices = []
        for m in range(5):
            for q in range(2):
                ids = np.flatnonzero((bank[:, 0] == m)&(bank[:, 1] == q)&(bank[:, 3] == 1)&(bank[:, 4] > meta['threshold']))
                if len(ids): choices.append(bank[ids[np.argmax(bank[ids, 4])]])
        if not choices:
            if state['stationary']:
                history.append(dict(search=meta, added=0)); stop = 'no_detected_increment'; break
            xx, new = h.fit(x, f, y, paths, n)
            decrease = new['cost'] <= state['cost']+1e-10
            history.append(dict(search=meta, fit=new, K=len(paths), added=0, decrease=decrease, terminal_polish=True))
            if not decrease: stop = 'non_decrease_failure'; break
            x, state = xx, new
            if not state['stationary']: stop = 'terminal_nonstationary'; break
            continue
        choices = sorted(choices, key=lambda z: -z[4])[:64-len(paths)]
        pp = paths+[(int(z[0]), int(z[1])) for z in choices]
        xx = ref.project(np.r_[x, np.array([[z[5], z[6], z[2]/ref.FS*ref.B] for z in choices]).ravel()], n)
        if s.objective(xx, f, y, pp)[0] > state['cost']: xx[len(x):].reshape(-1, 3)[:, :2] = 0
        xx, new = working_fit(xx, f, y, pp, n)
        decrease = new['cost'] <= state['cost']+1e-10
        history.append(dict(search=meta, fit=new, K=len(pp), added=len(choices), decrease=decrease))
        if not decrease: stop = 'non_decrease_failure'; break
        x, state, paths = xx, new, pp
        if not new['work_ready']: stop = 'working_update_failure'; break
    return dict(final=state, los=los, los_states=states, paths=paths, history=history, stop=stop,
        normal=stop == 'no_detected_increment' and state['stationary'], method=method,
        seconds=time.perf_counter()-start, frontend_seconds=front, two_regions=True, accelerated=True)


if __name__ == '__main__':
    runner.s.estimate = estimate
    if '--methods' not in sys.argv: sys.argv += ['--methods', 'STAGED']
    with threadpool_limits(limits=2): runner.main()
