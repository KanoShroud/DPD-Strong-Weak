"""Native stopping, common initial grid, final solution and timed incumbents."""
import time
from contextlib import contextmanager
import numpy as np
import torch
from scipy.optimize import minimize
from ..config import METHODS
from ..physics import grid, shortlist
from ..methods import peaks, coordinate, newton_single, initial_pairs, sbx_mutate
from .physics import Model, Spectrum
from .config import DEFAULT


class Trace:
    def __init__(self, guard):
        torch.cuda.synchronize()
        self.start = time.perf_counter()
        self.guard, self.last_check = guard, 0.
        self.events, self.history, self.stages = [], [], {}
        self.best = np.inf
        self.reason = 'COMPLETED_SEARCH'
        self.prefix_seconds = self.recording_seconds = self.resume_setup_seconds = 0.

    def check(self):
        now = time.perf_counter()
        if now-self.last_check >= 2:
            self.guard()
            self.last_check = now

    def elapsed(self):
        return self.prefix_seconds+time.perf_counter()-self.start-self.recording_seconds

    def save(self, positions, objective):
        p = np.asarray(positions).reshape(-1, 2)
        if not np.isfinite(objective) or not np.isfinite(p).all():
            raise ArithmeticError('Nonfinite native result')
        if objective < self.best:
            self.best = float(objective)
            self.events.append(dict(seconds=self.elapsed(), positions=p.tolist(), native_objective=self.best))

    @contextmanager
    def stage(self, name):
        torch.cuda.synchronize()
        start = self.elapsed()
        yield
        torch.cuda.synchronize()
        self.stages[name] = self.stages.get(name, 0.)+self.elapsed()-start

    def iteration(self, p, value, **extra):
        self.history.append(dict(seconds=self.elapsed(), positions=np.asarray(p).tolist(),
                                 native_objective=float(value), **extra))


def refine(m, p, maxiter=100):
    p = np.asarray(p).reshape(-1, 2)
    old = m.joint(p)
    result = minimize(lambda x: m.joint(x, True), p.ravel(), jac=True, method='L-BFGS-B',
                      bounds=[(-1000., 1000.)]*p.size,
                      options=dict(maxiter=maxiter, ftol=1e-12, gtol=1e-8, maxls=25))
    v = m.joint(result.x)
    return (result.x.reshape(-1, 2), v) if v <= old else (p, old)


def spectrum_solver(m, t, k, cfg, method):
    with t.stage('spectrum_and_grid'):
        spec = None if method == 'LS' else Spectrum(m, method, k, cfg['segments'],
                                                    cfg['loading'], cfg['music_weight'])
        score = m.conditional if spec is None else spec.evaluate
        pp = grid(cfg['grid'])
        ss = score(pp)
        ids = peaks(pp, ss, cfg['top'])
        options = [(float(ss[i]), pp[i].copy()) for i in ids]
        m.native_map = dict(points=pp.tolist(), power=ss.tolist(), meaning=method+'_native_score')

    def publish():
        pos = np.asarray([p for _, p in options])
        val = np.asarray([v for v, _ in options])
        take = shortlist(pos, val, k)
        t.save(pos[take], -float(val[take].sum()))
    publish()
    with t.stage('native_continuous_refinement'):
        for i, (value, p) in enumerate(options):
            if spec is None:
                q, v = refine(m, p[None])
                q, new = q[0], 1-v
            else:
                z = minimize(lambda x: spec.value_grad(x, value), p, jac=True, method='L-BFGS-B',
                             bounds=[(-1000., 1000.)]*2,
                             options=dict(maxiter=100, ftol=1e-12, gtol=1e-8, maxls=25))
                q, new = z.x, float(score(z.x[None])[0])
            if new >= value:
                options[i] = new, q
            publish()


def ap_solver(m, t, k, cfg):
    with t.stage('initialization'):
        pp = grid(cfg['grid'])
        if k == 2 and cfg.get('dictionary_cache', True):
            m.enable_response_cache(pp, cfg['fft_chunk'])
        scores = m.conditional(pp)
        starts = pp[peaks(pp, scores, cfg['top'])]
    any_limit = False
    with t.stage('conditional_search_and_refinement'):
        for initial in starts:
            positions, _ = refine(m, initial[None])
            if k == 2:
                scores = m.conditional(pp, positions)
                choices = pp[peaks(pp, scores, cfg['top'])]
                fitted = [refine(m, np.vstack([positions, p])) for p in choices]
                positions, _ = min(fitted, key=lambda x: x[1])
            value = m.joint(positions)
            t.save(positions, value)
            if k == 1:
                # No other source changes the conditional spectrum: rescanning it
                # after the same single-source optimization is redundant.
                continue
            stable = 0
            for iteration in range(cfg['cycles']):
                before, old = value, positions.copy()
                for j in range(k):
                    others = np.delete(positions, j, axis=0)
                    score = m.conditional(pp, others)
                    ids = peaks(pp, score, cfg['top'])
                    current = positions.copy()
                    for ind in ids:
                        candidate = positions.copy()
                        candidate[j] = pp[ind]
                        if m.joint(candidate) < m.joint(current):
                            current = candidate
                    positions = coordinate(m, current, j)
                    value = m.joint(positions)
                    t.save(positions, value)
                shift = float(np.linalg.norm(positions-old, axis=1).max())
                stable = stable+1 if before-value < cfg['objective_tolerance'] and shift < cfg['position_tolerance'] else 0
                t.iteration(positions, value, iteration=iteration, max_shift_m=shift)
                m.counts['iterations'] += 1
                if stable >= 2:
                    break
            else:
                any_limit = True
    t.reason = 'ITERATION_LIMIT' if any_limit else 'NATIVE_STABLE'


def nomp_solver(m, t, k, cfg):
    pp, residual = [], m.x
    points = grid(cfg['grid'])
    with t.stage('discovery_newton_cyclic_ls'):
        for _ in range(k):
            first = points[int(np.argmax(m.conditional(points, residual=residual)))]
            pp.append(newton_single(m, first, residual, steps=5))
            pp = np.asarray(pp)
            stable = 0
            for iteration in range(cfg['cycles']):
                old, before = pp.copy(), m.joint(pp)
                for j in range(len(pp)):
                    a, s, r = m.fitted(pp)
                    residual_j = r+a[:, :, j]*s[:, j, None]
                    proposal = pp.copy()
                    proposal[j] = newton_single(m, pp[j], residual_j, steps=5)
                    if m.joint(proposal) <= m.joint(pp):
                        pp = proposal
                value = m.joint(pp)
                if len(pp) == k:
                    t.save(pp, value)
                shift = float(np.linalg.norm(pp-old, axis=1).max())
                t.iteration(pp, value, iteration=iteration, max_shift_m=shift)
                m.counts['iterations'] += 1
                stable = stable+1 if before-value < cfg['objective_tolerance'] and shift < cfg['position_tolerance'] else 0
                if stable >= 2:
                    break
            else:
                t.reason = 'ITERATION_LIMIT'
            _, _, residual = m.fitted(pp)
            pp = list(pp)


def global_solver(m, t, k, cfg, rng, method):
    with t.stage('initialization'):
        starts = initial_pairs(m, k, cfg, rng)
    if method == 'MULTISTART':
        with t.stage('multistart_local_optimization'):
            for i, p in enumerate(starts[:cfg['starts']]):
                t.save(p, m.joint(p))
                q, value = refine(m, p)
                t.save(q, value)
                t.iteration(q, value, start=i)
                m.counts['iterations'] += 1
        return
    pop = np.asarray(starts[:cfg['population']]).reshape(-1, k*2)
    before, stale = np.inf, 0
    with t.stage('genetic_search'):
        for iteration in range(cfg['generations']):
            values = m.joint_batch(pop.reshape(-1, k, 2))
            order = np.argsort(values)
            best = float(values[order[0]])
            t.save(pop[order[0]].reshape(k, 2), best)
            stale = stale+1 if before-best < cfg['objective_tolerance'] else 0
            before = min(before, best)
            t.iteration(pop[order[0]].reshape(k, 2), best, generation=iteration, stagnant=stale)
            m.counts['iterations'] += 1
            if stale >= cfg['stagnation']:
                t.reason = 'GENETIC_STAGNATION'
                break
            tournament = rng.integers(0, len(pop), (len(pop)-4, 3))
            winners = tournament[np.arange(len(tournament)), np.argmin(values[tournament], axis=1)]
            pop = np.vstack([pop[order[:4]], sbx_mutate(pop[winners], rng)])
        else:
            t.reason = 'ITERATION_LIMIT'
    with t.stage('final_local_refinement'):
        q, value = refine(m, t.events[-1]['positions'])
        t.save(q, value)


def run(iq, method, count=2, config=None, seed=0, guard=lambda: None, state_store=None):
    if method not in METHODS or count not in (1, 2):
        raise ValueError('Unsupported method or K')
    cfg = dict(DEFAULT, **(config or {}))
    if cfg['grid'] != cfg['sbl_grid']:
        raise ValueError('All initial grids must match')
    t = Trace(guard)
    with t.stage('input_fft_transfer'):
        m = Model(iq, t.check, sample_rate_hz=cfg['sample_rate_hz'])
    if method in ('LS', 'MUSIC', 'MVDR'):
        spectrum_solver(m, t, count, cfg, method)
    elif method == 'AP':
        ap_solver(m, t, count, cfg)
    elif method == 'BNOMP':
        nomp_solver(m, t, count, cfg)
    elif method == 'SBL':
        from .sbl import solve
        solve(m, t, count, cfg, observer=state_store)
    else:
        global_solver(m, t, count, cfg, np.random.default_rng(seed), method)
    torch.cuda.synchronize()
    return dict(method=method, count=count, config=cfg, seed=seed, seconds=t.elapsed(),
                status='FINISHED', stop_reason=t.reason, events=t.events,
                history=t.history, stages=t.stages, counts=m.counts,
                positions=t.events[-1]['positions'] if t.events else [],
                local_optimizers=getattr(m, 'local_optimizers', []),
                recording_seconds=t.recording_seconds, resume_setup_seconds=t.resume_setup_seconds,
                native_map=getattr(m, 'native_map', None))


def at_time(result, limit):
    events = [e for e in result['events'] if e['seconds'] <= limit]
    return events[-1]['positions'] if events else []
