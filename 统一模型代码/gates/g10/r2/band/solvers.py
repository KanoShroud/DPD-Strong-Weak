"""Same shared-band input, native objectives; structural candidates stay named."""
import copy
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats import qmc
from ...physics import grid, shortlist
from ...methods import peaks, sbx_mutate
from ...r1.methods import Trace, ap_solver, nomp_solver
from ...r1.sbl import solve as sbl_solve
from ..solvers import ap, extra_cycles, local
from .physics import Model, Spectrum
from .config import DEFAULT_CONFIG, METHODS
from .stopping import StopController


def spectrum(m, t, method, count, cfg):
    with t.stage('spectrum_and_grid'):
        spec = None if method == 'LS' else Spectrum(m, method, count, cfg['segments'], cfg['loading'], cfg['music_weight'])
        score = m.conditional if spec is None else spec.evaluate
        points = grid(cfg['grid'])
        values = score(points)
        ids = peaks(points, values, cfg['top'])
        options = [(float(values[i]), points[i].copy()) for i in ids]
        m.native_map = dict(points=points.tolist(), power=values.tolist(), meaning=method+'_native_score')

    def publish():
        positions = np.asarray([p for _, p in options])
        scores = np.asarray([v for v, _ in options])
        ids = shortlist(positions, scores, count)
        t.save(positions[ids], -float(scores[ids].sum()))

    publish()
    with t.stage('native_continuous_refinement'):
        for i, (value, p) in enumerate(options):
            if spec is None:
                candidate, objective, _ = local(m, p[None], cfg)
                candidate, new = candidate[0], 1-objective
            else:
                opt = minimize(lambda x: spec.value_grad(x, value), p, jac=True, method='L-BFGS-B',
                    bounds=[(-1000., 1000.)]*2, options=dict(maxiter=cfg['local_maxiter'],
                    ftol=cfg['ftol'], gtol=cfg['gtol'], maxls=cfg['maxls']))
                candidate, new = opt.x, float(score(opt.x[None])[0])
            if new >= value:
                options[i] = new, candidate
            publish()


def global_search(m, t, method, count, cfg, seed):
    rng = np.random.default_rng(seed)
    with t.stage('initialization'):
        points = grid(cfg['grid'])
        choices = points[peaks(points, m.conditional(points), max(8, count))]
        needed = cfg['starts'] if method == 'MULTISTART' else cfg['population']
        # A 128-member GA must really receive 128 individuals, not a truncated 64-start list.
        exponent = max(6, int(np.ceil(np.log2(needed))))
        cover = qmc.Sobol(2*count, scramble=True, seed=int(rng.integers(2**31))).random_base2(exponent)
        starts = (-1000+2000*cover).reshape(-1, count, 2)
        starts[0] = choices[:count]
        if count == 2:
            for i in range(1, min(len(choices), 8)):
                starts[i] = [choices[0], choices[i]]
    if method == 'MULTISTART':
        with t.stage('multistart_local_optimization'):
            for i, start in enumerate(starts[:needed]):
                t.save(start, m.joint(start))
                p, value, _ = local(m, start, cfg)
                t.save(p, value)
                t.iteration(p, value, start=i)
                m.counts['iterations'] += 1
        return
    pop = starts[:needed].reshape(needed, -1).copy()
    before, stale = np.inf, 0
    with t.stage('genetic_search'):
        for iteration in range(cfg['generations']):
            values = m.joint_batch(pop.reshape(-1, count, 2))
            order = np.argsort(values)
            best = float(values[order[0]])
            t.save(pop[order[0]].reshape(count, 2), best)
            stale = stale+1 if before-best < cfg['objective_tolerance'] else 0
            before = min(before, best)
            t.iteration(pop[order[0]].reshape(count, 2), best, generation=iteration, stagnant=stale)
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
        p, value, _ = local(m, t.events[-1]['positions'], cfg)
        t.save(p, value)


def run(iq, method, count=2, config=None, seed=0, guard=lambda: None):
    if method not in METHODS or count not in (1, 2):
        raise ValueError('Unknown method or source count')
    cfg = dict(DEFAULT_CONFIG, **(config or {}))
    t, details = Trace(guard), {}
    with t.stage('input_fft_transfer_band_selection'):
        m = Model(iq, t.check, sample_rate_hz=cfg['sample_rate_hz'], half_band_hz=cfg['half_band_hz'])
    if method in ('LS', 'MUSIC', 'MVDR'):
        spectrum(m, t, method, count, cfg)
    elif method == 'AP_FULL':
        ap_solver(m, t, count, cfg)
    elif method.startswith('AP_'):
        if count != 2:
            raise ValueError('Dual-source ablations require K=2')
        details = ap(m, t, method, cfg)
    elif method.startswith('NOMP'):
        nomp_solver(m, t, count, cfg)
        start = np.asarray(t.events[-1]['positions'])
        details = dict(prefix_positions=start.tolist(), prefix_seconds=t.elapsed(), prefix_objective=t.best)
        if method == 'NOMP_EXTRA12':
            with t.stage('extra_cycles'):
                t.reason = extra_cycles(m, t, start, cfg)
        elif method == 'NOMP_JOINT':
            with t.stage('joint_refinement'):
                _, _, info = local(m, start, cfg, t)
            details['local_optimizers'] = [info]
    elif method == 'SBL':
        controller = StopController(cfg) if cfg['sbl_stop_rule'] != 'legacy' else None
        sbl_solve(m, t, count, cfg, stop_controller=controller)
        if controller is not None:
            details.update(stop_checkpoints=controller.checkpoints, candidate_stop=controller.first_stop)
    else:
        global_search(m, t, method, count, cfg, seed)
    torch.cuda.synchronize()
    if not t.events:
        raise ArithmeticError('No finite output')
    return dict(method=method, count=count, config=cfg, seed=seed, seconds=t.elapsed(),
        status='FINISHED', stop_reason=t.reason, events=t.events, history=t.history,
        stages=t.stages, counts=copy.deepcopy(m.counts), positions=t.events[-1]['positions'],
        mechanism=details, native_map=getattr(m, 'native_map', None),
        local_optimizers=getattr(m, 'local_optimizers', []))
