"""Observation-only adapters. Existing solvers frozen; GA duplicate starts repaired."""
import copy
import time
import numpy as np
import torch
from scipy.stats import qmc
from ..physics import grid
from ..methods import peaks, sbx_mutate
from ..r1.methods import ap_solver
from ..r1.sbl import solve as sbl_solve
from ..r2.band.solvers import spectrum
from ..r2.band.stopping import StopController
from ..r2.solvers import local
from ..r3.solvers import Model, Trace, canonical_positions, projected_gradient, run as r3_run
from .solvers import run as core_run
from .adaptation_review import run as profile_run
from .formal_protocol import CORE


def ga(m, t, count, cfg, seed):
    """Same frozen GA operators; only replace permutation-duplicate initial pairs."""
    rng = np.random.default_rng(seed)
    with t.stage('initialization'):
        pp = grid(cfg['grid'])
        choices = pp[peaks(pp, m.conditional(pp), max(8, count))]
        size = cfg['population']
        # Doubling the Sobol buffer preserves its original prefix; unused tail
        # supplies deterministic replacements without consuming the GA RNG.
        exp = max(6, int(np.ceil(np.log2(size))))
        cover = qmc.Sobol(2*count, scramble=True, seed=int(rng.integers(2**31))).random_base2(exp+1)
        starts = (-1000+2000*cover).reshape(-1, count, 2)
        starts[0] = choices[:count]
        for i in range(1, min(len(choices), 8)):
            starts[i] = [choices[0], choices[i]]
        seen, replacements, tail = set(), [], size
        for i in range(size):
            if canonical_positions(starts[i]) in seen:
                while canonical_positions(starts[tail]) in seen:
                    tail += 1
                starts[i] = starts[tail]; tail += 1; replacements.append(i)
            seen.add(canonical_positions(starts[i]))
        assert len(seen)==size
        initial = starts[:size].copy()
    pop = initial.reshape(size, -1).copy()
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
                t.reason = 'GENETIC_STAGNATION'; break
            tournament = rng.integers(0, len(pop), (len(pop)-4, 3))
            winners = tournament[np.arange(len(tournament)), np.argmin(values[tournament], axis=1)]
            pop = np.vstack([pop[order[:4]], sbx_mutate(pop[winners], rng)])
        else:
            t.reason = 'ITERATION_LIMIT'
    with t.stage('final_local_refinement'):
        p, value, _ = local(m, t.events[-1]['positions'], cfg)
        t.save(p, value)
    return dict(initial_population=initial.tolist(), duplicate_replacements=replacements,
                actual_population=size, actual_generations=iteration+1)


def run(iq, method, cfg, seed=0, guard=lambda: None, keep_map=False):
    if np.asarray(iq).shape != (4, 8192):
        raise ValueError('Expected full 4x8192 observation')
    if (cfg['sample_rate_hz'], cfg['half_band_hz'], cfg['grid']) != (50e6, 7e6, 20):
        raise ValueError('Common input/grid contract changed')
    torch.cuda.synchronize()
    started = time.perf_counter()
    if method in CORE or method=='BNOMP_REF':
        result = core_run(iq, method, cfg, guard)
    elif method=='PROFILE_DIAG':
        result = profile_run(iq, cfg, True, guard)
        result.update(method=method, config=copy.deepcopy(cfg))
        # The profile diagnostic only publishes its final complete pair.
        result['events'] = [dict(seconds=result['wall_solver_seconds'], positions=result['positions'],
                                 native_objective=result['native_objective'])]
    elif method=='MULTISTART_FIXED':
        result = r3_run(iq, method, 2, cfg, seed, guard)
    else:
        t = Trace(guard)
        with t.stage('input_fft_transfer_band_selection'):
            m = Model(iq, t.check, sample_rate_hz=cfg['sample_rate_hz'], half_band_hz=cfg['half_band_hz'])
            t.model = m
        details = {}
        if method in ('LS','MUSIC','MVDR'):
            spectrum(m,t,method,2,cfg)
        elif method=='AP_FULL':
            ap_solver(m,t,2,cfg)
        elif method=='GA':
            details=ga(m,t,2,cfg,seed)
        elif method=='SBL':
            controller=StopController(cfg) if cfg['sbl_stop_rule']!='legacy' else None
            sbl_solve(m,t,2,cfg,stop_controller=controller)
        else:
            raise ValueError(method)
        torch.cuda.synchronize()
        core, counts = t.elapsed()+t.recording_seconds, copy.deepcopy(m.counts)
        if not t.events:
            raise ArithmeticError('No finite output')
        p=np.asarray(t.events[-1]['positions'])
        cert_start=time.perf_counter()
        joint, gradient=m.joint(p,True)
        torch.cuda.synchronize()
        result=dict(method=method, config=copy.deepcopy(cfg), positions=p.tolist(),
            native_objective=t.best, joint_objective=joint,
            projected_gradient=projected_gradient(p,gradient), stop_reason=t.reason,
            seconds=core-t.recording_seconds, wall_solver_seconds=core,
            recording_seconds=t.recording_seconds, certificate_seconds=time.perf_counter()-cert_start,
            counts=counts, mechanism=details, events=t.events, history=t.history, stages=t.stages,
            native_map=getattr(m,'native_map',None) if keep_map else None)
        # Trace.model and Model.check otherwise form a reference cycle, which
        # can retain a large SBL/AP dictionary until a later GC collection.
        t.model = None
    torch.cuda.synchronize()
    result['complete_seconds']=time.perf_counter()-started
    result['seed']=seed
    result.setdefault('joint_objective', result['native_objective'])
    # Existing frozen traces exclude some recording/certificate overhead. Charge
    # the full overhead to every saved event as a conservative availability bound.
    offset=max(0.,result['complete_seconds']-result['seconds'])
    result['report_events']=[dict(seconds=e['seconds']+offset,positions=e['positions'])
                             for e in result.get('events',[]) if len(e['positions'])==2]
    result['report_events'].append(dict(seconds=result['complete_seconds'],positions=result['positions']))
    result['report_events'].sort(key=lambda e:e['seconds'])
    # Native objectives differ between spectral/ML/SBL methods; joint_objective
    # is a diagnostic, NEVER used to replace their own selected outputs.
    if not np.isfinite(result['positions']).all() or not np.isfinite(result['joint_objective']):
        raise ArithmeticError('Nonfinite output')
    return result
