"""Observation-only API, nested AP prefixes and unchanged native NOMP prefix."""
import copy
import time
import numpy as np
import torch
from ..physics import grid
from ..methods import peaks
from ..r1.methods import nomp_solver
from ..r2.solvers import local
from ..r3.solvers import Model, Trace, projected_gradient
from ..r3.common_stop import refine_pair
from .config import METHODS, REVISION
from .bnomp import solve as bnomp


def ap(model, trace, method, cfg):
    points = grid(cfg['grid'])
    with trace.stage('first_source'):
        first = points[peaks(points, model.conditional(points), cfg['top'])[0]]
        first, _, info = local(model, first[None], cfg)
    with trace.stage('conditional_candidates'):
        scores = model.conditional(points, first)
        ids = peaks(points, scores, cfg['top'])
        candidates = [np.vstack([first, points[i]]) for i in ids]
        used = int(method.split('_')[1])
        if len(candidates) != 5:
            raise RuntimeError('Expected five nested candidates')
    branches = []
    with trace.stage('branch_refinement'):
        for rank, start in enumerate(candidates[:used], 1):
            before = model.joint(start)
            p, value, detail = refine_pair(model, trace, start,
                'AP_1_ALT' if method.endswith('_ALT') else 'AP_1_JOINT', cfg)
            branches.append(dict(rank=rank, initial_positions=start.tolist(), positions=p.tolist(),
                initial_objective=before, objective=value, **detail))
    chosen = min(branches, key=lambda b: (b['objective'], b['rank']))
    trace.reason = chosen['stop_reason']
    return dict(output_positions=chosen['positions'], selected_rank=chosen['rank'],
                first_position=first.tolist(), first_optimizer=info,
                candidate_list=[p.tolist() for p in candidates], branches=branches)


def run(iq, method, cfg, guard=lambda: None, count=2, history=True, memo=True):
    if method not in METHODS or count not in (1, 2):
        raise ValueError('Unsupported method/count')
    if count != 2 and method not in ('BNOMP_REF', 'NOMP'):
        raise ValueError('Single-source checks only use the native source solvers')
    if np.asarray(iq).shape != (4, 8192):
        raise ValueError('Expected full 4x8192 IQ')
    if (cfg['sample_rate_hz'], cfg['half_band_hz'], cfg['grid']) != (50e6, 7e6, 20):
        raise ValueError('Frozen input or grid changed')
    t = Trace(guard, history=history)
    with t.stage('input_fft_transfer_band_selection'):
        m = Model(iq, t.check, memo=memo, sample_rate_hz=cfg['sample_rate_hz'],
                  half_band_hz=cfg['half_band_hz'])
        t.model = m
    if method == 'BNOMP_REF':
        detail = bnomp(m, t, count, cfg)
    elif method.startswith('AP_'):
        detail = ap(m, t, method, cfg)
    else:
        nomp_solver(m, t, count, cfg)
        torch.cuda.synchronize()
        prefix = np.array(t.events[-1]['positions'])
        detail = dict(prefix_positions=prefix.tolist(), prefix_objective=t.best,
                      prefix_seconds=t.elapsed(), prefix_counts=copy.deepcopy(m.counts),
                      prefix_stop_reason=t.reason, output_positions=prefix.tolist())
        if method == 'NOMP_JOINT':
            with t.stage('terminal_joint_refinement'):
                p, _, info = refine_pair(m, t, prefix, 'AP_1_JOINT', cfg)
            detail.update(output_positions=p.tolist(), terminal_refinement=info)
    torch.cuda.synchronize()
    core_time, counts = t.elapsed(), copy.deepcopy(m.counts)
    p = np.array(detail['output_positions'])
    start = time.perf_counter()
    value, gradient = m.joint(p, True)
    torch.cuda.synchronize()
    cert_time = time.perf_counter()-start
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise ArithmeticError('Nonfinite final solution')
    return dict(method=method, revision=REVISION, config=copy.deepcopy(cfg), count=count,
        positions=p.tolist(), native_objective=value, projected_gradient=projected_gradient(p, gradient),
        stop_reason=t.reason, seconds=core_time, wall_solver_seconds=core_time+t.recording_seconds,
        certificate_seconds=cert_time, recording_seconds=t.recording_seconds, counts=counts,
        mechanism=detail, stages=t.stages, events=t.events, history=t.history)
