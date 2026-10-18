"""Exact memoization and observation-only certificates; frozen native solvers.

R2 is never monkeypatched. AP can use the separately approved common-stop
controller; NOMP and MULTISTART_FIXED retain their native stopping.
"""
import copy
import itertools
import time
import numpy as np
import torch
from scipy.stats import qmc
from ..r2.band.physics import Model as BandModel
from ..r2.solvers import ap, local
from ..r1.methods import Trace as OriginalTrace, nomp_solver
from ..methods import peaks
from ..physics import grid
from .config import METHODS, REVISION


def canonical_positions(positions):
    p = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    return tuple(map(tuple, p[np.lexsort((p[:, 1], p[:, 0]))]))


def initial_starts(choices, count, seed):
    """Exactly four unique spectrum combinations and four unique Sobol points."""
    if count not in (1, 2):
        raise ValueError('Only K=1/2 supported')
    choices = np.asarray(choices, float).reshape(-1, 2)
    pairs = list(itertools.combinations(range(len(choices)), count))
    # Round-robin ranked pairs, rather than pinning every pair to the first peak.
    pairs.sort(key=lambda pair: (max(pair), sum(pair), pair))
    starts, labels, seen = [], [], set()
    for pair in pairs:
        p = choices[list(pair)]
        key = canonical_positions(p)
        if key not in seen:
            starts.append(p.copy()); labels.append('spectrum'); seen.add(key)
        if len(starts) == 4:
            break
    if len(starts) != 4:
        raise ValueError('Insufficient unique spectrum starts')
    cover = qmc.Sobol(2*count, scramble=True, seed=int(seed)).random_base2(6)
    for flat in cover:
        p = (-1000+2000*flat).reshape(count, 2)
        key = canonical_positions(p)
        if key not in seen:
            starts.append(p); labels.append('sobol'); seen.add(key)
        if len(starts) == 8:
            break
    if len(starts) != 8 or labels.count('sobol') != 4:
        raise RuntimeError('Initialization coverage contract failed')
    return np.asarray(starts), labels


def projected_gradient(positions, gradient):
    """Box KKT residual, in the original metre coordinates (same gtol)."""
    p, g = np.asarray(positions).ravel(), np.asarray(gradient).ravel().copy()
    g[(p <= -1000) & (g > 0)] = 0
    g[(p >= 1000) & (g < 0)] = 0
    return float(np.max(abs(g)))


class Model(BandModel):
    def __init__(self, *args, memo=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.memo_enabled, self.memo = memo, {}
        self.counts.update(objective_requests=0, exact_cache_hits=0)

    def joint(self, positions, gradient=False):
        self.counts['objective_requests'] += 1
        p = np.ascontiguousarray(np.asarray(positions, dtype=np.float64).reshape(-1, 2))
        key = (p.shape, p.tobytes())  # exact identity, no rounding or source permutation
        old = self.memo.get(key) if self.memo_enabled else None
        if old is not None and (not gradient or old[1] is not None):
            self.counts['exact_cache_hits'] += 1
            return (old[0], old[1].copy()) if gradient else old[0]
        result = super().joint(p, gradient)
        if gradient:
            value, g = result
            item = (value, g.copy())
        else:
            item = (result, None)
        if self.memo_enabled:
            self.memo[key] = item
        return result


class Trace(OriginalTrace):
    """Native selections unchanged. Serialization cost is measured separately."""
    def __init__(self, guard, history=True):
        super().__init__(guard)
        self.keep_history = history
        self.accepted_certificates = []
        self.model = None

    def certificate(self, p):
        if self.model is None or len(np.asarray(p).reshape(-1, 2)) != 2:
            return
        pp = np.asarray(p, dtype=np.float64).reshape(-1, 2)
        item = self.model.memo.get((pp.shape, pp.tobytes()))
        if item is not None and item[1] is not None and item[0] <= self.best:
            self.accepted_certificates.append(dict(seconds=self.elapsed(),
                wall_seconds=time.perf_counter()-self.start,
                projected_gradient=projected_gradient(p, item[1]), objective=item[0],
                positions=pp.tolist(), joint_evaluations=self.model.counts['joint_evaluations'],
                gradient_calls=self.model.counts['gradient_calls']))

    def save(self, positions, objective):
        start = time.perf_counter()
        super().save(positions, objective)
        self.certificate(positions)
        self.recording_seconds += time.perf_counter()-start

    def iteration(self, p, value, **extra):
        start = time.perf_counter()
        if self.keep_history:
            super().iteration(p, value, **extra)
        # Only an actually accepted iterate is eligible; line-search trials are not.
        self.certificate(p)
        self.recording_seconds += time.perf_counter()-start


def solve(m, t, method, count, cfg, seed):
    if method.startswith('AP_'):
        if count != 2:
            raise ValueError('AP two-source comparison requires K=2')
        if cfg.get('common_stop'):
            from .common_stop import ap_common
            return ap_common(m, t, method, cfg)
        return ap(m, t, method, cfg)
    if method == 'NOMP':
        nomp_solver(m, t, count, cfg)
        return {}
    if method != 'MULTISTART_FIXED' or cfg['starts'] != 8:
        raise ValueError('R3 multistart must use exactly 8 starts')
    with t.stage('initialization'):
        points = grid(cfg['grid'])
        choices = points[peaks(points, m.conditional(points), 8)]
        starts, kinds = initial_starts(choices, count, seed)
    native = []
    with t.stage('multistart_local_optimization'):
        for i, start in enumerate(starts):
            t.save(start, m.joint(start))
            p, value, info = local(m, start, cfg, t)
            t.save(p, value)
            t.iteration(p, value, start=i)
            m.counts['iterations'] += 1
            native.append(info)
    return dict(starts=starts.tolist(), start_kinds=kinds, actual_starts=len(native),
                local_optimizers=native)


def run(iq, method, count, config, seed=0, guard=lambda: None, memo=True, history=True):
    if method not in METHODS or count not in (1, 2):
        raise ValueError('Unsupported method/count')
    cfg = copy.deepcopy(config)
    if cfg['sample_rate_hz'] != 50e6 or cfg['grid'] != 20 or cfg['half_band_hz'] != 7e6:
        raise ValueError('Frozen physical input contract differs')
    if np.asarray(iq).shape != (4, 8192):
        raise ValueError('Expected full 4x8192 IQ; band selection is internal')
    t = Trace(guard, history)
    with t.stage('input_fft_transfer_band_selection'):
        m = Model(iq, t.check, memo=memo, sample_rate_hz=cfg['sample_rate_hz'],
                  half_band_hz=cfg['half_band_hz'])
        t.model = m
    details = solve(m, t, method, count, cfg, seed)
    torch.cuda.synchronize()
    core_seconds, core_counts = t.elapsed(), copy.deepcopy(m.counts)
    if not t.events:
        raise ArithmeticError('No finite output')
    # Common stopping selects the accepted terminal state, including exact ties.
    p = np.asarray(details.get('output_positions', t.events[-1]['positions']))
    diagnostic_start = time.perf_counter()
    value, gradient = m.joint(p, True)
    torch.cuda.synchronize()
    certificate_seconds = time.perf_counter()-diagnostic_start
    pg = projected_gradient(p, gradient)
    result = dict(method=method, config=cfg, implementation_revision=REVISION, seed=seed, count=count,
        positions=p.tolist(), status='FINISHED', stop_reason=t.reason,
        seconds=core_seconds, recording_seconds=t.recording_seconds,
        wall_solver_seconds=core_seconds+t.recording_seconds,
        certificate_seconds=certificate_seconds, counts=core_counts,
        certificate_counts={k: m.counts[k]-core_counts[k] for k in core_counts},
        native_objective=value, projected_gradient=pg,
        precision_met=pg <= cfg['gtol'], events=t.events, history=t.history,
        stages=t.stages, mechanism=details, accepted_certificates=t.accepted_certificates,
        memo_enabled=memo, history_enabled=history)
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise ArithmeticError('Nonfinite final objective/gradient')
    return result
