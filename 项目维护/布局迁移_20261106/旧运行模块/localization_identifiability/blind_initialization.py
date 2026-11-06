# ruff: noqa: F401
# Preserve imported globals for legacy function cloning and reexports.
"""Observation-only initialization pilot. No truth is accepted by solve()."""
from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import json
import time
import warnings

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment
from threadpoolctl import threadpool_limits

from structural_probe import ROOT, STATIONS, C, B, response, sha, write
from solver_reliability import evaluate, datasets
from alternative_fit import SOURCE
from position_reduction import channel, position_jac, valid, reduced_step

PROTOCOL = ROOT/'盲初始化核验协议.md'
GRID_AXIS = np.arange(-2990., 3000., 20.)
GRID = np.stack(np.meshgrid(GRID_AXIS, GRID_AXIS), axis=-1).reshape(-1, 2)
BOUND = 3000.


def signature(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def score_positions(y, f, points):
    """Eigenvalue of aligned station covariance: best rank-one energy."""
    scores = []
    for start in range(0, len(points), 256):
        d = np.linalg.norm(points[start:start+256, None]-STATIONS[None], axis=2)
        phase = np.exp(2j*np.pi*f[None, :, None]*d[:, None, :]/C)
        z = phase*y[None]
        cov = np.einsum('bfm,bfn->bmn', z, z.conj())
        scores.extend(np.linalg.eigvalsh(cov)[:, -1].real)
    return np.asarray(scores)


def rank_one(y, f, p):
    d = np.linalg.norm(p[None]-STATIONS, axis=1)
    phase = np.exp(-2j*np.pi*f[:, None]*d[None]/C)
    z = y*phase.conj()
    _, vec = np.linalg.eigh(z.T@z.conj())
    g = vec[:, -1]
    if abs(g[0]) < 1e-12:
        raise ValueError('Reference gain gauge unusable; no truth-based replacement.')
    g = g/g[0]
    s = z@g.conj()/np.vdot(g, g).real
    fit = phase*g[None]*s[:, None]
    return g, s, fit


def peaks(score, count=3, exclude=()):
    chosen = []
    for index in np.argsort(-score, kind='stable'):
        p = GRID[index]
        if all(np.linalg.norm(p-q) >= 100 for q in [*exclude, *chosen]):
            chosen.append(p.copy())
            if len(chosen) == count:
                break
    return chosen


def residual(v, p, f, y):
    g, d = channel(v)
    h, _ = response(p, d, g, f)
    coef = np.einsum('fqm,fm->fq', np.linalg.pinv(h, rcond=1e-12), y)
    r = y-np.einsum('fmq,fq->fm', h, coef)
    return np.sqrt(2)/np.linalg.norm(y)*np.r_[r.real.ravel(), r.imag.ravel()], coef


def los_vector(g):
    return np.r_[g[1:].real.ravel(), g[1:].imag.ravel(), np.zeros(40), np.tile([8., 8.], 10)]


def local_los(p, v, f, y):
    """Joint continuous LOS fit, unknown constant gains (no 1/d oracle)."""
    x0 = np.r_[p.ravel()/(C/B), v[:16]]
    cached = [None, None, None]

    def update(x):
        if cached[0] is None or not np.array_equal(x, cached[0]):
            pp = x[:4].reshape(2, 2)*(C/B)
            vv = v.copy()
            vv[:16] = x[4:]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)  # DRR=inf at zero reflection
                r, j, _ = evaluate(vv, pp, f, y, 'physical')
            cached[:] = [x.copy(), r, np.column_stack([position_jac(vv, pp, f, y), j[:, :16]])]
        return cached[1], cached[2]

    lower = np.r_[np.full(4, -BOUND/(C/B)), np.full(16, -np.inf)]
    upper = -lower
    opt = least_squares(lambda x: update(x)[0], x0, jac=lambda x: update(x)[1],
                        bounds=(lower, upper), x_scale='jac', method='trf',
                        max_nfev=100, ftol=1e-10, xtol=1e-10, gtol=1e-7)
    vv = v.copy()
    vv[:16] = opt.x[4:]
    pp = opt.x[:4].reshape(2, 2)*(C/B)
    return pp, vv, dict(cost=float(opt.cost), nfev=opt.nfev, status=int(opt.status),
                        message=opt.message, optimality=float(opt.optimality))


def seed_paths(p, v, f, y):
    """Greedy delayed-copy LS with estimated, not true, source spectra."""
    _, s = residual(v, p, f, y)
    dist = np.linalg.norm(p[None]-STATIONS[:, None], axis=2)
    base = np.exp(-2j*np.pi*f[:, None, None]*dist[None]/C)*s[:, None]
    gains = np.zeros((5, 2, 3), complex)
    delays = np.zeros((5, 2, 3))
    selected = []
    for m in range(5):
        support = [[], []]

        def fit(active):
            columns, keys = [], []
            for q in range(2):
                for delay in [0., *active[q]]:
                    columns.append(base[:, m, q]*np.exp(-2j*np.pi*f*delay/B))
                    keys.append((q, delay))
            a = np.column_stack(columns)
            coef = np.linalg.lstsq(a, y[:, m], rcond=1e-12)[0]
            power_ok = True
            for q in range(2):
                c = [coef[i] for i, (qq, _) in enumerate(keys) if q == qq]
                power_ok &= abs(c[0]) > 1e-12 and np.sum(np.abs(c[1:])**2) < abs(c[0])**2
            return float(np.linalg.norm(y[:, m]-a@coef)**2), coef, keys, bool(power_ok)

        current = fit(support)
        for _ in range(4):
            best = None
            for q in range(2):
                if len(support[q]) == 2:
                    continue
                for delay in np.arange(3., 40.001, .25):
                    trial = [list(t) for t in support]
                    trial[q] = sorted([*trial[q], float(delay)])
                    gaps = np.diff([0., *trial[q]])
                    if min(gaps) < 3 or max(gaps) > 20:
                        continue
                    fitted = fit(trial)
                    if fitted[3] and fitted[0] < current[0]-1e-12*np.linalg.norm(y[:, m])**2:
                        if best is None or fitted[0] < best[0][0]:
                            best = fitted, trial
            if best is None:
                break
            current, support = best
        # Unselected paths have zero gain; placeholders impose no signal.
        for q in range(2):
            ds = list(support[q])
            if not ds:
                ds = [8., 16.]
            elif len(ds) == 1:
                ds.append(ds[0]+8.)
            delays[m, q, 1:] = ds
            for coef, (qq, delay) in zip(current[1], current[2]):
                if qq == q:
                    idx = 0 if delay == 0 else 1+ds.index(delay)
                    gains[m, q, idx] = coef
        selected.append([list(t) for t in support])
    gains /= gains[:1, :, :1]
    los = gains[:, :, 0]
    ratios = gains[:, :, 1:]/los[:, :, None]
    vv = np.r_[los[1:].real.ravel(), los[1:].imag.ravel(), ratios.real.ravel(),
               ratios.imag.ravel(), np.diff(delays, axis=2).ravel()]
    if not valid(vv):
        return v.copy(), dict(status='infeasible_path_seed_fallback_LOS', selected=selected)
    # A path fit used fixed estimated waveforms: do not accept worse profiled fit.
    r0, _ = residual(v, p, f, y)
    r1, _ = residual(vv, p, f, y)
    accepted = r1@r1 <= r0@r0
    return (vv if accepted else v.copy()), dict(status='accepted' if accepted else 'worse_seed_fallback_LOS', selected=selected)


def feasible(p, v):
    return bool(np.all(np.isfinite(p)) and np.all(np.isfinite(v))
                and np.max(abs(p)) <= BOUND and valid(v)
                and np.min(np.linalg.norm(p[None]-STATIONS[:, None], axis=2)) > 1e-6)


def refine(p, v, f, y):
    """Same-model damped GN in physical coordinates; feasible acceptance only."""
    p, v = p.copy(), v.copy()
    damping = 1e-3
    history = []
    reason = 'iteration_limit'
    for iteration in range(60):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            r, jn, metrics = evaluate(v, p, f, y, 'physical')
        jp = position_jac(v, p, f, y)
        j = np.column_stack([jp, jn])
        scale = np.maximum(np.linalg.norm(j, axis=0), 1e-12)
        value = .5*float(r@r)
        grad = float(np.max(abs(j.T@r/scale)))
        record = dict(iteration=iteration, cost=value, positions=p.copy(), gradient=grad,
                      reflection_power=metrics['max_reflection_power_ratio'], accepted=False)
        history.append(record)
        if grad <= 1e-7:
            reason = 'scaled_unconstrained_gradient'
            break
        for trial in range(12):
            # This regularized joint solve equals eliminating nuisance from
            # the augmented linear system, tested independently in selftest.
            a = np.vstack([j/scale, np.sqrt(damping)*np.eye(80)])
            rhs = -np.r_[r, np.zeros(80)]
            step = np.linalg.lstsq(a, rhs, rcond=1e-12)[0]/scale
            pp = p+step[:4].reshape(2, 2)*(C/B)
            vv = v+step[4:]
            okay = feasible(pp, vv)
            newcost = None
            if okay:
                rr, _ = residual(vv, pp, f, y)
                newcost = .5*float(rr@rr)
            if okay and newcost < value:
                p, v = pp, vv
                record.update(accepted=True, damping=damping, trial=trial, next_cost=newcost)
                damping = max(1e-12, damping/3)
                break
            damping *= 10
        if not record['accepted']:
            reason = 'no_feasible_decrease_in_trial_budget'
            break
    rr, _ = residual(v, p, f, y)
    return p, v, dict(cost=.5*float(rr@rr), reason=reason, history=history, feasible=feasible(p, v))


def solve(y, f):
    """Blind interface. Public constants only; no layout/channel/source inputs."""
    before = signature(y, f)
    start = time.perf_counter()
    score = score_positions(y, f, GRID)
    first = peaks(score)
    pairs = []
    for i, p0 in enumerate(first):
        g0, _, fit0 = rank_one(y, f, p0)
        rest = y-fit0
        second = peaks(score_positions(rest, f, GRID), exclude=[p0])
        for k, p1 in enumerate(second):
            g1, _, _ = rank_one(rest, f, p1)
            p = np.array([p0, p1])
            v = los_vector(np.column_stack([g0, g1]))
            pp, vv, meta = local_los(p, v, f, y)
            pairs.append(dict(id=f'{i}_{k}', coarse=p, los_p=pp, los_v=vv, los=meta))
    order = sorted(range(len(pairs)), key=lambda i: pairs[i]['los']['cost'])[:3]
    branches = []
    for index in order:
        item = pairs[index]
        p, v = item['los_p'], item['los_v']
        vv, seed = seed_paths(p, v, f, y)
        r, _ = residual(vv, p, f, y)
        pp, final_v, meta = refine(p, vv, f, y)
        branches.append(dict(pair_id=item['id'], initial_p=p, initial_v=vv,
                             seed=seed, seed_cost=.5*float(r@r), final_p=pp, final_v=final_v, final=meta))
    chosen = min(range(len(branches)), key=lambda i: branches[i]['final']['cost'])
    assert signature(y, f) == before
    return dict(input_signature=before, seconds=time.perf_counter()-start, pairs=pairs,
                branches=branches, chosen=chosen, output=branches[chosen]['final_p'])


def errors(p, truth):
    cost = np.sum((p[:, None]-truth[None])**2, axis=2)
    row, col = linear_sum_assignment(cost)
    err = np.empty(2)
    err[col] = np.sqrt(cost[row, col])
    return dict(source_errors_m=err, rmse_m=float(np.sqrt(np.mean(err**2))), both20=bool(np.all(err <= 20)))


def selftest():
    rng = np.random.default_rng(20261028011)
    f = np.linspace(-.49*B, .49*B, 31)
    y = rng.normal(size=(31, 5))+1j*rng.normal(size=(31, 5))
    points = np.array([[143., -256.], [1500., 1234.]])
    checks = {}
    for i, p in enumerate(points):
        g, s, fit = rank_one(y, f, p)
        z = y*np.exp(2j*np.pi*f[:, None]*np.linalg.norm(p[None]-STATIONS, axis=1)[None]/C)
        direct = np.linalg.svd(z, compute_uv=False)[0]**2
        score = score_positions(y, f, p[None])[0]
        checks[f'rank_one_{i}'] = bool(np.isclose(score, direct, rtol=1e-12))
        checks[f'fit_energy_{i}'] = bool(np.isclose(np.linalg.norm(y-fit)**2, np.linalg.norm(y)**2-score, rtol=1e-12))
        checks[f'wave_{i}'] = bool(np.allclose(s, z@g.conj()/np.vdot(g, g).real))
    p = points
    v = los_vector(np.ones((5, 2), complex))
    v[16:56] = rng.normal(0, .1, 40)
    v[56:] = 8.
    r, jn, _ = evaluate(v, p, f, y, 'physical')
    jp = position_jac(v, p, f, y)
    allj = np.column_stack([jp, jn])
    eps = 1e-6
    fd = []
    for k in range(4):
        dp = np.eye(4)[k].reshape(2, 2)*(C/B)*eps
        fd.append((residual(v, p+dp, f, y)[0]-residual(v, p-dp, f, y)[0])/(2*eps))
    checks['blind_position_jacobian'] = bool(np.linalg.norm(np.column_stack(fd)-jp)/np.linalg.norm(jp) < 1e-6)
    scale = np.linalg.norm(allj, axis=0)
    aug = np.vstack([allj, np.sqrt(.001)*np.diag(scale)])
    rr = np.r_[r, np.zeros(80)]
    a, b, _, _ = reduced_step(rr, aug[:, :4], aug[:, 4:], 1e-12)
    full = np.linalg.lstsq(aug, -rr, rcond=1e-12)[0]
    checks['damped_block_equivalence'] = bool(np.allclose(np.r_[a, b], full, rtol=1e-7, atol=1e-8))
    checks['matching_swap'] = errors(points[::-1], points)['rmse_m'] == 0
    checks['twenty_boundary'] = errors(points+np.array([20., 0]), points)['both20']
    checks['truth_not_grid_aligned'] = bool(np.all(GRID_AXIS % 20 == 10))
    checks['repeat_score'] = bool(np.array_equal(score_positions(y, f, points), score_positions(y, f, points)))
    assert all(checks.values()), checks
    return checks


# Historical main retained only in the evidence archive.


# Historical batch entry disabled; use the public mp_dpd API.
