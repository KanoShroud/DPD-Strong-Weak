# ruff: noqa: F401
# Preserve imported globals for legacy function cloning and reexports.
"""Sequential score-test multipath reference, not an IARD reproduction."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
pass  # Explicit package imports replace historical search-path mutation.
from 研究方向.多径DPD.代码.mp_dpd.numerics.vp_core import B, C, FS, UNIT, STATIONS, cols, real, sha, write, error, cases, fit as los_fit  # noqa: E402


def response(x, f, paths):
    p = x[:4].reshape(2, 2)*UNIT
    v = p[None]-STATIONS[:, None]
    dist = np.linalg.norm(v, axis=2)
    if dist.min() < 1e-8:
        raise ValueError('Station coincidence')
    g = np.ones((5, 2), complex)
    g[1:] = (x[4:12]+1j*x[12:20]).reshape(4, 2)
    phase = np.exp(-2j*np.pi*f[:, None, None]*dist[None]/C)
    base = phase*g[None]
    fac = np.ones_like(base)
    for i, (m, q) in enumerate(paths):
        a, b, delay = x[20+3*i:23+3*i]
        fac[:, m, q] += (a+1j*b)*np.exp(-2j*np.pi*f*delay/B)
    h = base*fac
    dh = np.zeros((*h.shape, len(x)), complex)
    for q in range(2):
        for xy in range(2):
            dh[:, :, q, 2*q+xy] = h[:, :, q]*(-2j*np.pi*f[:, None]/B)*(v[:, q, xy]/dist[:, q])[None]
        for m in range(1, 5):
            k = 4+2*(m-1)+q
            dh[:, m, q, k] = phase[:, m, q]*fac[:, m, q]
            dh[:, m, q, k+8] = 1j*dh[:, m, q, k]
    for i, (m, q) in enumerate(paths):
        k = 20+3*i
        z = base[:, m, q]*np.exp(-2j*np.pi*f*x[k+2]/B)
        dh[:, m, q, k] = z
        dh[:, m, q, k+1] = 1j*z
        dh[:, m, q, k+2] = z*(x[k]+1j*x[k+1])*(-2j*np.pi*f/B)
    return h, dh, base


def evaluate(x, f, y, paths, tangent=False):
    h, dh, base = response(x, f, paths)
    inv = np.linalg.pinv(h, rcond=1e-12)
    s = np.einsum('fqm,fm->fq', inv, y)
    r = y-np.einsum('fmq,fq->fm', h, s)
    ds = np.einsum('fmqk,fq->fmk', dh, s)
    jp = ds-np.einsum('fmq,fqk->fmk', h, np.einsum('fqm,fmk->fqk', inv, ds))
    if tangent:
        return real(r), cols(jp), (h, inv, s, base)
    j = -jp-np.einsum('fqm,fqk->fmk', inv.conj(), np.einsum('fmqk,fm->fqk', dh.conj(), r))
    return real(r)/np.linalg.norm(y), cols(j)/np.linalg.norm(y)


def project(x, n):
    z = x.copy()
    z[:4] = np.clip(z[:4], -3000/UNIT, 3000/UNIT)
    for k in range(20, len(z), 3):
        z[k:k+2] /= max(1., np.linalg.norm(z[k:k+2]))
        z[k+2] = np.clip(z[k+2], 0., n/FS*B)
    return z


def fit(x0, f, y, paths, n, budget=300, recenter=True):
    start = time.perf_counter()
    x0 = project(x0, n)
    r0, j0 = evaluate(x0, f, y, paths)
    scale = np.maximum(np.linalg.norm(j0, axis=0), 1e-10)
    cost_scale = max(.5*r0@r0, 1e-12)
    cache = [None, None]
    calls = 0
    def unpack(z):
        return x0+z/scale
    def obj(z):
        nonlocal calls
        if cache[0] is None or not np.array_equal(cache[0], z):
            cache[:] = [z.copy(), evaluate(unpack(z), f, y, paths)]
            calls += 1
        r, j = cache[1]
        return .5*r@r/cost_scale, (j.T@r)/scale/cost_scale
    def cons(z):
        v = unpack(z)[20:].reshape(-1, 3)
        return 1-np.sum(v[:, :2]**2, axis=1)
    def jac(z):
        x = unpack(z)
        a = np.zeros((len(paths), len(x)))
        for i in range(len(paths)):
            k = 20+3*i
            a[i, k:k+2] = -2*x[k:k+2]/scale[k:k+2]
        return a
    low = np.full(len(x0), -np.inf)
    high = -low.copy()
    low[:4], high[:4] = -3000/UNIT, 3000/UNIT
    low[22::3], high[22::3] = 0., n/FS*B
    opt = minimize(obj, np.zeros(len(x0)), jac=True, method='SLSQP',
                   bounds=list(zip((low-x0)*scale, (high-x0)*scale)),
                   constraints=[dict(type='ineq', fun=cons, jac=jac)] if paths else [],
                   options=dict(maxiter=budget, ftol=1e-12))
    x = unpack(opt.x)
    r, j = evaluate(x, f, y, paths)
    mapping = float(np.max(abs(x-project(x-j.T@r, n))))
    result = dict(x=x, p=x[:4].reshape(2, 2)*UNIT, cost=float(.5*r@r),
                  projected_gradient=mapping, stationary=mapping <= 1e-7,
                  iterations=int(opt.nit), success=bool(opt.success), message=str(opt.message),
                  calls=calls, seconds=time.perf_counter()-start,
                  closure_error=float(np.linalg.norm(x-project(x, n))))
    # One numerical recentering ONLY after optimizer tolerance termination.
    # Total iterations remain <=300; no localization truth used for this decision.
    if recenter and opt.success and not result['stationary'] and opt.nit < budget:
        xx, other = fit(x, f, y, paths, n, budget-int(opt.nit), False)
        attempts = [result.copy(), other.copy()]
        if other['cost'] <= result['cost']:
            x, result = xx, other
        result['attempts'] = attempts
        result['iterations'] = sum(a['iterations'] for a in attempts)
        result['calls'] = sum(a['calls'] for a in attempts)
        result['seconds'] = time.perf_counter()-start
    return x, result


def search(x, paths, f, y, n, use, sigma2, step):
    start = time.perf_counter()
    r, j, (h, inv, s, base) = evaluate(x, f, y, paths, True)
    u, singular, _ = np.linalg.svd(j, full_matrices=False)
    rank = int(np.sum(singular > singular[0]*1e-10))
    u = u[:, :rank]
    rp = r-u@(u.T@r)
    vectors = np.column_stack([u, rp])
    nf = len(f)*5
    vc = (vectors[:nf]+1j*vectors[nf:]).reshape(len(f), 5, -1)
    ph = np.einsum('fmq,fqm->fm', h, inv).real
    threshold = float(chi2.isf(.05/(step*(step+1)*10*(n-1)), 2))
    checks, banks = {}, []
    for m in range(5):
        for q in range(2):
            signal = base[:, m, q]*s[:, q]
            padded = np.zeros((n, vectors.shape[1]), complex)
            padded[use] = vc[:, m].conj()*signal[:, None]
            dot = np.sqrt(2)*np.fft.fft(padded, axis=0)
            cross = np.stack([dot.real, -dot.imag], axis=-1)
            norm2 = float(2*np.sum(abs(signal)**2*(1-ph[:, m])))
            gram = norm2*np.eye(2)[None]-np.einsum('nki,nkj->nij', cross[:, :rank], cross[:, :rank])
            val = np.linalg.eigvalsh(gram)
            valid = val[:, 0] > norm2*1e-10
            valid[0] = False
            amp = np.zeros((n, 2))
            amp[valid] = np.linalg.solve(gram[valid], cross[valid, rank, :, None])[..., 0]
            evidence = np.einsum('ni,ni->n', amp, cross[:, rank])/sigma2
            for delay in (1, n//3, n-1):
                bc = np.zeros((len(f), 5, 2), complex)
                bc[:, m, 0] = signal*np.exp(-2j*np.pi*f*delay/FS)
                bc[:, m, 1] = 1j*bc[:, m, 0]
                bc -= np.einsum('fmq,fqk->fmk', h, np.einsum('fqm,fmk->fqk', inv, bc))
                direct = cols(bc)
                checks[f'fft_{m}_{q}_{delay}'] = bool(np.allclose(vectors.T@direct, cross[delay], rtol=2e-7, atol=1e-8*np.linalg.norm(direct)))
            banks.append(np.column_stack([np.full(n-1, m), np.full(n-1, q), np.arange(1, n), valid[1:], evidence[1:], amp[1:]]))
    bank = np.vstack(banks)
    ids = np.flatnonzero((bank[:, 3] == 1) & (bank[:, 4] > threshold))
    pick = None
    if len(ids):
        a = bank[ids[np.argmax(bank[ids, 4])]]
        pick = dict(m=int(a[0]), q=int(a[1]), sample_delay=int(a[2]), evidence=float(a[4]),
                    rho=[float(a[5]), float(a[6])], delay=float(a[2]/FS*B))
    return pick, bank, dict(rank=rank, columns=len(x), threshold=threshold, checks=checks,
                            seconds=time.perf_counter()-start, candidates=len(bank))


def selftest(f, y, n, x):
    checks = {}
    rng = np.random.default_rng(2026103021)
    paths = [(1, 0), (1, 0), (0, 1), (3, 1)]
    xx = np.r_[x, .1, .07, 1.3, -.2, .1, 7.1, .05, -.12, 22.8, 0., 0., 3.9]
    smallf, smally = f[::29], y[::29]
    rr, jj = evaluate(xx, smallf, smally, paths)
    eps = 1e-5
    num = np.column_stack([(evaluate(xx+eps*np.eye(len(xx))[k], smallf, smally, paths)[0]-evaluate(xx-eps*np.eye(len(xx))[k], smallf, smally, paths)[0])/(2*eps) for k in range(len(xx))])
    checks['multi_path_exact_vp_jacobian'] = bool(np.linalg.norm(num-jj)/np.linalg.norm(num) < 2e-6)
    from 研究方向.多径DPD.代码.mp_dpd.numerics.vp_core import evaluate as old_evaluate
    oldr, oldj = old_evaluate(x, f, y)
    newr, newj = evaluate(x, f, y, [])
    checks['los_equivalence'] = bool(np.allclose(oldr, newr) and np.allclose(oldj, newj))
    one = xx[:23]
    oldr, oldj = old_evaluate(one, f, y, paths[0])
    newr, newj = evaluate(one, f, y, paths[:1])
    checks['one_path_equivalence'] = bool(np.allclose(oldr, newr) and np.allclose(oldj, newj))
    worst = 0.
    for _ in range(100):
        a = rng.normal(size=len(xx))*5
        pa, pb = project(a, n), project(rng.normal(size=len(xx))*5, n)
        worst = max(worst, float((a-pa)@(pb-pa)))
    checks['convex_product_projection'] = worst < 1e-9
    return checks


# Historical main retained only in the evidence archive.


# Historical batch entry disabled; use the public mp_dpd API.
