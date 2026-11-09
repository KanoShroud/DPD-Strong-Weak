# ruff: noqa: F401
# Preserve imported globals for legacy function cloning and reexports.
"""Single nonlinear reflection-addition audit; no adaptive-loop claim."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares, minimize, linear_sum_assignment
from scipy.stats import chi2
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
pass  # Explicit package imports replace historical search-path mutation.
from 研究方向.多径DPD.代码.mp_dpd.models.frequency_operator import B, C, STATIONS  # noqa: E402

UNIT = C/B
FS = 50e6


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, obj):
    def conv(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    p.write_text(json.dumps(obj, default=conv, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def real(z):
    return np.sqrt(2)*np.r_[z.real.ravel(), z.imag.ravel()]


def cols(z):
    return np.sqrt(2)*np.concatenate([z.real.reshape(-1, z.shape[-1]), z.imag.reshape(-1, z.shape[-1])])


def response(x, f, path=None):
    p = x[:4].reshape(2, 2)*UNIT
    g = np.ones((5, 2), complex)
    g[1:] = (x[4:12]+1j*x[12:20]).reshape(4, 2)
    vector = p[None]-STATIONS[:, None]
    distance = np.linalg.norm(vector, axis=2)
    if np.min(distance) < 1e-8:
        raise ValueError('Candidate at station; derivative undefined')
    direction = vector/distance[:, :, None]
    phase = np.exp(-2j*np.pi*f[:, None, None]*distance[None]/C)
    h = phase*g[None]
    factor = np.ones_like(h)
    if path is not None:
        m, q = path
        rho = x[20]+1j*x[21]
        extra = np.exp(-2j*np.pi*f*x[22]/B)
        factor[:, m, q] += rho*extra
        h *= factor
    dh = np.zeros((*h.shape, len(x)), complex)
    for q in range(2):
        for xy in range(2):
            dh[:, :, q, 2*q+xy] = h[:, :, q]*(-2j*np.pi*f[:, None]/B)*direction[None, :, q, xy]
        for m in range(1, 5):
            k = 4+2*(m-1)+q
            dh[:, m, q, k] = phase[:, m, q]*factor[:, m, q]
            dh[:, m, q, k+8] = 1j*dh[:, m, q, k]
    if path is not None:
        m, q = path
        base = phase[:, m, q]*g[m, q]*extra
        dh[:, m, q, 20] = base
        dh[:, m, q, 21] = 1j*base
        dh[:, m, q, 22] = base*rho*(-2j*np.pi*f/B)
    return h, dh


def evaluate(x, f, y, path=None, mean_tangent=False):
    h, dh = response(x, f, path)
    inv = np.linalg.pinv(h, rcond=1e-12)
    s = np.einsum('fqm,fm->fq', inv, y)
    r = y-np.einsum('fmq,fq->fm', h, s)
    ds = np.einsum('fmqk,fq->fmk', dh, s)
    pw_ds = ds-np.einsum('fmq,fqk->fmk', h, np.einsum('fqm,fmk->fqk', inv, ds))
    if mean_tangent:
        return real(r), cols(pw_ds), (h, inv, s)
    j = -pw_ds-np.einsum('fqm,fqk->fmk', inv.conj(), np.einsum('fmqk,fm->fqk', dh.conj(), r))
    norm = np.linalg.norm(y)
    return real(r)/norm, cols(j)/norm


def closure(x, n):
    z = x.copy()
    z[:4] = np.clip(z[:4], -3000/UNIT, 3000/UNIT)
    if len(z) == 23:
        z[20:22] /= max(1., np.linalg.norm(z[20:22]))
        z[22] = np.clip(z[22], 0., n/FS*B)
    return z


def fit(x, f, y, n, path=None):
    start = time.perf_counter()
    cache = [None, None]
    def update(z):
        if cache[0] is None or not np.array_equal(z, cache[0]):
            cache[:] = [z.copy(), evaluate(z, f, y, path)]
        return cache[1]
    if path is None:
        lo = np.r_[np.full(4, -3000/UNIT), np.full(16, -np.inf)]
        opt = least_squares(lambda z: update(z)[0], x, jac=lambda z: update(z)[1],
                            bounds=(lo, -lo), x_scale='jac', max_nfev=300,
                            ftol=1e-12, xtol=1e-12, gtol=1e-10)
        nit = int(opt.nfev)
    else:
        lo = np.r_[np.full(4, -3000/UNIT), np.full(18, -np.inf), 0.]
        hi = np.r_[np.full(4, 3000/UNIT), np.full(18, np.inf), n/FS*B]
        def obj(z):
            r, j = update(z)
            return .5*r@r, j.T@r
        def cj(z):
            v = np.zeros(23)
            v[20:22] = -2*z[20:22]
            return v
        opt = minimize(obj, closure(x, n), jac=True, method='SLSQP', bounds=list(zip(lo, hi)),
                       constraints=[dict(type='ineq', fun=lambda z: 1-z[20:22]@z[20:22], jac=cj)],
                       options=dict(maxiter=300, ftol=1e-12))
        nit = int(opt.nit)
    rr, jj = evaluate(opt.x, f, y, path)
    grad = jj.T@rr
    mapping = np.linalg.norm(opt.x-closure(opt.x-grad, n), ord=np.inf)
    return opt.x, dict(cost=float(.5*rr@rr), status=int(opt.status), success=bool(opt.success),
                       message=str(opt.message), iterations=nit, seconds=time.perf_counter()-start,
                       projected_gradient=float(mapping), stationary=bool(mapping <= 1e-7),
                       p=opt.x[:4].reshape(2, 2)*UNIT,
                       reflection_power=None if path is None else float(opt.x[20:22]@opt.x[20:22]),
                       normalized_delay=None if path is None else float(opt.x[22]), x=opt.x)


def search(x, f, y, n, use, sigma2):
    start = time.perf_counter()
    r, j, (h, inv, s) = evaluate(x, f, y, mean_tangent=True)
    un, sn, _ = np.linalg.svd(j[:, 4:], full_matrices=False)
    if sn[-1] <= sn[0]*1e-10:
        raise ValueError('LOS gain tangent singular')
    a = j[:, :4]-un@(un.T@j[:, :4])
    l0 = np.linalg.pinv(a, rcond=1e-10)
    uq, sq, _ = np.linalg.svd(j, full_matrices=False)
    if sq[-1] <= sq[0]*1e-10:
        raise ValueError('Full LOS tangent singular')
    rp = r-uq@(uq.T@r)
    vectors = np.column_stack([uq, l0.T, rp])
    count = len(f)*5
    vc = (vectors[:count]+1j*vectors[count:]).reshape(len(f), 5, -1)
    all_rows, numerical = [], {}
    candidates = 10*(n-1)
    threshold = float(chi2.isf(.05/candidates, 2))
    ph = np.einsum('fmq,fqm->fm', h, inv).real
    for m in range(5):
        for q in range(2):
            signal = h[:, m, q]*s[:, q]
            padded = np.zeros((n, vectors.shape[1]), complex)
            padded[use] = vc[:, m].conj()*signal[:, None]
            dot = np.sqrt(2)*np.fft.fft(padded, axis=0)
            cross = np.stack([dot.real, -dot.imag], axis=-1)
            norm2 = float(2*np.sum(abs(signal)**2*(1-ph[:, m])))
            residual_gram = norm2*np.eye(2)[None]-np.einsum('nki,nkj->nij', cross[:, :20], cross[:, :20])
            eigen = np.linalg.eigvalsh(residual_gram)
            valid = (eigen[:, 0] > norm2*1e-10)
            valid[0] = False
            ca = np.zeros_like(residual_gram)
            ca[valid] = np.linalg.inv(residual_gram[valid])
            btr = cross[:, 24]
            amplitude = np.einsum('nij,nj->ni', ca, btr)
            d = cross[:, 20:24]
            shift = np.einsum('nki,ni->nk', d, amplitude)*UNIT
            # j position coordinates are C/B; convert both risk terms to m^2.
            penalty = sigma2*np.einsum('nki,nij,nkj->n', d, ca, d)*UNIT**2
            evidence = np.einsum('ni,ni->n', amplitude, btr)/sigma2
            gain = np.sum(shift**2, axis=1)-2*penalty
            eligible = valid & (evidence > threshold) & (gain > 0)
            for delay_index in (1, n//3, n-1):
                b = np.zeros((len(f), 5, 2), complex)
                b[:, m, 0] = signal*np.exp(-2j*np.pi*f*delay_index/FS)
                b[:, m, 1] = 1j*b[:, m, 0]
                b -= np.einsum('fmq,fqk->fmk', h, np.einsum('fqm,fmk->fqk', inv, b))
                direct = cols(b)
                numerical[f'fft_{m}_{q}_{delay_index}'] = bool(np.allclose(vectors.T@direct, cross[delay_index], rtol=2e-7, atol=1e-8*np.linalg.norm(direct)))
            # Every candidate retained, not only favorable ones.
            for k in range(1, n):
                all_rows.append([m, q, k, int(valid[k]), evidence[k], gain[k], penalty[k],
                                 amplitude[k, 0], amplitude[k, 1], int(eligible[k])])
    bank = np.asarray(all_rows)
    picks = {}
    for name, col in [('RISK', 5), ('FIT', 4)]:
        allowed = (bank[:, 3] == 1) & (bank[:, 4] > threshold)
        if name == 'RISK':
            allowed &= bank[:, 5] > 0
        ids = np.flatnonzero(allowed)
        idx = None if not len(ids) else int(ids[np.argmax(bank[ids, col])])
        if idx is None:
            picks[name] = None
        else:
            row = bank[idx]
            picks[name] = dict(station=int(row[0]), source=int(row[1]), sample_delay=int(row[2]),
                               normalized_delay=float(row[2]/FS*B), evidence=float(row[4]),
                               predicted_gain_m2=float(row[5]), penalty_m2=float(row[6]),
                               rho_real=float(row[7]), rho_imag=float(row[8]))
    return picks, bank, dict(seconds=time.perf_counter()-start, threshold=threshold,
                            candidates=candidates, regular_count=int(np.sum(bank[:, 3])), checks=numerical,
                            scope='Estimated-design plug-in; nominal multiple-test threshold is not an exact nonlinear guarantee')


def error(p, truth):
    dist = np.linalg.norm(np.asarray(p)[:, None]-truth[None], axis=2)
    ii, jj = linear_sum_assignment(dist**2)
    e = np.zeros(2)
    e[jj] = dist[ii, jj]
    return e


def cases(short):
    for folder, inputs, layouts, reps in [
        ('consensus_development_20261029_223235_334946', 'independent_backend_20261029_222111_026721', range(20, 24), range(200, 203)),
        ('consensus_validation_20261029_223520_495993', 'consensus_validation_20261029_223520_495993', range(40, 44), range(300, 303))]:
        for li in layouts:
            for rep in reps:
                if short and (li not in (20, 40) or rep not in (200, 300)):
                    continue
                for mode in ('LOS', 'MP'):
                    tag = f'L{li}_R{rep}_{mode}'
                    yield tag, BASE/folder/(tag+'.json'), BASE/inputs/f'L{li}_R{rep}_input.npz', mode


# Historical main retained only in the evidence archive.


# Historical batch entry disabled; use the public mp_dpd API.
