"""Station-local RF EM/MM bridge on a given candidate support, not a geolocator.

The candidate support is estimator state, never the unknown physical path list.
Each source retains ONE unknown coefficient per frequency, shared by its paths.
The frequency-domain and coordinate conventions exactly match frozen reference.py.
"""
from pathlib import Path
import sys

import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / 'unknown_order_20261030'))
import reference as ref  # noqa: E402


def components(x, f, paths):
    """Return sparse station-local coefficients C,F and derivatives C,F,D.

    First ten components are LOS, ordered m then q. The remaining components
    are candidate reflections. x is the ORIGINAL dimensionless optimizer
    coordinate: p/(c/B), complex relative LOS gains, then (Re rho, Im rho, B*tau).
    """
    x, f = np.asarray(x, float), np.asarray(f, float)
    if x.shape != (20 + 3 * len(paths),) or f.ndim != 1:
        raise ValueError('parameter/frequency shape')
    if not np.isfinite(x).all() or not np.isfinite(f).all():
        raise ValueError('nonfinite parameter/frequency')
    if any(m not in range(5) or q not in range(2) for m, q in paths):
        raise ValueError('candidate source/station outside Q2/M5')
    p = x[:4].reshape(2, 2) * ref.UNIT
    v = p[None] - ref.STATIONS[:, None]
    dist = np.linalg.norm(v, axis=2)
    if dist.min() < 1e-8:
        raise ValueError('station coincidence')
    gain = np.ones((5, 2), complex)
    gain[1:] = (x[4:12] + 1j * x[12:20]).reshape(4, 2)
    sites = [(m, q, None) for m in range(5) for q in range(2)]
    sites += [(m, q, i) for i, (m, q) in enumerate(paths)]
    a = np.zeros((len(sites), len(f)), complex)
    da = np.zeros((*a.shape, len(x)), complex)
    for c, (m, q, i) in enumerate(sites):
        phase = np.exp(-2j * np.pi * f * dist[m, q] / ref.C)
        factor = np.ones(len(f), complex)
        if i is not None:
            k = 20 + 3 * i
            delay_phase = np.exp(-2j * np.pi * f * x[k + 2] / ref.B)
            factor = (x[k] + 1j * x[k + 1]) * delay_phase
        a[c] = phase * gain[m, q] * factor
        da[c, :, 2*q:2*q+2] = (a[c, :, None] * (-2j*np.pi*f[:, None]/ref.B)
                                      * v[m, q][None] / dist[m, q])
        if m:
            k_gain = 4 + 2*(m-1) + q
            da[c, :, k_gain] = phase * factor
            da[c, :, k_gain + 8] = 1j * phase * factor
        if i is not None:
            k = 20 + 3*i
            da[c, :, k] = phase * gain[m, q] * delay_phase
            da[c, :, k+1] = 1j * da[c, :, k]
            da[c, :, k+2] = a[c] * (-2j*np.pi*f/ref.B)
    station = np.asarray([m for m, _, _ in sites])
    source = np.asarray([q for _, q, _ in sites])
    return a, da, station, source


def aggregate(a, da, station, source):
    h = np.zeros((a.shape[1], 5, 2), complex)
    dh = np.zeros((*h.shape, da.shape[-1]), complex)
    for c, (m, q) in enumerate(zip(station, source)):
        h[:, m, q] += a[c]
        dh[:, m, q] += da[c]
    return h, dh


def station_beta(station):
    """Uniform positive allocation WITHIN each station; not noise divided twice."""
    counts = np.bincount(station, minlength=5)
    if np.any(counts == 0):
        raise ValueError('every station requires a component')
    return 1.0 / counts[station]


def means(a, s, source):
    return a * s[:, source].T


def observed_cost(y, mu, station):
    fitted = np.zeros_like(y)
    for c, m in enumerate(station):
        fitted[:, m] += mu[c]
    return float(np.sum(abs(y-fitted)**2) / np.sum(abs(y)**2))


def expectation(y, mu, station, beta):
    beta = np.asarray(beta, float)
    if beta.shape != station.shape or np.any(beta <= 0):
        raise ValueError('positive beta required')
    if not np.allclose(np.bincount(station, weights=beta, minlength=5), 1., atol=1e-12):
        raise ValueError('beta must sum to one at EACH station')
    fitted = np.zeros_like(y)
    for c, m in enumerate(station):
        fitted[:, m] += mu[c]
    return mu + beta[:, None] * (y-fitted)[:, station].T


def surrogate(x, f, paths, s, z, beta, norm2):
    """Normalized surrogate and exact gradient in ORIGINAL optimizer coordinates."""
    a, da, _, source = components(x, f, paths)
    r = means(a, s, source) - z
    dm = da * s[:, source].T[:, :, None]
    cost = float(np.sum(abs(r)**2/beta[:, None])/norm2)
    grad = 2*np.einsum('cf,cfk,c->k', r.conj(), dm, 1/beta).real/norm2
    return cost, grad


def shared_wave_mstep(a, source, z, beta):
    """Minimize frozen surrogate; never give each path its own waveform."""
    s = np.zeros((a.shape[1], 2), complex)
    for q in range(2):
        use = source == q
        denominator = np.sum(abs(a[use])**2/beta[use, None], axis=0)
        if np.any(denominator <= 0):
            raise ValueError('unexcited source coefficient')
        s[:, q] = np.sum(a[use].conj()*z[use]/beta[use, None], axis=0)/denominator
    return s


def profile_wave(h, y):
    return np.einsum('fqm,fm->fq', np.linalg.pinv(h, rcond=1e-12), y)


def bounded_mm_step(x, f, y, paths, n, max_backtracks=30):
    """ONE correctness-test generalized EM step, not a frozen production backend.

    E and M share the current profiled waves. x update lowers the surrogate,
    then observed-data common-wave LS cannot increase the observed objective.
    The reference convex projection is applied in ORIGINAL optimizer coordinates.
    """
    a, da, station, source = components(x, f, paths)
    h, _ = aggregate(a, da, station, source)
    s = profile_wave(h, y)
    mu = means(a, s, source)
    beta = station_beta(station)
    z = expectation(y, mu, station, beta)
    norm2 = float(np.sum(abs(y)**2))
    old, grad = surrogate(x, f, paths, s, z, beta, norm2)
    candidate, accepted, calls = x.copy(), False, 0
    for k in range(max_backtracks):
        trial = ref.project(x-(.5**k)*grad, n)
        value, _ = surrogate(trial, f, paths, s, z, beta, norm2)
        calls += 1
        if value < old:
            candidate, accepted = trial, True
            break
    aa, dd, _, _ = components(candidate, f, paths)
    hh, _ = aggregate(aa, dd, station, source)
    ss = profile_wave(hh, y)
    final = observed_cost(y, means(aa, ss, source), station)
    return candidate, dict(accepted=accepted, old_cost=old, final_cost=final,
                          surrogate_calls=calls, surrogate_cost=value,
                          projected_gradient=float(np.max(abs(x-ref.project(x-grad, n)))))
