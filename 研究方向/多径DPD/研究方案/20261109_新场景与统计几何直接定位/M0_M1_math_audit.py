#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure NumPy mathematical sanity checks for the four-station dual-source M0/M1 study.

This script does NOT run IQ localization, estimate a fourth-order polyspectrum,
prove finite-sample identifiability, or modify the original repository.
"""
from __future__ import annotations
import itertools
import json
import numpy as np

C = 299792458.0
A = 3000.0
BS = np.array([[A, A], [-A, A], [-A, -A], [A, -A]], dtype=float)


def cumulant_counterexample():
    """Each of two sources contributes TWO independent time-indexed innovations.

    Demonstrates that the zero-time fourth-order sensor cumulant is generically
    rank FOUR, not automatically the sum of only two rank-one spatial tensors.
    """
    rng = np.random.default_rng(2026110901)
    coeff = rng.normal(size=(4, 4))  # 4 receivers x 4 innovations (2 per source)
    bits = np.asarray(list(itertools.product((-1.0, 1.0), repeat=4)))
    x = bits @ coeff.T
    cov = x.T @ x / len(x)
    moment4 = np.einsum("ni,nj,nk,nl->ijkl", x, x, x, x) / len(x)
    cum4 = (moment4 - np.einsum("ij,kl->ijkl", cov, cov)
            - np.einsum("ik,jl->ijkl", cov, cov)
            - np.einsum("il,jk->ijkl", cov, cov))
    # Each independent BPSK symbol has fourth cumulant 1 - 3 = -2.
    expected = -2.0 * np.einsum("ir,jr,kr,lr->ijkl", coeff, coeff, coeff, coeff)
    err = float(np.max(abs(cum4 - expected)))
    singular = np.linalg.svd(cum4.reshape(16, 16), compute_uv=False)
    rank = int(np.sum(singular > singular[0] * 1e-9))
    assert err < 1e-11, err
    assert rank == 4, (rank, singular)
    return {"case": "naive_zero_lag_FOC_rank", "sources": 2,
            "innovations_per_source": 2, "spatial_cumulant_matrix_rank": rank,
            "max_exact_enumeration_error": err,
            "warning": "zero-lag four-sensor tensor is not generically CP-rank-two"}


def frequency_tensor_algebra():
    """Only checks population algebra with ASSUMED nonzero source trispectra."""
    rng = np.random.default_rng(2026110902)
    h = [rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2)) for _ in range(4)]
    weights = np.array([-2.0, -0.8])
    tensor = sum(weights[q] * np.einsum("i,j,k,l->ijkl",
                    h[0][:, q], h[1][:, q], h[2][:, q].conj(), h[3][:, q].conj())
                 for q in range(2))
    singular = np.linalg.svd(tensor.reshape(16, 16), compute_uv=False)
    assert all(np.linalg.matrix_rank(x) == 2 for x in h)
    assert np.sum(singular > singular[0] * 1e-9) == 2
    return {"case": "ideal_frequency_fourway_tensor", "sources": 2,
            "unfold_rank": 2, "mode_kruskal_ranks": [2, 2, 2, 2],
            "third_order_grouped_kruskal_sum": 6,
            "third_order_unique_threshold": 2 * 2 + 2,
            "warning": "assumed population tensor, not an estimator on finite IQ"}


def dft_bpsk_fourth_scaling():
    """Exactly enumerate all BPSK sequences to verify C4(Uk,Ul,Uk*,Ul*)=-2/N.

    This is a single-frequency-quartet cumulant of *normalized full DFT bins*.
    Its magnitude tends to zero as the Fourier frame length increases.
    """
    cases = []
    for n in (4, 8, 12):
        k, ell = 1, 2
        symbols = np.array(list(itertools.product((-1.0, 1.0), repeat=n)))
        nn = np.arange(n)
        u = symbols @ np.exp(-2j * np.pi * k * nn / n) / np.sqrt(n)
        v = symbols @ np.exp(-2j * np.pi * ell * nn / n) / np.sqrt(n)
        av = [u, v, u.conj(), v.conj()]
        avg = lambda x: np.mean(x)
        c4 = (avg(av[0]*av[1]*av[2]*av[3])
              -avg(av[0]*av[1])*avg(av[2]*av[3])
              -avg(av[0]*av[2])*avg(av[1]*av[3])
              -avg(av[0]*av[3])*avg(av[1]*av[2]))
        target = -2.0/n
        assert abs(c4-target) < 1e-11, (n,c4,target)
        cases.append({"symbol_count":n,"cumulant_real":float(c4.real),
                      "target":target,"error":float(abs(c4-target))})
    return {"case":"BPSK_DFT_fourth_cumulant_shrinks", "exact_enumerations":cases,
            "approx_magnitude_at_6554_symbols":2.0/6554.0,
            "warning":"population information nonzero; naive full-frame one-shot estimation is ill-conditioned"}


def tdoa(p):
    r = np.linalg.norm(BS - np.asarray(p), axis=1)
    return r - r[0]


def bisect_monotone(fn, target, low=-A, high=A):
    for _ in range(75):
        mid = (low + high) / 2
        if fn(mid) < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def invert_square_tdoa(d):
    """Algebraic inverse except on axes, where monotonic 1D inversion is used."""
    d2, d3, d4 = np.asarray(d)[1:]
    gap = d3 - d2 - d4
    if abs(gap) > 1e-6:
        r1 = (d2*d2 + d4*d4 - d3*d3)/(2*gap)
        return np.array([d2*(2*r1+d2)/(4*A), d4*(2*r1+d4)/(4*A)])
    if abs(d2) <= 1e-7:
        yy = bisect_monotone(lambda y: np.hypot(A, y+A) - np.hypot(A, y-A), d4)
        return np.array([0.0, yy])
    if abs(d4) <= 1e-7:
        xx = bisect_monotone(lambda x: np.hypot(x+A, A) - np.hypot(x-A, A), d2)
        return np.array([xx, 0.0])
    raise AssertionError("incompatible exact TDOA: singular algebra and both d2,d4 nonzero")


def geometry_inverse():
    rng = np.random.default_rng(2026110903)
    positions = np.vstack([rng.uniform(-2400, 2400, size=(2000, 2)),
                           np.array([[0, 0], [0, 1100], [1000, 0],
                                     [-2200, 0], [0, -2000]])])
    estimated = np.asarray([invert_square_tdoa(tdoa(p)) for p in positions])
    errors = np.linalg.norm(estimated - positions, axis=1)
    assert errors.max() < 1e-4, (errors.max(), positions[errors.argmax()])
    return {"case": "square_four_station_TDOA_inverse", "positions_checked": len(positions),
            "max_position_reconstruction_error_m": float(errors.max()),
            "warning": "numerical sanity only; analytic injectivity proof is in the Markdown memo"}


def direction_gauge_cross_relation():
    f = np.linspace(-6.25e6, 6.25e6, 161)
    true_p = np.array([730.0, -1100.0])
    false_p = np.array([-910.0, 750.0])
    tau = np.linalg.norm(BS - true_p, axis=1) / C
    tau_wrong = np.linalg.norm(BS - false_p, axis=1) / C
    gain = np.array([1 + 0.4j, .85 - .2j, -.64 + .3j, .94 + .16j])
    gain_wrong = np.array([.74+.2j, .99-.1j, -.54+.1j, .9+.22j])
    delta = np.array([.31, .72, 1.1, .61]) * 1e-6
    alpha = np.array([.2+.15j, -.24+.08j, .13-.26j, .28+.04j])
    def H(t, g):
        return g[:, None] * np.exp(-2j*np.pi*t[:, None]*f[None, :]) * (
            1 + alpha[:, None] * np.exp(-2j*np.pi*delta[:, None]*f[None, :]))
    h = H(tau, gain)
    d = 1.0 + .2*np.exp(-2j*np.pi*f*.36e-6)  # source-common gauge, independent of station
    equivalent = h * d[None, :]
    false_h = H(tau_wrong, gain_wrong)
    def closure(a, b):
        return float(np.linalg.norm(a[1:]*b[:1]-a[:1]*b[1:]) / np.linalg.norm(h[1:]*h[:1]))
    equiv_error = closure(equivalent, h)
    false_error = closure(false_h, h)
    assert equiv_error < 1e-12
    assert false_error > 1e-3
    return {"case": "frequency_scalar_gauge_and_cross_relation",
            "same_position_arbitrary_shared_filter_closure": equiv_error,
            "different_position_trial_closure": false_error,
            "warning": "illustration only; arbitrary wrong channels may approach the true manifold in noise"}


def main():
    results = [cumulant_counterexample(), frequency_tensor_algebra(),
               dft_bpsk_fourth_scaling(), geometry_inverse(),
               direction_gauge_cross_relation()]
    return {"status": "PASS", "scope": "standalone exact algebra and numerical checks, no RF IQ", "tests": results}


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
