# ruff: noqa: F401
# Preserve imported globals for legacy function cloning and reexports.
"""Local position-update elimination audit, not a blind localization solver."""
from pathlib import Path
from datetime import datetime
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
from structural_probe import ROOT, STATIONS, LAYOUTS, C, B, response, sha, write
from alternative_fit import SOURCE
from solver_reliability import datasets, physical, evaluate


def channel(v):
    los = np.ones((5, 2), complex)
    los[1:] = v[:8].reshape(4, 2) + 1j*v[8:16].reshape(4, 2)
    ratio = v[16:36].reshape(5, 2, 2) + 1j*v[36:56].reshape(5, 2, 2)
    gain = np.concatenate([los[:, :, None], los[:, :, None]*ratio], axis=2)
    delay = np.concatenate([np.zeros((5, 2, 1)), np.cumsum(v[56:].reshape(5, 2, 2), axis=2)], axis=2)
    return gain, delay


def position_jac(v, p, f, y):
    """Exact profiled residual derivative, coordinates measured in C/B metres."""
    gain, delay = channel(v)
    h, _ = response(p, delay, gain, f)
    inverse = np.linalg.pinv(h, rcond=1e-12)
    coef = np.einsum('fqm,fm->fq', inverse, y)
    residual = y-np.einsum('fmq,fq->fm', h, coef)
    unit = p[None]-STATIONS[:, None]
    unit /= np.linalg.norm(unit, axis=2)[:, :, None]
    dh = np.zeros((len(f), 5, 2, 4), complex)
    for q in range(2):
        for xy in range(2):
            dh[:, :, q, 2*q+xy] = h[:, :, q]*(-2j*np.pi*f[:, None]/B)*unit[None, :, q, xy]
    ds = np.einsum('fmqv,fq->fmv', dh, coef)
    j = -ds+np.einsum('fmq,fqv->fmv', h, np.einsum('fqm,fmv->fqv', inverse, ds))
    j -= np.einsum('fqm,fqv->fmv', inverse.conj(), np.einsum('fmqv,fm->fqv', dh.conj(), residual))
    scale = np.sqrt(2)/np.linalg.norm(y)
    return scale*np.concatenate([j.real.reshape(-1, 4), j.imag.reshape(-1, 4)])


def valid(v):
    gain, delay = channel(v)
    return bool(np.all(np.sum(abs(gain[:, :, 1:])**2, axis=2) < abs(gain[:, :, 0])**2)
                and np.all(abs(gain[:, :, 0]) > 0)
                and np.min(np.diff(delay, axis=2)) >= 3
                and np.max(np.diff(delay, axis=2)) <= 20)


def reduced_step(r, jp, jn, tol):
    # Column scaling preserves the nuisance column space. No weak directions
    # are removed except the declared relative SVD numerical-rank threshold.
    scale = np.linalg.norm(jn, axis=0)
    assert np.all(scale > 0)
    n = jn/scale
    u, s, vh = np.linalg.svd(n, full_matrices=False)
    rank = int(np.sum(s > s[0]*tol))
    basis = u[:, :rank]
    a = jp-basis@(basis.T@jp)
    b = r-basis@(basis.T@r)
    dp = np.linalg.lstsq(a, -b, rcond=tol)[0]
    dn = -(vh[:rank].T@((basis.T@(r+jp@dp))/s[:rank]))/scale
    return dp, dn, a, rank


# Historical main retained only in the evidence archive.


# Historical batch entry disabled; use the public mp_dpd API.
