"""AZ Algorithms 2.1/3.2 on the same ridge system, finite-point adaptation audit."""
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.linalg import svd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.research_discovery.finite_fft_operator import FiniteOperator
from 统一模型代码.gates.research_discovery.boundary_blind_pilot import Direct
from 统一模型代码.gates.research_discovery.residual_competition import HOLDOUT
from 统一模型代码.gates.g11.precision_diagnostic import sha, write


def az(op, y, rank, seed=20261023025):
    phase = (op.phase*op.gains[:, :, None]).transpose(2, 1, 0)
    d = op.length*np.einsum('flk,flj->fkj', phase.conj(), phase)+1e-6*np.eye(op.k)[None]
    values, vectors = np.linalg.eigh(d)
    inverse_root = (vectors/np.sqrt(values)[:, None, :])@vectors.conj().transpose(0, 2, 1)

    def root(x):
        x = np.asarray(x)
        shape = x.shape
        block = x.reshape(op.k, op.width, -1).transpose(1, 0, 2)
        return (inverse_root@block).transpose(1, 0, 2).reshape(shape)

    missing = np.exp(2j*np.pi*np.arange(op.n, op.length)[:, None]*op.f[None]/50e6)
    raw = op.phase*op.gains[:, :, None]
    v = np.concatenate([np.concatenate([missing*raw[k, l][None] for k in range(op.k)], axis=1) for l in range(4)], axis=0)
    w = root(v.conj().T).conj().T

    def t(x):
        return w.conj().T@(w@x)

    rhs = root(op.H@y)
    omega = np.random.default_rng(seed).normal(size=(op.shape[1], rank))
    first = t(omega)
    projected = first-t(first)  # (K-K^2)Omega, K=I-W^H W.
    target = t(rhs)
    u, s, vh = svd(projected, full_matrices=False, lapack_driver='gesdd')
    keep = s >= 1e-12
    z = vh[keep].conj().T@((u[:, keep].conj().T@target)/s[keep])
    x1 = omega@z
    answer = root(rhs+t(x1))
    return answer, dict(sketch_width=rank, retained_rank=int(keep.sum()),
        smallest_retained=float(s[keep][-1]), absolute_svd_cutoff=1e-12)


def main():
    out = HOLDOUT.parent/('round24_az_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    assert not out.exists() and out.resolve().is_relative_to(ROOT/'outputs_e2e')
    path = (HOLDOUT/'inputs/0_24000.npz').resolve(strict=True)
    identity = sha(path)
    out.mkdir()
    write(out/'protocol.json', dict(input=str(path), input_sha=identity, n=4096, gap=30,
        source='https://doi.org/10.1137/19M1306385', algorithms=['2.1', '3.2'],
        adaptation='symmetrically preconditioned same ridge normal system; Z*=I',
        sketch_widths=[256, 512, 768], cutoff=1e-12, seed=20261023025,
        caveat='numerical adaptation pilot, not a finished performance baseline', truth_use='fixed diagnostic point only'))
    with np.load(path, allow_pickle=False) as z:
        p = z['positions']+[[5, -3], [-4, 6]]
        y = (z['signal'][0, :, :4096]*10**(30/20)+z['signal'][1, :, :4096]+z['noise'][:, :4096]).ravel()
    records = []
    with threadpool_limits(limits=2):
        op = FiniteOperator(p, n=4096)
        reference = Direct(op).solve(op, y)
        for width in (256, 512, 768):
            start = time.perf_counter()
            c, detail = az(op, y, width)
            elapsed = time.perf_counter()-start
            normal = op.H@(op@c-y)+1e-6*c
            records.append(dict(detail=detail, seconds=elapsed,
                relative_normal_residual=float(np.linalg.norm(normal)/np.linalg.norm(op.H@y)),
                relative_coefficient_error=float(np.linalg.norm(c-reference)/np.linalg.norm(reference))))
            write(out/'results.json', records)
            print(width, records[-1], flush=True)
    assert sha(path) == identity
    write(out/'audit.json', dict(completed=True, input_unchanged=True,
        adaptation_not_performance_claim=True))
    print(out, flush=True)


if __name__ == '__main__':
    main()
