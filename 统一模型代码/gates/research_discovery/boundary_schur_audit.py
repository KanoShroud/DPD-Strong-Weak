"""Controlled missing-row compression pilot; no blind localization or novelty claim."""
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve, svd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.research_discovery.finite_fft_operator import FiniteOperator
from 统一模型代码.gates.research_discovery.finite_gram_audit import gram, objective
from 统一模型代码.gates.research_discovery.residual_competition import HOLDOUT
from 统一模型代码.gates.g11.precision_diagnostic import sha, write


def prepare(op, relative=1e-6, ridge=1e-6):
    start = time.perf_counter()
    t = np.arange(op.n, op.length)
    v = np.exp(2j*np.pi*t[:, None]*op.f[None]/50e6)
    u, singular, _ = svd(v, full_matrices=False, lapack_driver='gesdd')
    tail = np.r_[np.cumsum(singular[::-1]**2)[::-1], 0.]
    threshold = relative*ridge*op.n/op.k
    rank = int(np.flatnonzero(tail <= threshold)[0])
    q = u[:, :rank]
    h = q.conj().T@v
    actual_tail = float(np.linalg.norm(v-q@h, 'fro')**2)
    assert actual_tail*op.k/op.n <= relative*ridge*1.01
    return h, dict(rank=rank, row_dimension=4*rank, coefficient_dimension=op.shape[1],
        full_missing_rows=4*(op.length-op.n), coefficient_relative_bound=actual_tail*op.k/op.n/ridge,
        preparation_seconds=time.perf_counter()-start)


def schur(op, y, h, ridge=1e-6):
    phase = op.phase*op.gains[:, :, None]
    blocks = phase.transpose(2, 1, 0)
    d = op.length*np.einsum('flk,flj->fkj', blocks.conj(), blocks)+ridge*np.eye(op.k)[None]
    small = np.concatenate([np.concatenate([h*phase[k, l][None] for k in range(op.k)], axis=1)
                            for l in range(4)], axis=0)
    rhs = op.H@y
    dinv_rhs = np.linalg.solve(d, rhs.reshape(op.k, op.width).T[..., None]).squeeze(-1).T.ravel()
    reshaped = small.conj().T.reshape(op.k, op.width, -1).transpose(1, 0, 2)
    dinv_vh = np.linalg.solve(d, reshaped).transpose(1, 0, 2).reshape(op.shape[1], -1)
    s = np.eye(small.shape[0])-small@dinv_vh
    s = (s+s.conj().T)/2
    correction = cho_solve(cho_factor(s, lower=True), small@dinv_rhs)
    return dinv_rhs+dinv_vh@correction


def main():
    out = HOLDOUT.parent/('round20_boundary_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    assert not out.exists() and out.resolve().is_relative_to(ROOT/'outputs_e2e')
    path = (HOLDOUT/'inputs/0_24000.npz').resolve(strict=True)
    identity = sha(path)
    out.mkdir()
    write(out/'protocol.json', dict(input=str(path), sha256=identity, lengths=[1024, 4096],
        ridge=1e-6, requested_uniform_relative_bound=1e-6, gap=15,
        points='truth and fixed offset', repeats=2, model='full time finite extension, not band projection',
        purpose='rank, stability and total same-objective cost only'))
    with np.load(path, allow_pickle=False) as z:
        truth, signal, noise = z['positions'], z['signal'], z['noise']
    records = []
    with threadpool_limits(limits=2):
        for n in (1024, 4096):
            y = (signal[0, :, :n]*10**(15/20)+signal[1, :, :n]+noise[:, :n]).ravel()
            h, preparation = prepare(FiniteOperator(truth, n=n))
            for label, p in [('truth', truth), ('offset', truth+[[5, -3], [-4, 6]])]:
                methods = {'direct': [], 'compressed_schur': []}
                coefficients = {}
                for repeat in range(2):
                    for method in (list(methods) if repeat == 0 else list(reversed(methods))):
                        start = time.perf_counter()
                        op = FiniteOperator(p, n=n)
                        if method == 'direct':
                            matrix = gram(op)
                            matrix.flat[::matrix.shape[0]+1] += 1e-6
                            coeff = cho_solve(cho_factor(matrix, lower=True), op.H@y)
                        else:
                            coeff = schur(op, y, h)
                        elapsed = time.perf_counter()-start
                        normal = op.H@(op@coeff-y)+1e-6*coeff
                        methods[method].append(dict(seconds=elapsed, objective=objective(op, y, coeff, 1e-3),
                            relative_normal_residual=float(np.linalg.norm(normal)/np.linalg.norm(op.H@y))))
                        coefficients[method] = coeff
                error = float(np.linalg.norm(coefficients['compressed_schur']-coefficients['direct'])/np.linalg.norm(coefficients['direct']))
                records.append(dict(n=n, point=label, preparation=preparation, methods=methods,
                    actual_coefficient_error=error, requested_bound_passed=bool(error <= 1e-6)))
                write(out/'results.json', records)
                print(n, label, 'coefficient_error', error, flush=True)
    assert sha(path) == identity
    write(out/'audit.json', dict(completed=True, n=len(records), inputs_unchanged=True,
        all_bounds_passed=all(r['requested_bound_passed'] for r in records), localization_not_run=True))
    print(out, flush=True)


if __name__ == '__main__':
    main()
