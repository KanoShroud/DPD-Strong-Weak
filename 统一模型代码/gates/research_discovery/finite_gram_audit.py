"""Analytic finite-record Gram versus LSMR; conventional numerical references."""
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import lsmr
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.research_discovery.finite_fft_operator import FiniteOperator
from 统一模型代码.gates.research_discovery.finite_extension_probe import dictionary
from 统一模型代码.gates.research_discovery.residual_competition import HOLDOUT
from 统一模型代码.gates.g11.precision_diagnostic import sha, write


def gram(op):
    frequency = op.f/50e6
    delta = frequency[None, :]-frequency[:, None]
    kernel = op.n*np.sinc(op.n*delta)/np.sinc(delta)*np.exp(1j*np.pi*(op.n-1)*delta)
    rows = (op.phase*op.gains[:, :, None]).transpose(1, 0, 2).reshape(4, -1)
    return np.tile(kernel, (op.k, op.k))*(rows.conj().T@rows)


def objective(op, y, c, damp):
    residual = y-op@c
    return float((np.vdot(residual, residual).real+damp**2*np.vdot(c, c).real)/np.vdot(y, y).real)


def main():
    out = HOLDOUT.parent/('round18_gram_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    assert not out.exists() and out.resolve().is_relative_to(ROOT/'outputs_e2e')
    files = sorted((HOLDOUT/'inputs').glob('*_24000.npz'))
    identities = {str(p.resolve(strict=True)): sha(p) for p in files}
    out.mkdir()
    write(out/'protocol.json', dict(inputs=identities, cases=4, gap=15, damp=1e-3,
        target='full finite time record, NOT round16 band-projected target',
        points=['truth', 'truth+[[5,-3],[-4,6]]'], repeats=3,
        claim='same-objective numerical benchmark, not blind localization',
        lsmr=dict(atol=1e-9, btol=1e-9, maxiter=1000, conlim=1e8)))
    records = []
    with threadpool_limits(limits=2):
        for path in files:
            with np.load(path, allow_pickle=False) as z:
                truth = z['positions']
                y = (z['signal'][0, :, :1024]*10**(15/20)+z['signal'][1, :, :1024]+z['noise'][:, :1024]).ravel()
            for name, positions in [('truth', truth), ('offset', truth+[[5, -3], [-4, 6]])]:
                op = FiniteOperator(positions)
                b = dictionary(positions)
                g = gram(op)
                relative = np.linalg.norm(g-b.conj().T@b)/np.linalg.norm(g)
                assert relative < 1e-11
                rows = (op.phase*op.gains[:, :, None]).transpose(1, 0, 2).reshape(4, -1)
                phase = np.tile(np.exp(2j*np.pi*op.f/50e6), op.k)
                end = rows*phase[None]**op.n
                boundary = end.conj().T@end-rows.conj().T@rows
                displacement = phase.conj()[:, None]*g*phase[None]-g
                derr = np.linalg.norm(displacement-boundary)/np.linalg.norm(boundary)
                assert derr < 1e-10
                methods = {'analytic_cholesky': [], 'fft_lsmr': []}
                for repeat in range(3):
                    order = list(methods) if repeat % 2 == 0 else list(reversed(methods))
                    for method in order:
                        start = time.perf_counter()
                        local = FiniteOperator(positions)
                        if method == 'analytic_cholesky':
                            matrix = gram(local)
                            matrix.flat[::matrix.shape[0]+1] += 1e-6
                            rhs = local.H@y
                            c = cho_solve(cho_factor(matrix, lower=True), rhs)
                            iterations, stop = 1, 'factorization'
                        else:
                            result = lsmr(local, y, damp=1e-3, atol=1e-9, btol=1e-9, maxiter=1000, conlim=1e8)
                            c, stop, iterations = result[:3]
                        seconds = time.perf_counter()-start
                        normal = local.H@(local@c-y)+1e-6*c
                        nr = np.linalg.norm(normal)/np.linalg.norm(local.H@y)
                        methods[method].append(dict(seconds=seconds, objective=objective(local, y, c, 1e-3),
                            relative_normal_residual=float(nr), iterations=int(iterations), stop=stop))
                records.append(dict(case=path.stem, point=name, gram_error=float(relative),
                    displacement_error=float(derr), dimension=g.shape[0],
                    missing_row_schur_dimension=4*(op.length-op.n), methods=methods))
                write(out/'results.json', records)
                print(path.stem, name, {k: round(np.median([v['seconds'] for v in vs]), 4) for k, vs in methods.items()}, flush=True)
    for path, identity in identities.items():
        assert sha(path) == identity
    write(out/'audit.json', dict(completed=True, points=len(records), repetitions=3,
        inputs_unchanged=True, gram_identity_passed=True, displacement_identity_passed=True,
        not_localization=True, not_bandlimited_control=True))
    print(out, flush=True)


if __name__ == '__main__':
    main()
