"""Conventional full-period preconditioned CG challenge to boundary Schur."""
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.research_discovery.finite_fft_operator import FiniteOperator
from 统一模型代码.gates.research_discovery.boundary_blind_pilot import Direct
from 统一模型代码.gates.research_discovery.boundary_schur_audit import prepare, schur
from 统一模型代码.gates.research_discovery.residual_competition import HOLDOUT
from 统一模型代码.gates.g11.precision_diagnostic import sha, write


def pcg(op, y, tolerance):
    phase = (op.phase*op.gains[:, :, None]).transpose(2, 1, 0)
    d = op.length*np.einsum('flk,flj->fkj', phase.conj(), phase)+1e-6*np.eye(op.k)[None]
    # Explicit K=2 inverse blocks are tiny; this is not a dense problem inverse.
    inverse = np.linalg.inv(d)

    def precondition(x):
        blocks = np.asarray(x).reshape(op.k, op.width).T
        return np.einsum('fkj,fj->fk', inverse, blocks).T.ravel()

    matrix = LinearOperator((op.shape[1], op.shape[1]), matvec=lambda x: op.H@(op@x)+1e-6*x, dtype=complex)
    pre = LinearOperator(matrix.shape, matvec=precondition, dtype=complex)
    count = 0

    def callback(x):
        nonlocal count
        count += 1

    result, info = cg(matrix, op.H@y, M=pre, rtol=tolerance, atol=0, maxiter=2000, callback=callback)
    return result, dict(info=int(info), iterations=count)


def main():
    out = HOLDOUT.parent/('round22_pcg_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    assert not out.exists() and out.resolve().is_relative_to(ROOT/'outputs_e2e')
    files = sorted((HOLDOUT/'inputs').glob('*_24000.npz'))
    identities = {str(p.resolve(strict=True)): sha(p) for p in files}
    out.mkdir()
    write(out/'protocol.json', dict(inputs=identities, n=4096, gap=30,
        fixed_points='truth plus [[5,-3],[-4,6]]', tolerances=[1e-10, 1e-12],
        preconditioner='full-period KxK frequency blocks', maxiter=2000, ridge=1e-6,
        reference='cached analytic Gram Cholesky', repeats=2, no_localization=True))
    records = []
    h = None
    with threadpool_limits(limits=2):
        for index, path in enumerate(files):
            with np.load(path, allow_pickle=False) as z:
                p = z['positions']+[[5, -3], [-4, 6]]
                y = (z['signal'][0, :, :4096]*10**(30/20)+z['signal'][1, :, :4096]+z['noise'][:, :4096]).ravel()
            op = FiniteOperator(p, n=4096)
            start = time.perf_counter()
            direct = Direct(op)
            direct_prep = time.perf_counter()-start
            if h is None:
                h, prep = prepare(op)
            reference = direct.solve(op, y)
            methods = ['direct', 'boundary', 'pcg_10', 'pcg_12']
            measures = []
            for repeat in range(2):
                for name in methods if (repeat+index) % 2 == 0 else methods[::-1]:
                    start = time.perf_counter()
                    local = FiniteOperator(p, n=4096)
                    if name == 'direct':
                        c, detail = direct.solve(local, y), {}
                    elif name == 'boundary':
                        c, detail = schur(local, y, h), {}
                    else:
                        c, detail = pcg(local, y, 1e-10 if name == 'pcg_10' else 1e-12)
                    seconds = time.perf_counter()-start
                    normal = local.H@(local@c-y)+1e-6*c
                    measures.append(dict(method=name, seconds=seconds, detail=detail,
                        coefficient_error=float(np.linalg.norm(c-reference)/np.linalg.norm(reference)),
                        normal_residual=float(np.linalg.norm(normal)/np.linalg.norm(local.H@y))))
            records.append(dict(case=path.stem, direct_preparation_seconds=direct_prep,
                shared_boundary_preparation=prep, measures=measures))
            write(out/'results.json', records)
            print(path.stem, {name: round(np.mean([v['seconds'] for v in measures if v['method'] == name]), 3) for name in methods}, flush=True)
    for path, identity in identities.items():
        assert sha(path) == identity
    write(out/'audit.json', dict(completed=True, cases=len(records), inputs_unchanged=True, numerical_comparison_only=True))
    print(out, flush=True)


if __name__ == '__main__':
    main()
