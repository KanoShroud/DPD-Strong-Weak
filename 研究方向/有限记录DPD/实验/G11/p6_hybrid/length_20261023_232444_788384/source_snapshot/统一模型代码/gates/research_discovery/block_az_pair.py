"""Cached, nested-sketch AZ versus boundary and direct, same ridge accuracy."""
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
from 统一模型代码.gates.research_discovery.boundary_schur_audit import prepare, schur
from 统一模型代码.gates.research_discovery.residual_competition import HOLDOUT
from 统一模型代码.gates.g11.precision_diagnostic import sha, write


class AZCache:
    def __init__(self, op, seed):
        self.missing = np.exp(2j*np.pi*np.arange(op.n, op.length)[:, None]*op.f[None]/50e6)
        self.omega = np.random.default_rng(seed).normal(size=(op.shape[1], 512))

    def solve(self, op, y):
        phase = (op.phase*op.gains[:, :, None]).transpose(2, 1, 0)
        d = op.length*np.einsum('flk,flj->fkj', phase.conj(), phase)+1e-6*np.eye(op.k)[None]
        values, vectors = np.linalg.eigh(d)
        ir = (vectors/np.sqrt(values)[:, None, :])@vectors.conj().transpose(0, 2, 1)

        def root(x):
            x = np.asarray(x)
            return (ir@x.reshape(op.k, op.width, -1).transpose(1, 0, 2)).transpose(1, 0, 2).reshape(x.shape)

        raw = op.phase*op.gains[:, :, None]
        v = np.concatenate([np.concatenate([self.missing*raw[k, l][None] for k in range(op.k)], axis=1) for l in range(4)], axis=0)
        w = root(v.conj().T).conj().T

        def t(x):
            return w.conj().T@(w@x)

        b = op.H@y
        rhs = root(b)
        target = t(rhs)
        history = []
        diagnostic_seconds = 0.
        for width in (320, 512):
            omega = self.omega[:, :width]
            first = t(omega)
            projected = first-t(first)
            u, s, vh = svd(projected, full_matrices=False, lapack_driver='gesdd')
            comparisons = []
            for cutoff in (1e-11, 1e-12, 1e-13):
                diagnostic_start = time.perf_counter()
                keep = s >= cutoff
                x1 = omega@(vh[keep].conj().T@((u[:, keep].conj().T@target)/s[keep]))
                c = root(rhs+t(x1))
                normal = op.H@(op@c-y)+1e-6*c
                beta = float(np.linalg.norm(normal)/(1e-6*np.linalg.norm(c)))
                certificate = beta/(1-beta) if beta < 1 else None
                comparisons.append(dict(cutoff=cutoff, retained=int(keep.sum()), certificate=certificate))
                if cutoff == 1e-12:
                    answer = c.copy()
                    passed = certificate is not None and certificate <= 1e-6
                else:
                    diagnostic_seconds += time.perf_counter()-diagnostic_start
            history.append(dict(width=width, cutoff_checks=comparisons, accepted=passed))
            if passed:
                break
        if not passed:
            raise RuntimeError('Block-AZ exhausted nested widths without the requested residual certificate')
        return answer, history, diagnostic_seconds


def main():
    out = HOLDOUT.parent/('round25_az_pair_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    assert not out.exists() and out.resolve().is_relative_to(ROOT/'outputs_e2e')
    files = sorted((HOLDOUT/'inputs').glob('*_24000.npz'))
    identities = {str(p.resolve(strict=True)): sha(p) for p in files}
    out.mkdir()
    write(out/'protocol.json', dict(inputs=identities, n=4096, gap=30,
        seeds=[20261023025, 20261023026], sketch_widths=[320, 512], nested=True,
        default_cutoff=1e-12, diagnostics_cutoffs=[1e-11, 1e-12, 1e-13],
        online_acceptance='normal residual bound beta/(1-beta)<=1e-6, no true position/coefficients',
        cache='all position-independent kernels, bases and random probes shared across geometry',
        timing='includes operator construction and complete per-position solving; preparation separate; auxiliary cutoff sensitivity cost excluded and recorded',
        reference='same ridge direct cached analytic Gram', repetitions=2))
    records = []
    with threadpool_limits(limits=2):
        with np.load(files[0], allow_pickle=False) as z:
            op = FiniteOperator(z['positions'], n=4096)
        start = time.perf_counter()
        direct = Direct(op)
        prep_times = {'direct': time.perf_counter()-start}
        h, prep = prepare(op)
        prep_times['boundary'] = prep['preparation_seconds']
        azs = {}
        for seed in (20261023025, 20261023026):
            start = time.perf_counter()
            azs[seed] = AZCache(op, seed)
            prep_times[str(seed)] = time.perf_counter()-start
        write(out/'preparation.json', prep_times)
        for index, path in enumerate(files):
            with np.load(path, allow_pickle=False) as z:
                p = z['positions']+[[5, -3], [-4, 6]]
                y = (z['signal'][0, :, :4096]*10**(30/20)+z['signal'][1, :, :4096]+z['noise'][:, :4096]).ravel()
            reference = direct.solve(FiniteOperator(p, n=4096), y)
            for repeat in range(2):
                names = ['direct', 'boundary', 20261023025, 20261023026]
                if (index+repeat) % 2:
                    names.reverse()
                for name in names:
                    start = time.perf_counter()
                    local = FiniteOperator(p, n=4096)
                    history = []
                    diagnostic_seconds = 0.
                    if name == 'direct':
                        c = direct.solve(local, y)
                    elif name == 'boundary':
                        c = schur(local, y, h)
                    else:
                        c, history, diagnostic_seconds = azs[name].solve(local, y)
                    elapsed = time.perf_counter()-start-diagnostic_seconds
                    records.append(dict(case=path.stem, repeat=repeat, method=str(name), seconds=elapsed,
                        cutoff_diagnostic_seconds=diagnostic_seconds,
                        coefficient_error=float(np.linalg.norm(c-reference)/np.linalg.norm(reference)), history=history))
                    write(out/'results.json', records)
            print(path.stem, '配对完成', flush=True)
    for path, identity in identities.items():
        assert sha(path) == identity
    write(out/'audit.json', dict(completed=True, records=len(records), inputs_unchanged=True,
        all_coefficient_errors_below_1e_minus6=all(r['coefficient_error'] <= 1e-6 for r in records)))
    print(out, flush=True)


if __name__ == '__main__':
    main()
