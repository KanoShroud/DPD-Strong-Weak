"""Exact FFT implementation of the finite Fourier-extension matrix action.

LSMR is a conventional numerical reference (Fong/Saunders 2011), not novel.
Its early iterative solution is not identical to truncated-SVD regularization.
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.research_discovery.finite_extension_probe import SOURCE, ATTR, dictionary
from 统一模型代码.gates.g8.physics import C, RECEIVERS
from 统一模型代码.gates.g11.precision_diagnostic import sha, write


class FiniteOperator(LinearOperator):
    def __init__(self, positions, n=1024, fs=50e6):
        positions = np.asarray(positions).reshape(-1, 2)
        self.n, self.k = n, len(positions)
        self.length = n+2*int(np.ceil(fs*1000/C))
        edge = int(np.floor(7e6*self.length/fs))
        bins = np.arange(-edge, edge+1)
        self.indices = bins % self.length
        self.f = bins*fs/self.length
        self.width = len(bins)
        d = np.linalg.norm(positions[:, None]-np.asarray(RECEIVERS)[None], axis=-1)
        delays = (d-d.min(axis=1)[:, None])/C
        gains = 1/d
        self.gains = gains/np.sqrt(n*np.sum(gains**2, axis=1))[:, None]
        self.phase = np.exp(-2j*np.pi*delays[:, :, None]*self.f[None, None])
        self.forward_calls = self.adjoint_calls = 0
        super().__init__(dtype=np.dtype('complex128'), shape=(4*n, self.k*self.width))

    def _matvec(self, x):
        self.forward_calls += 1
        x = np.asarray(x).reshape(self.k, self.width)
        spectrum = np.zeros((self.k, 4, self.length), dtype=complex)
        spectrum[:, :, self.indices] = self.phase*x[:, None]
        waves = np.fft.ifft(spectrum, axis=-1)*self.length
        return np.sum(waves[:, :, :self.n]*self.gains[:, :, None], axis=0).ravel()

    def _rmatvec(self, y):
        self.adjoint_calls += 1
        y = np.asarray(y).reshape(4, self.n)
        spectrum = np.fft.fft(y, n=self.length, axis=-1)[:, self.indices]
        return np.sum(self.gains[:, :, None]*self.phase.conj()*spectrum[None], axis=1).ravel()


def main():
    out = SOURCE.parent/('round11_fft_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    assert not out.exists() and out.resolve().is_relative_to(ROOT/'outputs_e2e')
    protected = {str(p): sha(p) for p in (ROOT/'统一模型代码/gates').rglob('*.py')}
    inputs = {str(p): sha(p) for folder in (SOURCE, ATTR) for p in folder.rglob('*') if p.is_file()}
    out.mkdir()
    write(out/'protocol.json', dict(cases='four layouts realization22000 gap15, true and blindAP5 points',
        tolerances=[1e-6, 1e-8], maxiter=500, conlim=1e8, damp=0,
        hypothesis='same matrix action, conventional LSMR, no SVD-equivalence claim',
        source='https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsmr.html',
        protected=protected, inputs=inputs))
    rng = np.random.default_rng(20261022011)
    records = []
    for path in sorted((ATTR/'pairs').glob('*.json')):
        row = json.loads(path.read_text(encoding='utf-8'))
        if row['gap'] != 15 or not row['case'].endswith('_22000'):
            continue
        _, group, rep = row['case'].split('_')
        with np.load(SOURCE/'inputs'/f'{group}_{rep}.npz', allow_pickle=False) as z:
            y = (z['signal'][0, :, :1024]*10**(15/20)+z['signal'][1, :, :1024]+z['noise'][:, :1024]).ravel()
        for label, positions in [('truth', row['truth']), ('blind_ap5', row['results']['rect']['ap5']['positions'])]:
            a = dictionary(np.asarray(positions))
            op = FiniteOperator(positions)
            c = rng.normal(size=op.shape[1])+1j*rng.normal(size=op.shape[1])
            v = rng.normal(size=op.shape[0])+1j*rng.normal(size=op.shape[0])
            fe = float(np.linalg.norm(op@c-a@c)/np.linalg.norm(a@c))
            ae = float(np.linalg.norm(op.H@v-a.conj().T@v)/np.linalg.norm(a.conj().T@v))
            assert fe < 1e-11 and ae < 1e-11
            for tol in (1e-6, 1e-8):
                op = FiniteOperator(positions)
                start = time.perf_counter()
                result = lsmr(op, y, atol=tol, btol=tol, maxiter=500, conlim=1e8)
                elapsed = time.perf_counter()-start
                residual = y-op@result[0]
                records.append(dict(case=row['case'], label=label, tol=tol,
                    residual_energy_fraction=float(np.vdot(residual, residual).real/np.vdot(y, y).real),
                    istop=int(result[1]), iterations=int(result[2]), seconds=elapsed,
                    norm_normal_residual=float(result[4]), condition_estimate=float(result[6]),
                    forward_error=fe, adjoint_error=ae, forward_calls=op.forward_calls,
                    adjoint_calls=op.adjoint_calls))
        write(out/'results.json', records)
        print(row['case'], 'FFT及伴随等价检查通过', flush=True)
    for path, identity in {**protected, **inputs}.items():
        assert sha(path) == identity, path
    write(out/'audit.json', dict(completed=True, cases=4, solves=len(records),
        inputs_unchanged=True, sources_unchanged=True, diagnostic_only=True))
    print(out, flush=True)


if __name__ == '__main__':
    main()
