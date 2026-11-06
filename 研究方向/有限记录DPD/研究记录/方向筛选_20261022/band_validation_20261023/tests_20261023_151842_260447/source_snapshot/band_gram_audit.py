"""Same-band Gram identity and direct profile derivative, no localization run."""
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.research_discovery.finite_fft_operator import FiniteOperator
from 统一模型代码.gates.research_discovery.finite_fft_blind import derivative_action
from 统一模型代码.gates.research_discovery.finite_extension_probe import dictionary
from 统一模型代码.gates.research_discovery.residual_competition import HOLDOUT
from 统一模型代码.gates.g11.precision_diagnostic import sha, write
from 统一模型代码.gates.g8.physics import C


class BandGram:
    def __init__(self, iq, damp=1e-3):
        start = time.perf_counter()
        self.n = iq.shape[-1]
        self.keep = np.flatnonzero(abs(np.fft.fftfreq(self.n, 1/50e6)) <= 7e6)
        length = self.n+2*int(np.ceil(50e6*1000/C))
        bins = np.arange(-int(np.floor(7e6*length/50e6)), int(np.floor(7e6*length/50e6))+1)
        v = np.exp(2j*np.pi*np.arange(self.n)[:, None]*bins[None]/length)
        self.h = np.fft.fft(v, axis=0)[self.keep]/np.sqrt(self.n)
        self.kernel = self.h.conj().T@self.h
        self.y = self.analysis(iq.ravel())
        self.rhs_projection = self.y.reshape(4, -1)@self.h.conj()
        self.energy = np.vdot(self.y, self.y).real
        self.damp = damp
        self.preparation_seconds = time.perf_counter()-start

    def analysis(self, v):
        return (np.fft.fft(np.asarray(v).reshape(4, self.n), axis=-1)[:, self.keep]/np.sqrt(self.n)).ravel()

    def evaluate(self, positions, gradient=True):
        op = FiniteOperator(positions, n=self.n)
        rows = (op.phase*op.gains[:, :, None]).transpose(1, 0, 2).reshape(4, -1)
        g = np.tile(self.kernel, (op.k, op.k))*(rows.conj().T@rows)
        g.flat[::g.shape[0]+1] += self.damp**2
        rhs = np.sum(rows.conj()*np.tile(self.rhs_projection, (1, op.k)), axis=0)
        coeff = cho_solve(cho_factor(g, lower=True), rhs)
        predicted = (rows*coeff[None]).reshape(4, op.k, op.width).sum(axis=1)@self.h.T
        residual = self.y-predicted.ravel()
        value = float((np.vdot(residual, residual).real+self.damp**2*np.vdot(coeff, coeff).real)/self.energy)
        nr = float(np.linalg.norm(g@coeff-rhs)/np.linalg.norm(rhs))
        if not gradient:
            return value
        derivative = derivative_action(op, np.asarray(positions), coeff)
        projected = np.column_stack([self.analysis(derivative[:, j]) for j in range(4)])
        grad = -2*(residual.conj()@projected).real/self.energy
        return value, grad, coeff, nr


def main():
    out = HOLDOUT.parent/('round19_band_gram_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    assert not out.exists() and out.resolve().is_relative_to(ROOT/'outputs_e2e')
    files = sorted((HOLDOUT/'inputs').glob('*_24000.npz'))
    identities = {str(p.resolve(strict=True)): sha(p) for p in files}
    out.mkdir()
    write(out/'protocol.json', dict(inputs=identities, cases=4, gap=15, damp=1e-3,
        target='same retained N-point FFT bins as round16', half_band_hz=7e6,
        point='truth+[[5,-3],[-4,6]], diagnosis only', derivative_steps=[0.01, 0.001],
        test='Gram/adjoint/direct dense solve/profile gradient', no_localization=True))
    records = []
    with threadpool_limits(limits=2):
        for path in files:
            with np.load(path, allow_pickle=False) as z:
                p = z['positions']+[[5, -3], [-4, 6]]
                iq = z['signal'][0, :, :1024]*10**(15/20)+z['signal'][1, :, :1024]+z['noise'][:, :1024]
            model = BandGram(iq)
            start = time.perf_counter()
            value, grad, coeff, nr = model.evaluate(p)
            seconds = time.perf_counter()-start
            b = dictionary(p)
            a = np.column_stack([model.analysis(b[:, j]) for j in range(b.shape[1])])
            direct_g = a.conj().T@a+1e-6*np.eye(a.shape[1])
            direct = cho_solve(cho_factor(direct_g, lower=True), a.conj().T@model.y)
            fit_error = float(np.linalg.norm(a@(direct-coeff))/np.linalg.norm(a@direct))
            assert fit_error < 1e-7 and nr < 1e-10
            checks = []
            for step in (0.01, 0.001):
                fd = []
                for axis in range(4):
                    plus, minus = p.copy().ravel(), p.copy().ravel()
                    plus[axis] += step
                    minus[axis] -= step
                    fd.append((model.evaluate(plus.reshape(2, 2), False)-model.evaluate(minus.reshape(2, 2), False))/(2*step))
                relative = float(np.linalg.norm(np.asarray(fd)-grad)/np.linalg.norm(grad))
                assert relative < 1e-3
                checks.append(dict(step=step, relative_gradient_error=relative))
            records.append(dict(case=path.stem, preparation_seconds=model.preparation_seconds,
                solve_and_gradient_seconds=seconds, value=value, normal_residual=nr,
                fitted_signal_error=fit_error, checks=checks))
            write(out/'results.json', records)
            print(path.stem, '同频带目标与梯度核验通过', flush=True)
    for path, identity in identities.items():
        assert sha(path) == identity
    write(out/'audit.json', dict(completed=True, cases=len(records), inputs_unchanged=True,
        numerical_only=True, not_blind_performance=True))
    print(out, flush=True)


if __name__ == '__main__':
    main()
