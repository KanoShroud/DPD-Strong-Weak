"""Conventional finite-record Slepian representation: development reference.

Source: Slepian Beamforming (arXiv:2312.03922), sections 2 and 3.1;
unknown-position use is a near-field adaptation, not a paper reproduction.
All solvers receive only the same retained DFT data. No truth in solvers.
"""
import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, cho_factor, cho_solve, lstsq
from scipy.special import roots_legendre
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.research_discovery.band_validation import (
    Family, Problem, FS, BAND, RIDGE, INPUT, BASE, digest, write, input_data,
)
from 统一模型代码.gates.g8.physics import RECEIVERS, C
from 统一模型代码.gates.g10.r4.paper_checks.assignment_metrics import evaluate

EPSILONS = (1e-3, 1e-6, 1e-9)


class SlepianFamily:
    def __init__(self, n, eps=1e-6, quadrature_factor=1):
        tick = time.perf_counter()
        self.n, self.eps = n, eps
        sites = np.asarray(RECEIVERS)
        baseline = np.linalg.norm(sites[:, None]-sites[None], axis=-1).max()
        self.delay_bound = float(baseline*FS/C)
        self.duration = n-1+2*self.delay_bound
        self.center = (n-1)/2
        self.keep = np.flatnonzero(abs(np.fft.fftfreq(n, 1/FS)) <= BAND)
        order = int(np.ceil(4*(BAND/FS)*self.duration))+32
        self.order = order*quadrature_factor
        nodes, weights = roots_legendre(self.order)
        self.f, weights = nodes*(BAND/FS), weights*(BAND/FS)
        # Dual concentration kernel on a centered continuous time interval.
        root = np.sqrt(weights)
        kernel = (self.duration*np.sinc(self.duration*(self.f[:, None]-self.f[None]))
                  *root[:, None]*root[None])
        values, vectors = eigh(kernel, driver='evd')
        self.values = np.maximum(values[::-1], 0)
        vectors = vectors[:, ::-1]
        tails = np.r_[np.cumsum(self.values[::-1])[::-1], 0.]
        self.rank = int(np.flatnonzero(tails <= eps*self.values.sum())[0])
        self.tail_fraction = float(tails[self.rank]/self.values.sum())
        self.basis = root[:, None]*vectors[:, :self.rank]/np.sqrt(self.values[:self.rank])[None]
        time_matrix = np.exp(2j*np.pi*(np.arange(n)-self.center)[:, None]*self.f[None])
        self.h = np.fft.fft(time_matrix, axis=0)[self.keep]/np.sqrt(n)
        self.seconds = time.perf_counter()-tick

    def analysis(self, iq):
        return np.fft.fft(iq, axis=-1)[:, self.keep]/np.sqrt(self.n)

    def geometry(self, positions):
        p = np.asarray(positions).reshape(-1, 2)
        vec = p[:, None]-np.asarray(RECEIVERS)[None]
        distance = np.linalg.norm(vec, axis=-1)
        unit = vec/distance[:, :, None]
        gain = (1/distance)/np.sqrt(np.sum(distance**-2, axis=1))[:, None]
        weights = distance**-2/np.sum(distance**-2, axis=1)[:, None]
        log_gain = -unit/distance[:, :, None]+np.sum(weights[:, :, None]*unit/distance[:, :, None], axis=1)[:, None]
        delay = (distance-distance[:, :1])*FS/C
        derivative = (unit-unit[:, :1])*FS/C
        phase = np.exp(-2j*np.pi*delay[:, :, None]*self.f)
        return gain, log_gain, phase, derivative

    def matrix(self, positions):
        gain, _, phase, _ = self.geometry(positions)
        return np.vstack([np.hstack([(self.h*phase[k, l])@self.basis*gain[k, l]
                                     for k in range(len(gain))]) for l in range(4)])

    def derivative(self, positions, coefficients):
        gain, log_gain, phase, dt = self.geometry(positions)
        spectra = coefficients.reshape(len(gain), self.rank)@self.basis.T
        result = []
        for k in range(len(gain)):
            for axis in range(2):
                v = []
                for l in range(4):
                    multiplier = log_gain[k, l, axis]-2j*np.pi*self.f*dt[k, l, axis]
                    v.append(self.h@(phase[k, l]*spectra[k]*multiplier)*gain[k, l])
                result.append(np.concatenate(v))
        return np.column_stack(result)

    def metadata(self):
        return dict(rank=self.rank, eps=self.eps, quadrature=self.order,
                    duration_samples=self.duration, tail_fraction=self.tail_fraction,
                    last_eigenvalue=float(self.values[self.rank-1]), preparation_seconds=self.seconds)


class SlepianProblem:
    def __init__(self, family, iq):
        self.family = family
        self.z = family.analysis(iq).ravel()
        self.energy = float(np.vdot(self.z, self.z).real)

    def evaluate(self, positions):
        a = self.family.matrix(positions)
        gram = a.conj().T@a
        gram.flat[::len(gram)+1] += RIDGE
        b = a.conj().T@self.z
        c = cho_solve(cho_factor(gram), b)
        residual = self.z-a@c
        normal = -a.conj().T@residual+RIDGE*c
        relative_bound = float(np.linalg.norm(normal)/(RIDGE*np.linalg.norm(c)))
        if not np.isfinite(c).all() or relative_bound > 1e-5:
            raise ArithmeticError(f'Slepian inner solve failed certificate: {relative_bound}')
        value = float((np.vdot(residual, residual).real+RIDGE*np.vdot(c, c).real)/self.energy)
        derivative = self.family.derivative(positions, c)
        gradient = -2*(residual.conj()@derivative).real/self.energy
        return value, gradient, c, dict(normal_bound=relative_bound,
            fit_fraction=float(np.vdot(residual, residual).real/self.energy))


def reconstruct(z, n, keep):
    spectrum = np.zeros((4, n), complex)
    spectrum[:, keep] = z.reshape(4, len(keep))
    return np.fft.ifft(spectrum, axis=-1)*np.sqrt(n)


def common_starts(iq):
    import torch
    from 统一模型代码.gates.g10.r3.solvers import Model
    from 统一模型代码.gates.g10.r2.solvers import local
    from 统一模型代码.gates.g10.physics import grid
    from 统一模型代码.gates.g10.methods import peaks
    cfg = json.loads((INPUT/'protocol.json').read_text(encoding='utf-8'))['config']
    torch.cuda.synchronize()
    tick = time.perf_counter()
    m = Model(iq, sample_rate_hz=FS, half_band_hz=BAND)
    points = grid(20)
    first = points[peaks(points, m.conditional(points), 1)[0]]
    first, _, _ = local(m, first[None], cfg)
    ids = peaks(points, m.conditional(points, others=first), 5)
    starts = [np.vstack([first, points[i]]) for i in ids]
    torch.cuda.synchronize()
    return starts, time.perf_counter()-tick


def optimize(callback, starts):
    cache = {}

    def objective(x):
        key = np.asarray(x).tobytes()
        if key not in cache:
            val, grad, _, info = callback(np.asarray(x).reshape(2, 2))
            cache[key] = (val, np.asarray(grad).ravel(), info)
        return cache[key][:2]

    scores = [objective(p.ravel())[0] for p in starts]
    selected = int(np.argmin(scores))
    opt = minimize(objective, starts[selected].ravel(), jac=True, method='L-BFGS-B',
                   bounds=[(-1000., 1000.)]*4,
                   options=dict(maxiter=100, ftol=1e-12, gtol=1e-8, maxls=25))
    final = opt.x.reshape(2, 2) if np.isfinite(opt.fun) and opt.fun <= scores[selected] else starts[selected]
    value, grad = objective(final.ravel())
    return dict(positions=final.tolist(), selected=selected, scores=scores, objective=value,
                gradient_max=float(abs(grad).max()), calls=len(cache), iterations=int(opt.nit),
                optimizer_success=bool(opt.success), message=str(opt.message),
                inner=[entry[2] for entry in cache.values()])


def tests(folder):
    rng = np.random.default_rng(20261023030)
    fam = SlepianFamily(128, 1e-6)
    finer = SlepianFamily(128, 1e-6, 2)
    p = np.array([[173., 117.], [-137., -213.]])
    a, fine = fam.matrix(p), finer.matrix(p)
    # Degenerate concentrated eigenfunctions may rotate: compare projectors/Gram, not columns.
    quadrature_error = float(np.linalg.norm(a@a.conj().T-fine@fine.conj().T)/np.linalg.norm(fine@fine.conj().T))
    assert fam.rank == finer.rank and quadrature_error < 1e-7
    c = rng.normal(size=a.shape[1])+1j*rng.normal(size=a.shape[1])
    analytic = fam.derivative(p, c)
    finite = []
    for j in range(4):
        delta = np.zeros((2, 2)); delta.flat[j] = .001
        finite.append(((fam.matrix(p+delta)-fam.matrix(p-delta))@c)/.002)
    derivative_error = float(np.linalg.norm(np.column_stack(finite)-analytic)/np.linalg.norm(analytic))
    assert derivative_error < 1e-6
    iq = reconstruct(a@c, fam.n, fam.keep)
    pb = SlepianProblem(fam, iq)
    v, g, estimate, info = pb.evaluate(p)
    augmented = np.vstack([a, np.sqrt(RIDGE)*np.eye(a.shape[1])])
    direct = lstsq(augmented, np.r_[pb.z, np.zeros(a.shape[1])], lapack_driver='gelsy')[0]
    coefficient_error = float(np.linalg.norm(estimate-direct)/np.linalg.norm(direct))
    assert coefficient_error < 1e-6
    fd = []
    for j in range(4):
        delta = np.zeros((2, 2)); delta.flat[j] = .001
        fd.append((pb.evaluate(p+delta)[0]-pb.evaluate(p-delta)[0])/.002)
    gradient_error = float(np.linalg.norm(np.asarray(fd)-g)/max(np.linalg.norm(g), 1e-10))
    assert gradient_error < .01
    noise = rng.normal(size=iq.shape)+1j*rng.normal(size=iq.shape)
    spectrum = np.fft.fft(noise, axis=-1); spectrum[:, fam.keep] = 0
    outside = SlepianProblem(fam, iq+np.fft.ifft(spectrum, axis=-1))
    value2, grad2, _, _ = outside.evaluate(p)
    assert abs(value2-v) < 1e-12 and np.linalg.norm(grad2-g) < 1e-10
    swapped = pb.evaluate(p[::-1])[0]
    assert abs(swapped-v) < 1e-12
    write(folder/'tests.json', dict(passed=True, quadrature_error=quadrature_error,
        derivative_error=derivative_error, coefficient_error=coefficient_error,
        gradient_error=gradient_error, outside_band=True, permutation=True, inner=info,
        family=fam.metadata()))


def pilot(folder, files, n, limit):
    import torch
    from 统一模型代码.gates.g10.r3.solvers import Model
    # Explicit public warm-up, excluded from per-case times and listed separately.
    tick = time.perf_counter()
    torch.zeros(1, device='cuda'); torch.cuda.synchronize()
    write(folder/'warmup.json', dict(seconds=time.perf_counter()-tick))
    records = []
    methods = ['periodic', 'hann_after_band', 'finite_fourier']+[f'slepian_{eps:g}' for eps in EPSILONS]
    for index, path in enumerate(files[:limit]):
        for gap in (0, 15):
            iq, truth = input_data(path, n, gap)
            keep = np.flatnonzero(abs(np.fft.fftfreq(n, 1/FS)) <= BAND)
            tick = time.perf_counter()
            z = np.fft.fft(iq, axis=-1)[:, keep]/np.sqrt(n)
            band_iq = reconstruct(z, n, keep)
            preprocessing = time.perf_counter()-tick
            starts, start_seconds = common_starts(band_iq)
            results = {}
            order = methods if (index+(gap == 15)) % 2 == 0 else methods[::-1]
            for method in order:
                tick = time.perf_counter()
                metadata = {}
                if method.startswith('slepian_'):
                    family = SlepianFamily(n, float(method.split('_')[1]))
                    pb = SlepianProblem(family, band_iq)
                    callback = pb.evaluate
                    metadata = family.metadata()
                elif method == 'finite_fourier':
                    family = Family(n); family.setup('direct')
                    pb = Problem(family, band_iq)
                    callback = lambda p: pb.evaluate(p, 'direct')
                    metadata = dict(rank=family.q, preparation=family.times)
                else:
                    window = np.hanning(n) if method == 'hann_after_band' else np.ones(n)
                    model = Model(band_iq*window, sample_rate_hz=FS, half_band_hz=BAND)

                    def callback(p):
                        value, gradient = model.joint(p, True)
                        return value, gradient, None, {}
                prepared = time.perf_counter()-tick
                result = optimize(callback, starts)
                torch.cuda.synchronize()
                seconds = time.perf_counter()-tick
                result.update(seconds=seconds+start_seconds+preprocessing, solver_seconds=seconds-prepared,
                              preparation_seconds=prepared, common_start_seconds=start_seconds,
                              common_preprocessing_seconds=preprocessing, model=metadata,
                              metrics=evaluate(truth, result['positions']))
                # Truth point evaluation is an offline diagnostic, not search/timing/selection.
                diag_tick = time.perf_counter()
                diagnostic = callback(truth)
                result['truth_diagnostic'] = dict(objective=float(diagnostic[0]),
                    gradient_max=float(abs(np.asarray(diagnostic[1])).max()), info=diagnostic[3],
                    seconds=time.perf_counter()-diag_tick)
                results[method] = result
                write(folder/'progress.json', dict(case=path.stem, gap=gap, n=n,
                    starts=[p.tolist() for p in starts], results=results))
                print(n, path.stem, gap, method, f'{result["seconds"]:.2f}s', result['metrics'], flush=True)
            records.append(dict(case=path.stem, gap=gap, n=n, truth=truth.tolist(),
                                starts=[p.tolist() for p in starts], order=order, results=results))
            write(folder/'results.json', records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['tests', 'pilot'])
    parser.add_argument('--n', type=int, default=1024)
    parser.add_argument('--limit', type=int, default=4)
    args = parser.parse_args()
    files = sorted((INPUT/'inputs').glob('*_25000.npz'))
    identities = json.loads((INPUT/'input_identities.json').read_text(encoding='utf-8'))
    for path in files:
        assert path.resolve().is_relative_to(ROOT/'outputs_e2e') and digest(path) == identities[str(path)]
    out = BASE/'slepian_reference_20261023'/(args.mode+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    assert not out.exists() and out.resolve().is_relative_to(ROOT/'outputs_e2e')
    out.mkdir(parents=True)
    protected = {str(p): digest(p) for p in (ROOT/'统一模型代码/gates').rglob('*.py')}
    snapshot = out/'source_snapshot'
    for name in protected:
        p = Path(name); target = snapshot/p.relative_to(ROOT/'统一模型代码/gates')
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(p.read_bytes())
    write(out/'protocol.json', dict(args=vars(args), files={str(p): digest(p) for p in files},
        sources=protected, epsilon=list(EPSILONS), ridge=RIDGE, input_band_hz=BAND, fs=FS,
        common_prefix='same periodic conditional five candidates, no truth',
        search='own objective selects one branch then joint L-BFGS; same limits',
        data_role='reused development data, not independent validation',
        reference='Slepian Beamforming sec2/3.1 + R025 localization relation; near-field adaptation',
        timing='cold preparation charged each method, common prefix charged each, truth diagnostic excluded'))
    print('结果目录', out, flush=True)
    started = time.perf_counter()
    try:
        with threadpool_limits(limits=2):
            if args.mode == 'tests':
                tests(out)
            else:
                pilot(out, files, args.n, args.limit)
    except Exception:
        write(out/'failure.json', dict(traceback=traceback.format_exc(), seconds=time.perf_counter()-started))
        raise
    assert all(digest(p) == expected for p, expected in protected.items())
    assert all(digest(p) == identities[str(p)] for p in files)
    write(out/'audit.json', dict(completed=True, inputs_and_sources_unchanged=True,
                                seconds=time.perf_counter()-started, formal=False))
    print('完成', out, flush=True)


if __name__ == '__main__':
    main()
