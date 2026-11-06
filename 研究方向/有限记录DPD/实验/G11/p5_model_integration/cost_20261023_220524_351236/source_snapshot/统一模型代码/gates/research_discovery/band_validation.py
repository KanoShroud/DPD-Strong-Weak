"""Same observed-band ridge problem: direct, boundary, AZ and LSMR.

All methods receive only the retained unitary N-point DFT coefficients.
AZ: Coppe et al., 2020, Algorithms 2.1/3.2 on a symmetrically
block-preconditioned normal system; LSMR: Fong & Saunders, 2011.
These are explicitly adapted numerical baselines, not original DPD reproductions.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import cho_factor, cho_solve, svd
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator, lsmr
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.research_discovery.finite_fft_operator import FiniteOperator
from 统一模型代码.gates.research_discovery.finite_fft_blind import derivative_action
from 统一模型代码.gates.g8.physics import C

BASE = ROOT/'outputs_e2e/research_discovery_20261022'
INPUT = BASE/'experiments/A_window_baselines/round23_independent_20261023_113113_199745'
OUT = BASE/'band_validation_20261023'
FS, BAND, RIDGE, TOL = 50e6, 7e6, 1e-6, 1e-6


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


class Family:
    """Only public N/Fs/band settings; no source positions or observations."""
    def __init__(self, n):
        tick = time.perf_counter()
        self.n, self.p = n, n+2*int(np.ceil(FS*1000/C))
        e = int(np.floor(BAND*self.p/FS))
        self.f = np.arange(-e, e+1)*FS/self.p
        self.indices = np.arange(-e, e+1) % self.p
        self.q = len(self.f)
        mask = abs(np.fft.fftfreq(n, 1/FS)) <= BAND
        self.keep = np.flatnonzero(mask)
        self.mask = mask
        self.times = {'common': time.perf_counter()-tick}
        self.kernel = self.v = self.small = self.h = self.out = self.omega = None

    def setup(self, method):
        if method in ('direct', 'boundary', 'az') and self.h is None:
            tick = time.perf_counter()
            t = np.exp(2j*np.pi*np.arange(self.n)[:, None]*self.f[None]/FS)
            ft = np.fft.fft(t, axis=0)/np.sqrt(self.n)
            self.h, self.out = ft[self.mask], ft[~self.mask]
            self.times['basis'] = time.perf_counter()-tick
        if method in ('direct', 'az') and self.kernel is None:
            tick = time.perf_counter()
            self.kernel = self.h.conj().T@self.h
            self.times['kernel'] = time.perf_counter()-tick
        if method == 'boundary' and self.small is None:
            tick = time.perf_counter()
            missing = np.exp(2j*np.pi*np.arange(self.n, self.p)[:, None]*self.f[None]/FS)
            self.v = np.vstack([missing, self.out])
            _, singular, vh = svd(self.v, full_matrices=False, lapack_driver='gesdd')
            tails = np.r_[np.cumsum(singular[::-1]**2)[::-1], 0.]
            rank = int(np.flatnonzero(2*tails/self.n/RIDGE <= TOL)[0])
            self.small = singular[:rank, None]*vh[:rank]
            self.rank = rank
            self.bound = float(2*tails[rank]/self.n/RIDGE)
            self.times['boundary'] = time.perf_counter()-tick
        if method == 'az' and self.omega is None:
            tick = time.perf_counter()
            self.omega = np.random.default_rng(20261023026).normal(size=(2*self.q, min(640, 2*self.q)))
            self.times['az_probes'] = time.perf_counter()-tick

    def prep_cost(self, method):
        if method == 'lsmr':
            return self.times['common']
        return (self.times['common']+self.times['basis']
                +self.times.get({'boundary': 'boundary', 'direct': 'kernel', 'az': 'kernel'}[method], 0.)
                +(self.times['az_probes'] if method == 'az' else 0.))

    def analysis(self, x):
        return np.fft.fft(np.asarray(x).reshape(4, self.n), axis=-1)[:, self.keep]/np.sqrt(self.n)

    def synthesis(self, z):
        x = np.zeros((4, self.n), complex)
        x[:, self.keep] = z
        return np.fft.ifft(x, axis=-1)*np.sqrt(self.n)


class Problem:
    def __init__(self, fam, iq):
        self.fam = fam
        self.z = fam.analysis(iq)
        self.projected = np.fft.fft(fam.synthesis(self.z), n=fam.p, axis=-1)[:, fam.indices]
        self.energy = float(np.vdot(self.z, self.z).real)

    def parts(self, p):
        op = FiniteOperator(p, n=self.fam.n)
        rows = (op.phase*op.gains[:, :, None]).transpose(1, 0, 2).reshape(4, -1)
        b = np.sum(rows.conj()*np.tile(self.projected, (1, op.k)), axis=0)
        return op, rows, b

    def forward(self, rows, c):
        k = rows.shape[1]//self.fam.q
        c = np.asarray(c)
        if c.ndim == 1:
            spectrum = np.zeros((4, self.fam.p), complex)
            spectrum[:, self.fam.indices] = (rows*c[None]).reshape(4, k, self.fam.q).sum(axis=1)
            wave = np.fft.ifft(spectrum, axis=-1)*self.fam.p
            return self.fam.analysis(wave[:, :self.fam.n])
        return np.column_stack([self.forward(rows, x).ravel() for x in c.T])

    def adjoint(self, rows, z):
        k = rows.shape[1]//self.fam.q
        x = np.fft.fft(self.fam.synthesis(np.asarray(z).reshape(4, len(self.fam.keep))), n=self.fam.p, axis=-1)[:, self.fam.indices]
        return np.sum(rows.conj()*np.tile(x, (1, k)), axis=0)

    def matrix(self, rows):
        k = rows.shape[1]//self.fam.q
        a = np.tile(self.fam.kernel, (k, k))*(rows.conj().T@rows)
        a.flat[::len(a)+1] += RIDGE
        return a

    def root(self, op):
        phase = (op.phase*op.gains[:, :, None]).transpose(2, 1, 0)
        d = op.length*np.einsum('flk,flj->fkj', phase.conj(), phase)+RIDGE*np.eye(op.k)[None]
        values, vec = np.linalg.eigh(d)
        inverse = (vec/np.sqrt(values)[:, None, :])@vec.conj().transpose(0, 2, 1)

        def apply(x):
            x = np.asarray(x)
            return (inverse@x.reshape(op.k, op.width, -1).transpose(1, 0, 2)).transpose(1, 0, 2).reshape(x.shape)
        return d, apply

    def check(self, rows, b, c):
        r = self.forward(rows, c)-self.z
        normal = self.adjoint(rows, r)+RIDGE*c
        beta = np.linalg.norm(normal)/(RIDGE*np.linalg.norm(c))
        return dict(normal_relative=float(np.linalg.norm(normal)/np.linalg.norm(b)),
                    coefficient_certificate=float(beta/(1-beta)) if beta < 1 else None)

    def solve(self, p, method):
        op, rows, b = self.parts(p)
        info = {}
        if method == 'direct':
            c = cho_solve(cho_factor(self.matrix(rows), lower=True), b)
        elif method == 'boundary':
            d, _ = self.root(op)
            phase = op.phase*op.gains[:, :, None]
            h = self.fam.small
            small = np.concatenate([np.concatenate([h*phase[k, l][None] for k in range(op.k)], axis=1) for l in range(4)], axis=0)
            db = np.linalg.solve(d, b.reshape(op.k, op.width).T[..., None]).squeeze(-1).T.ravel()
            rhs = small.conj().T.reshape(op.k, op.width, -1).transpose(1, 0, 2)
            dv = np.linalg.solve(d, rhs).transpose(1, 0, 2).reshape(len(b), -1)
            s = np.eye(len(small))-small@dv
            s = (s+s.conj().T)/2
            c = db+dv@cho_solve(cho_factor(s, lower=True), small@db)
            info = dict(rank=self.fam.rank, uniform_coefficient_bound=self.fam.bound)
        elif method == 'az':
            _, root = self.root(op)
            matrix = self.matrix(rows)

            def defect(x):
                return x-root(matrix@root(x))

            rhs = root(b)
            target = defect(rhs)
            attempts = []
            for width in sorted(set(min(w, len(b)) for w in (160, 320, 640))):
                omega = self.fam.omega[:len(b), :width]
                first = defect(omega)
                a = first-defect(first)
                u, s, vh = svd(a, full_matrices=False, lapack_driver='gesdd')
                keep = s >= 1e-12
                x1 = omega@(vh[keep].conj().T@((u[:, keep].conj().T@target)/s[keep]))
                c = root(rhs+defect(x1))
                cert = self.check(rows, b, c)['coefficient_certificate']
                attempts.append(dict(width=width, retained=int(keep.sum()), certificate=cert))
                if cert is not None and cert <= TOL:
                    break
            info = dict(attempts=attempts, sufficient=cert is not None and cert <= TOL)
        elif method == 'lsmr':
            _, root = self.root(op)
            ndata, d = self.z.size, len(b)

            def mv(x):
                c0 = root(x)
                return np.r_[self.forward(rows, c0).ravel(), np.sqrt(RIDGE)*c0]

            def rmv(x):
                return root(self.adjoint(rows, x[:ndata])+np.sqrt(RIDGE)*x[ndata:])

            a = LinearOperator((ndata+d, d), matvec=mv, rmatvec=rmv, dtype=np.complex128)
            result = lsmr(a, np.r_[self.z.ravel(), np.zeros(d)], atol=1e-14, btol=1e-14, conlim=1e12, maxiter=2000)
            c = root(result[0])
            info = dict(istop=int(result[1]), iterations=int(result[2]))
        else:
            raise ValueError(method)
        info.update(self.check(rows, b, c))
        cert = info['coefficient_certificate']
        info['sufficient'] = cert is not None and cert <= TOL
        return c, info

    def evaluate(self, p, method):
        p = np.asarray(p).reshape(2, 2)
        c, info = self.solve(p, method)
        op, rows, _ = self.parts(p)
        residual = self.z-self.forward(rows, c)
        value = float((np.vdot(residual, residual).real+RIDGE*np.vdot(c, c).real)/self.energy)
        deriv = derivative_action(op, p, c)
        projected = np.column_stack([self.fam.analysis(deriv[:, j]).ravel() for j in range(4)])
        gradient = -2*(residual.ravel().conj()@projected).real/self.energy
        return value, gradient, c, info


def preflight(mode):
    folder = OUT/(mode+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    assert not folder.exists() and folder.resolve().is_relative_to(ROOT/'outputs_e2e')
    inputs = json.loads((INPUT/'input_identities.json').read_text(encoding='utf-8'))
    files = sorted((INPUT/'inputs').glob('*_25000.npz'))
    for p in files:
        assert p.resolve().is_relative_to(ROOT/'outputs_e2e') and digest(p) == inputs[str(p)]
    src = sorted((ROOT/'统一模型代码/gates').rglob('*.py'))
    identities = {str(p): digest(p) for p in src+files}
    folder.mkdir(parents=True)
    (folder/'source_snapshot').mkdir()
    for p in src:
        target = folder/'source_snapshot'/p.relative_to(ROOT/'统一模型代码/gates')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    write(folder/'protocol.json', dict(mode=mode, identities=identities, python=sys.version, platform=platform.platform(),
        numpy=np.__version__, scipy=scipy.__version__, fs=FS, band=BAND, ridge=RIDGE, tolerance=TOL,
        threads=2, input_use='reused development data; truth only for fixed diagnostic points and offline metrics',
        targets='same retained observation bins and same ridge; preparation charged separately and in full cold calls',
        az=dict(paper='10.1137/19M1306385', algorithms=['2.1', '3.2'], widths=[160, 320, 640], cutoff=1e-12),
        lsmr=dict(paper='10.1137/10079687X', atol=1e-14, btol=1e-14, maxiter=2000,
                  actions='FFT retained-band augmented system, block right preconditioning')))
    return folder, files, identities


def input_data(path, n, gap):
    with np.load(path, allow_pickle=False) as z:
        iq = z['signal'][0, :, :n]*10**(gap/20)+z['signal'][1, :, :n]+z['noise'][:, :n]
        return iq, z['positions']


def tests(folder, path, n=1024):
    iq, truth = input_data(path, n, 15)
    fam = Family(n)
    fam.setup('direct')
    fam.setup('boundary')
    q = fam.q
    error = np.linalg.norm(fam.kernel+fam.v.conj().T@fam.v-fam.p*np.eye(q))/fam.p/np.sqrt(q)
    assert error < 1e-10
    pb = Problem(fam, iq)
    p = truth+[[5, -3], [-4, 6]]
    vd, gd, cd, di = pb.evaluate(p, 'direct')
    vb, gb, cb, bi = pb.evaluate(p, 'boundary')
    ce = float(np.linalg.norm(cb-cd)/np.linalg.norm(cd))
    ge = float(np.linalg.norm(gb-gd)/np.linalg.norm(gd))
    assert ce <= TOL and ge <= 1e-5
    op, rows, b = pb.parts(p)
    blocks = [np.concatenate([fam.h*rows[l, k*q:(k+1)*q][None] for k in range(2)], axis=1) for l in range(4)]
    dense = np.vstack(blocks)
    ae = np.linalg.norm(dense.conj().T@pb.z.ravel()-b)/np.linalg.norm(b)
    assert ae < 1e-11
    fe = np.linalg.norm(dense@cd-pb.forward(rows, cd).ravel())/np.linalg.norm(dense@cd)
    assert fe < 1e-10
    finite = []
    for j in range(4):
        plus, minus = p.copy(), p.copy()
        plus.flat[j] += .001
        minus.flat[j] -= .001
        finite.append((pb.evaluate(plus, 'direct')[0]-pb.evaluate(minus, 'direct')[0])/.002)
    fd = float(np.linalg.norm(np.asarray(finite)-gd)/np.linalg.norm(gd))
    assert fd < 1e-3
    rng = np.random.default_rng(20261023027)
    outside = rng.normal(size=iq.shape)+1j*rng.normal(size=iq.shape)
    spectrum = np.fft.fft(outside, axis=-1)
    spectrum[:, fam.keep] = 0
    altered = Problem(fam, iq+10*np.fft.ifft(spectrum, axis=-1))
    zerr = float(np.linalg.norm(pb.z-altered.z)/np.linalg.norm(pb.z))
    assert zerr < 1e-12
    v2, g2, c2, _ = altered.evaluate(p, 'boundary')
    assert np.linalg.norm(c2-cb)/np.linalg.norm(cb) < TOL
    result = dict(n=n, q=q, retained=len(fam.keep), rank=fam.rank, rank_lower=q-len(fam.keep),
        uniform_bound=fam.bound, gram_identity=error, adjoint=ae, forward=fe, coefficient_error=ce,
        gradient_error=ge, finite_difference_error=fd, outside_band_input_error=zerr,
        outside_band_objective_change=abs(v2-vb), outside_band_gradient_change=float(np.linalg.norm(g2-gb)),
        direct=di, boundary=bi, preparation=fam.times)
    write(folder/f'tests_{n}.json', result)
    print('正确性通过', n, 'rank', fam.rank, 'coeff', ce, 'gradient', ge, flush=True)


def numerical(folder, files, limit):
    rows, preps = [], []
    for n in (1024, 4096):
        fam = Family(n)
        fam.setup('direct')
        fam.setup('boundary')
        fam.setup('az')
        preps.append(dict(n=n, q=fam.q, retained=len(fam.keep), rank=fam.rank, bound=fam.bound, times=fam.times))
        write(folder/'preparation.json', preps)
        for index, path in enumerate(files[:limit]):
            iq, truth = input_data(path, n, 30)
            pb = Problem(fam, iq)
            for label, offset in [('near', [[5, -3], [-4, 6]]), ('far', [[170, -130], [-160, 120]])]:
                p = truth+offset
                reference = pb.solve(p, 'direct')[0]
                for repeat in range(2):
                    methods = ['direct', 'boundary', 'az', 'lsmr']
                    if (index+repeat) % 2:
                        methods.reverse()
                    for method in methods:
                        tick = time.perf_counter()
                        c, info = pb.solve(p, method)
                        elapsed = time.perf_counter()-tick
                        err = float(np.linalg.norm(c-reference)/np.linalg.norm(reference))
                        rows.append(dict(n=n, case=path.stem, point=label, repeat=repeat, method=method,
                            seconds=elapsed, coefficient_error=err, sufficient=bool(err <= TOL and info['sufficient']), info=info))
                        write(folder/'results.json', rows)
                        print(n, path.stem, label, repeat, method, round(elapsed, 3), 'err', f'{err:.2e}', flush=True)


def blind(folder, files, methods, limit, outside=False):
    import torch
    from 统一模型代码.gates.g10.r3.solvers import Model, Trace
    from 统一模型代码.gates.g10.r2.solvers import local
    from 统一模型代码.gates.g10.physics import grid
    from 统一模型代码.gates.g10.methods import peaks
    from 统一模型代码.gates.g10.r4.paper_checks.assignment_metrics import evaluate
    cfg = json.loads((INPUT/'protocol.json').read_text(encoding='utf-8'))['config']
    rows = []
    for index, path in enumerate(files[:limit]):
        for gap in (15, 30):
            iq, truth = input_data(path, 4096, gap)
            if outside:
                rng = np.random.default_rng(20261023028)
                perturb = rng.normal(size=iq.shape)+1j*rng.normal(size=iq.shape)
                ft = np.fft.fft(perturb, axis=-1)
                ft[:, abs(np.fft.fftfreq(iq.shape[1], 1/FS)) <= BAND] = 0
                iq = iq+10*np.fft.ifft(ft, axis=-1)
            result = {}
            order = methods if (index+(gap == 30)) % 2 == 0 else list(reversed(methods))
            for method in order:
                torch.cuda.synchronize()
                tick = time.perf_counter()
                fam = Family(4096)
                fam.setup('direct')  # Shared first-source finite fit, counted for all.
                fam.setup(method)
                pb = Problem(fam, iq)
                trace = Trace(lambda: None, history=False)
                m = Model(fam.synthesis(pb.z), trace.check, sample_rate_hz=FS, half_band_hz=BAND)
                trace.model = m
                points = grid(20)
                first = points[peaks(points, m.conditional(points), 1)[0]]
                first, _, _ = local(m, first[None], cfg)
                c, _ = pb.solve(first[None], 'direct')
                _, rr, _ = pb.parts(first[None])
                residual = pb.z-pb.forward(rr, c)
                rm = Model(fam.synthesis(residual), trace.check, sample_rate_hz=FS, half_band_hz=BAND)
                ids = peaks(points, rm.conditional(points), 5)
                starts = [np.vstack([first, points[j]]) for j in ids]
                cache, infos = {}, []

                def objective(x):
                    key = np.asarray(x).tobytes()
                    if key not in cache:
                        val, grad, _, info = pb.evaluate(np.asarray(x).reshape(2, 2), method)
                        if not info['sufficient']:
                            raise RuntimeError(f'{method}: common coefficient certificate not reached')
                        cache[key] = val, grad
                        infos.append(info)
                    return cache[key]

                scores = [objective(p.ravel())[0] for p in starts]
                selected = int(np.argmin(scores))
                opt = minimize(objective, starts[selected].ravel(), jac=True, method='L-BFGS-B',
                    bounds=[(-1000., 1000.)]*4, options=dict(maxiter=100, ftol=1e-12, gtol=1e-8, maxls=25))
                final = opt.x.reshape(2, 2) if np.isfinite(opt.fun) and opt.fun <= scores[selected] else starts[selected]
                value, gradient = objective(final.ravel())
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-tick
                result[method] = dict(positions=final.tolist(), seconds=elapsed, preparation=fam.times,
                    starts=[p.tolist() for p in starts], selected=selected, objective=value,
                    gradient_max=float(abs(gradient).max()), iterations=int(opt.nit), calls=len(cache),
                    success=bool(opt.success), message=str(opt.message), inner=infos,
                    metrics=evaluate(truth, final))
                write(folder/'progress.json', dict(case=path.stem, gap=gap, completed_methods=result,
                                                  previous_cases=len(rows), outside_perturbation=outside))
                print('盲配对', path.stem, gap, method, round(elapsed, 2), result[method]['metrics']['both20'], flush=True)
            if 'direct' in result:
                for method in methods:
                    np.testing.assert_allclose(result['direct']['starts'], result[method]['starts'], atol=1e-9, rtol=0)
            rows.append(dict(case=path.stem, gap=gap, results=result))
            write(folder/'results.json', rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['tests', 'numerical', 'blind'])
    parser.add_argument('--limit', type=int, default=4)
    parser.add_argument('--methods', default='direct,boundary')
    parser.add_argument('--outside', action='store_true')
    args = parser.parse_args()
    folder, files, identities = preflight(args.mode)
    print('结果目录', folder, flush=True)
    started = time.perf_counter()
    write(folder/'invocation.json', vars(args))
    try:
        with threadpool_limits(limits=2):
            if args.mode == 'tests':
                for n in (1024, 4096):
                    tests(folder, files[0], n)
            elif args.mode == 'numerical':
                numerical(folder, files, args.limit)
            else:
                blind(folder, files, args.methods.split(','), args.limit, args.outside)
    except Exception:
        write(folder/'failure.json', dict(completed=False, traceback=traceback.format_exc(),
                                          seconds=time.perf_counter()-started))
        raise
    assert all(digest(p) == expected for p, expected in identities.items())
    write(folder/'audit.json', dict(completed=True, identities_unchanged=True, seconds=time.perf_counter()-started,
        argv=sys.argv, formal_independent_evidence=False))
    print('完成', folder, flush=True)


if __name__ == '__main__':
    main()
