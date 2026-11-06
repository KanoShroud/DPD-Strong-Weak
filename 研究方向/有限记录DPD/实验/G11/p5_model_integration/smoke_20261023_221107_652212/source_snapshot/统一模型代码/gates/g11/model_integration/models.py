"""G11-P5 observation-only representations, with model-consistent LS/VP.

The finite bases are conventional Fourier-extension/Slepian models, NOT new
algorithms. All observations are the same selected unitary DFT coefficients.
Hann is the historical windowed-data approximate criterion, not white-noise ML.
"""
import time
from collections import Counter

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from ...g8.physics import C, RECEIVERS
from ...research_discovery.band_validation import Family, FS, BAND
from ...research_discovery.slepian_reference import SlepianFamily, reconstruct


def geometry(p, frequencies):
    p = np.asarray(p, float).reshape(-1, 2)
    v = p[:, None] - np.asarray(RECEIVERS)[None]
    raw = np.linalg.norm(v, axis=-1)
    d = np.maximum(raw, 1.)
    u = np.where((raw > 1.)[..., None], v/d[..., None], 0.)
    gain = (1/d)/np.sqrt(np.sum(d**-2, axis=1))[:, None]
    weights = gain**2
    log_gain = -u/d[..., None] + np.sum(weights[..., None]*u/d[..., None], axis=1)[:, None]
    delay = (d-d[:, :1])/C
    dt = (u-u[:, :1])/C
    phase = np.exp(-2j*np.pi*delay[..., None]*frequencies)
    multiplier = log_gain[:, :, None, :] - 2j*np.pi*frequencies[None, None, :, None]*dt[:, :, None, :]
    return gain[..., None]*phase, multiplier


def single_second(p, frequencies):
    """Second log-response derivative, including normalized 1/d amplitude."""
    v = np.asarray(p)[None]-np.asarray(RECEIVERS)
    raw = np.linalg.norm(v, axis=1)
    if np.any(raw <= 1.):
        raise ArithmeticError('Newton point inside the 1 m receiver clamp; nonsmooth location')
    d, u = raw, v/raw[:, None]
    uu = u[:, :, None]*u[:, None, :]
    w = d**-2/np.sum(d**-2)
    mean = np.sum(w[:, None]*u/d[:, None], axis=0)
    hmean = np.sum(w[:, None, None]*(np.eye(2)-4*uu)/d[:, None, None]**2, axis=0)+2*np.outer(mean, mean)
    hlog = -(np.eye(2)-2*uu)/d[:, None, None]**2+hmean
    hd = (np.eye(2)-uu)/d[:, None, None]
    return hlog[:, None]-2j*np.pi*frequencies[None, :, None, None]*(hd-hd[:1])[:, None]/C


class Periodic:
    """Independent coefficients per retained frequency, as in the frozen model."""
    def __init__(self, iq, hann=False, **unused):
        start = time.perf_counter()
        self.n = iq.shape[-1]
        self.keep = np.flatnonzero(abs(np.fft.fftfreq(self.n, 1/FS)) <= BAND)
        z = np.fft.fft(iq, axis=-1)[:, self.keep]/np.sqrt(self.n)
        if hann:
            z = np.fft.fft(reconstruct(z, self.n, self.keep)*np.hanning(self.n), axis=-1)[:, self.keep]/np.sqrt(self.n)
        self.z = z
        self.f = np.fft.fftfreq(self.n, 1/FS)[self.keep]
        self.energy = float(np.vdot(z, z).real)
        self.counts = Counter()
        self.ridge = 0.
        self.preparation_seconds = time.perf_counter()-start
        self.metadata = dict(n=self.n, frequencies=len(self.f), coefficients_per_source=len(self.f),
                             ridge=0., criterion='Hann approximate' if hann else 'periodic LS')

    def response(self, p):
        a, mult = geometry(p, self.f)
        return a.transpose(2, 1, 0), mult

    def fit(self, p, data=None, gradient=False):
        self.counts['fits'] += 1
        p = np.asarray(p).reshape(-1, 2)
        data = self.z if data is None else data
        a, mult = self.response(p)
        x = data.T
        # SVD for the tiny station x source matrices also handles duplicate sources.
        inv = np.linalg.pinv(a, rcond=1e-12)
        coef = np.einsum('fkl,fl->fk', inv, x)
        contributions = (a*coef[:, None, :]).transpose(2, 1, 0)
        residual = data-contributions.sum(axis=0)
        value = float(np.vdot(residual, residual).real/self.energy)
        g = None
        if gradient:
            self.counts['gradients'] += 1
            deriv = contributions[..., None]*mult
            g = -2*np.einsum('lf,klfd->kd', residual.conj(), deriv).real.ravel()/self.energy
        return dict(value=value, gradient=g, coefficients=coef, contributions=contributions,
                    residual=residual, normal_relative=0.)

    def scan(self, points, others=None, data=None, progress=None):
        data = self.z if data is None else data
        self.counts['scans'] += 1
        self.counts['grid_points'] += len(points)
        if others is not None and len(others):
            a, _ = self.response(others)
            inv = np.linalg.pinv(a, rcond=1e-12)
            proj = np.eye(4)[None]-a@inv
            r = np.einsum('flm,mf->lf', proj, data)
        else:
            proj, r = None, data
        scores = []
        for lo in range(0, len(points), 32):
            a, _ = geometry(points[lo:lo+32], self.f)
            v = a if proj is None else np.einsum('flm,bmf->blf', proj, a)
            norm = np.sum(abs(v)**2, axis=1)
            num = abs(np.einsum('blf,lf->bf', v.conj(), r))**2
            scores.extend(np.sum(np.divide(num, norm, out=np.zeros_like(num), where=norm > 1e-20), axis=1)/self.energy)
            if progress:
                progress(min(lo+32, len(points)), len(points))
        return np.asarray(scores)

    def hessian(self, p, data):
        fit = self.fit(np.asarray(p)[None], data, gradient=True)
        a, t = geometry(np.asarray(p)[None], self.f)
        a, t = a[0], t[0]
        q = fit['coefficients'][:, 0]
        da = a[..., None]*t
        daa = a[..., None, None]*(t[..., :, None]*t[..., None, :]+single_second(p, self.f))
        r = fit['residual']
        j = da*q[None, :, None]
        fixed = 2*np.einsum('lfi,lfj->ij', j.conj(), j).real
        fixed -= 2*np.einsum('lf,lfij,f->ij', r.conj(), daa, q).real
        cross = np.einsum('lfi,lf->fi', da.conj(), r)-np.einsum('lf,lfi->fi', a.conj(), j)
        correction = 2*np.einsum('fi,fj,f->ij', cross.conj(), cross, 1/np.sum(abs(a)**2, axis=0)).real
        self.counts['profile_hessians'] += 1
        return fit, (fixed-correction)/self.energy


class Finite:
    """Dense reference with full cross-frequency coupling and exact Schur scans.

Basis coordinates are whitened on the same extended discrete time interval.
Thus ridge penalizes *reference-waveform energy*, not arbitrary column scale.
All geometry/normalization derivatives are included. Preparation is charged.
"""
    def __init__(self, iq, kind='finite_fourier', eps=1e-6, ridge=1e-6):
        tick = time.perf_counter()
        self.n = iq.shape[-1]
        self.kind, self.ridge = kind, float(ridge)
        self.keep = np.flatnonzero(abs(np.fft.fftfreq(self.n, 1/FS)) <= BAND)
        self.z = np.fft.fft(iq, axis=-1)[:, self.keep]/np.sqrt(self.n)
        self.energy = float(np.vdot(self.z, self.z).real)
        self.counts = Counter()
        pad = int(np.ceil(FS*1000/C))
        self.support = np.arange(-pad, self.n+pad)
        if kind == 'finite_fourier':
            f = Family(self.n)
            self.f = f.f
            t = np.exp(2j*np.pi*np.arange(self.n)[:, None]*self.f[None]/FS)
            self.h = np.fft.fft(t, axis=0)[self.keep]/np.sqrt(self.n)
            self.transform = np.eye(f.q)/np.sqrt(len(self.support))
            self.rank = f.q
            # Public Gram reused at every location; do not rebuild A^H A from
            # a tall time/frequency matrix for each grid point.
            self.kernel = self.h.conj().T@self.h/len(self.support)
            basis_error = 0.
        elif kind == 'slepian':
            family = SlepianFamily(self.n, eps)
            self.f = family.f*FS
            self.h = family.h
            ext = np.exp(2j*np.pi*(self.support-family.center)[:, None]*family.f[None])@family.basis
            gram = ext.conj().T@ext
            # Symmetric whitening, not pruning weak modes or changing retained rank.
            values, vectors = np.linalg.eigh((gram+gram.conj().T)/2)
            if values[0] <= 1e-12*values[-1]:
                raise ArithmeticError('Slepian support metric is numerically rank deficient')
            transform = (vectors/np.sqrt(values)[None])@vectors.conj().T
            self.transform = family.basis@transform
            self.rank = family.rank
            basis_error = float(np.linalg.norm(transform.conj().T@gram@transform-np.eye(self.rank))/np.sqrt(self.rank))
        else:
            raise ValueError(kind)
        self._cache = {}
        self.max_normal_relative = 0.
        self.metadata = dict(n=self.n, frequencies=len(self.keep), coefficients_per_source=self.rank,
            ridge=self.ridge, eps=eps if kind == 'slepian' else None,
            coefficient_units='reference waveform energy over common extended discrete interval',
            support=[int(self.support[0]), int(self.support[-1])], basis_isometry_error=basis_error,
            time_reference='station 0; fixed symmetric support, not nearest-station switching')
        self.preparation_seconds = time.perf_counter()-tick

    def block(self, p):
        key = np.asarray(p, dtype=float).tobytes()
        if key not in self._cache:
            a, _ = geometry(np.asarray(p)[None], self.f)
            if self.kind == 'finite_fourier':
                scale = self.transform[0, 0]
                block = (self.h[None]*a[0, :, None, :]*scale).reshape(-1, self.rank)
            else:
                block = np.vstack([(self.h*a[0, l])@self.transform for l in range(4)])
            if len(self._cache) >= 3:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = block
        return self._cache[key]

    def derivative(self, p, coef):
        a, mult = geometry(p, self.f)
        out = []
        for k in range(len(p)):
            spectrum = self.transform@coef[k*self.rank:(k+1)*self.rank]
            for axis in range(2):
                out.append(np.concatenate([self.h@(a[k, l]*spectrum*mult[k, l, :, axis]) for l in range(4)]))
        return np.column_stack(out)

    def matrix(self, p):
        return np.hstack([self.block(x) for x in np.asarray(p).reshape(-1, 2)])

    def fit(self, p, data=None, gradient=False):
        self.counts['fits'] += 1
        p = np.asarray(p).reshape(-1, 2)
        z = (self.z if data is None else data).ravel()
        if self.kind == 'finite_fourier':
            phase, _ = geometry(p, self.f)
            gram = np.block([[self.kernel*(phase[i].conj().T@phase[j])
                              for j in range(len(p))] for i in range(len(p))])
            projected = self.h.conj().T@z.reshape(self.z.shape).T/np.sqrt(len(self.support))
            b = np.concatenate([np.sum(phase[i].conj()*projected.T, axis=0) for i in range(len(p))])
            a = None
        else:
            a = self.matrix(p)
            gram = a.conj().T@a
            b = a.conj().T@z
        gram.flat[::len(gram)+1] += self.ridge
        c = cho_solve(cho_factor(gram, lower=True), b)
        if a is None:
            contributions = np.stack([(self.h@(phase[i]*c[i*self.rank:(i+1)*self.rank]).T).T/np.sqrt(len(self.support))
                                      for i in range(len(p))])
        else:
            contributions = np.stack([(self.block(point)@c[i*self.rank:(i+1)*self.rank]).reshape(self.z.shape)
                                      for i, point in enumerate(p)])
        residual = z-contributions.sum(axis=0).ravel()
        normal = gram@c-b
        nr = float(np.linalg.norm(normal)/max(np.linalg.norm(b), 1e-30))
        if not np.isfinite(c).all() or nr > 1e-8:
            raise ArithmeticError(f'Full coupled coefficient solve failed: {nr}')
        self.max_normal_relative = max(self.max_normal_relative, nr)
        g = None
        if gradient:
            self.counts['gradients'] += 1
            g = -2*(residual.conj()@self.derivative(p, c)).real/self.energy
        return dict(value=float((np.vdot(residual, residual).real+self.ridge*np.vdot(c, c).real)/self.energy),
            gradient=g, coefficients=c, contributions=contributions, residual=residual.reshape(self.z.shape),
            normal_relative=nr)

    def scan(self, points, others=None, data=None, progress=None):
        """Exact regularized conditional gain; not residual-only matching for AP."""
        self.counts['scans'] += 1
        self.counts['grid_points'] += len(points)
        z = (self.z if data is None else data).ravel()
        if self.kind == 'finite_fourier':
            return self.fourier_scan(points, others, z, progress)
        a0 = None
        if others is not None and len(others):
            a0 = self.matrix(others)
            g0 = a0.conj().T@a0+self.ridge*np.eye(a0.shape[1])
            factor = cho_factor(g0, lower=True)
            q0 = cho_solve(factor, a0.conj().T@z)
        scores = []
        for i, point in enumerate(points):
            a = self.block(point)
            g = a.conj().T@a+self.ridge*np.eye(self.rank)
            b = a.conj().T@z
            if a0 is not None:
                cross = a0.conj().T@a
                g -= cross.conj().T@cho_solve(factor, cross)
                b -= cross.conj().T@q0
            g = (g+g.conj().T)/2
            c = cho_solve(cho_factor(g, lower=True), b)
            scores.append(float(np.vdot(b, c).real/self.energy))
            if progress:
                progress(i+1, len(points))
        return np.asarray(scores)

    def fourier_scan(self, points, others, z, progress):
        """Algebraically identical shared-Gram acceleration, not a new method."""
        projected = (self.h.conj().T@z.reshape(self.z.shape).T).T/np.sqrt(len(self.support))
        prior = None
        if others is not None and len(others):
            prior, _ = geometry(others, self.f)
            g0 = np.block([[self.kernel*(a.conj().T@b) for b in prior] for a in prior])
            g0.flat[::len(g0)+1] += self.ridge
            factor = cho_factor(g0, lower=True)
            b0 = np.concatenate([np.sum(a.conj()*projected, axis=0) for a in prior])
            q0 = cho_solve(factor, b0)
        scores = []
        for i, point in enumerate(points):
            current = geometry(point[None], self.f)[0][0]
            gram = self.kernel*(current.conj().T@current)
            gram.flat[::len(gram)+1] += self.ridge
            b = np.sum(current.conj()*projected, axis=0)
            if prior is not None:
                cross = np.vstack([self.kernel*(a.conj().T@current) for a in prior])
                gram -= cross.conj().T@cho_solve(factor, cross)
                b -= cross.conj().T@q0
            gram = (gram+gram.conj().T)/2
            coef = cho_solve(cho_factor(gram, lower=True), b)
            scores.append(float(np.vdot(b, coef).real/self.energy))
            if progress:
                progress(i+1, len(points))
        return np.asarray(scores)

    def hessian(self, p, data):
        fit = self.fit(np.asarray(p)[None], data, gradient=True)
        a, t = geometry(np.asarray(p)[None], self.f)
        a, t = a[0], t[0]
        second = t[..., :, None]*t[..., None, :]+single_second(p, self.f)
        matrix = self.block(p)
        c, r = fit['coefficients'], fit['residual'].ravel()

        def differentiated(mult):
            if self.kind == 'finite_fourier':
                return (self.h[None]*(a*mult)[:, None]*self.transform[0, 0]).reshape(-1, self.rank)
            return np.vstack([(self.h*a[l]*mult[l])@self.transform for l in range(4)])

        ds = [differentiated(t[..., i]) for i in range(2)]
        j = np.column_stack([d@c for d in ds])
        fixed = 2*(j.conj().T@j).real
        spectrum = self.transform@c
        for i in range(2):
            for k in range(2):
                action = np.concatenate([self.h@(a[l]*second[l, :, i, k]*spectrum) for l in range(4)])
                fixed[i, k] -= 2*np.vdot(r, action).real
        cross = np.column_stack([d.conj().T@r for d in ds])-matrix.conj().T@j
        gram = matrix.conj().T@matrix+self.ridge*np.eye(self.rank)
        correction = 2*(cross.conj().T@cho_solve(cho_factor(gram, lower=True), cross)).real
        self.counts['profile_hessians'] += 1
        return fit, (fixed-correction)/self.energy


def build(iq, kind, **kwargs):
    if kind in ('periodic', 'hann'):
        return Periodic(iq, hann=kind == 'hann')
    return Finite(iq, kind=kind, **kwargs)
