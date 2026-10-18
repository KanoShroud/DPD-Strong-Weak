"""Same full-frequency known-response model; exact small-K LS acceleration."""
import numpy as np
import torch
from ..physics import Model as OriginalModel, Spectrum as OriginalSpectrum, atom, fit
from .config import FS


def small_fit(a, x):
    k = a.shape[-1]
    b = (a.mH@x[..., None]).squeeze(-1)
    gram = a.mH@a
    if k == 1:
        s = b/gram[..., 0, 0].real[..., None]
    elif k == 2:
        g0, g1, cross = gram[..., 0, 0].real, gram[..., 1, 1].real, gram[..., 0, 1]
        det = g0*g1-abs(cross)**2
        safe = det.clamp_min(1e-15)
        s = torch.stack(((g1*b[..., 0]-cross*b[..., 1])/safe,
                         (g0*b[..., 1]-cross.conj()*b[..., 0])/safe), -1)
        bad = det < 1e-8*g0*g1
        if bool(bad.any()):
            s = s.clone()
            s[bad] = fit(a[bad], x[bad])[0]
    else:
        return fit(a, x)
    return s, x-(a@s[..., None]).squeeze(-1)


class Model(OriginalModel):
    def __init__(self, iq, check=lambda: None, device='cuda', sample_rate_hz=FS):
        if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
            raise ValueError('Invalid sampling rate')
        super().__init__(iq, check, device)
        self.sample_rate_hz = float(sample_rate_hz)
        self.f = torch.as_tensor(np.fft.fftfreq(self.n, 1/self.sample_rate_hz), device=device)

    def enable_response_cache(self, positions, chunk=256):
        """Optional exact complex128 dictionary cache; build cost belongs to solver."""
        pp = np.asarray(positions, float).copy()
        required = self.n*4*len(pp)*16
        free, _ = torch.cuda.mem_get_info()
        # Reserve workspace and respect the existing 14 GiB allocation ceiling.
        reusable = torch.cuda.memory_reserved()-torch.cuda.memory_allocated()
        allowance = min(free+reusable-2*2**30, 13*2**30-torch.cuda.memory_allocated())
        if required > allowance:
            self.counts['dictionary_cache_bytes'] = 0
            return False
        # atom() is station-contiguous. Preserve its strides so caching does not
        # silently select a different reduction/matrix-multiply layout.
        self._dictionary = torch.empty_strided((self.n, 4, len(pp)), (4*len(pp), 1, 4),
                                               dtype=torch.complex128, device=self.device)
        self._dictionary_positions = pp
        self._dictionary_step = float(pp[1, 0]-pp[0, 0]) if len(pp)>1 else 0.
        for lo in range(0, self.n, chunk):
            self.check()
            self._dictionary[lo:lo+chunk] = super().response(pp, self.f[lo:lo+chunk])
        self.counts['dictionary_cache_bytes'] = required
        self.counts['dictionary_cache_hits'] = 0
        return True

    def refresh_response_cache(self, positions, chunk=256):
        if not hasattr(self, '_dictionary'):
            return
        pp = np.asarray(positions)
        changed = np.flatnonzero(np.any(pp != self._dictionary_positions, axis=1))
        if len(changed):
            for lo in range(0, self.n, chunk):
                self._dictionary[lo:lo+chunk, :, changed] = super().response(pp[changed], self.f[lo:lo+chunk])
            self._dictionary_positions[changed] = pp[changed]

    def response(self, positions, freq=None):
        if hasattr(self, '_dictionary') and not torch.is_tensor(positions):
            pp = np.asarray(positions).reshape(-1, 2)
            f = self.f if freq is None else freq
            lo = f.storage_offset()-self.f.storage_offset()
            shared = f.untyped_storage().data_ptr() == self.f.untyped_storage().data_ptr()
            if pp.shape == self._dictionary_positions.shape and np.array_equal(pp, self._dictionary_positions):
                if shared:
                    self.counts['dictionary_cache_hits'] += 1
                    return self._dictionary[lo:lo+len(f)]
            if shared and self._dictionary_step > 0:
                side = int(round(np.sqrt(len(self._dictionary_positions))))
                xy = np.rint((pp+1000)/self._dictionary_step).astype(int)
                if np.all((xy>=0)&(xy<side)):
                    ids = xy[:,1]*side+xy[:,0]
                    if np.array_equal(pp, self._dictionary_positions[ids]):
                        self.counts['dictionary_cache_hits'] += 1
                        if len(ids)==1 or np.all(np.diff(ids)==1):
                            return self._dictionary[lo:lo+len(f), :, ids[0]:ids[-1]+1]
                        return self._dictionary[lo:lo+len(f), :, ids]
        return super().response(positions, freq)

    def fitted(self, positions):
        a = self.response(positions)
        s, residual = small_fit(a, self.x)
        return a, s, residual

    def joint_tensor(self, p):
        return abs(small_fit(self.response(p.reshape(-1, 2)), self.x)[1]).square().sum()/self.energy

    def joint_batch(self, population, chunk=16):
        out = []
        for lo in range(0, len(population), chunk):
            self.check()
            p = torch.as_tensor(np.asarray(population[lo:lo+chunk]), dtype=torch.float64, device=self.device)
            batch, k = p.shape[:2]
            a = self.response(p.reshape(-1, 2)).reshape(self.n, 4, batch, k).permute(2, 0, 1, 3)
            residual = small_fit(a, self.x.expand(batch, -1, -1))[1]
            out.extend((abs(residual).square().sum((1, 2))/self.energy).cpu().tolist())
            self.counts['joint_evaluations'] += batch
        return np.asarray(out)


class Spectrum(OriginalSpectrum):
    def __init__(self, model, method, count, segments, loading, weight='uniform'):
        super().__init__(model, method, count, segments, loading)
        # Covariance construction is rate-independent; the response frequency axis is not.
        self.freq = torch.as_tensor(np.fft.fftfreq(model.n//segments, 1/model.sample_rate_hz),
                                    device=model.device)
        self.weight_mode = weight
        energy = self.cov.diagonal(dim1=1, dim2=2).real.sum(1)
        excess = (energy-4*segments/model.n).clamp_min(0)
        self.weight = (excess if weight == 'excess_power' else torch.ones_like(energy))
        if float(self.weight.sum().cpu()) <= 0:
            self.weight = torch.ones_like(energy)
        self.weight = self.weight/self.weight.sum()

    def tensor(self, points):
        a = atom(points, self.freq, self.m.stations)
        den = (a.conj()*(self.matrix@a)).sum(1).real.clamp_min(1e-15)
        if self.method == 'MUSIC':
            return 1/(self.weight[:, None]*den).sum(0)
        return (1/den).mean(0)

    def evaluate(self, points, chunk=64):
        pp = np.asarray(points).reshape(-1, 2)
        values = []
        with torch.no_grad():
            for lo in range(0, len(pp), chunk):
                self.m.check()
                p = torch.as_tensor(pp[lo:lo+chunk], dtype=torch.float64, device=self.m.device)
                values.extend(self.tensor(p).cpu().tolist())
                self.m.counts['candidate_points'] += len(p)
                self.m.counts['frequency_response_points'] += len(p)*len(self.freq)
        return np.asarray(values)

    def value_grad(self, position, scale):
        p = torch.tensor(np.asarray(position).reshape(1, 2), dtype=torch.float64,
                         device=self.m.device, requires_grad=True)
        value = -self.tensor(p)[0]/scale
        gradient = torch.autograd.grad(value, p)[0]
        self.m.counts['frequency_response_points'] += len(self.freq)
        self.m.counts['candidate_points'] += 1
        return float(value.detach().cpu()), gradient.detach().cpu().numpy().ravel()
