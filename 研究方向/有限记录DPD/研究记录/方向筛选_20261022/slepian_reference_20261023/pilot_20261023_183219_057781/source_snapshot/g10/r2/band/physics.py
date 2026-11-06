"""DFT selection, not resampling: sample count and selected-bin count differ."""
import numpy as np
import torch
from ...r1.physics import Spectrum as ParentSpectrum
from ..solvers import CountModel


class Model(CountModel):
    def __init__(self, iq, check=lambda: None, device='cuda', sample_rate_hz=50e6, half_band_hz=7e6):
        super().__init__(iq, check, device, sample_rate_hz=sample_rate_hz)
        self.sample_count = self.n
        self.half_band_hz = float(half_band_hz)
        if not 0 < half_band_hz <= sample_rate_hz/2:
            raise ValueError('Invalid common analysis band')
        mask = abs(self.f) <= half_band_hz
        self.frequency_indices = torch.where(mask)[0]
        self.f, self.x = self.f[mask], self.x[mask]
        # Legacy solvers use n for frequency-loop length. FFT scaling/noise
        # retain sample_count; covariance snapshots use their own segment size.
        self.n = len(self.f)
        self.frequency_noise_variance = 1/self.sample_count
        self.energy = abs(self.x).square().sum()
        self.counts.update(input_samples=self.sample_count, retained_frequencies=self.n)


class Spectrum(ParentSpectrum):
    def __init__(self, model, method, count, segments, loading, weight='uniform'):
        self.m, self.method, self.weight_mode = model, method, weight
        if model.sample_count % segments or segments < 4:
            raise ValueError('At least four within-record snapshots required')
        length = model.sample_count//segments
        frequencies = np.fft.fftfreq(length, 1/model.sample_rate_hz)
        keep = abs(frequencies) <= model.half_band_hz
        data = np.fft.fft(model.iq.reshape(4, segments, length), axis=-1)/length
        data = torch.as_tensor(data[:, :, keep], device=model.device).permute(2, 0, 1)
        self.freq = torch.as_tensor(frequencies[keep], device=model.device)
        self.cov = data@data.mH/segments
        ev, vec = torch.linalg.eigh(self.cov)
        self.rank_min = int((ev > ev[:, -1:]*1e-10).sum(-1).min().cpu())
        model.counts['covariance_decompositions'] += len(self.freq)
        if method == 'MUSIC':
            en = vec[:, :, :4-count]
            self.matrix = en@en.mH
        else:
            scale = self.cov.diagonal(dim1=1, dim2=2).real.mean(-1)
            self.matrix = torch.linalg.inv(self.cov+loading*scale[:, None, None]*torch.eye(4, device=model.device))
            model.counts['covariance_decompositions'] += len(self.freq)
        energy = self.cov.diagonal(dim1=1, dim2=2).real.sum(1)
        excess = (energy-4/length).clamp_min(0)
        self.weight = excess if weight == 'excess_power' else torch.ones_like(energy)
        if float(self.weight.sum().cpu()) <= 0:
            self.weight = torch.ones_like(energy)
        self.weight = self.weight/self.weight.sum()
