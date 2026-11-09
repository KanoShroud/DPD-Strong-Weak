# ruff: noqa: F401
# Preserve imported globals for legacy function cloning and reexports.
"""Numerical evaluation of Xia's unchanged finite Fourier spatial score.

FFT tables plus local Taylor evaluation replace exponential summation only.
No location candidate generation, threshold, path model or estimator lives here.
"""
import math
import time
import numpy as np

C = 299792458.0


class CrossSpectrumKernel:
    """Evaluate sum_f cross[m,f]*exp(2j*pi*f*tau)/N at arbitrary delays."""

    def __init__(self, y, frequency_hz, fft_length, sampling_hz,
                 oversample=8, order=8):
        start = time.perf_counter()
        y = np.asarray(y, dtype=np.complex128)
        f = np.asarray(frequency_hz, dtype=np.float64)
        if y.ndim != 2 or f.ndim != 1 or y.shape[1] != len(f):
            raise ValueError('Expected y[station,frequency] and frequency[frequency]')
        if not np.isfinite(y).all() or not np.isfinite(f).all():
            raise ValueError('Nonfinite observations or frequencies')
        if int(fft_length) != fft_length or fft_length < 1:
            raise ValueError('Positive integer original FFT length required')
        if not np.isfinite(sampling_hz) or sampling_hz <= 0:
            raise ValueError('Positive sampling frequency required')
        if int(oversample) != oversample or oversample < 1:
            raise ValueError('Positive integer oversampling required')
        if int(order) != order or not 0 <= order <= 20:
            raise ValueError('Taylor order must be an integer from 0 to 20')
        n = int(fft_length)
        bins_float = f*n/sampling_hz
        bins = np.rint(bins_float).astype(np.int64)
        if not np.allclose(bins_float, bins, rtol=0, atol=1e-8):
            raise ValueError('Frequency set is not on the original FFT lattice')
        if len(np.unique(bins % n)) != len(bins):
            raise ValueError('Repeated original FFT bin')
        if np.any(abs(f) > sampling_hz/2):
            raise ValueError('Frequency outside the original Nyquist interval')
        denominator = abs(y[0])**2
        if np.any(denominator == 0):
            raise ValueError('REFERENCE_SPECTRAL_ZERO: no regularization added')
        cross = y[0].conj()[None]*y/denominator[None]
        if not np.isfinite(cross).all():
            raise ValueError('Nonfinite original reference ratio')
        self.n = n
        self.fs = float(sampling_hz)
        self.oversample = int(oversample)
        self.order = int(order)
        self.stations_count = y.shape[0]
        self.nfft = self.n*self.oversample
        self.grid_hz = self.fs*self.oversample
        self.period = self.n/self.fs
        self.fmax = float(abs(f).max()) if len(f) else 0.
        if self.fmax <= 0:
            raise ValueError('At least one nonzero frequency required')
        self.cross_l1_over_n = np.sum(abs(cross),axis=1)/n
        self.max_u = np.pi*self.fmax/(self.fs*self.oversample)
        self.truncation_bound_per_station = (
            self.cross_l1_over_n*np.exp(self.max_u)
            *self.max_u**(self.order+1)/math.factorial(self.order+1))
        self.tables = np.empty((self.order+1,self.stations_count,self.nfft),
                               dtype=np.complex128)
        phase_derivative = 1j*f/self.fmax
        for j in range(self.order+1):
            coefficients = np.zeros((self.stations_count,self.nfft),dtype=np.complex128)
            # Signed original bins must map into the oversized FFT, not n.
            coefficients[:,bins % self.nfft] = cross*phase_derivative**j/math.factorial(j)
            self.tables[j] = np.fft.ifft(coefficients,axis=1)*(self.nfft/n)
        self.preparation_seconds = time.perf_counter()-start

    def components_from_delays(self, delays_s):
        """Return complex per-station scores for arbitrary [query,station] delays."""
        delay = np.asarray(delays_s,dtype=np.float64)
        if delay.ndim != 2 or delay.shape[1] != self.stations_count:
            raise ValueError('Expected delay[query,station]')
        if not np.isfinite(delay).all():
            raise ValueError('Nonfinite delay')
        grid_coordinate = delay*self.grid_hz
        if np.any(abs(grid_coordinate) >= np.iinfo(np.int64).max/2):
            raise ValueError('Delay too large for safe FFT grid indexing')
        nearest = np.rint(grid_coordinate).astype(np.int64)
        delta = delay-nearest/self.grid_hz
        u = (2*np.pi*self.fmax)*delta
        indices = nearest % self.nfft
        stations = np.arange(self.stations_count)[None,:]
        ans = self.tables[self.order,stations,indices].copy()
        for j in range(self.order-1,-1,-1):
            ans *= u
            ans += self.tables[j,stations,indices]
        return ans

    def components(self, points, stations):
        points = np.asarray(points,dtype=np.float64)
        stations = np.asarray(stations,dtype=np.float64)
        if points.ndim != 2 or stations.shape != (self.stations_count,points.shape[1]):
            raise ValueError('Incompatible candidate/station coordinates')
        if not np.isfinite(points).all() or not np.isfinite(stations).all():
            raise ValueError('Nonfinite coordinates')
        distance = np.linalg.norm(points[:,None]-stations[None],axis=-1)
        delay = (distance-distance[:,:1])/C
        # The reference delay is exactly zero, hence its factor is constant;
        # table index zero retains the same computed cross ratio as direct code.
        return self.components_from_delays(delay)

    def scores(self, points, stations, chunk_size=65536):
        points = np.asarray(points,dtype=np.float64)
        if int(chunk_size)!=chunk_size or chunk_size<1:
            raise ValueError('Positive integer chunk size required')
        result=np.empty(len(points),dtype=np.float64)
        for start in range(0,len(points),int(chunk_size)):
            stop=min(start+int(chunk_size),len(points))
            result[start:stop]=np.prod(abs(self.components(points[start:stop],stations)),axis=1)
        return result

    def product_truncation_bound(self, approximate_components):
        """Exact-arithmetic bound only; excludes FFT and floating roundoff."""
        amplitude = abs(np.asarray(approximate_components))
        error=np.zeros(amplitude.shape[0])
        product=np.ones(amplitude.shape[0])
        for m in range(self.stations_count):
            epsilon=self.truncation_bound_per_station[m]
            error=error*(amplitude[:,m]+epsilon)+product*epsilon
            product*=amplitude[:,m]
        return error

    def metadata(self):
        return dict(original_fft_length=self.n,oversample=self.oversample,
                    taylor_order=self.order,table_shape=list(self.tables.shape),
                    table_bytes=int(self.tables.nbytes),max_abs_u=self.max_u,
                    truncation_bound_per_station=self.truncation_bound_per_station.tolist(),
                    truncation_bound_excludes_float_roundoff=True,
                    preparation_seconds=self.preparation_seconds,
                    statistical_score_changed=False)
