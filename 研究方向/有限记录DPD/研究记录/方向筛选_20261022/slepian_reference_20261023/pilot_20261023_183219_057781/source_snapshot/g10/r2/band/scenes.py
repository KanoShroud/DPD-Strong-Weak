"""Geometry-only selection; validation waveforms are never used for calibration."""
import numpy as np
from ....g8.physics import RECEIVERS
from ...r1.scenes import make as original_make, observation, digest
from .config import GEOMETRY_SEED, WAVEFORM_SEED, LAYOUTS, HALF_BAND_HZ


def geometry_valid(pp, category):
    p = np.asarray(pp)
    expected = ((True, True), (False, True), (True, False), (False, False))[category]
    inside = abs(p).sum(1) < 500
    if tuple(inside) != expected or abs(p).max() > 1000:
        return False
    if np.linalg.norm(p, axis=1).min() < 100 or np.linalg.norm(p[0]-p[1]) < 300:
        return False
    if np.linalg.norm(p[:, None]-RECEIVERS, axis=-1).min() < 150:
        return False
    for a in range(4):
        for b in range(a+1, 4):
            v, delta = RECEIVERS[b]-RECEIVERS[a], p-RECEIVERS[a]
            if (abs(v[0]*delta[:, 1]-v[1]*delta[:, 0])/np.linalg.norm(v)).min() < 50:
                return False
    return True


def layouts():
    rng = np.random.default_rng(GEOMETRY_SEED)
    result = []
    for group in range(4):
        for _ in range(100000):
            p = rng.uniform(-950, 950, (2, 2))
            if geometry_valid(p, group):
                result.append(p.tolist())
                break
        else:
            raise RuntimeError('No valid preregistered geometry')
    return result


def make(task, formal=False):
    group, repeat, gap, _ = task
    positions = layouts()[group] if formal else LAYOUTS[group]
    result = original_make(group, repeat, gap, positions=positions, seed=WAVEFORM_SEED)
    result['meta'].update(role='independent' if formal else 'development', geometry_seed=GEOMETRY_SEED if formal else None)
    return result


def band_energy(scene):
    signal = np.asarray(scene['signal'])
    freq = np.fft.fftfreq(signal.shape[-1], 1/scene['meta']['sample_rate_hz'])
    power = abs(np.fft.fft(signal, axis=-1))**2
    fractions = power[..., abs(freq) <= HALF_BAND_HZ].sum(-1)/power.sum(-1)
    return dict(minimum_fraction=float(fractions.min()), fractions=fractions.tolist(), half_band_hz=HALF_BAND_HZ)


__all__ = ['make', 'layouts', 'geometry_valid', 'observation', 'digest', 'band_energy']
