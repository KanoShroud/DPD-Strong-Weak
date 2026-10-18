"""Continuous RRC samples before cropping; paired randomness and exact record powers."""
import hashlib
import numpy as np
from ..g8.scenes import rrc, RATE, BW
from ..g8.physics import FS, C, RECEIVERS
from .config import N, SEED


def digest(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(a.tobytes()).hexdigest()


def make(role, group, realization=0, snr_db=3., gap_db=12., center='zero', n=N):
    roles = {'smoke': 0, 'center': 1, 'calibration': 2, 'tuning': 3, 'formal': 4}
    rid = roles[role]
    geo = np.random.default_rng(np.random.SeedSequence([SEED, rid, group, 0]))
    while True:
        pos = geo.uniform(-1000, 1000, (2, 2))
        distance = np.linalg.norm(pos[:, None] - RECEIVERS, axis=-1)
        if distance.min() >= 150 and np.linalg.norm(pos[0] - pos[1]) >= 300:
            break
    rng = np.random.default_rng(np.random.SeedSequence([SEED, rid, group, realization, 1]))
    nrng = np.random.default_rng(np.random.SeedSequence([SEED, rid, group, realization, 2]))
    drawn_center = rng.uniform(-FS/2+BW, FS/2-BW)
    rng.random()  # preserve the historical frequency-order draw, even at zero centre
    f0 = drawn_center if center == 'random' else float(0 if center == 'zero' else center)
    pulse_energy = np.sum(rrc(np.arange(-10, 10, .001))**2)*.001
    signals = np.empty((2, 4, n), complex)
    for k in range(2):
        bits = rng.choice([-1., 1.], size=int(n*RATE/FS)+512)
        timing, phase = rng.random(), rng.uniform(-np.pi, np.pi)
        for station in range(4):
            t = np.arange(n)/FS - distance[k, station]/C
            u = t*RATE-timing
            base = np.floor(u).astype(int)
            value = np.zeros(n)
            for delta in range(-10, 11):
                value += bits[base+delta+256]*rrc(u-base-delta)
            signals[k, station] = (value/np.sqrt(pulse_energy)
                * np.exp(2j*np.pi*f0*t+1j*phase)/distance[k, station])
    # One scalar per source, not station RMS: enforces the approved actual record gamma.
    target = BW/FS * 10**(snr_db/10) * np.array([10**(gap_db/10), 1.])
    signals *= np.sqrt(target/np.mean(abs(signals)**2, axis=(1, 2)))[:, None, None]
    noise = (nrng.normal(size=(4, n))+1j*nrng.normal(size=(4, n)))/np.sqrt(2)
    power = np.mean(abs(signals)**2, axis=(1, 2))
    meta = dict(role=role, group=group, realization=realization, seed=SEED, N=n,
                center_hz=f0, drawn_center_hz=drawn_center, positions=pos.tolist(),
                gap_db=float(gap_db), actual_gap_db=float(10*np.log10(power[0]/power[1])),
                snr_inband_db=float(snr_db), snr_fullband_db=float(snr_db+10*np.log10(BW/FS)),
                powers=power.tolist(), noise_sha=digest(noise), components_sha=digest(signals))
    return dict(signal=signals, noise=noise, positions=pos, meta=meta)


def observation(scene, kind):
    if kind == 'mixed':
        return scene['signal'].sum(0)+scene['noise'], scene['positions'].copy(), 2
    index = 1 if kind == 'weak' else 0
    return scene['signal'][index]+scene['noise'], scene['positions'][index:index+1].copy(), 1
