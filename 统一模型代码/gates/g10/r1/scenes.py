"""Fixed approved geometry; independent realizations, exactly paired power scans."""
import numpy as np
from ...g8.scenes import rrc, RATE, BW
from ...g8.physics import C, RECEIVERS
from ..scenes import digest, observation as observation
from .config import SEED, LAYOUTS, LAYOUT_NAMES, N, SNR_DB, FS, FORMAL_REPEATS


def geometry_audit():
    rows = []
    for group, positions in enumerate(LAYOUTS):
        pp = np.asarray(positions)
        distance = np.linalg.norm(pp[:, None]-RECEIVERS, axis=-1)
        line = []
        for a in range(4):
            for b in range(a+1, 4):
                v = RECEIVERS[b]-RECEIVERS[a]
                delta = pp-RECEIVERS[a]
                line.extend((abs(v[0]*delta[:, 1]-v[1]*delta[:, 0])/np.linalg.norm(v)).tolist())
        inside = (abs(pp).sum(1) < 500).tolist()
        expected = ((True, True), (False, True), (True, False), (False, False))[group]
        assert inside == list(expected)
        assert distance.min() >= 150 and np.linalg.norm(pp, axis=1).min() >= 100
        assert min(line) >= 50 and np.linalg.norm(pp[0]-pp[1]) >= 300
        rows.append(dict(group=group, name=LAYOUT_NAMES[group], positions=pp.tolist(),
                         inside=inside, minimum_station_m=float(distance.min()),
                         minimum_line_m=float(min(line)),
                         separation_m=float(np.linalg.norm(pp[0]-pp[1]))))
    return rows


def make(group, realization, gap=0., n=N, snr=SNR_DB, fs=FS, positions=None, seed=SEED):
    if fs <= BW:
        raise ValueError('Complex sampling rate must leave bandwidth guard')
    pp = np.asarray(LAYOUTS[group] if positions is None else positions, float)
    distance = np.linalg.norm(pp[:, None]-RECEIVERS, axis=-1)
    rng = np.random.default_rng(np.random.SeedSequence([seed, group, realization, 1]))
    nrng = np.random.default_rng(np.random.SeedSequence([seed, group, realization, 2]))
    pulse_energy = np.sum(rrc(np.arange(-10, 10, .001))**2)*.001
    signals = np.empty((2, 4, n), complex)
    for k in range(2):
        bits = rng.choice([-1., 1.], size=int(n*RATE/fs)+512)
        timing, phase = rng.random(), rng.uniform(-np.pi, np.pi)
        for station in range(4):
            u = (np.arange(n)/fs-distance[k, station]/C)*RATE-timing
            base = np.floor(u).astype(int)
            value = np.zeros(n)
            for delta in range(-10, 11):
                value += bits[base+delta+256]*rrc(u-base-delta)
            signals[k, station] = value*np.exp(1j*phase)/np.sqrt(pulse_energy)/distance[k, station]
    target = BW/fs*10**(snr/10)*np.array([10**(gap/10), 1.])
    signals *= np.sqrt(target/np.mean(abs(signals)**2, axis=(1, 2)))[:, None, None]
    noise = (nrng.normal(size=(4, n))+1j*nrng.normal(size=(4, n)))/np.sqrt(2)
    meta = dict(role='r1_development', group=group, realization=realization, seed=seed,
                N=n, sample_rate_hz=fs, duration_seconds=n/fs, bandwidth_hz=BW,
                noise_variance=1., snr_definition='mean_station_power_over_expected_inband_noise',
                gap_db=float(gap), snr_inband_db=snr, center_hz=0.,
                positions=pp.tolist(), noise_sha=digest(noise), components_sha=digest(signals))
    return dict(signal=signals, noise=noise, positions=pp, meta=meta)


def task_list(repeats=FORMAL_REPEATS, gaps=(0, 3, 6, 9, 12, 15), realization_start=0):
    for group in range(4):
        for realization in range(realization_start, realization_start+repeats):
            yield group, realization, 0, 'weak'
            for gap in gaps:
                for kind in ('strong', 'mixed'):
                    yield group, realization, gap, kind
