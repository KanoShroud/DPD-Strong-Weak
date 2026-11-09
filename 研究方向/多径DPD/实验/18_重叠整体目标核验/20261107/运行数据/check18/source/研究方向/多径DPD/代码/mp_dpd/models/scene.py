# ruff: noqa: F401
# Preserve imported globals for legacy function cloning and reexports.
"""Frozen physical RF generator. Truth is never an estimator argument."""
from itertools import combinations
from pathlib import Path
import hashlib
import json
import os
import time
import uuid
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = next(p for p in HERE.parents if (p/'AGENTS.md').is_file())
C = 299792458.
FC = 2.4e9
FS = 50e6
B = 12.5e6
N = 8192
RS = 10e6
BOUND = 4500.
STATIONS = np.array([[3000., 3000.], [-3000., 3000.], [-3000., -3000.], [3000., -3000.]])
NOISE_PSD = 1.380649e-23 * 290 * 10**(.5)
NOISE_VARIANCE = NOISE_PSD * FS
CONDITIONS = [dict(name='L0', L=0, rho=0., ids=[])] + [
    dict(name=f'L{l}_P{round(r*100):03d}', L=l, rho=r, ids={1: [0], 2: [0, 3], 3: [0, 1, 3]}[l])
    for l in (1, 2, 3) for r in ([.5, .75] if l == 1 else [.5, .75, 1., 1.25, 1.5])]


def clean(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.pending')
    tmp.write_text(json.dumps(value, ensure_ascii=False, default=clean, allow_nan=False), encoding='utf-8')
    for attempt in range(8):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.05 * (attempt + 1))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def geometry_ok(p):
    p = np.asarray(p)
    if np.min(np.linalg.norm(p[:, None] - STATIONS[None], axis=-1)) < 500:
        return False
    if np.linalg.norm(p[0]-p[1]) < 500:
        return False
    for q in p:
        if max(abs(q)) <= 3000:
            continue
        for i, j in combinations(range(4), 2):
            a, v = STATIONS[i], STATIONS[j]-STATIONS[i]
            t = (q-a)@v/(v@v)
            distance = abs(v[0]*(q-a)[1]-v[1]*(q-a)[0])/np.linalg.norm(v)
            if (t < 0 or t > 1) and distance < 300:
                return False
    return True


def layouts(phase):
    rng = np.random.default_rng(np.random.SeedSequence([2026110424, 0 if phase == 'pilot' else 1]))
    kinds = ['IN', 'OUT', 'MIX', 'MIX'] if phase == 'pilot' else ['IN']*4+['OUT']*4+['MIX']*4
    out = []
    for index, kind in enumerate(kinds):
        for attempts in range(100000):
            flags = [kind != 'OUT', kind == 'IN']
            points = []
            for inside in flags:
                while True:
                    z = rng.uniform(-2400 if inside else -4200, 2400 if inside else 4200, 2)
                    if inside or max(abs(z)) >= 3300:
                        break
                points.append(z)
            p = np.array(points)
            if geometry_ok(p):
                # Labels have no information meaning; randomize without duplicating a layout.
                p = p[rng.permutation(2)]
                distance = np.linalg.norm(STATIONS[:, None]-p[None], axis=-1)
                gain_power = (C/(4*np.pi*FC*distance))**2
                pt = 10**.6 * NOISE_PSD * B / gain_power.mean(axis=0)
                out.append(dict(index=index, kind=kind, p=p, pt_w=pt, pt_dbm=10*np.log10(pt*1000), attempts=attempts+1))
                break
        else:
            raise RuntimeError('Geometry generation exhausted')
    return out


def contract():
    return dict(version='FOUR_RF_24GHZ_V1', stations=STATIONS, fc=FC, fs=FS, bandwidth=B, n=N,
                bound=BOUND, temperature_k=290., noise_figure_db=5., noise_psd=NOISE_PSD,
                noise_variance=NOISE_VARIANCE, nominal_snr_db=[6., 6.], antenna_gain_dbi=[0., 0.],
                conditions=CONDITIONS, pilot_layouts=layouts('pilot'), formal_layouts=layouts('formal'),
                formal_repeats=20, pilot_repeats=1, capacity=64, pg=1e-7, threads=2,
                geometry_rules=dict(inner_max=2400, outer_min=3300, outer_max=4200,
                                    min_source_station=500, min_source_source=500, extended_line_clearance=300),
                source_seed=2026110425, algorithm_seed=2026110426)


def make_bank(layout, phase, repeat, synthesize, rrc):
    seed = [2026110425, 0 if phase == 'pilot' else 1, layout['index'], repeat]
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    p = np.asarray(layout['p']); pt = np.asarray(layout['pt_w'])
    # Public delay/geometry protection, not oracle record-dependent padding.
    first = -int(np.ceil((np.sqrt(2)*(BOUND+3000)/C+40/B)*RS))-68
    last = int(np.ceil(N/FS*RS))+68
    bits = rng.choice([-1., 1.], (2, last-first+1))
    offsets = rng.uniform(0, 1/RS, 2)
    distance = np.linalg.norm(STATIONS[:, None]-p[None], axis=-1)
    gains = np.sqrt(pt)[None]*C/(4*np.pi*FC*distance)*np.exp(1j*rng.uniform(-np.pi, np.pi, (4, 2)))
    delays = np.array([3., 7., 11., 15.])[None, None]+rng.uniform(0, .5, (4, 2, 4))
    phases = np.exp(1j*rng.uniform(-np.pi, np.pi, (4, 2, 4)))
    weights = rng.uniform(.8, 1.2, (4, 2, 4))
    # Deterministic ensemble-average normalization of the existing finite RRC sum.
    phase = (np.arange(4096)+.5)/4096
    norm2 = float(np.mean(np.sum(rrc(phase[:, None]-np.arange(-31, 33)[None])**2, axis=1)))
    t = np.arange(N)/FS
    los = np.empty((2, 4, N), complex)
    reflected = np.empty((2, 4, 4, N), complex)
    for q in range(2):
        for m in range(4):
            tau = distance[m, q]/C
            los[q, m] = gains[m, q]*synthesize(bits[q], first, offsets[q], t-tau)/np.sqrt(norm2)
            for ell in range(4):
                reflected[q, m, ell] = gains[m, q]*phases[m, q, ell]*synthesize(
                    bits[q], first, offsets[q], t-tau-delays[m, q, ell]/B)/np.sqrt(norm2)
    noise = np.sqrt(NOISE_VARIANCE/2)*(rng.normal(size=(4, N))+1j*rng.normal(size=(4, N)))
    freq = np.fft.fftfreq(N, 1/FS); mask = abs(freq) <= B/2
    z = np.fft.fft(los, axis=-1, norm='ortho')
    band_power = np.sum(abs(z[..., mask])**2, axis=-1)/N
    snr = 10*np.log10(band_power/(NOISE_VARIANCE*mask.mean()))
    return dict(los=los, reflected=reflected, noise=noise, bits=bits, first=first, offsets=offsets,
                delays=delays, phases=phases, weights=weights, gains=gains, p=p, pt_w=pt,
                noise_variance=NOISE_VARIANCE, realized_los_snr_db=snr, pulse_norm2=norm2,
                seed=np.asarray(seed))


def observation(bank, condition):
    ids = condition['ids']
    powers = np.zeros((4, 2, 4))
    signal = bank['los'].sum(axis=0).copy()
    if ids:
        w = bank['weights'][:, :, ids]
        values = condition['rho']*w/w.sum(axis=2, keepdims=True)
        powers[:, :, ids] = values
        assert np.all(values < 1) and np.allclose(values.sum(axis=2), condition['rho'])
        signal += np.sum(bank['reflected']*np.sqrt(powers.transpose(1, 0, 2))[..., None], axis=(0, 2))
    return signal+bank['noise'], powers
