"""Read-only frozen evidence adapters for the approved two-task audit."""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
PROJECT = BASE.parents[2]
sys.path.insert(0, str(BASE))
from structural_probe import B, C, STATIONS  # noqa: E402

UNIT = C / B
DEV = BASE / 'consensus_development_20261029_223235_334946'
VAL = BASE / 'consensus_validation_20261029_223520_495993'
INPUT_DEV = BASE / 'independent_backend_20261029_222111_026721'
CODE = ['structural_probe.py', 'solver_reliability.py', 'position_reduction.py',
        'blind_initialization.py', 'boundary_update.py', 'blind_backend_pilot.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    def default(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    default=default, allow_nan=False), encoding='utf-8')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def cases(short=False):
    for folder, source, layouts, reps in [(DEV, INPUT_DEV, range(20, 24), range(200, 203)),
                                         (VAL, VAL, range(40, 44), range(300, 303))]:
        for li in layouts:
            for rep in reps:
                if short and (li >= 40 or rep != 200):
                    continue
                tag = f'L{li}_R{rep}'
                yield tag, source / (tag + '_input.npz'), folder


def preflight(kind, short=False):
    paths = [HERE / '执行协议.md', *HERE.glob('*.py'), *[BASE / n for n in CODE]]
    for _, inp, folder in cases(short):
        paths += [inp, folder / 'protocol.json']
        tag = inp.stem.removesuffix('_input')
        paths += [folder / (tag + '_' + mode + '.json') for mode in ['LOS', 'MP']]
    hashes = {str(p.resolve(strict=True)): sha(p) for p in paths}
    assert all(Path(p).is_relative_to(PROJECT) for p in hashes)
    for folder in [DEV, VAL]:
        frozen = load(folder / 'protocol.json')['hashes']
        for name in CODE:
            key = str(BASE / name)
            if key in frozen:
                assert sha(key) == frozen[key], key
    out = HERE / (kind + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    (out / 'source').mkdir()
    for p in paths[:1 + len(list(HERE.glob('*.py'))) + len(CODE)]:
        (out / 'source' / p.name).write_bytes(p.read_bytes())
    write(out / 'protocol.json', dict(kind=kind, short=short, hashes=hashes,
          task3_authorized=False, threads=2, independent_new_data=False))
    print('结果目录', out, flush=True)
    return out, hashes


def verify(hashes):
    return {p: sha(p) == h for p, h in hashes.items()}


def spectra(data, noisy=False, mp=False):
    f = np.fft.fftfreq(8192, 1 / 50e6)
    keep = abs(f) <= B / 2
    iq = data['los'].sum(axis=0).copy()
    if mp:
        iq += data['reflected'].sum(axis=0)
    if noisy:
        iq += data['noise']
    y = np.fft.fft(iq, axis=1, norm='ortho')[:, keep].T
    e = np.fft.fft(data['reflected'], axis=2, norm='ortho')[:, :, keep].transpose(0, 2, 1)
    return f[keep], y, e


def source_indices(q):
    return np.r_[np.arange(2*q, 2*q+2), np.arange(4+q, 12, 2),
                 np.arange(12+q, 20, 2),
                 [20+4*m+2*q+k for m in range(5) for k in range(2)],
                 [40+4*m+2*q+k for m in range(5) for k in range(2)],
                 [60+4*m+2*q+k for m in range(5) for k in range(2)]].astype(int)


def regularity(x, f):
    from position_reduction import channel
    from structural_probe import response
    from blind_initialization import feasible
    p = x[:4].reshape(2, 2) * UNIT
    gain, delay = channel(x[4:])
    h, _ = response(p, delay, gain, f)
    ss = np.linalg.svd(h, compute_uv=False)
    rho = x[20:40].reshape(10, 2) + 1j*x[40:60].reshape(10, 2)
    return dict(feasible=feasible(p, x[4:]), min_los=float(abs(gain[:, :, 0]).min()),
                min_rank_ratio=float(np.min(ss[:, 1]/ss[:, 0])),
                min_station_distance=float(np.linalg.norm(p[None]-STATIONS[:, None], axis=2).min()),
                min_ball_slack=float(np.min(1-np.sum(abs(rho)**2, axis=1))),
                min_gap_margin=float(min(np.min(x[60:]-3), np.min(20-x[60:]))))
