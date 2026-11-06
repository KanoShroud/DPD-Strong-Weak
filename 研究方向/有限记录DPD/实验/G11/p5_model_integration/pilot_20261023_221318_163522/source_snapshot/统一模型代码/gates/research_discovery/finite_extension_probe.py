"""Finite Fourier-extension LS target audit, not a localization algorithm."""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.linalg import lstsq
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.g11.precision_diagnostic import sha, write
from 统一模型代码.gates.g8.physics import C, RECEIVERS
from 统一模型代码.gates.research_discovery.short_record_attribution import SOURCE

ATTR = SOURCE.parent/'round3_attribution_20261022_215937_701453'


def dictionary(positions, n=1024, fs=50e6):
    length = n+2*int(np.ceil(fs*1000/C))
    bins = np.arange(-int(np.floor(7e6*length/fs)), int(np.floor(7e6*length/fs))+1)
    frequencies = bins*fs/length
    blocks = []
    for pos in positions:
        distance = np.linalg.norm(np.asarray(RECEIVERS)-pos, axis=1)
        # A common temporal shift is absorbed by the freely estimated coefficients.
        delays = (distance-distance.min())/C
        times = np.arange(n)[None, :]/fs-delays[:, None]
        block = np.exp(2j*np.pi*times.reshape(-1, 1)*frequencies[None, :])
        block *= np.repeat(1/distance, n)[:, None]
        block /= np.linalg.norm(block, axis=0)[None, :]
        blocks.append(block)
    return np.concatenate(blocks, axis=1)


def main():
    out = SOURCE.parent/('round5_extension_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    assert not out.exists() and out.resolve().is_relative_to(ROOT/'outputs_e2e')
    protected = {str(p): sha(p) for p in (ROOT/'统一模型代码/gates').rglob('*.py')}
    inputs = {str(p): sha(p) for folder in (SOURCE, ATTR) for p in folder.rglob('*') if p.is_file()}
    out.mkdir()
    write(out/'protocol.json', dict(input_roots=[str(SOURCE), str(ATTR)], diagnostic_only=True,
        n=1024, fs=50e6, half_band_hz=7e6, svd_rcond=1e-8, realization=22000,
        protected=protected, inputs=inputs))
    records = []
    tick = time.perf_counter()
    with threadpool_limits(limits=2):
        for path in sorted((ATTR/'pairs').glob('*.json')):
            row = json.loads(path.read_text(encoding='utf-8'))
            if row['gap'] != 15 or not row['case'].endswith('_22000'):
                continue
            _, group, rep = row['case'].split('_')
            with np.load(SOURCE/'inputs'/f'{group}_{rep}.npz', allow_pickle=False) as z:
                clean = z['signal'][0, :, :1024]*10**(15/20)+z['signal'][1, :, :1024]
                noisy = clean+z['noise'][:, :1024]
            y = np.column_stack([clean.ravel(), noisy.ravel()])
            results = {}
            for label, positions in [('truth', row['truth']),
                                     ('blind_ap5', row['results']['rect']['ap5']['positions'])]:
                start = time.perf_counter()
                a = dictionary(np.asarray(positions))
                coeff, _, rank, singular = lstsq(a, y, cond=1e-8, lapack_driver='gelsd')
                residual = y-a@coeff
                fractions = np.sum(abs(residual)**2, axis=0)/np.sum(abs(y)**2, axis=0)
                assert np.isfinite(fractions).all() and np.all(fractions >= 0)
                results[label] = dict(positions=positions, residual_energy_fraction=fractions.tolist(),
                    rank=int(rank), shape=list(a.shape), seconds=time.perf_counter()-start,
                    smallest_retained_relative=float(singular[rank-1]/singular[0]))
            item = dict(case=row['case'], results=results, right_hand_sides=['clean', 'noisy'])
            records.append(item)
            write(out/'results.json', records)
            print(row['case'], {k: v['residual_energy_fraction'] for k, v in results.items()}, flush=True)
    assert len(records) == 4
    for path, identity in {**protected, **inputs}.items():
        assert sha(path) == identity, path
    write(out/'audit.json', dict(completed=True, cases=len(records), factorizations=8,
        seconds=time.perf_counter()-tick, sources_unchanged=True, inputs_unchanged=True,
        diagnostic_only=True))
    print(out, flush=True)


if __name__ == '__main__':
    main()
