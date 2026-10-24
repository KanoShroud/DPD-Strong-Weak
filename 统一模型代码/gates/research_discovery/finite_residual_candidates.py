"""Conventional finite-record SIC candidate discovery pilot, not final estimator.

The cheap residual spectrum is a screening surrogate, not the projected finite
record joint LS. Truth is consumed only after candidate generation for coverage.
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.linalg import svd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.research_discovery.finite_extension_probe import SOURCE, ATTR, dictionary
from 统一模型代码.gates.g11.precision_diagnostic import sha, write
from 统一模型代码.gates.g10.r3.solvers import Model, Trace
from 统一模型代码.gates.g10.physics import grid
from 统一模型代码.gates.g10.methods import peaks


def candidates(iq, first, bandlimited=False):
    a = dictionary(np.asarray(first).reshape(1, 2))
    n = iq.shape[-1]
    keep = np.flatnonzero(abs(np.fft.fftfreq(n, 1/50e6)) <= 7e6)
    y = iq.ravel()
    if bandlimited:
        a = (np.fft.fft(a.reshape(4, n, -1), axis=1)[:, keep, :]/np.sqrt(n)).reshape(4*len(keep), -1)
        y = (np.fft.fft(iq, axis=-1)[:, keep]/np.sqrt(n)).ravel()
    assert np.isfinite(a).all()
    driver = 'gesdd'
    try:
        u, s, vh = svd(a, full_matrices=False, lapack_driver=driver)
    except np.linalg.LinAlgError:
        driver = 'gesvd'
        u, s, vh = svd(a, full_matrices=False, lapack_driver=driver)
    reconstruction_error = float(np.linalg.norm(a-(u*s)@vh)/np.linalg.norm(a))
    assert reconstruction_error < 1e-10
    rank = int((s > s[0]*1e-6).sum())
    q = u[:, :rank]
    residual = y-q@(q.conj().T@y)
    if bandlimited:
        padded = np.zeros((4, n), dtype=complex)
        padded[:, keep] = residual.reshape(4, len(keep))
        residual = np.fft.ifft(padded, axis=-1)*np.sqrt(n)
    else:
        residual = residual.reshape(iq.shape)
    trace = Trace(lambda: None, history=False)
    model = Model(residual, trace.check, sample_rate_hz=50e6, half_band_hz=7e6)
    points = grid(20)
    scores = model.conditional(points)
    ids = peaks(points, scores, 5)
    return dict(positions=points[ids].tolist(), scores=scores[ids].tolist(), rank=rank,
        svd_driver=driver, reconstruction_error=reconstruction_error, bandlimited=bandlimited,
        residual_energy_fraction=float(np.vdot(residual, residual).real/np.vdot(iq, iq).real))


def main():
    out = SOURCE.parent/('round9_sic_candidates_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    assert not out.exists() and out.resolve().is_relative_to(ROOT/'outputs_e2e')
    protected = {str(p): sha(p) for p in (ROOT/'统一模型代码/gates').rglob('*.py')}
    inputs = {str(p): sha(p) for folder in (SOURCE, ATTR) for p in folder.rglob('*') if p.is_file()}
    out.mkdir()
    write(out/'protocol.json', dict(cases='all12 short15dB observations', first_source='saved blind AP initial first source',
        controls='same five candidates,20m grid, fixed1e-6 finite waveform SVD',
        caveat='residual spectrum is only surrogate screening, not a joint LS objective',
        source=str(SOURCE), protected=protected, inputs=inputs))
    records = []
    tick = time.perf_counter()
    with threadpool_limits(limits=2):
        for path in sorted((ATTR/'pairs').glob('*.json')):
            row = json.loads(path.read_text(encoding='utf-8'))
            if row['gap'] != 15:
                continue
            _, group, rep = row['case'].split('_')
            with np.load(SOURCE/'inputs'/f'{group}_{rep}.npz', allow_pickle=False) as z:
                iq = z['signal'][0, :, :1024]*10**(15/20)+z['signal'][1, :, :1024]+z['noise'][:, :1024]
            detail = row['results']['rect']['ap1']['detail']
            first = np.asarray(detail['first_position'])
            start = time.perf_counter()
            new = candidates(iq, first)
            elapsed = time.perf_counter()-start
            # Truth enters only here; names strong/weak do not select candidates.
            truth = np.asarray(row['truth'])
            old = np.asarray(detail['candidate_list'])[:, 1]
            r = dict(case=row['case'], first=first.tolist(), new=new, old=old.tolist(),
                first_errors_to_true_sources=np.linalg.norm(truth-first.reshape(2), axis=1).tolist(),
                old_min_weak_distance=float(np.linalg.norm(old-truth[1], axis=1).min()),
                new_min_weak_distance=float(np.linalg.norm(np.asarray(new['positions'])-truth[1], axis=1).min()),
                seconds=elapsed, coverage_only=True)
            records.append(r)
            write(out/'results.json', records)
            print(row['case'], round(r['old_min_weak_distance'], 2), round(r['new_min_weak_distance'], 2), flush=True)
    assert len(records) == 12
    for path, identity in {**protected, **inputs}.items():
        assert sha(path) == identity, path
    write(out/'audit.json', dict(completed=True, n=12, seconds=time.perf_counter()-tick,
        old_within20=sum(r['old_min_weak_distance'] <= 20 for r in records),
        new_within20=sum(r['new_min_weak_distance'] <= 20 for r in records),
        sources_unchanged=True, inputs_unchanged=True, coverage_only=True))
    print(out, flush=True)


if __name__ == '__main__':
    main()
