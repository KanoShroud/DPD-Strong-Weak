"""Same-frequency-information control, frozen holdout replay, not fresh evidence."""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.research_discovery.finite_fft_blind import solve
from 统一模型代码.gates.research_discovery.finite_residual_candidates import candidates
from 统一模型代码.gates.research_discovery.residual_competition import HOLDOUT
from 统一模型代码.gates.g11.precision_diagnostic import sha, write
from 统一模型代码.gates.g10.r4.paper_checks.assignment_metrics import evaluate


def main():
    out = HOLDOUT.parent/('round16_band_pair_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    assert not out.exists() and out.resolve().is_relative_to(ROOT/'outputs_e2e')
    inputs = {str(p): sha(p) for p in HOLDOUT.rglob('*') if p.is_file()}
    protected = {str(p): sha(p) for p in (ROOT/'统一模型代码/gates').rglob('*.py')}
    out.mkdir()
    write(out/'protocol.json', dict(source=str(HOLDOUT), cases=16, same_frequencies=True,
        half_band_hz=7e6, n=1024, fs=50e6, fft_scale='unitary, original N retained',
        damp=1e-3, no_retuning=True, not_new_independent_data=True,
        protected=protected, inputs=inputs))
    oldrows = json.loads((HOLDOUT/'results.json').read_text(encoding='utf-8'))
    records = []
    checks = {}
    with threadpool_limits(limits=2):
        for index, row in enumerate(oldrows):
            group, rep, gap = row['group'], row['realization'], row['gap']
            with np.load(HOLDOUT/'inputs'/f'{group}_{rep}.npz', allow_pickle=False) as z:
                iq = z['signal'][0, :, :1024]*10**(gap/20)+z['signal'][1, :, :1024]+z['noise'][:, :1024]
                truth = z['positions']
            first = row['results']['rect']['detail']['first_position']
            if index == 0:
                rng = np.random.default_rng(20261023016)
                perturbation = rng.normal(size=iq.shape)+1j*rng.normal(size=iq.shape)
                f = np.fft.fftfreq(1024, 1/50e6)
                spectrum = np.fft.fft(perturbation, axis=-1)
                spectrum[:, abs(f) <= 7e6] = 0
                changed = iq+10*np.fft.ifft(spectrum, axis=-1)
                kept = abs(f) <= 7e6
                np.testing.assert_allclose(np.fft.fft(iq, axis=-1)[:, kept],
                    np.fft.fft(changed, axis=-1)[:, kept], atol=1e-10)
                a, b = candidates(iq, first, True), candidates(changed, first, True)
                np.testing.assert_allclose(a['positions'], b['positions'], atol=0)
                np.testing.assert_allclose(a['scores'], b['scores'], atol=1e-10)
                checks['outside_band_perturbation_does_not_change_candidates'] = True
                write(out/'checks.json', checks)
            detail = solve(iq, first, damp=1e-3, bandlimited=True)
            record = dict(group=group, realization=rep, gap=gap, detail=detail,
                metrics=evaluate(truth, detail['positions']),
                fulltime_metrics=row['results']['finite_ridge']['metrics'])
            records.append(record)
            write(out/'results.json', records)
            print(f'{len(records)}/16 相同频带配对完成', flush=True)
    summary = []
    for gap in (0, 15):
        part = [r for r in records if r['gap'] == gap]
        e = np.asarray([r['metrics']['errors_m'] for r in part])
        summary.append(dict(gap=gap, n=len(part), both20=int(np.all(e <= 20, axis=1).sum()),
            weak_rmse=float(np.sqrt(np.mean(e[:, 1]**2))),
            fulltime_both20=sum(r['fulltime_metrics']['both20'] for r in part),
            mean_seconds=float(np.mean([r['detail']['seconds'] for r in part]))))
    write(out/'summary.json', summary)
    for path, identity in {**protected, **inputs}.items():
        assert sha(path) == identity, path
    write(out/'audit.json', dict(completed=True, n=len(records), inputs_unchanged=True,
        sources_unchanged=True, not_new_independent_data=True, prefix_cost_excluded=True))
    print(out, flush=True)


if __name__ == '__main__':
    main()
