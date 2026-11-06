"""Oracle attribution only: remove finite-record strong-source model residual.

Not a proposed algorithm: true source components/positions are used to construct
controls. All 12 existing 15 dB development records are retained, three windows.
"""
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.research_discovery.short_record_attribution import SOURCE
from 统一模型代码.gates.research_discovery.window_baselines import weight, VARIANTS
from 统一模型代码.gates.g11.precision_diagnostic import sha, write, components, set_x
from 统一模型代码.gates.g10.r3.solvers import Model, Trace
from 统一模型代码.gates.g10.r4.solvers import ap
from 统一模型代码.gates.g10.r4.paper_checks.assignment_metrics import evaluate


def main():
    cfg = json.loads((SOURCE/'protocol.json').read_text(encoding='utf-8'))['config']
    selected = []
    for path in sorted((SOURCE/'pairs').glob('*.json')):
        r = json.loads(path.read_text(encoding='utf-8'))
        if r['kind'] == 'mixed' and r['n'] == 1024 and r['gap'] == 15:
            selected.append(r)
    assert len(selected) == 12
    protected = {str(p): sha(p) for p in (ROOT/'统一模型代码/gates').rglob('*.py')}
    inputs = {str(p): sha(p) for p in SOURCE.rglob('*') if p.is_file()}
    out = SOURCE.parent/('round4_closure_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    assert not out.exists() and out.resolve().is_relative_to(ROOT/'outputs_e2e')
    out.mkdir()
    (out/'pairs').mkdir()
    write(out/'protocol.json', dict(source=str(SOURCE), cases=[r['case'] for r in selected],
        modes=['remove_strong_model_residual', 'remove_both_model_residuals'],
        diagnostic_only=True, no_noise_or_power_rescaling=True, config=cfg,
        protected=protected, inputs=inputs))
    rows = []
    tick = time.perf_counter()
    for ix, row in enumerate(selected):
        _, group, rep = row['case'].split('_')
        with np.load(SOURCE/'inputs'/f'{group}_{rep}.npz', allow_pickle=False) as z:
            sig = z['signal'][..., :1024].copy()
            noise = z['noise'][:, :1024]
        sig[0] *= 10**(15/20)
        iq = sig.sum(0)+noise
        truth = np.asarray(row['truth'])
        for name in VARIANTS:
            win = weight(name, 1024, 50e6)
            for mode in ('remove_strong_model_residual', 'remove_both_model_residuals'):
                trace = Trace(lambda: None, history=False)
                model = Model(iq*win, trace.check, sample_rate_hz=50e6, half_band_hz=7e6)
                trace.model = model
                observed, w, _, _, exact, residual = components(model, sig, noise, truth, win)
                np.testing.assert_allclose(observed.sum(0)+w, model.x.cpu().numpy(), atol=1e-12)
                data = exact[0]+observed[1]+w if mode == 'remove_strong_model_residual' else exact.sum(0)+w
                set_x(model, data)
                detail = ap(model, trace, 'AP_5_JOINT', copy.deepcopy(cfg))
                positions = detail['output_positions']
                r = dict(case=row['case'], window=name, mode=mode, positions=positions,
                    metrics=evaluate(truth, positions), detail=detail,
                    residual_energy_fraction=(np.sum(abs(residual)**2, axis=(1, 2))/np.sum(abs(observed)**2, axis=(1, 2))).tolist(),
                    objective=model.joint(np.asarray(positions)), diagnostic_only=True)
                rows.append(r)
                write(out/'pairs'/f'{len(rows):03d}.json', r)
        print(f'{ix+1}/12 完成有限窗残差归因', flush=True) if (ix+1) % 4 == 0 else None
    summary = []
    for name in VARIANTS:
        for mode in ('remove_strong_model_residual', 'remove_both_model_residuals'):
            part = [r for r in rows if r['window'] == name and r['mode'] == mode]
            errors = np.asarray([r['metrics']['errors_m'] for r in part])
            summary.append(dict(window=name, mode=mode, n=len(part),
                both20=int(np.all(errors <= 20, axis=1).sum()),
                weak_rmse=float(np.sqrt(np.mean(errors[:, 1]**2)))))
    write(out/'summary.json', summary)
    for path, identity in {**protected, **inputs}.items():
        assert sha(path) == identity, path
    write(out/'audit.json', dict(completed=True, solves=len(rows), diagnostic_only=True,
        inputs_unchanged=True, sources_unchanged=True, seconds=time.perf_counter()-tick))
    print(out, flush=True)


if __name__ == '__main__':
    main()
