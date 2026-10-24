"""Separate blind candidate coverage from local criterion bias on 1024 samples."""
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.research_discovery.window_baselines import weight, VARIANTS
from 统一模型代码.gates.g11.precision_diagnostic import sha, write, local
from 统一模型代码.gates.g10.r3.solvers import Model, Trace
from 统一模型代码.gates.g10.r4.solvers import ap
from 统一模型代码.gates.g10.r4.paper_checks.assignment_metrics import evaluate

SOURCE = ROOT/'outputs_e2e/research_discovery_20261022/experiments/A_window_baselines/round2_20261022_215420_921608'


def main():
    cfg = json.loads((SOURCE/'protocol.json').read_text(encoding='utf-8'))['config']
    selected = []
    for path in sorted((SOURCE/'pairs').glob('*.json')):
        row = json.loads(path.read_text(encoding='utf-8'))
        if row['kind'] == 'mixed' and row['n'] == 1024:
            selected.append((path, row))
    assert len(selected) == 24
    protected = {str(p): sha(p) for p in (ROOT/'统一模型代码/gates').rglob('*.py')}
    inputs = {str(p): sha(p) for p in SOURCE.rglob('*') if p.is_file()}
    out = SOURCE.parent/('round3_attribution_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    assert not out.exists() and out.resolve().is_relative_to(ROOT/'outputs_e2e')
    out.mkdir()
    (out/'pairs').mkdir()
    write(out/'protocol.json', dict(source=str(SOURCE), selected=[str(p) for p, _ in selected],
        compare='saved_AP1 versus blind_AP5 versus diagnostic_truth_local', variants=VARIANTS,
        oracle_used_only_in_named_diagnostic=True, configs=cfg, protected=protected, inputs=inputs))
    rows = []
    tick = time.perf_counter()
    for ix, (_, row) in enumerate(selected):
        _, group, rep = row['case'].split('_')
        with np.load(SOURCE/'inputs'/f'{group}_{rep}.npz', allow_pickle=False) as z:
            sig, noise = z['signal'][..., :1024], z['noise'][:, :1024]
        iq = sig[0]*10**(row['gap']/20)+sig[1]+noise
        truth = np.asarray(row['truth'])
        results = {}
        order = VARIANTS[ix % 3:] + VARIANTS[:ix % 3]
        for name in order:
            processed = iq*weight(name, 1024, 50e6)
            trace = Trace(lambda: None, history=False)
            model = Model(processed, trace.check, sample_rate_hz=50e6, half_band_hz=7e6)
            trace.model = model
            detail = ap(model, trace, 'AP_5_JOINT', copy.deepcopy(cfg))
            blind = dict(positions=detail['output_positions'], detail=detail,
                         metrics=evaluate(truth, detail['output_positions']),
                         objective=model.joint(np.asarray(detail['output_positions'])))
            diagnostic = local(model, truth)
            diagnostic['metrics'] = evaluate(truth, diagnostic['positions'])
            diagnostic['not_deployable'] = True
            results[name] = dict(ap1=row['results'][name], ap5=blind, truth_local=diagnostic)
        result = dict(case=row['case'], gap=row['gap'], truth=row['truth'], results=results)
        write(out/'pairs'/f'{ix:03d}.json', result)
        rows.append(result)
        if (ix+1) % 8 == 0:
            print(f'{ix+1}/24：完成盲搜索和真值起点诊断，{time.perf_counter()-tick:.1f}s', flush=True)
    summary = []
    for gap in (0, 15):
        part = [r for r in rows if r['gap'] == gap]
        for name in VARIANTS:
            for mode in ('ap1', 'ap5', 'truth_local'):
                e = np.asarray([r['results'][name][mode]['metrics']['errors_m'] for r in part])
                summary.append(dict(gap=gap, window=name, mode=mode, n=len(part),
                    both20=int(np.all(e <= 20, axis=1).sum()),
                    weak_rmse=float(np.sqrt(np.mean(e[:, 1]**2))),
                    median_weak=float(np.median(e[:, 1]))))
    write(out/'summary.json', summary)
    for path, identity in {**protected, **inputs}.items():
        assert sha(path) == identity, path
    write(out/'audit.json', dict(completed=True, cases=len(rows), blind_solves=72,
        oracle_diagnostics=72, inputs_unchanged=True, sources_unchanged=True,
        seconds=time.perf_counter()-tick))
    print(out, flush=True)


if __name__ == '__main__':
    main()
