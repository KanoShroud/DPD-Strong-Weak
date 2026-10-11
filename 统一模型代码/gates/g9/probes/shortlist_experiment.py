"""G9-R2: retain 32 candidates at 10m before paying for a 5m full map."""
import argparse
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from ...g8.physics import peaks
from ..runtime import BASE, Guard, identity, preflight, read, verify, write
from ..solver import far, local
from . import coverage_experiment as engine
from .coverage_search import BatchedScore
from .failure_attribution import load_inputs, regenerate


def dense32(calc, other, initial, step, guard=lambda: None):
    torch.cuda.synchronize()
    started = time.perf_counter()
    calc.set_sources([other])
    axis = np.arange(-1000., 1000.01, step)
    x, y = np.meshgrid(axis, axis)
    points = np.stack([x.ravel(), y.ravel()], -1)
    points = points[far(points, [other])]
    scores = calc.evaluate(points)
    starts = points[peaks(points, scores, 32)]
    candidates = [np.asarray(initial)]
    for p in starts:
        guard()
        candidates.extend([p, local(calc, p, [other])])
    values = calc.evaluate(candidates)
    index = int(np.argmax(values))
    torch.cuda.synchronize()
    return dict(position=np.asarray(candidates[index]).tolist(), score=float(values[index]),
                grid_points=len(points), candidate_count=32, seconds=time.perf_counter()-started)


def develop(out, records, contract, config, guard):
    selected = read(BASE/'r1_failure_20261011_01/contract.json')['selected']
    rows = []
    for item in records:
        if item['tag'] not in selected:
            continue
        iq, truth, predicted, original = regenerate(item, contract)
        calc = BatchedScore(iq, guard, config['chunks'][str(item['row']['N'])])
        result = dense32(calc, predicted[0], predicted[1], 10, guard)
        result['error_m'] = float(np.linalg.norm(np.asarray(result['position'])-truth[1]))
        row = dict(tag=item['tag'], N=item['row']['N'], selection=selected[item['tag']],
                   baseline_metrics=original, result=result)
        rows.append(row)
        write(out/'rows'/f'{item["tag"]}.json', row)
        if len(rows) % 4 == 0:
            print(f'Top32开发核验 {len(rows)}/24', flush=True)
    summary = []
    for kind in ('failure', 'success_control'):
        sub = [r for r in rows if r['selection'] == kind]
        summary.append(dict(kind=kind, cases=len(sub),
            success=sum(r['result']['error_m'] <= 100 for r in sub),
            seconds_mean=float(np.mean([r['result']['seconds'] for r in sub]))))
    write(out/'report.json', dict(status='DEVELOPMENT_COMPLETE', summary=summary))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('develop', 'fresh'), required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.zeros(1, device='cuda').sum().item()
    report, records, inputs = load_inputs()
    previous = read(BASE/'r1_failure_20261011_01/ledger.json')['cumulative_seconds']
    extra = sum(read(p)['seconds_this_run'] for p in BASE.glob('r2_coverage_*/ledger.json'))
    stage_guard, total_guard = Guard(45*60, extra), Guard(8*3600, previous+extra)
    def guard():
        stage_guard()
        total_guard()
    out = preflight(BASE/args.run_id)
    config = read(args.config)
    sources = report['contract']['sources']+[identity(Path(__file__).with_name(name)) for name in
        ('shortlist_experiment.py', 'coverage_experiment.py', 'coverage_search.py', 'failure_attribution.py')]
    inputs += [identity(args.config), identity(BASE/'r1_failure_20261011_01/contract.json')]
    write(out/'contract.json', dict(stage=args.stage, sources=sources, inputs=inputs, config=config,
        scope='10m/Top32 fixed before fresh; all methods share batched score'))
    start = time.perf_counter()
    try:
        if args.stage == 'develop':
            develop(out, records, report['contract'], config, guard)
        else:
            if config['step_m'] != 10 or config['topk'] != 32:
                raise ValueError('新检查仅接受冻结10m/Top32配置')
            old = engine.dense
            engine.dense = dense32
            try:
                engine.fresh(out, config, guard)
            finally:
                engine.dense = old
        verify(sources)
        verify(inputs)
        write(out/'final_audit_report.json', dict(status='PASS', sources=sources, inputs=inputs,
            report=identity(out/'report.json'), test_access=False))
    except Exception:
        write(out/'failure.json', dict(status='FAILED', traceback=traceback.format_exc()))
        raise
    finally:
        elapsed = time.perf_counter()-start
        write(out/'ledger.json', dict(seconds_this_run=elapsed, r2_cumulative_seconds=extra+elapsed,
                                    cumulative_seconds=previous+extra+elapsed))
    print('完成：'+str(out/'report.json'), flush=True)


if __name__ == '__main__':
    main()
