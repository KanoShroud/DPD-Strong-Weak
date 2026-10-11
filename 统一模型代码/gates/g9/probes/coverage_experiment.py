"""G9-R2 numerical equivalence, dense coverage, conditional fresh paired check."""
import argparse
from functools import partial
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from .. import solver_v2
from ..metrics import aggregate, evaluate
from ..runner import last_positions
from ..runtime import BASE, Guard, identity, preflight, read, verify, write
from ..scenes import make
from ..solver import Score, local
from .coverage_search import BatchedScore, dense
from .failure_attribution import load_inputs, regenerate, full_score


def timed(call):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = call()
    torch.cuda.synchronize()
    return value, time.perf_counter()-start


def benchmark(records, contract, guard):
    output, chosen = [], {}
    rng = np.random.default_rng(2026101102)
    points = rng.uniform(-1000, 1000, (512, 2))
    for n in (4096, 16384):
        item = next(r for r in records if r['row']['N'] == n)
        iq, _, predicted, _ = regenerate(item, contract)
        reference = Score(iq, guard)
        reference.set_sources([predicted[0]])
        expected = reference.evaluate(points)
        for chunk in (16, 64, 128, 256):
            calc = BatchedScore(iq, guard, chunk)
            calc.set_sources([predicted[0]])
            actual = calc.evaluate(points)
            error = float(np.max(np.abs(actual-expected)))
            assert np.allclose(actual, expected, rtol=1e-10, atol=1e-12)
            times = [timed(lambda: calc.evaluate(points))[1] for _ in range(3)]
            output.append(dict(N=n, chunk=chunk, max_abs_error=error, seconds=float(np.median(times))))
        chosen[str(n)] = min([x for x in output if x['N'] == n], key=lambda x: x['seconds'])['chunk']
    return dict(rows=output, chunks=chosen)


def develop(out, records, contract, guard):
    bench = benchmark(records, contract, guard)
    write(out/'benchmark.json', bench)
    print('评分分块基准：'+str(bench), flush=True)
    selection = read(BASE/'r1_failure_20261011_01/contract.json')['selected']
    rows = []
    for item in records:
        if item['tag'] not in selection:
            continue
        iq, truth, predicted, original = regenerate(item, contract)
        calc = BatchedScore(iq, guard, bench['chunks'][str(item['row']['N'])])
        row = dict(tag=item['tag'], N=item['row']['N'], selection=selection[item['tag']],
                   baseline_metrics=original, results={})
        for step in (10, 5):
            result = dense(calc, predicted[0], predicted[1], step, guard)
            result['error_m'] = float(np.linalg.norm(np.asarray(result['position'])-truth[1]))
            row['results'][str(step)] = result
        rows.append(row)
        write(out/'rows'/f'{item["tag"]}.json', row)
        print(f'覆盖验证 {len(rows)}/24', flush=True)
    summary = []
    for kind in ('failure', 'success_control'):
        for step in ('10', '5'):
            sub = [r for r in rows if r['selection'] == kind]
            summary.append(dict(kind=kind, step_m=int(step), cases=len(sub),
                success=sum(r['results'][step]['error_m'] <= 100 for r in sub),
                seconds_mean=float(np.mean([r['results'][step]['seconds'] for r in sub]))))
    write(out/'report.json', dict(status='DEVELOPMENT_COMPLETE', benchmark=bench, summary=summary))


def repaired(iq, predicted, chunk, step, guard):
    calc = BatchedScore(iq, guard, chunk)
    positions = np.asarray(predicted).copy()
    events = []
    # One global sweep over both original slots; no truth-based reordering.
    for k in range(2):
        other = positions[1-k]
        before = full_score(calc, positions)
        result = dense(calc, other, positions[k], step, guard)
        proposal = positions.copy()
        proposal[k] = result['position']
        if full_score(calc, proposal) >= before:
            positions = proposal
        events.append(dict(slot=k, positions=positions.tolist(), score=full_score(calc, positions)))
    # Two shared local alternating cycles after global replacement.
    for _ in range(2):
        for k in range(2):
            others = positions[1-k:2-k]
            calc.set_sources(others)
            candidate = local(calc, positions[k], others)
            if calc.evaluate([candidate])[0] >= calc.evaluate([positions[k]])[0]:
                positions[k] = candidate
    return dict(positions=positions.tolist(), score=full_score(calc, positions), events=events)


def fresh(out, config, guard):
    rows = []
    for group in config['groups']:
        for overlap in (.5, 1.):
            for gap in (6, 12):
                scene = make('check', group, 0, overlap, 3, gap)
                for n in (4096, 16384):
                    guard()
                    iq = scene['signal'][:, :, :n].sum(0)+scene['noise'][:, :n]
                    chunk = config['chunks'][str(n)]
                    original_class = solver_v2.Score
                    solver_v2.Score = partial(BatchedScore, chunk=chunk)
                    try:
                        base = solver_v2.run(iq, 'B', 60., guard=guard)
                    finally:
                        solver_v2.Score = original_class
                    if base['status'] != 'FINISHED':
                        raise RuntimeError('小规模常规B未完成，停止而非扩大预算')
                    positions = last_positions(base, 60.)
                    result, elapsed = timed(lambda: repaired(iq, positions, chunk, config['step_m'], guard))
                    row = dict(group=group, overlap=overlap, gap=gap, N=n,
                        B=dict(metrics=evaluate(scene['positions'], positions), seconds=base['seconds'],
                               positions=positions),
                        D=dict(metrics=evaluate(scene['positions'], result['positions']),
                               seconds=base['seconds']+elapsed, positions=result['positions']),
                        update_events=result['events'])
                    rows.append(row)
                    write(out/'rows'/f'{group}_{overlap}_{gap}_{n}.json', row)
        print(f'新布局完成 {group-config["groups"][0]+1}/{len(config["groups"])}', flush=True)
    summary = {m: aggregate([r[m] for r in rows]) for m in ('B', 'D')}
    by_length = {str(n): {m: aggregate([r[m] for r in rows if r['N'] == n]) for m in ('B', 'D')}
                 for n in (4096, 16384)}
    write(out/'report.json', dict(status='FRESH_CHECK_COMPLETE', summary=summary, by_length=by_length,
                                groups=len(config['groups']), conditions=len(rows), test_access=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('develop', 'fresh'), required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--config')
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
    sources = report['contract']['sources']+[identity(Path(__file__)),
        identity(Path(__file__).with_name('coverage_search.py')),
        identity(Path(__file__).with_name('failure_attribution.py'))]
    inputs += [identity(BASE/'r1_failure_20261011_01/contract.json')]
    config = read(args.config) if args.config else None
    if args.config:
        inputs.append(identity(args.config))
    write(out/'contract.json', dict(stage=args.stage, sources=sources, inputs=inputs, config=config,
        total_prior_seconds=previous+extra, scope='small coverage/cost study, no training/test'))
    start = time.perf_counter()
    try:
        if args.stage == 'develop':
            develop(out, records, report['contract'], guard)
        else:
            if config is None:
                raise ValueError('新布局检查必须先冻结配置')
            fresh(out, config, guard)
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
