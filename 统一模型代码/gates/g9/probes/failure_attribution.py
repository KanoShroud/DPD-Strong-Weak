"""G9-R1: frozen B failures, same IQ, fixed-source score/search attribution."""
import argparse
import hashlib
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from ...g8.physics import peaks
from ..metrics import evaluate
from ..runner import key_hash, last_positions
from ..runtime import BASE, Guard, identity, preflight, read, verify, write
from ..scenes import make
from ..solver import Score, far, local

META = ('group', 'realization', 'overlap', 'gap', 'N', 'kind')
TOL = 1e-10


def load_inputs():
    root = BASE/'formal_20261011_repaired'
    report = read(root/'comparison_report.json')
    audit = read(root/'final_audit_report.json')
    verify([audit['report']])
    verify(report['contract']['sources'])
    verify([report['contract']['dev_contract']])
    inputs = [identity(root/name) for name in (
        'comparison_report.json', 'final_audit_report.json', 'ledger.json')]
    records = []
    for path in sorted((root/'rows').glob('*_mixed_B.json')):
        row = read(path)
        seal = row.pop('row_sha256')
        assert key_hash(row) == seal, path
        records.append(dict(tag=path.stem, row=row))
        inputs.append(identity(path))
    assert len(records) == 512
    records.sort(key=lambda v: tuple(v['row'][k] for k in META))
    return report, records, inputs


def regenerate(item, contract):
    row = item['row']
    meta = {k: row[k] for k in META}
    scene = make('check', row['group'], row['realization'], row['overlap'],
                 contract['selected_snr_db'], row['gap'])
    iq = scene['signal'][:, :, :row['N']].sum(0)+scene['noise'][:, :row['N']]
    request = dict(tag=item['tag'], method='B', budget=row['budget_seconds'],
        metadata=meta, hr=list(contract['hr'][str(row['N'])]), solver='v2',
        iq_sha256=hashlib.sha256(np.ascontiguousarray(iq).tobytes()).hexdigest())
    assert key_hash(request) == row['request_hash'], item['tag']
    predicted = np.asarray(last_positions(row, contract['T0_seconds'][str(row['N'])]))
    truth = scene['positions']
    ids, cols = linear_sum_assignment(np.linalg.norm(truth[:, None]-predicted[None], axis=-1)**2)
    assert list(ids) == [0, 1]
    predicted = predicted[cols]
    return iq, truth, predicted, evaluate(truth, predicted)


def select(records, contract):
    chosen = {}
    for n in (4096, 16384):
        for overlap in (.5, 1.):
            for gap in (6, 12):
                pool = [v for v in records if (v['row']['N'], v['row']['overlap'], v['row']['gap'])
                        == (n, overlap, gap)]
                failures, successes, groups = [], [], set()
                for v in pool:
                    # Reconstruct labels only for selection; no new scoring is used.
                    _, _, _, metric = regenerate(v, contract)
                    flags = metric['success']['100']
                    if not flags[1] and v['row']['group'] not in groups and len(failures) < 2:
                        failures.append(v['tag'])
                        groups.add(v['row']['group'])
                    if all(flags) and not successes:
                        successes.append(v['tag'])
                    if len(failures) == 2 and successes:
                        break
                assert len(failures) == 2 and len(successes) == 1
                chosen.update({tag: 'failure' for tag in failures})
                chosen.update({tag: 'success_control' for tag in successes})
    assert len(chosen) == 24
    return chosen


def full_score(calc, pair):
    calc.set_sources(pair)
    return 1-float(calc.residual.abs().square().sum().cpu())/calc.energy


def fixed_probe(calc, strong, weak_predicted, weak_truth):
    calc.set_sources([strong])
    old, exact = calc.evaluate([weak_predicted, weak_truth])
    oracle = local(calc, weak_truth, [strong])
    oracle_score = float(calc.evaluate([oracle])[0])
    if oracle_score < exact:
        oracle, oracle_score = weak_truth.copy(), float(exact)
    return dict(returned_score=float(old), truth_score=float(exact),
        truth_higher=bool(exact > old+TOL), oracle_position=oracle.tolist(),
        oracle_score=oracle_score, oracle_error_m=float(np.linalg.norm(oracle-weak_truth)),
        oracle_higher=bool(oracle_score > old+TOL))


def dense_probe(calc, strong, weak_predicted, weak_truth, guard):
    calc.set_sources([strong])
    axis = np.arange(-1000., 1000.1, 10.)
    x, y = np.meshgrid(axis, axis)
    points = np.stack([x.ravel(), y.ravel()], -1)
    points = points[far(points, [strong])]
    values = calc.evaluate(points)
    starts = points[peaks(points, values, 5)]
    candidates = [weak_predicted.copy()]
    for seed in starts:
        guard()
        refined = local(calc, seed, [strong])
        candidates.extend([seed, refined])
    scores = calc.evaluate(candidates)
    best = np.asarray(candidates[int(np.argmax(scores))])
    nearest = int(np.argmin(np.linalg.norm(points-weak_truth, axis=-1)))
    oracle = fixed_probe(calc, strong, weak_predicted, weak_truth)
    best_score = float(np.max(scores))
    best_error = float(np.linalg.norm(best-weak_truth))
    return dict(position=best.tolist(), error_m=best_error, score=best_score,
        grid_points=len(points), nearest_truth_grid_score=float(values[nearest]),
        truth_grid_distance_m=float(np.linalg.norm(points[nearest]-weak_truth)),
        top5_initial_positions=starts.tolist(), oracle=oracle,
        oracle_beats_dense=bool(oracle['oracle_score'] > best_score+TOL),
        wrong_beats_truth_basin=bool(best_error > 100 and oracle['oracle_error_m'] <= 100
                                   and best_score > oracle['oracle_score']+TOL))


def joint_oracle(calc, truth):
    positions = truth.copy()
    for _ in range(2):
        for k in range(2):
            others = np.delete(positions, k, axis=0)
            calc.set_sources(others)
            refined = local(calc, positions[k], others)
            if calc.evaluate([refined])[0] >= calc.evaluate([positions[k]])[0]:
                positions[k] = refined
    return dict(positions=positions.tolist(), score=full_score(calc, positions),
                metrics=evaluate(truth, positions))


def summarize(rows, detailed):
    summary = []
    for n in (4096, 16384):
        subset = [r for r in rows if r['N'] == n]
        bad = [r for r in subset if not r['baseline_metrics']['success']['100'][1]]
        summary.append(dict(N=n, cases=len(subset), weak_failures=len(bad),
            failed_strong_sse_within100=sum(r['strong_error_m'] <= 100 for r in bad),
            truth_pair_higher=sum(r['truth_pair_score'] > r['returned_pair_score']+TOL for r in bad),
            estimated_truth_higher=sum(r['estimated']['truth_higher'] for r in bad),
            estimated_good_oracle_higher=sum(r['estimated']['oracle_higher'] and
                r['estimated']['oracle_error_m'] <= 100 for r in bad),
            true_strong_truth_higher=sum(r['true_strong']['truth_higher'] for r in bad),
            true_strong_good_oracle_higher=sum(r['true_strong']['oracle_higher'] and
                r['true_strong']['oracle_error_m'] <= 100 for r in bad)))
    details = []
    for kind in ('failure', 'success_control'):
        subset = [r for r in detailed if r['selection'] == kind]
        details.append(dict(selection=kind, cases=len(subset),
            estimated_dense_success=sum(r['estimated']['error_m'] <= 100 for r in subset),
            true_strong_dense_success=sum(r['true_strong']['error_m'] <= 100 for r in subset),
            estimated_missed_better_truth_basin=sum(r['estimated']['error_m'] > 100 and
                r['estimated']['oracle']['oracle_error_m'] <= 100 and
                r['estimated']['oracle_beats_dense'] for r in subset),
            true_strong_missed_better_truth_basin=sum(r['true_strong']['error_m'] > 100 and
                r['true_strong']['oracle']['oracle_error_m'] <= 100 and
                r['true_strong']['oracle_beats_dense'] for r in subset),
            true_strong_wrong_beats_truth_basin=sum(r['true_strong']['wrong_beats_truth_basin'] for r in subset),
            joint_oracle_both_success=sum(all(r['joint_oracle']['metrics']['success']['100']) for r in subset)))
    return dict(status='COMPLETED_NEEDS_REVIEW', score_summary=summary, selected_summary=details,
                selected_case_scope='purpose-selected diagnostic, not population performance', test_access=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.zeros(1, device='cuda').sum().item()
    report, records, inputs = load_inputs()
    prior = read(BASE/'formal_20261011_repaired/ledger.json')['charged_seconds']
    # Prior probe attempts, including smoke, count against the same approved limit.
    elapsed_prior = sum(read(p)['seconds_this_run'] for p in BASE.glob('r1_failure_*/ledger.json'))
    total_guard = Guard(8*3600, prior+elapsed_prior)
    probe_guard = Guard(30*60, elapsed_prior)
    def guard():
        total_guard()
        probe_guard()
    out = preflight(BASE/args.run_id)
    sources = report['contract']['sources']+[identity(Path(__file__))]
    start = time.perf_counter()
    rows, detailed = [], []
    try:
        selected = select(records, report['contract'])
        if args.smoke:
            tags = [next(r['tag'] for r in records if r['row']['N'] == n and
                         selected.get(r['tag']) == 'failure') for n in (4096, 16384)]
            selected = {tag: selected[tag] for tag in tags}
            records = [r for r in records if r['tag'] in selected]
        write(out/'contract.json', dict(sources=sources, inputs=inputs, selected=selected,
            scope='engineering smoke' if args.smoke else 'G9-R1 diagnosis',
            prior_total_seconds=prior+elapsed_prior, probe_limit_seconds=1800,
            truth_policy='all oracle branches diagnostic only; frozen B@T0 unchanged'))
        last = time.perf_counter()
        print(f'阶段1/2：{len(records)}条同IQ评分核验', flush=True)
        for i, item in enumerate(records):
            guard()
            iq, truth, predicted, metric = regenerate(item, report['contract'])
            calc = Score(iq, guard)
            record = dict(tag=item['tag'], **{k: item['row'][k] for k in META},
                truth=truth.tolist(), predicted=predicted.tolist(), baseline_metrics=metric,
                strong_error_m=float(np.linalg.norm(predicted[0]-truth[0])),
                returned_pair_score=full_score(calc, predicted), truth_pair_score=full_score(calc, truth),
                estimated=fixed_probe(calc, predicted[0], predicted[1], truth[1]),
                true_strong=fixed_probe(calc, truth[0], predicted[1], truth[1]))
            rows.append(record)
            write(out/'score_rows'/f'{item["tag"]}.json', record)
            if time.perf_counter()-last > 30:
                print(f'评分核验 {i+1}/{len(records)}，已用{(time.perf_counter()-start)/60:.1f}分钟', flush=True)
                last = time.perf_counter()
        print(f'阶段2/2：{len(selected)}条固定样本密搜及强源替换', flush=True)
        for item in records:
            if item['tag'] not in selected:
                continue
            guard()
            iq, truth, predicted, _ = regenerate(item, report['contract'])
            calc = Score(iq, guard)
            record = dict(tag=item['tag'], selection=selected[item['tag']],
                estimated=dense_probe(calc, predicted[0], predicted[1], truth[1], guard),
                true_strong=dense_probe(calc, truth[0], predicted[1], truth[1], guard),
                joint_oracle=joint_oracle(calc, truth))
            detailed.append(record)
            write(out/'dense_rows'/f'{item["tag"]}.json', record)
            print(f'密搜 {len(detailed)}/{len(selected)}，已用{(time.perf_counter()-start)/60:.1f}分钟', flush=True)
        write(out/'report.json', summarize(rows, detailed))
        verify(sources)
        verify(inputs)
        write(out/'final_audit_report.json', dict(status='PASS', sources=sources, inputs=inputs,
            report=identity(out/'report.json'), regenerated_request_identity='PASS',
            score_rows=len(rows), dense_rows=len(detailed), test_access=False))
    except Exception:
        write(out/'failure.json', dict(status='FAILED', traceback=traceback.format_exc()))
        raise
    finally:
        elapsed = time.perf_counter()-start
        write(out/'ledger.json', dict(seconds_this_run=elapsed, probe_cumulative_seconds=elapsed_prior+elapsed,
                                    cumulative_seconds=prior+elapsed_prior+elapsed))
    print('完成：'+str(out/'report.json'), flush=True)


if __name__ == '__main__':
    main()
