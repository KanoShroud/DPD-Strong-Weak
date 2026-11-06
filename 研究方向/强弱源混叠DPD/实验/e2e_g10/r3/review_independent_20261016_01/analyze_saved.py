"""Review frozen R3 results without invoking any localization solver."""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'AGENTS.md').is_file() and (p / '统一模型代码').is_dir())
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.g10.r3.runtime import (
    check_contract, read, sha, seal, load_input, validate, job_key)
from 统一模型代码.gates.g10.r3.report import flatten, summarize
from 统一模型代码.gates.g10.r2.report import interval
from 统一模型代码.gates.g10.metrics import evaluate


def main():
    out = ROOT / 'outputs_e2e/unified/e2e_g10/r3/independent_20261016_210849_013753'
    target = Path(__file__).with_name('analysis.json')
    if target.exists():
        raise FileExistsError(target)
    contract = check_contract(out)
    audit = read(out / 'final_audit_report.json')
    assert audit['complete'] and audit['completed'] == 480
    for rel, digest in contract['sources'].items():
        assert sha(out / 'source_snapshot' / rel) == digest, rel
    cache, rows = {}, []
    for job in contract['jobs']:
        task = tuple(job['task'])
        if task not in cache:
            _, _, truth, identity = load_input(out, task, contract)
            cache[task] = truth, identity
        truth, identity = cache[task]
        name = job_key(job) + '.json'
        path = out / 'rows' / name
        assert sha(path) == audit['rows'][name], name
        row = read(path)
        assert seal(row) == row['record_sha']
        validate(row, job, identity, contract['configurations'][job['method']], truth)
        distances = np.linalg.norm(truth[:, None] - np.array(row['result']['positions'])[None], axis=2)
        ii, jj = linear_sum_assignment(distances**2)
        np.testing.assert_allclose(distances[ii, jj], row['metrics']['errors_m'], atol=1e-10)
        assert np.isfinite(distances).all()
        rows.append(row)
    assert len(rows) == len({(tuple(r['task']), r['method']) for r in rows}) == 480
    flat = flatten(rows)
    report = read(out / 'comparison_report.json')
    for saved in report['summaries']:
        actual = summarize([x for x in flat if x['method'] == saved['method']])
        for key, value in actual.items():
            np.testing.assert_allclose(value, saved[key], rtol=1e-12, atol=1e-12)
    paired, failures = [], []
    for task in cache:
        methods = {r['method']: r for r in rows if tuple(r['task']) == task}
        a, b, c = (methods[m] for m in ('AP_1_ALT', 'AP_1_JOINT', 'NOMP'))
        ar, br, cr = (r['result'] for r in (a, b, c))
        ai, bi = (r['mechanism']['branches'][0]['initial_positions'] for r in (ar, br))
        np.testing.assert_allclose(ai, bi, atol=1e-10, rtol=0)
        d = np.linalg.norm(np.array(ar['positions'])[:, None] - np.array(br['positions'])[None], axis=2)
        ii, jj = linear_sum_assignment(d**2)
        paired.append(dict(group=task[0], realization=task[1], gap_db=task[2],
                           seconds_saved=ar['wall_solver_seconds']-br['wall_solver_seconds'],
                           max_position_difference_m=float(d[ii, jj].max()),
                           same_success=a['metrics']['success']['20']==b['metrics']['success']['20']))
        if not all(b['metrics']['success']['20']):
            first_two = next(e for e in cr['events'] if len(e['positions']) == 2)
            failures.append(dict(task=list(task), truth=b['truth'],
                initial_positions=bi, initial_errors=evaluate(b['truth'], bi)['errors_m'],
                final_positions=br['positions'], errors=b['metrics']['errors_m'],
                projected_gradient=br['projected_gradient'], objective=br['native_objective'],
                nomp_errors=c['metrics']['errors_m'], nomp_objective=cr['native_objective'],
                nomp_first_two_errors=evaluate(b['truth'], first_two['positions'])['errors_m']))
    summaries = {x['method']: x for x in report['summaries']}
    a, b = (summaries[m] for m in ('AP_1_ALT', 'AP_1_JOINT'))
    details = {}
    for method in summaries:
        items = [x for x in flat if x['method'] == method]
        raw = [r['result'] for r in rows if r['method'] == method]
        details[method] = dict(
            stops=dict(Counter(r['stop_reason'] for r in raw)),
            time_p95_seconds=float(np.quantile([x['wall_solver_seconds'] for x in items], .95)),
            weak_max_m=max(x['weak_error_m'] for x in items),
            weak_median_m=float(np.median([x['weak_error_m'] for x in items])),
            by_layout={str(g):summarize([x for x in items if x['group']==g]) for g in range(4)},
            success_by_gap={str(g):sum(x['both20'] for x in items if x['gap_db']==g)
                            for g in (0,3,6,9,12,15)},
            mean_stages={key:float(np.mean([r['stages'].get(key,0.) for r in raw]))
                         for key in sorted({k for r in raw for k in r['stages']})})
    joint_items = [x for x in flat if x['method']=='AP_1_JOINT']
    fail_squared = sum(x['weak_error_m']**2 for x in joint_items if not x['both20'])
    total_squared = sum(x['weak_error_m']**2 for x in joint_items)
    successes = [x['weak_error_m'] for x in joint_items if x['both20']]
    ml = [r['result']['mechanism'] for r in rows if r['method']=='MULTISTART_FIXED']
    assert all(m['actual_starts']==8 and Counter(m['start_kinds']) == Counter(ml[0]['start_kinds']) for m in ml)
    result = dict(source=str(out), contract_sha=sha(out/'contract.json'),
        audit_sha=sha(out/'final_audit_report.json'), verified_rows=len(rows), verified_inputs=len(cache),
        source_files_verified=len(contract['sources']), identity_metrics_pairing_passed=True,
        summaries=report['summaries'], paired=report['paired'], methods=details, failures=failures,
        ap_same_success_all=all(x['same_success'] for x in paired),
        ap_max_position_difference_m=max(x['max_position_difference_m'] for x in paired),
        ap_seconds_saved_ci95_replayed=interval(paired,'seconds_saved'),
        ap_joint_faster_count=sum(x['seconds_saved']>0 for x in paired),
        ap_time_reduction_percent=100*(1-b['wall_solver_seconds']/a['wall_solver_seconds']),
        ap_joint_call_reduction_percent=100*(1-b['count_joint_evaluations']/a['count_joint_evaluations']),
        ap_gradient_reduction_percent=100*(1-b['count_gradient_calls']/a['count_gradient_calls']),
        failure_weak_squared_error_share=fail_squared/total_squared,
        ap_success_subset_weak_rmse_m=float(np.sqrt(np.mean(np.square(successes)))),
        multistart_actual_counts=dict(Counter(m['actual_starts'] for m in ml)),
        multistart_kinds=ml[0]['start_kinds'], ledger=read(out/'ledger.json'),
        limitations='Four fixed geometries, five realizations each; six power gaps are paired variants. No new solve or performance-selected data.')
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['methods','failures','summaries']},ensure_ascii=False,indent=2))
    print('Failure tasks:', [x['task'] for x in failures])
    for method, value in details.items():
        print(method, 'stops', value['stops'], 'success_by_gap', value['success_by_gap'],
              'mean_stages', value['mean_stages'])


if __name__ == '__main__':
    main()
