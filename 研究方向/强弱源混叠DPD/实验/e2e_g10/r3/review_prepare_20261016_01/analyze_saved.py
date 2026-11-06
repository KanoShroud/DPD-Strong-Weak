"""Read-only R3 preparation review: no solver, new evidence files only."""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'AGENTS.md').is_file() and (p / '统一模型代码').is_dir())
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.g10.r3.runtime import check_contract, read, sha, seal
from 统一模型代码.gates.g10.metrics import evaluate


def main():
    out = ROOT / 'outputs_e2e/unified/e2e_g10/r3/prepare_20261016_201814_740356'
    check_contract(out)
    audit = read(out / 'final_audit_report.json')
    rows = []
    for filename, digest in audit['rows'].items():
        path = out / 'rows' / filename
        assert sha(path) == digest, filename
        row = read(path)
        assert seal(row) == row['record_sha'], filename
        assert evaluate(row['truth'], row['result']['positions']) == row['metrics']
        rows.append(row)
    first = [r for r in rows if r['job']['repetition'] == 0]
    methods = {}
    for method in sorted({r['method'] for r in first}):
        results = [r['result'] for r in first if r['method'] == method]
        gradients = [r['projected_gradient'] for r in results]
        messages = [i['message'] for r in results
                    for i in r['mechanism'].get('local_optimizers', [])]
        methods[method] = dict(
            n=len(results), precision_count=sum(r['precision_met'] for r in results),
            gradient_min_median_max=np.quantile(gradients, [0, .5, 1]).tolist(),
            optimizer_messages=dict(Counter(messages)),
            branch_stops=dict(Counter(b['stop_reason'] for r in results
                                     for b in r['mechanism'].get('branches', []))))
    pairs = []
    for row in first:
        if row['method'] != 'AP_1_ALT':
            continue
        other = next(r for r in first if r['method'] == 'AP_1_JOINT'
                     and r['task'] == row['task'])
        a, b = np.array(row['result']['positions']), np.array(other['result']['positions'])
        d = np.linalg.norm(a[:, None] - b[None], axis=2)
        ii, jj = linear_sum_assignment(d**2)
        pairs.append(dict(task=row['task'], max_position_difference_m=float(d[ii, jj].max())))
    report = read(out / 'comparison_report.json')
    summary = {r['method']: r for r in report['summaries']}
    a, b = summary['AP_1_ALT'], summary['AP_1_JOINT']
    result = dict(source=str(out), verified_rows=len(rows), identity_and_metrics_passed=True,
                  readiness=read(out / 'readiness.json'), methods=methods,
                  ap_position_comparison=pairs,
                  max_ap_position_difference_m=max(p['max_position_difference_m'] for p in pairs),
                  relative_time_saved=1-b['wall_solver_seconds']/a['wall_solver_seconds'],
                  relative_joint_evaluations_saved=1-b['count_joint_evaluations']/a['count_joint_evaluations'],
                  relative_gradient_calls_saved=1-b['count_gradient_calls']/a['count_gradient_calls'],
                  source_audit_sha=sha(out / 'final_audit_report.json'))
    target = Path(__file__).with_name('analysis.json')
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
