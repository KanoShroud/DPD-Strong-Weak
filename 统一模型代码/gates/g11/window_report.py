"""Read-only numerical certificates plus derived G11-P2 report; no solves."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.g11.precision_diagnostic import (
    P0, Model, load_case, read, write, resolve, sha, derivatives,
)
from 统一模型代码.gates.g11.window_validation import window, statistics
from 统一模型代码.gates.g10.r4.paper_checks.assignment_metrics import evaluate


def main():
    out = resolve(sys.argv[1])
    assert out.is_relative_to(ROOT/'outputs_e2e/unified/e2e_g11')
    assert not (out/'terminal_audit.json').exists()
    d = read(out/'report.json')
    protocol = read(out/'protocol.json')
    expected = read(P0/'source_manifest.json')
    folder = resolve(read(ROOT/'outputs_e2e/unified/e2e_g10/r4/review_supplement_release.json')['out'])
    assert d['summary'] == statistics(d['rows'])
    certificates = []
    for row in d['rows']:
        saved = read(out/'pairs'/f"{row['kind']}_{row['case']}.json")
        assert saved == row
        for name, result in row['results'].items():
            assert result['metrics'] == evaluate(row['truth'], result['positions'])
        if row['kind'] != 'mixed':
            continue
        _, signal, noise, truth, _ = load_case(folder, expected, (row['layout'], row['repeat'], row['gap']))
        for name, result in row['results'].items():
            iq = (signal.sum(0)+noise)*window(name, 8192)[None]
            m = Model(iq)
            value, g, h = derivatives(m, result['positions'])
            assert abs(value-result['native_objective']) < 1e-11
            assert abs(np.max(abs(g))-result['projected_gradient']) < 1e-10
            eig = np.linalg.eigvalsh((h+h.T)/2)
            certificates.append(dict(case=row['case'], window=name,
                 hessian_min=float(eig.min()), newton_correction_m=float(np.linalg.norm(np.linalg.solve(h,g))),
                 gradient_max=float(abs(g).max()), stop_reason=result['stop_reason']))
    assert all(sha(p)==h for p,h in {**protocol['source_code'],**protocol['protected']}.items())
    all_mixed = [r for r in d['rows'] if r['kind']=='mixed']
    times = {w:np.array([r['results'][w]['complete_seconds'] for r in all_mixed]) for w in ('rect','hann')}
    counts = {w:{name:float(np.mean([r['results'][w]['counts'].get(name,0) for r in all_mixed]))
                 for name in ('gradient_calls','joint_evaluations','frequency_response_points','candidate_points')}
              for w in ('rect','hann')}
    summary = dict(
        mixed_checks=len(certificates), all_hessians_positive=all(r['hessian_min']>0 for r in certificates),
        max_newton_correction_m=max(r['newton_correction_m'] for r in certificates),
        mean_complete_seconds={w:float(x.mean()) for w,x in times.items()},
        mean_paired_time_difference_seconds=float((times['hann']-times['rect']).mean()),
        median_paired_time_difference_seconds=float(np.median(times['hann']-times['rect'])),
        arithmetic_mean_counts=counts, metrics_and_summaries_reproduced=True,
        frozen_sources_and_papers_unchanged=True, localization_solves=0)
    write(out/'terminal_audit.json',dict(summary=summary, rows=certificates, script_sha=sha(__file__),
                 source_report_sha=sha(out/'report.json')))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
