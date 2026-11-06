"""Offline descriptive replay; all failures retained, no parameter selection."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.research_discovery.band_validation import BASE, digest, write
from 统一模型代码.gates.g10.r4.paper_checks.assignment_metrics import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pilots', nargs='+', type=Path)
    parser.add_argument('--closure', required=True, type=Path)
    args = parser.parse_args()
    out = BASE/'slepian_reference_20261023'/'summary'
    assert not out.exists()
    out.mkdir()
    raw, identities = [], {}
    for folder in args.pilots:
        assert json.loads((folder/'audit.json').read_text(encoding='utf-8'))['completed']
        identities[str(folder/'results.json')] = digest(folder/'results.json')
        rows = json.loads((folder/'results.json').read_text(encoding='utf-8'))
        assert len(rows) == 8
        for row in rows:
            assert len(row['starts']) == 5 and len(row['results']) == 6
            for result in row['results'].values():
                check = evaluate(row['truth'], result['positions'])
                assert check['both20'] == result['metrics']['both20']
                np.testing.assert_allclose(check['errors_m'], result['metrics']['errors_m'], atol=1e-10)
                assert result['seconds'] >= result['preparation_seconds']+result['solver_seconds']
        raw.extend(rows)
    assert json.loads((args.closure/'audit.json').read_text(encoding='utf-8'))['completed']
    diagnoses = json.loads((args.closure/'closure.json').read_text(encoding='utf-8'))
    identities[str(args.closure/'closure.json')] = digest(args.closure/'closure.json')
    summary = []
    text = ['# 常规有限记录参照：逐例统计回放', '',
            '本表保留全部开发观测；0 dB的源A/B为固定身份。耗时包含冷准备与共同候选成本。', '',
            '|N|功率差dB|方法|双源20m|源A RMSE/m|源B RMSE/m|平均完整/s|平均准备/s|平均调用|',
            '|---|---|---|---|---|---|---|---|---|']
    for n in sorted({r['n'] for r in raw}):
        for gap in (0, 15):
            rows = [r for r in raw if r['n'] == n and r['gap'] == gap]
            for method in rows[0]['results']:
                records = [r['results'][method] for r in rows]
                err = np.array([r['metrics']['errors_m'] for r in records])
                row = dict(n=n, gap=gap, method=method, count=len(records),
                    both20=sum(r['metrics']['both20'] for r in records),
                    rmse=np.sqrt(np.mean(err**2, axis=0)).tolist(),
                    median=np.median(err, axis=0).tolist(), max_error=float(err.max()),
                    seconds=float(np.mean([r['seconds'] for r in records])),
                    preparation=float(np.mean([r['preparation_seconds'] for r in records])),
                    calls=float(np.mean([r['calls'] for r in records])),
                    gradient_stopped=sum(r['gradient_max'] <= 1e-8 for r in records),
                    rank=records[0]['model'].get('rank'))
                summary.append(row)
                text.append(f'|{n}|{gap}|{method}|{row["both20"]}/{len(records)}|'
                            f'{row["rmse"][0]:.4f}|{row["rmse"][1]:.4f}|{row["seconds"]:.3f}|'
                            f'{row["preparation"]:.3f}|{row["calls"]:.1f}|')
    comparisons = []
    for row in raw:
        reference = np.asarray(row['results']['finite_fourier']['positions'])
        distances = {}
        for method in ['slepian_0.001', 'slepian_1e-06', 'slepian_1e-09']:
            other = np.asarray(row['results'][method]['positions'])
            cost = np.linalg.norm(reference[:, None]-other[None], axis=-1)
            i, j = linear_sum_assignment(cost**2)
            distances[method] = float(cost[i, j].max())
        comparisons.append(dict(n=row['n'], case=row['case'], gap=row['gap'], max_position_difference=distances))
    text += ['', '## 失败归因（事后诊断，不替换盲结果）', '',
             '|N|构型|差/dB|方法|真值初始化误差A/B m|五支精修选优误差A/B m|常规有限残余候选误差A/B m|',
             '|---|---|---|---|---|---|---|']
    for row in diagnoses:
        truth = row['truth_initialized']['metrics']['errors_m']
        branch = row['all_five_refined'][row['selected_by_objective']]['metrics']['errors_m']
        residual = row['conventional_residual']['metrics']['errors_m']
        text.append(f'|{row["n"]}|{row["case"]}|{row["gap"]}|{row["method"]}|'
                    f'{truth[0]:.3f}/{truth[1]:.3f}|{branch[0]:.3f}/{branch[1]:.3f}|'
                    f'{residual[0]:.3f}/{residual[1]:.3f}|')
    (out/'统计表.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    write(out/'statistics.json', dict(groups=summary, comparisons=comparisons))
    write(out/'audit.json', dict(passed=True, input_identities=identities,
        observations=len(raw), outputs=sum(len(r['results']) for r in raw),
        metrics_replayed=True, failures_retained=True, no_inferential_statistics=True))
    print(out)


if __name__ == '__main__':
    main()
