"""Replay saved band-validation evidence; never invokes a localization solver."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'outputs_e2e/research_discovery_20261022/band_validation_20261023'
SOURCES = {
    'tests': BASE/'tests_20261023_151842_260447',
    'numerical': BASE/'numerical_20261023_151900_559104',
    'blind': BASE/'blind_20261023_152101_538088',
    'outside': BASE/'blind_20261023_152850_757208',
}
METHODS = ('direct', 'boundary', 'az', 'lsmr')


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def export(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(values):
    x = np.asarray(values, float)
    return dict(mean=float(x.mean()), median=float(np.median(x)),
                minimum=float(x.min()), maximum=float(x.max()))


def main():
    target = BASE/'summary'
    target.mkdir(exist_ok=True)
    source_files = [p for directory in SOURCES.values() for p in directory.glob('*.json')]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    audits = {name: load(p/'audit.json') for name, p in SOURCES.items()}
    assert all(a['completed'] and a['identities_unchanged'] for a in audits.values())
    numerical = load(SOURCES['numerical']/'results.json')
    blind = load(SOURCES['blind']/'results.json')
    outside = load(SOURCES['outside']/'results.json')
    assert len(numerical) == 128 and len(blind) == 8 and len(outside) == 2
    assert all(x['sufficient'] for x in numerical)
    num_summary, flat_num = [], []
    for row in numerical:
        flat_num.append({k: row[k] for k in ('n', 'case', 'point', 'repeat', 'method', 'seconds', 'coefficient_error', 'sufficient')}
                        | dict(certificate=row['info']['coefficient_certificate']))
    for n in (1024, 4096):
        for method in METHODS:
            items = [r for r in numerical if r['n'] == n and r['method'] == method]
            num_summary.append(dict(n=n, method=method, samples=len(items),
                seconds=stats([r['seconds'] for r in items]),
                max_coefficient_error=max(r['coefficient_error'] for r in items),
                max_certificate=max(r['info']['coefficient_certificate'] for r in items)))
    pairs, flat_blind, comparisons = [], [], {}
    for method in METHODS:
        records = [x['results'][method] for x in blind]
        errors = np.array([r['metrics']['errors_m'] for r in records])
        comparisons[method] = dict(
            seconds=stats([r['seconds'] for r in records]),
            prep_seconds=stats([sum(r['preparation'].values()) for r in records]),
            both20=sum(r['metrics']['both20'] for r in records),
            rmse_all=float(np.sqrt(np.mean(errors**2))),
            rmse_strong=float(np.sqrt(np.mean(errors[:, 0]**2))),
            rmse_weak=float(np.sqrt(np.mean(errors[:, 1]**2))),
            calls=sum(r['calls'] for r in records), iterations=sum(r['iterations'] for r in records),
            max_gradient=max(r['gradient_max'] for r in records),
            max_certificate=max(i['coefficient_certificate'] for r in records for i in r['inner']),
            stop_messages=dict(Counter(r['message'] for r in records)))
        assert all(r['success'] and all(i['sufficient'] for i in r['inner']) for r in records)
    for row in blind:
        direct = row['results']['direct']
        for method in METHODS:
            r = row['results'][method]
            start_error = float(np.max(np.abs(np.asarray(r['starts'])-direct['starts'])))
            position_error = float(np.max(np.linalg.norm(np.asarray(r['positions'])-direct['positions'], axis=1)))
            assert start_error <= 1e-9 and position_error <= 1e-4
            assert r['selected'] == direct['selected'] and r['calls'] == direct['calls'] and r['iterations'] == direct['iterations']
            pairs.append(dict(case=row['case'], gap=row['gap'], method=method,
                start_error_m=start_error, position_difference_m=position_error,
                objective_difference=abs(r['objective']-direct['objective']),
                cost_reduction_vs_direct=1-r['seconds']/direct['seconds']))
            flat_blind.append(dict(case=row['case'], gap=row['gap'], method=method,
                seconds=r['seconds'], preparation_seconds=sum(r['preparation'].values()),
                calls=r['calls'], iterations=r['iterations'], selected=r['selected'],
                strong_error_m=r['metrics']['errors_m'][0], weak_error_m=r['metrics']['errors_m'][1],
                both20=r['metrics']['both20'], objective=r['objective'], gradient_max=r['gradient_max']))
    outside_checks = []
    for row in outside:
        original = next(r for r in blind if (r['case'], r['gap']) == (row['case'], row['gap']))['results']['boundary']
        changed = row['results']['boundary']
        start_diff = float(np.max(np.abs(np.asarray(original['starts'])-changed['starts'])))
        final_diff = float(np.max(np.linalg.norm(np.asarray(original['positions'])-changed['positions'], axis=1)))
        assert start_diff < 1e-7 and final_diff < 1e-4 and original['selected'] == changed['selected']
        outside_checks.append(dict(case=row['case'], gap=row['gap'], start_difference_m=start_diff,
            final_difference_m=final_diff, objective_difference=abs(original['objective']-changed['objective']),
            same_calls=original['calls'] == changed['calls'], same_iterations=original['iterations'] == changed['iterations']))
    costs = {}
    for reference in ('direct', 'az', 'lsmr'):
        ratios = [r['results']['boundary']['seconds']/r['results'][reference]['seconds'] for r in blind]
        costs[reference] = dict(paired_reduction=stats([1-x for x in ratios]),
            mean_time_reduction=1-comparisons['boundary']['seconds']['mean']/comparisons[reference]['seconds']['mean'])
    speed_by_order = {}
    for first in (True, False):
        selected = [r for r in blind if (list(r['results']).index('boundary') < list(r['results']).index('direct')) == first]
        speed_by_order['boundary_first' if first else 'direct_first'] = dict(n=len(selected),
            reduction=stats([1-r['results']['boundary']['seconds']/r['results']['direct']['seconds'] for r in selected]))
    summary = dict(status='BAND_COMPUTATION_VALIDATED_THEORETICAL_NOVELTY_NOT_ESTABLISHED',
        sources={k: str(v.relative_to(ROOT)) for k, v in SOURCES.items()},
        audits=audits, numerical=num_summary, blind=comparisons, paired_cost=costs,
        maximum_position_difference_m=max(x['position_difference_m'] for x in pairs),
        maximum_objective_difference=max(x['objective_difference'] for x in pairs),
        order_check=speed_by_order, outside=outside_checks,
        sample_scope='4 reused development realizations x 2 power gaps; not 8 independent geometries',
        timing_scope='cold per-observation total, all necessary preparation, shared prefix, solving and coefficient certification',
        report_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    dump(target/'summary.json', summary)
    dump(target/'source_identities.json', hashes)
    export(target/'numerical_cases.csv', flat_num)
    export(target/'blind_cases.csv', flat_blind)
    export(target/'pair_checks.csv', pairs)
    export(target/'outside_checks.csv', outside_checks)
    lines = ['# 带限验证：保存结果的离线统计', '',
        '仅重读结果，不重新定位。耗时为同批冷启动完整成本，精度为20m双源标准。', '',
        '| 求解器 | 完整平均秒 | 完整中位秒 | 平均准备秒 | 双源成功 | 弱源RMSE/m | 目标调用/迭代总数 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for method, r in comparisons.items():
        lines.append(f"| {method} | {r['seconds']['mean']:.6f} | {r['seconds']['median']:.6f} | {r['prep_seconds']['mean']:.6f} | {r['both20']}/8 | {r['rmse_weak']:.9f} | {r['calls']}/{r['iterations']} |")
    lines += ['', '## 固定位置内层求解', '',
        '| N | 求解器 | 次数 | 平均秒 | 中位秒 | 最大系数相对误差 |', '|---:|---|---:|---:|---:|---:|']
    for r in num_summary:
        lines.append(f"| {r['n']} | {r['method']} | {r['samples']} | {r['seconds']['mean']:.6f} | {r['seconds']['median']:.6f} | {r['max_coefficient_error']:.3e} |")
    lines += ['', '固定位置耗时不含一次性准备；完整耗时包含。1024点边界法慢于直接法，不能省略该结果。',
              '', '每例相同初值、分支、目标调用和迭代；最大位置差：'+str(summary['maximum_position_difference_m'])+' m。',
              '原坐标证书、带外扰动与运行先后顺序复核见summary.json；全部逐例数据和来源SHA256随附。']
    (target/'统计表.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == value for p, value in hashes.items())
    dump(target/'audit.json', dict(passed=True, input_results_unchanged=True, numeric_records=128,
                                  blind_results=32, outside_cases=2, new_localizations=0))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
