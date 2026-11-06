"""Saved-pair-only statistics and figures. No optimizer imports or calls here."""
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from ..runtime import read, write, sha
from ..metrics import evaluate
from ..r2.runtime import seal
from ..r2.report import csv_write
from .formal_report import paired_boot, summary, table, export

METHODS = ('AP_5_ALT', 'AP_5_JOINT')
COLORS = ('#E69F00', '#0072B2')


def generate(out):
    out = Path(out)
    c = read(out/'contract.json')
    report = out/'reports'
    report.mkdir(exist_ok=True)
    rows, ids, stops = [], {}, Counter()
    expected = {'_'.join(map(str, t[:3])) for t in c['tasks']}
    found = set()
    for p in sorted((out/'pairs').glob('*.json')):
        row = read(p)
        if row['record_sha'] != seal(row) or p.stem not in expected:
            raise RuntimeError('Pair checksum or membership mismatch')
        found.add(p.stem)
        ids[str(p.relative_to(out))] = sha(p)
        task = row['task']
        g = c['layouts'][task[0]]
        for method in METHODS:
            r, m = row['results'][method], row['metrics'][method]
            if m != evaluate(row['truth'], r['positions']):
                raise RuntimeError('Metric replay differs')
            branches = r['mechanism']['branches']
            stops.update(method+':'+b['stop_reason'] for b in branches)
            rows.append(dict(layout=task[0], category=g['category'], role=g['role'],
                realization=task[1], gap_db=task[2], method=method,
                both20=int(all(m['success']['20'])), strong20=int(m['success']['20'][0]),
                weak20=int(m['success']['20'][1]), strong_error_m=m['errors_m'][0], weak_error_m=m['errors_m'][1],
                gospa_m=m['gospa_m'], seconds=r['seconds'], stop=branches[r['mechanism']['selected_rank']-1]['stop_reason'],
                objective=r['joint_objective'], gradient=r['projected_gradient'],
                joint_evaluations=r['counts']['joint_evaluations'], gradient_calls=r['counts']['gradient_calls'],
                selected_rank=r['mechanism']['selected_rank']))
    if found != expected:
        raise RuntimeError('Incomplete paired report; no selective complete-case ranking')
    csv_write(report/'per_case_metrics.csv', rows)
    write(report/'source_manifest.json', dict(rows=ids, contract_sha=sha(out/'contract.json')))
    summaries, paired = [], []
    for role in ('all', 'analysis', 'locked_validation'):
        part = [r for r in rows if role == 'all' or r['role'] == role]
        if not part:
            continue
        for method in METHODS:
            summaries.append(dict(method=method, scope=role, **summary([r for r in part if r['method'] == method])))
        left = {(r['layout'], r['realization'], r['gap_db']): r for r in part if r['method'] == METHODS[0]}
        diffs = []
        for b in [r for r in part if r['method'] == METHODS[1]]:
            a = left[b['layout'], b['realization'], b['gap_db']]
            diffs.append(dict(b, delta_success=b['both20']-a['both20'], delta_seconds=b['seconds']-a['seconds'],
                              delta_objective=b['objective']-a['objective']))
        alt = next(s for s in summaries if s['scope'] == role and s['method'] == METHODS[0])
        joint = next(s for s in summaries if s['scope'] == role and s['method'] == METHODS[1])
        paired.append(dict(scope=role, n=len(diffs), rescued=sum(d['delta_success'] > 0 for d in diffs),
            harmed=sum(d['delta_success'] < 0 for d in diffs),
            mean_seconds_change=float(np.mean([d['delta_seconds'] for d in diffs])),
            seconds_change_ci95=paired_boot(diffs, 'delta_seconds'),
            success_change_ci95=paired_boot(diffs, 'delta_success'),
            relative_mean_saving=1-joint['mean_seconds']/alt['mean_seconds'],
            mean_objective_change=float(np.mean([d['delta_objective'] for d in diffs]))))
    table(report, 'summary_table', summaries)
    csv_write(report/'paired_comparison.csv', paired)
    write(report/'comparison_report.json', dict(summaries=summaries, paired=paired,
        branch_stops=dict(stops), input_pairs=len(expected), solves=2*len(expected),
        note='Same five initial candidates, full-gradient stopping; native failures retained. Cost includes input model, search and refinement, excludes file IO.',
        uncertainty='Existing R4 stratified layout/repeat paired bootstrap, 1000 draws; power conditions stay grouped.'))
    label = ('DEVELOPMENT CHECK' if c.get('preparation') else 'Formal paired comparison')+f' / n={len(expected)}'
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei', 'DejaVu Sans'],
        'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.unicode_minus': False})
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    plot_data = []
    for i, method in enumerate(METHODS):
        part = [r for r in rows if r['method'] == method]
        for gap in sorted({r['gap_db'] for r in part}):
            sub = [r for r in part if r['gap_db'] == gap]
            s = summary(sub)
            plot_data.append(dict(method=method, gap_db=gap, **s))
        data = [v for v in plot_data if v['method'] == method]
        x = [v['gap_db'] for v in data]
        for ax, field in zip(axes, ('both20', 'weak_rmse_m', 'mean_seconds')):
            ax.plot(x, [v[field] for v in data], color=COLORS[i], linestyle=('--', '-')[i],
                    marker=('s', 'o')[i], label=method)
            ax.set_xlabel('功率差 (dB)')
    axes[0].set_ylim(0, 1.03); axes[0].set_ylabel('双源20 m成功率'); axes[0].legend(fontsize=8)
    axes[1].set_ylabel('弱源RMSE (m)'); axes[1].set_yscale('log'); axes[2].set_ylabel('平均完整耗时 (s)')
    positive = [max(v['weak_rmse_m'], 1e-8) for v in plot_data]
    low = 10**np.floor(np.log10(min(positive)))
    high = 10**np.ceil(np.log10(max(positive)))
    axes[1].set_ylim(low, max(high, low*10))
    axes[1].yaxis.set_minor_formatter(NullFormatter())
    axes[2].set_ylim(bottom=0)
    for ax in axes:
        ax.set_xticks(sorted({r['gap_db'] for r in rows}))
    export(report/'figures', 'paired_performance', fig, plot_data, 'reports/source_manifest.json',
        '同一观测、同一五候选的交替/联合精修；所有终止状态均纳入。逐条件点估计，配对区间见comparison_report.json。',
        'Paired five-candidate alternating and joint refinement; all termination states retained. Point estimates; paired intervals are in the report.', label)
    fig, ax = plt.subplots(figsize=(5.2, 4.2)); scatter = []
    left = {(r['layout'], r['realization'], r['gap_db']): r for r in rows if r['method'] == METHODS[0]}
    for b in [r for r in rows if r['method'] == METHODS[1]]:
        a = left[b['layout'], b['realization'], b['gap_db']]
        scatter.append(dict(layout=b['layout'], realization=b['realization'], gap_db=b['gap_db'],
                            alt_seconds=a['seconds'], joint_seconds=b['seconds'],
                            alt_calls=a['joint_evaluations'], joint_calls=b['joint_evaluations']))
    ax.scatter([r['alt_seconds'] for r in scatter], [r['joint_seconds'] for r in scatter], s=9, alpha=.5, color=COLORS[1])
    end = max(max(r['alt_seconds'], r['joint_seconds']) for r in scatter)*1.05
    ax.plot([0, end], [0, end], '--', color='gray', label='等耗时')
    ax.set_xlim(0, end); ax.set_ylim(0, end); ax.set_xlabel('交替精修完整耗时 (s)'); ax.set_ylabel('联合精修完整耗时 (s)'); ax.legend()
    export(report/'figures', 'paired_cost', fig, scatter, 'reports/source_manifest.json',
        '每点对应同一观测的一对求解；虚线下方表示联合精修更快。',
        'Each point is a matched observation. Below the diagonal means joint refinement is faster.', label)
    fig, ax = plt.subplots(figsize=(5.5, 3.5)); cdf = []
    for i, method in enumerate(METHODS):
        errors = np.sort([r['weak_error_m'] for r in rows if r['method'] == method])
        p = np.arange(1, len(errors)+1)/len(errors)
        ax.step(np.maximum(errors, 1e-8), p, where='post', label=method, color=COLORS[i], linestyle=('--', '-')[i])
        cdf += [dict(method=method, error_m=float(e), cdf=float(v)) for e, v in zip(errors, p)]
    ax.set_xscale('log'); ax.set_xlabel('弱源定位误差 (m)'); ax.set_ylabel('累计比例'); ax.legend()
    positive = [max(r['error_m'], 1e-8) for r in cdf]
    low = 10**np.floor(np.log10(min(positive)))
    high = 10**np.ceil(np.log10(max(positive)))
    ax.set_xlim(low, max(high, low*10)); ax.set_ylim(0, 1.03)
    ax.xaxis.set_minor_formatter(NullFormatter())
    export(report/'figures', 'weak_error_cdf', fig, cdf, 'reports/source_manifest.json',
        '全部样本的未截断弱源误差CDF，不剔除失败样本。',
        'Untruncated weak-source error CDF over all observations, including failures.', label)
    a, b = [next(s for s in summaries if s['scope'] == 'all' and s['method'] == m) for m in METHODS]
    text = (f'# 五候选配对运行摘要\n\n{label}\n\n'
            f'共{len(expected)}对、{2*len(expected)}次求解。\n\n'
            f'- 交替：{a["both_count"]}/{a["n"]}成功，平均{a["mean_seconds"]:.4f}秒。\n'
            f'- 联合：{b["both_count"]}/{b["n"]}成功，平均{b["mean_seconds"]:.4f}秒。\n\n'
            '请回读reports/comparison_report.json、final_audit_report.json及本摘要。'
            '图表在reports/figures，逐例与配对统计在reports目录。\n')
    (out/'运行摘要.md').write_text(text, encoding='utf-8')
