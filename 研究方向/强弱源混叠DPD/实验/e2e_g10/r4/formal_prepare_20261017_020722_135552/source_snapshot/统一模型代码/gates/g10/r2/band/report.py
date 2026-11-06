"""Saved rows only: stratified statistics, reusable tables and figures."""
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ...metrics import evaluate
from ...r1.methods import at_time
from ..report import csv_write, export, interval, COLORS, MARKERS, STYLES
from ....g8.physics import RECEIVERS
from .config import METHODS, TIMES
from .runtime import read, write, sha, seal


def flatten(row, positions=None):
    metrics = row['metrics'] if positions is None else evaluate(row['truth'], positions)
    errors, flags = metrics['errors_m'], metrics['success']['20']
    result = dict(tag=row['tag'], group=row['task'][0], realization=row['task'][1], gap_db=row['task'][2],
        kind=row['kind'], method=row['method'], variant=row['variant'], both20=int(all(flags)),
        strong20=int(flags[0]) if row['kind'] != 'weak' else None,
        weak20=int(flags[-1]) if row['kind'] != 'strong' else None,
        strong_error_m=errors[0] if row['kind'] != 'weak' else None,
        weak_error_m=errors[-1] if row['kind'] != 'strong' else None,
        gospa_m=metrics['gospa_m'], seconds=row['result']['seconds'],
        stop=row['result']['stop_reason'], iterations=row['result']['counts']['iterations'])
    for field, value in row['result']['counts'].items():
        result['count_'+field] = value
    return result


def summary(items):
    result = dict(n=len(items), success_count=sum(r['both20'] for r in items),
        success20=float(np.mean([r['both20'] for r in items])), ci95=interval(items, 'both20'),
        gospa_m=float(np.mean([r['gospa_m'] for r in items])),
        seconds=float(np.mean([r['seconds'] for r in items])),
        p95_seconds=float(np.quantile([r['seconds'] for r in items], .95)),
        mean_iterations=float(np.mean([r['iterations'] for r in items])),
        stopping=dict(Counter(r['stop'] for r in items)))
    for role in ('strong', 'weak'):
        values = [r[role+'_error_m'] for r in items if r[role+'_error_m'] is not None]
        flags = [r[role+'20'] for r in items if r[role+'20'] is not None]
        result[role+'_rmse_m'] = float(np.sqrt(np.mean(np.square(values)))) if values else None
        result[role+'_p95_m'] = float(np.quantile(values, .95)) if values else None
        result[role+'20'] = float(np.mean(flags)) if flags else None
    return result


def table(out, name, rows):
    csv_write(out/(name+'.csv'), rows)
    keys = ['method', 'kind', 'variant', 'n', 'success_count', 'success20', 'strong_rmse_m', 'weak_rmse_m', 'gospa_m', 'seconds']
    cells = [[str(r.get(k, '')) if not isinstance(r.get(k), float) else f'{r[k]:.5g}' for k in keys] for r in rows]
    text = '| '+' | '.join(keys)+' |\n|'+ '|'.join(['---']*len(keys))+'|\n'
    text += '\n'.join('| '+' | '.join(row)+' |' for row in cells)
    (out/(name+'.md')).write_text(text+'\n', encoding='utf-8')
    latex = '\\begin{tabular}{'+'l'*len(keys)+'}\n'+' & '.join(k.replace('_', '\\_') for k in keys)+' \\\\\n'
    latex += '\n'.join(' & '.join(c.replace('_', '\\_') for c in row)+' \\\\' for row in cells)
    (out/(name+'.tex')).write_text(latex+'\n\\end{tabular}\n', encoding='utf-8')


def generate(out, selected_only=False):
    out = Path(out).resolve(strict=True)
    contract = read(out/'contract.json')
    formal = contract['stage'] == 'independent'
    files = sorted((out/'rows').glob('*.json'))
    rows = [read(p) for p in files]
    if not rows:
        return
    for row in rows:
        if row['record_sha'] != seal(row) or row['metrics'] != evaluate(row['truth'], row['result']['positions']):
            raise RuntimeError('Invalid saved row in report')
    sources = {str(p.relative_to(out)): sha(p) for p in files}
    flat = [flatten(r) for r in rows]
    csv_write(out/'per_case_metrics.csv', flat)
    summaries = []
    for method in METHODS:
        for kind in ('weak', 'strong', 'mixed'):
            for variant in sorted({r['variant'] for r in flat if r['method'] == method and r['kind'] == kind}):
                subset = [r for r in flat if r['method'] == method and r['kind'] == kind and r['variant'] == variant]
                summaries.append(dict(method=method, kind=kind, variant=variant, **summary(subset)))
    table(out, 'summary_table', summaries)
    planned = Counter(j['method'] for j in contract['jobs'])
    actual = Counter(r['method'] for r in rows)
    write(out/'comparison_report.json', dict(stage=contract['stage'], completed=len(rows), planned=len(contract['jobs']),
        complete_methods=[m for m in METHODS if actual[m] == planned[m]], summaries=summaries,
        inference_scope='Four fixed geometries; paired resampling within layout, keep each realization across gaps.',
        note='At 0 dB strong/weak denote scan identities; equal mean received power.'))
    style = Path('C:/Users/Administrator/.codex/skills/scientific-toolkit-skill/references/scientific-skills/scientific-visualization/assets/publication.mplstyle')
    if style.exists():
        plt.style.use(str(style))
    plt.rcParams.update({'figure.constrained_layout.use': False, 'figure.autolayout': False,
                         'font.size': 8, 'legend.fontsize': 6.5})
    label = 'Independent / shared +/-7 MHz' if formal else 'Development calibration / NOT independent evidence'
    groups = [('classical', ('LS', 'MUSIC', 'MVDR', 'AP_FULL', 'NOMP', 'MULTISTART', 'GA', 'SBL')),
              ('ap', ('AP_FULL', 'AP_1_ALT', 'AP_1_JOINT', 'AP_5_ALT', 'AP_5_JOINT')),
              ('nomp', ('AP_5_JOINT', 'NOMP', 'NOMP_EXTRA12', 'NOMP_JOINT'))]
    mixed = [r for r in flat if r['kind'] == 'mixed' and r['variant'] == 0]
    for group_name, methods in groups:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.1))
        data = []
        for i, method in enumerate(methods):
            sub = [r for r in mixed if r['method'] == method]
            if not sub:
                continue
            for column, target in enumerate(('strong20', 'weak20', 'both20')):
                points = []
                for gap in sorted({r['gap_db'] for r in sub}):
                    values = [r for r in sub if r['gap_db'] == gap]
                    mean = float(np.mean([r[target] for r in values]))
                    ci = interval(values, target)
                    points.append((gap, mean, ci))
                    data.append(dict(method=method, gap_db=gap, target=target, success=mean, n=len(values), ci95=ci, variant=0))
                axes[column].plot([p[0] for p in points], [p[1] for p in points],
                    color=COLORS[i % len(COLORS)], marker=MARKERS[i % len(MARKERS)], linestyle=STYLES[i % len(STYLES)], label=method)
                if all(p[2] is not None for p in points):
                    axes[column].fill_between([p[0] for p in points], [p[2][0] for p in points],
                        [p[2][1] for p in points], color=COLORS[i % len(COLORS)], alpha=.08)
        for ax, title in zip(axes, ('Source A', 'Source B', 'Both sources')):
            ax.set(xlabel='Power gap (dB)', ylabel='Success within 20 m', title=title, ylim=(-.02, 1.02))
        axes[0].legend(loc='best')
        export(out, 'success_'+group_name, fig, data,
            '20 m成功率；开发图仅显示v0，不表示已选最优配置。阴影为布局内按随机实现重采样95%区间，n见CSV；全成功的退化区间不代表总体概率确定为1。',
            '20 m success; development plots show v0 only. Shading: conditional paired 95% bootstrap intervals; sample sizes in CSV.', sources, label)
    # Accuracy and runtime for every completed configuration, no silent cherry-picking.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    overview = [r for r in summaries if r['kind'] == 'mixed']
    names = [r['method']+(' v'+str(r['variant']) if not formal else '') for r in overview]
    x = np.arange(len(names))
    axes[0].bar(x, [r['weak_rmse_m'] or 0 for r in overview], color='#0072B2')
    axes[1].bar(x, [r['seconds'] for r in overview], color='#E69F00')
    for ax in axes:
        ax.set_xticks(x, names, rotation=90)
    axes[0].set_ylabel('Source B RMSE (m)'); axes[1].set_ylabel('Complete runtime (s)')
    export(out, 'accuracy_cost', fig, overview, '各完整配置的弱源RMSE与实际完整耗时；非同一目标的原生评分不横比。',
           'Source B RMSE and full measured runtime per configuration.', sources, label)
    # Saved anytime incumbents include missing output as failure, never use future estimates.
    timed = []
    for r in rows:
        if r['kind'] != 'mixed' or r['variant'] != 0:
            continue
        for moment in TIMES:
            timed.append(dict(flatten(r, at_time(r['result'], moment)), moment=moment))
    csv_write(out/'anytime_metrics.csv', timed)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, method in enumerate(('MUSIC', 'MVDR', 'AP_FULL', 'AP_5_JOINT', 'NOMP', 'NOMP_JOINT', 'GA', 'SBL')):
        sub = [r for r in timed if r['method'] == method]
        if sub:
            ax.plot(TIMES, [np.mean([r['both20'] for r in sub if r['moment'] == t]) for t in TIMES],
                marker=MARKERS[i % len(MARKERS)], linestyle=STYLES[i % len(STYLES)], label=method)
    ax.set(xscale='log', xlabel='Elapsed complete algorithm time (s)', ylabel='Both sources within 20 m', ylim=(-.02, 1.02))
    ax.legend(ncol=2)
    export(out, 'anytime', fig, timed, '共同时间节点的已保存最优输出；尚无输出计失败，不按秒截断算法。',
           'Saved incumbents at shared elapsed-time checkpoints; absent output counts as failure.', sources, label)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, method in enumerate(METHODS):
        values = sorted(r['weak_error_m'] for r in mixed if r['method'] == method and r['weak_error_m'] is not None)
        if values:
            ax.step(values, np.arange(1, len(values)+1)/len(values), where='post',
                color=COLORS[i % len(COLORS)], linestyle=STYLES[i % len(STYLES)], label=method)
    ax.set(xlabel='Source B error (m)', ylabel='Empirical CDF', ylim=(0, 1.02))
    ax.legend(ncol=2)
    export(out, 'weak_cdf', fig, mixed, '弱源未截断误差CDF；开发图为v0，P95见汇总表。',
           'Untruncated source B error CDF; P95 in summary table.', sources, label)
    # Preselected cases: final positions, never label these as algorithm spectra.
    repeat = 200 if formal else 40
    case_rows = [r for r in rows if r['task'][1] == repeat and r['kind'] == 'mixed' and r['task'][2] == 15 and r['variant'] == 0]
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    spatial_data = []
    for group, ax in enumerate(axes.ravel()):
        selected = [r for r in case_rows if r['task'][0] == group]
        ax.scatter(RECEIVERS[:, 0], RECEIVERS[:, 1], marker='^', c='black', label='Receivers')
        if selected:
            truth = np.asarray(selected[0]['truth'])
            ax.scatter(truth[:, 0], truth[:, 1], marker='*', s=100, c='black', label='Truth')
        for method in ('AP_5_JOINT', 'NOMP_JOINT', 'MUSIC', 'MVDR', 'SBL'):
            r = next((r for r in selected if r['method'] == method), None)
            if r:
                p = np.asarray(r['result']['positions'])
                ax.scatter(p[:, 0], p[:, 1], marker='x', label=method)
                for slot, xy in enumerate(p):
                    spatial_data.append(dict(group=group, method=method, slot=slot, x_m=xy[0], y_m=xy[1], tag=r['tag']))
        ax.set(xlim=(-1000, 1000), ylim=(-1000, 1000), xlabel='x (m)', ylabel='y (m)', title=f'Layout {group}')
        ax.set_aspect('equal')
    axes[0, 0].legend(fontsize=6)
    export(out, 'preselected_positions', fig, spatial_data, '预先指定：每布局首个实现、15 dB；展示最终位置，不是空间谱。',
           'Preregistered first realization per layout at 15 dB; final positions, not spectra.', sources, label)
    cost_rows = []
    for r in rows:
        for stage, seconds in r['result']['stages'].items():
            cost_rows.append(dict(tag=r['tag'], method=r['method'], variant=r['variant'], kind=r['kind'], stage=stage, seconds=seconds))
    csv_write(out/'stage_costs.csv', cost_rows)
    # Matched mixed-versus-isolated values preserve the same realization and noise.
    isolated = {(r['method'], r['variant'], r['task'][0], r['task'][1], r['task'][2], r['kind']): r for r in rows if r['kind'] != 'mixed'}
    paired = []
    for r in rows:
        if r['kind'] != 'mixed':
            continue
        method = r['method']
        if method not in ('LS', 'MUSIC', 'MVDR', 'AP_FULL', 'NOMP', 'MULTISTART', 'GA', 'SBL'):
            continue
        for role, index in (('weak', 1), ('strong', 0)):
            lookup = (method, r['variant'], r['task'][0], r['task'][1], 0 if role == 'weak' else r['task'][2], role)
            single = isolated.get(lookup)
            if single:
                paired.append(dict(method=method, variant=r['variant'], group=r['task'][0], realization=r['task'][1],
                    gap_db=r['task'][2], role=role, mixed_error_m=r['metrics']['errors_m'][index],
                    isolated_error_m=single['metrics']['errors_m'][0], mixed_success=r['metrics']['success']['20'][index],
                    isolated_success=single['metrics']['success']['20'][0], mixed_tag=r['tag'], isolated_tag=single['tag']))
    csv_write(out/'isolated_mixed_pairs.csv', paired)
    gap_stats = []
    for method in METHODS:
        for gap in sorted({r['gap_db'] for r in mixed}):
            sub = [r for r in mixed if r['method'] == method and r['gap_db'] == gap]
            if sub:
                gap_stats.append(dict(method=method, gap_db=gap, **summary(sub)))
    csv_write(out/'per_gap_summary.csv', gap_stats)
    for group_name, methods in groups:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.1))
        for i, method in enumerate(methods):
            sub = [r for r in gap_stats if r['method'] == method]
            for ax, field in zip(axes, ('strong_rmse_m', 'weak_rmse_m', 'gospa_m')):
                if sub:
                    ax.plot([r['gap_db'] for r in sub], [r[field] for r in sub],
                        color=COLORS[i % len(COLORS)], marker=MARKERS[i % len(MARKERS)],
                        linestyle=STYLES[i % len(STYLES)], label=method)
        for ax, title in zip(axes, ('Source A RMSE', 'Source B RMSE', 'GOSPA')):
            ax.set(xlabel='Power gap (dB)', ylabel=title+' (m)')
        axes[0].legend()
        export(out, 'errors_'+group_name, fig, [r for r in gap_stats if r['method'] in methods],
            '按功率差统计未截断RMSE和GOSPA；全部样本纳入，开发图为v0。',
            'Untruncated RMSE and GOSPA versus power gap; all completed cases, v0 for development.', sources, label)
    operation_rows = []
    for s in summaries:
        sub = [r for r in flat if r['method'] == s['method'] and r['kind'] == s['kind'] and r['variant'] == s['variant']]
        op = {k:s[k] for k in ('method', 'kind', 'variant', 'n', 'seconds')}
        for field in ('candidate_points', 'frequency_response_points', 'joint_evaluations', 'gradient_calls', 'covariance_decompositions', 'sbl_dictionary_columns'):
            op[field] = float(np.mean([r.get('count_'+field, 0) for r in sub]))
        operation_rows.append(op)
    csv_write(out/'operation_counts.csv', operation_rows)
    deltas = []
    for candidate, baseline in (('AP_5_JOINT', 'AP_1_JOINT'), ('AP_5_JOINT', 'AP_5_ALT'),
                                 ('NOMP_JOINT', 'NOMP'), ('NOMP_JOINT', 'NOMP_EXTRA12')):
        a = {(r['group'], r['realization'], r['gap_db']):r for r in mixed if r['method'] == candidate}
        b = {(r['group'], r['realization'], r['gap_db']):r for r in mixed if r['method'] == baseline}
        pairs = [dict(group=k[0], realization=k[1], gap_db=k[2], delta=a[k]['both20']-b[k]['both20']) for k in a.keys() & b.keys()]
        if pairs:
            deltas.append(dict(candidate=candidate, baseline=baseline, n=len(pairs),
                success_difference=float(np.mean([p['delta'] for p in pairs])), ci95=interval(pairs, 'delta')))
    write(out/'paired_comparisons.json', deltas)
    text = '# G10-R2带限比较运行摘要\n\n'
    text += f"阶段：{contract['stage']}；已保存{len(rows)}/{len(contract['jobs'])}次求解。\n\n"
    text += '公共频带±7 MHz；全部方法配置与输入身份见contract.json。开发结果用于校准，不是独立性能证据。\n\n'
    text += '回读comparison_report.json、summary_table.md、final_audit_report.json及figures。主批完成后可先分析，SBL后补。\n'
    (out/'运行摘要.md').write_text(text, encoding='utf-8')
