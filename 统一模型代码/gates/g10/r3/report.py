"""Offline paired reporting; timing repetitions never inflate sample size."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..r2.report import csv_write, export, interval, COLORS, MARKERS, STYLES
from ..metrics import evaluate
from .config import METHODS, REVISION
from .runtime import read, write, sha, seal, safe_out


def flatten(rows):
    grouped = {}
    for row in rows:
        if row['record_sha'] != seal(row) or row['metrics'] != evaluate(row['truth'], row['result']['positions']):
            raise RuntimeError('Invalid row in report')
        grouped.setdefault((tuple(row['job']['task']), row['job']['method']), []).append(row)
    result = []
    for (task, method), values in sorted(grouped.items()):
        values.sort(key=lambda r: r['job']['repetition'])
        first = values[0]; m = first['metrics']; r = first['result']
        for row in values[1:]:
            np.testing.assert_allclose(row['result']['positions'], r['positions'], atol=1e-7, rtol=0)
        item = dict(group=task[0], realization=task[1], gap_db=task[2], method=method,
            n_timing_repeats=len(values), strong20=int(m['success']['20'][0]),
            weak20=int(m['success']['20'][1]), both20=int(all(m['success']['20'])),
            strong_error_m=m['errors_m'][0], weak_error_m=m['errors_m'][1], gospa_m=m['gospa_m'],
            objective=r['native_objective'], projected_gradient=r['projected_gradient'],
            precision_met=int(r['precision_met']), stop=r['stop_reason'])
        for key in ('seconds', 'wall_solver_seconds', 'recording_seconds', 'certificate_seconds'):
            item[key] = float(np.median([row['result'][key] for row in values]))
        for key, value in r['counts'].items():
            item['count_'+key] = value
        hits = [[e for e in row['result']['accepted_certificates']
                 if e['projected_gradient'] <= row['result']['config']['gtol']] for row in values]
        item['first_precision_wall_seconds'] = float(np.median([h[0]['wall_seconds'] for h in hits])) if all(hits) else None
        item['first_precision_core_seconds'] = float(np.median([h[0]['seconds'] for h in hits])) if all(hits) else None
        item['first_precision_joint_calls'] = hits[0][0]['joint_evaluations'] if all(hits) else None
        for key in ('input_fft_transfer_band_selection', 'first_source', 'conditional_candidates', 'branch_refinement'):
            item['stage_'+key] = float(np.median([row['result']['stages'].get(key, 0.) for row in values]))
        result.append(item)
    return result


def summarize(items):
    s = dict(n=len(items), both_count=sum(x['both20'] for x in items))
    for key in ('strong20', 'weak20', 'both20', 'gospa_m', 'seconds', 'wall_solver_seconds',
                'recording_seconds', 'certificate_seconds', 'precision_met',
                'count_joint_evaluations', 'count_gradient_calls'):
        s[key] = float(np.mean([x[key] for x in items]))
    for target in ('strong', 'weak'):
        errors = [x[target+'_error_m'] for x in items]
        s[target+'_rmse_m'] = float(np.sqrt(np.mean(np.square(errors))))
        s[target+'_p95_m'] = float(np.quantile(errors, .95))
    s['both20_ci95'] = interval(items, 'both20')
    return s


def generate(out):
    out = safe_out(Path(out).resolve(strict=True))
    contract = read(out/'contract.json')
    if contract['revision'] != REVISION:
        raise RuntimeError('Not an R3 result; refusing to overwrite another report')
    files = sorted((out/'rows').glob('*.json'))
    rows = [read(p) for p in files]
    if not rows:
        raise RuntimeError('No rows to report')
    flat = flatten(rows)
    csv_write(out/'per_case_metrics.csv', flat)
    summaries = [dict(method=m, **summarize([x for x in flat if x['method'] == m]))
                 for m in METHODS if any(x['method'] == m for x in flat)]
    active_methods = [m for m in METHODS if any(x['method'] == m for x in flat)]
    csv_write(out/'summary_table.csv', summaries)
    pairs = []
    for key in sorted({(x['group'], x['realization'], x['gap_db']) for x in flat}):
        group = {x['method']: x for x in flat if (x['group'], x['realization'], x['gap_db']) == key}
        if not all(m in group for m in ('AP_1_ALT', 'AP_1_JOINT')):
            continue
        a, b = group['AP_1_ALT'], group['AP_1_JOINT']
        pairs.append(dict(group=key[0], realization=key[1], gap_db=key[2],
            seconds_saved=a['wall_solver_seconds']-b['wall_solver_seconds'],
            core_seconds_saved=a['seconds']-b['seconds'],
            objective_difference=b['objective']-a['objective'],
            both_precision=int(a['precision_met'] and b['precision_met']),
            success_difference=b['both20']-a['both20'],
            joint_calls_saved=a['count_joint_evaluations']-b['count_joint_evaluations'],
            gradient_calls_saved=a['count_gradient_calls']-b['count_gradient_calls']))
        pa, pb = a['first_precision_wall_seconds'], b['first_precision_wall_seconds']
        pairs[-1]['first_precision_seconds_saved'] = pa-pb if pa is not None and pb is not None else None
        ca, cb = a['first_precision_core_seconds'], b['first_precision_core_seconds']
        pairs[-1]['first_precision_core_seconds_saved'] = ca-cb if ca is not None and cb is not None else None
        ca, cb = a['first_precision_joint_calls'], b['first_precision_joint_calls']
        pairs[-1]['first_precision_calls_saved'] = ca-cb if ca is not None and cb is not None else None
    csv_write(out/'paired_cost.csv', pairs)
    report = dict(stage=contract['stage'], completed=len(rows), expected=len(contract['jobs']),
        summaries=summaries, paired=dict(n=len(pairs),
            mean_seconds_saved=float(np.mean([p['seconds_saved'] for p in pairs])),
            seconds_saved_ci95=interval(pairs, 'seconds_saved'),
            mean_core_seconds_saved=float(np.mean([p['core_seconds_saved'] for p in pairs])),
            both_precision_count=sum(p['both_precision'] for p in pairs),
            maximum_objective_difference=float(max(abs(p['objective_difference']) for p in pairs)),
            mean_joint_calls_saved=float(np.mean([p['joint_calls_saved'] for p in pairs])),
            mean_gradient_calls_saved=float(np.mean([p['gradient_calls_saved'] for p in pairs]))),
        scope='Fixed geometries; layout-stratified realization bootstrap, all gaps retained. Timing repeats collapsed by median.',
        timing='Headline wall_solver_seconds includes logging; seconds excludes measured record serialization. Certificates and file I/O are separate.')
    precision_pairs = [p for p in pairs if p['first_precision_seconds_saved'] is not None]
    report['paired']['first_precision_pairs'] = len(precision_pairs)
    report['paired']['mean_first_precision_seconds_saved'] = float(np.mean([p['first_precision_seconds_saved'] for p in precision_pairs])) if precision_pairs else None
    report['paired']['mean_first_precision_core_seconds_saved'] = float(np.mean([p['first_precision_core_seconds_saved'] for p in precision_pairs])) if precision_pairs else None
    report['paired']['mean_first_precision_calls_saved'] = float(np.mean([p['first_precision_calls_saved'] for p in precision_pairs])) if precision_pairs else None
    write(out/'comparison_report.json', report)
    if contract.get('previous_preparation'):
        prior = Path(contract['previous_preparation']['out'])
        old_report = read(prior/'comparison_report.json')
        comparison = []
        for current in summaries:
            native = next(x for x in old_report['summaries'] if x['method'] == current['method'])
            for kind, values in (('native_stop_prior_run', native), ('common_stop_current_run', current)):
                comparison.append(dict(stop_protocol=kind, **values))
        csv_write(out/'native_vs_common_summary.csv', comparison)
        report['native_stop_reference'] = contract['previous_preparation']
        report['native_comparison_note'] = 'Separate runs; use within-run AP pairing for timing claims. Prior NOMP/ML are not rerun.'
        write(out/'comparison_report.json', report)
    keys = ('method', 'n', 'both_count', 'strong_rmse_m', 'weak_rmse_m', 'gospa_m', 'wall_solver_seconds', 'seconds')
    cells = [[str(x[k]) if not isinstance(x[k], float) else f'{x[k]:.6g}' for k in keys] for x in summaries]
    md = '| '+' | '.join(keys)+' |\n|'+'|'.join(['---']*len(keys))+'|\n'
    md += '\n'.join('| '+' | '.join(c)+' |' for c in cells)
    (out/'summary_table.md').write_text(md+'\n', encoding='utf-8')
    tex = '\\begin{tabular}{'+'l'*len(keys)+'}\n'
    tex += ' & '.join(k.replace('_', '\\_') for k in keys)+' \\\\\n'
    tex += '\n'.join(' & '.join(c.replace('_', '\\_') for c in row)+' \\\\' for row in cells)
    (out/'summary_table.tex').write_text(tex+'\n\\end{tabular}\n', encoding='utf-8')
    style = Path('C:/Users/Administrator/.codex/skills/scientific-toolkit-skill/references/scientific-skills/scientific-visualization/assets/publication.mplstyle')
    if style.exists():
        plt.style.use(str(style))
    plt.rcParams.update({'figure.constrained_layout.use': False, 'figure.autolayout': False})
    sources = {str(p.relative_to(out)): sha(p) for p in files}
    sources['plot_code:'+str(Path(__file__).resolve())] = sha(__file__)
    label = 'R3 independent validation' if contract['stage'] == 'formal' else 'R3 DEVELOPMENT / timing repetitions are not independent samples'
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2)); data = []
    for i, method in enumerate(active_methods):
        items = [x for x in flat if x['method'] == method]
        if not items:
            continue
        summaries_gap = [dict(method=method, gap_db=gap, **summarize([x for x in items if x['gap_db'] == gap]))
                         for gap in sorted({x['gap_db'] for x in items})]
        data.extend(summaries_gap)
        for ax, field in zip(axes, ('both20', 'weak_rmse_m', 'wall_solver_seconds')):
            ax.plot([x['gap_db'] for x in summaries_gap], [x[field] for x in summaries_gap],
                    label=method, color=COLORS[i], marker=MARKERS[i], linestyle=STYLES[i])
            ax.set_xlabel('Power gap (dB)')
        ci = np.asarray([x['both20_ci95'] for x in summaries_gap], float)
        axes[0].fill_between([x['gap_db'] for x in summaries_gap], ci[:, 0], ci[:, 1], color=COLORS[i], alpha=.12)
    for ax, ylabel in zip(axes, ('Both sources within 20 m (fraction)', 'Weak-source RMSE (m)', 'Complete solver time (s)')):
        ax.set_ylabel(ylabel)
    axes[0].set_ylim(-.03, 1.03); axes[2].legend(fontsize=6)
    export(out, 'power_quality_cost', fig, data,
        '各功率差的双源成功率、弱源RMSE与完整算法时间。n见CSV；成功率阴影为固定布局内按随机实现配对重采样95%区间。',
        'Success, weak-source RMSE and complete cost; n in CSV; success bands: layout-stratified realization bootstrap 95% intervals.', sources, label)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2)); data = []
    for i, method in enumerate(active_methods):
        items = [x for x in flat if x['method'] == method]
        for ax, target in zip(axes, ('strong', 'weak')):
            errors = np.sort([x[target+'_error_m'] for x in items])
            if not len(errors):
                continue
            cdf = np.arange(1, len(errors)+1)/len(errors)
            ax.step(errors, cdf, where='post', label=method, color=COLORS[i], linestyle=STYLES[i])
            data.extend(dict(method=method, target=target, error_m=float(e), cdf=float(f)) for e, f in zip(errors, cdf))
            ax.set_xlabel(target.capitalize()+' error (m)'); ax.set_ylabel('Empirical CDF')
    axes[0].legend(fontsize=6)
    export(out, 'error_cdf', fig, data, '全部样本的未截断强弱源误差CDF；P95见汇总表。',
           'Untruncated strong/weak error CDF; P95 in summary table.', sources, label)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for i, method in enumerate(active_methods):
        items = [x for x in flat if x['method'] == method]
        if not items:
            continue
        axes[0].scatter([i]*len(items), [x['wall_solver_seconds'] for x in items], s=8, alpha=.4, color=COLORS[i], marker=MARKERS[i])
        axes[1].bar(i-.18, np.mean([x['count_joint_evaluations'] for x in items]), .36, color=COLORS[i], label=method)
        axes[1].bar(i+.18, np.mean([x['count_gradient_calls'] for x in items]), .36, facecolor='none', edgecolor=COLORS[i], hatch='//')
    for ax in axes:
        ax.set_xticks(range(len(active_methods)), active_methods, rotation=25, ha='right')
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(facecolor='gray', label='Objective'),
                            Patch(facecolor='none', edgecolor='gray', hatch='//', label='With gradient')], fontsize=6)
    axes[0].set_ylabel('Complete solver time (s)'); axes[1].set_ylabel('Actual evaluations (mean)')
    export(out, 'time_and_calls', fig, flat,
        '左：逐样本完整耗时，短测每点为五次计时中位数；右：实心为实际目标评价，斜线为其中带梯度的评价。记录和事后精度检查耗时另存CSV。',
        'Left: per-IQ complete time (median of five during preparation). Right: objective evaluations, hatched subset with gradients; recording/certificate costs in CSV.', sources, label)
    (out/'运行摘要.md').write_text('# G10-R3运行摘要\n\n'
        +f"阶段：{contract['stage']}；完成 {len(rows)}/{len(contract['jobs'])}。\n\n"
        +md+'\n\n完整耗时包含必要计算与轨迹记录；事后精度检查单列。短测不作为独立性能证据。\n'
        +'当前证据应结合 `paired_cost.csv`、`final_audit_report.json` 和 `readiness.json` 判断。\n', encoding='utf-8')
    return report
