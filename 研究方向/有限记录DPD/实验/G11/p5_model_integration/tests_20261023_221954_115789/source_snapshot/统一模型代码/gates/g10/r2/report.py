"""Saved-result-only paired statistics, mechanism tables and reusable figures."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..metrics import evaluate
from ..r1.methods import at_time
from ...g8.physics import RECEIVERS
from .config import METHODS, TIMES
from .runtime import read, write, sha, validate_row

COLORS = ('#000000', '#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00')
MARKERS = ('o', 's', '^', 'D', 'v', 'P', 'X')
STYLES = ('-', '--', '-.', ':', '-', '--', '-.')


def csv_write(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def interval(rows, field):
    """Conditional on fixed layouts; sample realizations with all gaps together."""
    rng = np.random.default_rng(2026101502)
    groups = sorted({r['group'] for r in rows})
    boot = np.zeros(2000)
    if not groups:
        return None
    for group in groups:
        items = [r for r in rows if r['group'] == group]
        repeats = sorted({r['realization'] for r in items})
        if len(repeats) < 2:
            return None
        values = np.array([np.mean([r[field] for r in items if r['realization'] == repeat]) for repeat in repeats])
        boot += values[rng.integers(len(values), size=(2000, len(values)))].mean(1)/len(groups)
    return np.quantile(boot, [.025, .975]).tolist()


def flat(row, positions, moment):
    metrics = evaluate(row['truth'], positions)
    flags, errors = metrics['success']['20'], metrics['errors_m']
    return dict(tag=row['tag'], group=row['task'][0], realization=row['task'][1], gap_db=row['task'][2],
                method=row['method'], moment=moment, strong20=int(flags[0]), weak20=int(flags[1]),
                both20=int(all(flags)), strong_error_m=errors[0], weak_error_m=errors[1],
                gospa_m=metrics['gospa_m'], coverage=metrics['matched']/2,
                seconds=row['result']['seconds'], stop=row['result']['stop_reason'])


def summarize(items):
    result = dict(n=len(items), both_count=sum(r['both20'] for r in items),
                  mean_seconds=float(np.mean([r['seconds'] for r in items])),
                  p95_seconds=float(np.quantile([r['seconds'] for r in items], .95)),
                  gospa_m=float(np.mean([r['gospa_m'] for r in items])))
    all_errors = []
    for target in ('strong', 'weak'):
        values = [r[target+'_error_m'] for r in items if r[target+'_error_m'] is not None]
        all_errors.extend(values)
        result[target+'20'] = float(np.mean([r[target+'20'] for r in items]))
        result[target+'_rmse_m'] = float(np.sqrt(np.mean(np.square(values)))) if len(values) == len(items) else None
        result[target+'_p95_m'] = float(np.quantile(values, .95)) if len(values) == len(items) else None
    result['both20'] = result['both_count']/len(items)
    result['both20_ci95'] = interval(items, 'both20')
    result['rmse_m'] = float(np.sqrt(np.mean(np.square(all_errors)))) if len(all_errors) == len(items)*2 else None
    return result


def export(out, stem, fig, data, cn, en, sources, label):
    folder = out/'figures'; folder.mkdir(exist_ok=True)
    fig.suptitle(label, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .96))
    for ext in ('pdf', 'png'):
        fig.savefig(folder/f'{stem}.{ext}', dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    csv_write(folder/f'{stem}.csv', data)
    write(folder/f'{stem}.json', dict(caption_cn=cn, caption_en=en, source_rows=sources,
          protocol='EVAL-20261011-v2', selection='All completed paired cases unless caption specifies',
          plot_code_sha=sha(__file__)))
    (folder/f'{stem}_caption.md').write_text(cn+'\n\n'+en+'\n', encoding='utf-8')


def generate(out):
    out = Path(out).resolve(strict=True)
    style = Path('C:/Users/Administrator/.codex/skills/scientific-toolkit-skill/references/scientific-skills/scientific-visualization/assets/publication.mplstyle')
    if style.exists():
        plt.style.use(str(style))
    # export() applies tight_layout once; do not also enable the style's
    # constrained-layout engine when creating figures.
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 9, 'legend.fontsize': 6.5,
                         'figure.constrained_layout.use': False, 'figure.autolayout': False})
    contract = read(out/'contract.json')
    label = 'G10-R2 / PREPARATION (not formal evidence)' if contract['stage'] == 'prepare' else 'G10-R2 / DEVELOPMENT (reused R1 inputs)'
    files = sorted((out/'rows').glob('*.json'))
    rows = [read(p) for p in files]
    if not rows:
        raise ValueError('No saved results')
    for row in rows:
        validate_row(row)
    sources = {str(p.relative_to(out)): sha(p) for p in files}
    methods = [m for m in METHODS if any(r['method'] == m for r in rows)]
    data = [flat(r, r['result']['positions'], 'final') for r in rows]
    timed = [flat(r, at_time(r['result'], t), str(t)) for r in rows for t in TIMES]
    csv_write(out/'per_case_metrics.csv', data+timed)
    summary = [dict(method=m, **summarize([r for r in data if r['method'] == m])) for m in methods]
    gaps = sorted({r['gap_db'] for r in data})
    by_gap = [dict(method=m, gap_db=g, **summarize([r for r in data if r['method'] == m and r['gap_db'] == g]))
              for m in methods for g in gaps]
    csv_write(out/'summary_table.csv', summary); csv_write(out/'gap_summary.csv', by_gap)
    by_key = {(r['group'], r['realization'], r['gap_db'], r['method']): r for r in data}
    pairs = []
    comparisons = [('AP_5_JOINT', 'AP_1_JOINT'), ('AP_5_ALT', 'AP_1_ALT'),
                   ('AP_1_JOINT', 'AP_1_ALT'), ('AP_5_JOINT', 'AP_5_ALT'),
                   ('NOMP_EXTRA12', 'NOMP'), ('NOMP_JOINT', 'NOMP'),
                   ('NOMP_JOINT', 'NOMP_EXTRA12'), ('NOMP_JOINT', 'AP_5_JOINT')]
    for candidate, baseline in comparisons:
        diff, rescued, harmed = [], 0, 0
        for r in data:
            key = (r['group'], r['realization'], r['gap_db'], baseline)
            if r['method'] != candidate or key not in by_key:
                continue
            b = by_key[key]
            delta = r['both20']-b['both20']
            rescued += delta > 0; harmed += delta < 0
            diff.append(dict(r, difference=delta))
        if diff:
            pairs.append(dict(candidate=candidate, baseline=baseline, n=len(diff), rescued=int(rescued),
                              harmed=int(harmed), success_difference=float(np.mean([r['difference'] for r in diff])),
                              ci95=interval(diff, 'difference')))
    csv_write(out/'paired_comparisons.csv', pairs)
    branches, curvature = [], []
    for r in rows:
        mechanism = r['result']['mechanism']
        for b in mechanism.get('branches', []):
            after = evaluate(r['truth'], b['final_positions'])
            branches.append(dict(tag=r['tag'], method=r['method'], group=r['task'][0],
                                 realization=r['task'][1], gap_db=r['task'][2],
                                 initial_rank=b['initial_rank'], selected=int(b['initial_rank'] == mechanism['selected_rank']),
                                 initial_objective=b['initial_objective'], final_objective=b['final_objective'],
                                 improvement=b['improvement'], both20=int(all(after['success']['20'])),
                                 stop=b['stop_reason']))
        if 'curvature' in r['result']:
            q = r['result']['curvature']
            curvature.append(dict(tag=r['tag'], method=r['method'], both20=int(all(r['metrics']['success']['20'])),
                                  positive_definite=q.get('positive_definite', False),
                                  ideal_block_gs_rho=q.get('ideal_block_gs_rho'),
                                  minimum_eigenvalue=min(q['eigenvalues']) if 'eigenvalues' in q else None))
    csv_write(out/'ap_branches.csv', branches); csv_write(out/'nomp_curvature.csv', curvature)
    stages = [dict(tag=r['tag'], method=r['method'], stage=k, seconds=v)
              for r in rows for k, v in r['result']['stages'].items()]
    csv_write(out/'stage_costs.csv', stages)
    costs = [dict(tag=r['tag'], method=r['method'], seconds=r['result']['seconds'],
                  diagnostics_seconds=r['result']['diagnostics_seconds'], **r['result']['counts']) for r in rows]
    csv_write(out/'operation_counts.csv', costs)
    write(out/'comparison_report.json', dict(stage=contract['stage'], label=label, source_rows=sources,
          summary=summary, gap_summary=by_gap, paired_comparisons=pairs,
          note='Fixed-layout conditional paired bootstrap; six gaps stay together. No significance stars or independent-validation claim.',
          independent_validation='NOT_STARTED', missing_outputs=sum(r['coverage'] < 1 for r in data),
          solver_seconds=sum(r['seconds'] for r in costs), diagnostics_seconds=sum(r['diagnostics_seconds'] for r in costs)))
    write(out/'mechanism_report.json', dict(ap_branches=branches, nomp_curvature=curvature,
          note='Rank and Hessian diagnostics are post-solve, never used to select or stop the algorithm. Local SPD theory is not an exact residual-NOMP rate guarantee.'))
    header = '| Method | N | Both@20m | Weak RMSE/m | GOSPA/m | Time/s |'
    table = [header, '|---|---:|---:|---:|---:|---:|']
    latex = ['\\begin{tabular}{lrrrrr}', 'Method & N & Success & Weak RMSE & GOSPA & Time \\\\']
    for r in summary:
        cells = [r['method'], str(r['n']), f"{r['both_count']}/{r['n']}",
                 f"{r['weak_rmse_m']:.3f}" if r['weak_rmse_m'] is not None else '--',
                 f"{r['gospa_m']:.3f}", f"{r['mean_seconds']:.3f}"]
        table.append('| '+' | '.join(cells)+' |')
        latex.append(' & '.join(c.replace('_', '\\_') for c in cells)+' \\\\')
    latex.append('\\end{tabular}')
    (out/'summary_table.md').write_text('\n'.join(table)+'\n', encoding='utf-8')
    (out/'summary_table.tex').write_text('\n'.join(latex)+'\n', encoding='utf-8')
    text = ['# G10-R2运行摘要', '', label, '', *table, '',
            '成功门槛20m；RMSE采用未截断平方距离匹配；GOSPA c=100m。',
            '图中n为每方法样本数。区间按固定布局内随机实现配对重采样；单重复准备不绘制区间。',
            'AP四格为短流程拆分；NOMP扩展包含共同前缀完整成本。事后Hessian诊断成本另列。',
            '这是准备/开发结果；不自动选择创新方法、不调参、不运行独立复核。',
            '回读：comparison_report.json、mechanism_report.json、final_audit_report.json、figures/。']
    (out/'运行摘要.md').write_text('\n'.join(text)+'\n', encoding='utf-8')

    def draw_lines(ax, field, source, xfield='gap_db', percent=False):
        for i, method in enumerate(methods):
            items = [r for r in source if r['method'] == method]
            x = [r[xfield] for r in items]
            y = np.array([r[field] for r in items], float)*(100 if percent else 1)
            ax.plot(x, y, color=COLORS[i], marker=MARKERS[i], linestyle=STYLES[i], label=method, markersize=3)
        ax.grid(alpha=.15)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, field, title in zip(axes, ('strong20', 'weak20', 'both20'), ('Strong', 'Weak', 'Both')):
        draw_lines(ax, field, by_gap, percent=True)
        ax.set(xlabel='Power gap (dB)', ylabel='Success within 20 m (%)', title=title, ylim=(-3, 103))
        for i, method in enumerate(methods):
            for gap in gaps:
                subset = [r for r in data if r['method'] == method and r['gap_db'] == gap]
                ci = interval(subset, field)
                if ci is not None:
                    center = np.mean([r[field] for r in subset])*100
                    ax.errorbar(gap, center, yerr=[[max(0., center-ci[0]*100)], [max(0., ci[1]*100-center)]],
                                color=COLORS[i], capsize=1, linewidth=.5)
    axes[-1].legend()
    export(out, 'success_by_gap', fig, data, f'各方法强/弱/双源20m成功率；每方法n={len(data)//len(methods)}。有重复时显示条件配对95%区间。',
           'Strong/weak/both success within 20 m; conditional paired 95% intervals when repeated realizations exist.', sources, label)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    for ax, field, title in zip(axes, ('weak_rmse_m', 'gospa_m'), ('Weak-source RMSE (m)', 'Mean GOSPA (m)')):
        draw_lines(ax, field, by_gap); ax.set(xlabel='Power gap (dB)', ylabel=title)
    axes[-1].legend()
    export(out, 'error_by_gap', fig, by_gap, '弱源未截断RMSE与GOSPA，包含全部样本。',
           'Untruncated weak-source RMSE and GOSPA over all cases.', sources, label)
    time_rows = [dict(method=m, time=t, both20=float(np.mean([r['both20'] for r in timed if r['method']==m and r['moment']==str(t)])))
                 for m in methods for t in TIMES]
    fig, ax = plt.subplots(figsize=(6, 3))
    draw_lines(ax, 'both20', time_rows, 'time', True)
    ax.set(xlabel='Complete algorithm time (s)', ylabel='Both-source success (%)', xscale='log', ylim=(-3, 103))
    ax.legend()
    export(out, 'success_vs_time', fig, timed, '各时刻已经保存的原生目标最优解；无输出计失败，不按真值挑时刻。',
           'Best native-objective incumbent available at each time; unavailable outputs count as failures.', sources, label)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    for i, method in enumerate(methods):
        items = [r for r in data if r['method']==method]
        errors = np.sort([r['weak_error_m'] for r in items if r['weak_error_m'] is not None])
        axes[0].step(errors, np.arange(1, len(errors)+1)/len(items), where='post', color=COLORS[i],
                     linestyle=STYLES[i], label=method)
        axes[1].scatter([i]*len(items), [r['seconds'] for r in items], s=8, alpha=.35, color=COLORS[i])
    axes[0].set(xlabel='Untruncated weak error (m)', ylabel='Cumulative probability', ylim=(0, 1.03))
    axes[0].legend(); axes[1].set(ylabel='Full solver time (s)', xticks=range(len(methods)), xticklabels=methods)
    axes[1].tick_params(axis='x', labelrotation=60)
    export(out, 'cdf_and_cost', fig, data, '弱源误差CDF与逐例完整耗时；未截断误差，全部观测为分母。P95见统计表。',
           'Untruncated weak-error CDF and individual complete solver times; P95 in tables.', sources, label)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    ap_methods = [m for m in methods if m.startswith('AP_5')]
    for i, method in enumerate(ap_methods):
        selected = [r['initial_rank'] for r in branches if r['method']==method and r['selected']]
        axes[0].bar(np.arange(1, 6)+(.15 if i else -.15), [selected.count(k) for k in range(1, 6)],
                    width=.3, color=COLORS[2+i], label=method)
    axes[0].set(xlabel='Selected initial candidate rank', ylabel='Cases', xticks=range(1, 6)); axes[0].legend()
    selected_pairs = [r for r in pairs if r['baseline']=='NOMP']
    for i, r in enumerate(selected_pairs):
        axes[1].bar(i-.15, r['rescued'], .3, color='#009E73', label='Rescued' if i==0 else None)
        axes[1].bar(i+.15, r['harmed'], .3, color='#D55E00', label='Harmed' if i==0 else None)
    axes[1].set(xticks=range(len(selected_pairs)), xticklabels=[r['candidate'] for r in selected_pairs], ylabel='Cases vs NOMP')
    axes[1].legend()
    export(out, 'mechanism', fig, branches+pairs, 'AP最终选择的初始候选排名；NOMP扩展相对原版的改对/改错数量。',
           'AP selected initial ranks and NOMP-extension rescued/harmed cases.', sources, label)
    first = 31 if contract['stage']=='prepare' else 100
    fig, axes = plt.subplots(2, 2, figsize=(7, 6))
    example_data = []
    for group, ax in enumerate(axes.ravel()):
        examples = [r for r in rows if r['task']==[group, first, 15]]
        ax.scatter(RECEIVERS[:, 0], RECEIVERS[:, 1], marker='^', color='gray', label='Receivers', s=20)
        if examples:
            truth = np.array(examples[0]['truth'])
            ax.scatter(truth[:, 0], truth[:, 1], marker='*', color='black', label='Truth', s=45)
            for i, method in enumerate(methods):
                r = next((r for r in examples if r['method']==method), None)
                if r:
                    pp = np.array(r['result']['positions'])
                    ax.scatter(pp[:, 0], pp[:, 1], marker=MARKERS[i], facecolors='none', edgecolors=COLORS[i], s=30, label=method)
                    example_data += [dict(group=group, method=method, source=j, x_m=float(p[0]), y_m=float(p[1])) for j, p in enumerate(pp)]
        ax.set(title=f'Layout {group}, realization {first}, 15 dB', xlabel='x (m)', ylabel='y (m)',
               xlim=(-1000, 1000), ylim=(-1000, 1000), aspect='equal')
    axes[0, 0].legend(fontsize=5, loc='lower left')
    export(out, 'prespecified_positions', fig, example_data, '预先固定：各布局首个实现、15dB；不是事后挑选成功或失败案例。',
           'Prespecified first realization at 15 dB in each layout; no outcome-based selection.', sources, label)
    return summary
