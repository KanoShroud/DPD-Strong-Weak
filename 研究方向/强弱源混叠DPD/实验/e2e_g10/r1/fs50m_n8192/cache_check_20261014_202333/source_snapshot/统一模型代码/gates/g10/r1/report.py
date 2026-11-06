"""Offline R1 statistics and publication figures from saved solver results only."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from ..report import COLORS, STYLES, MARKERS, csv_write as original_csv_write
from ..metrics import evaluate
from .methods import at_time
from .runtime import read, write, sha
from .config import METHODS as BASE_METHODS, TIMES, LAYOUTS
from ...g8.physics import RECEIVERS
from .sbl_analysis import expand_rows

METHODS = BASE_METHODS+('SBL_GROUP_GAMMA','SBL_GROUP_COHERENT')
COLORS = tuple(COLORS)+('#009E73','#D55E00')
STYLES = tuple(STYLES)+('--','-.')
MARKERS = tuple(MARKERS)+('s','^')


def csv_write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    original_csv_write(path, [{k: r.get(k) for k in keys} for r in rows])


def export(out, stem, fig, data, cn, en, source, label):
    folder = out/'figures'
    folder.mkdir(exist_ok=True)
    fig.suptitle(label, fontsize=10)
    for ext in ('pdf', 'png'):
        fig.savefig(folder/f'{stem}.{ext}', dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    csv_write(folder/f'{stem}.csv', data)
    write(folder/f'{stem}.json', dict(caption_cn=cn, caption_en=en, sources=source,
                                     protocol='EVAL-20261011-v2', label=label, plot_code_sha=sha(__file__)))
    (folder/f'{stem}_caption.md').write_text(cn+'\n\n'+en+'\n', encoding='utf-8')


def describe(rows, field, rms=False):
    values = [r[field] for r in rows if r[field] is not None]
    if not values:
        return None
    return float(np.sqrt(np.mean(np.square(values))) if rms else np.mean(values))


def interval(rows, field):
    """Conditional on four fixed geometries; resample realization IDs within each."""
    if not rows:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(2026101402)
    boot = np.zeros(2000)
    groups = sorted({r['group'] for r in rows})
    mean = 0.
    enough_repeats = True
    for group in groups:
        subset = [r for r in rows if r['group'] == group]
        rr = sorted({r['realization'] for r in subset})
        enough_repeats &= len(rr) >= 2
        vals = np.array([np.mean([r[field] for r in subset if r['realization'] == rep]) for rep in rr])
        boot += vals[rng.integers(len(vals), size=(2000, len(vals)))].mean(1)/len(groups)
        mean += vals.mean()/len(groups)
    return (float(mean), *np.quantile(boot, [.025, .975]).tolist()) if enough_repeats else (float(mean),np.nan,np.nan)


def flat_row(row, positions, moment):
    m = evaluate(row['truth'], positions)
    flags, errors = m['success']['20'], m['errors_m']
    result = row['result']
    return dict(tag=row['tag'], group=row['scene']['group'], realization=row['scene']['realization'],
                gap_db=row['scene']['gap_db'], kind=row['kind'], method=row['method'], moment=moment,
                strong20=int(flags[0]), weak20=int(flags[-1]), both20=int(all(flags)),
                strong_error=errors[0], weak_error=errors[-1], gospa=m['gospa_m'],
                coverage=m['matched']/m['targets'], seconds=result['seconds'], stop=result['stop_reason'])


def generate(out, label=None):
    style=Path('C:/Users/Administrator/.codex/skills/scientific-toolkit-skill/references/scientific-skills/scientific-visualization/assets/publication.mplstyle')
    if style.exists():
        plt.style.use(str(style))
    contract = read(out/'contract.json')
    label = label or ('G10-R1 / DEVELOPMENT' if contract['stage'] != 'formal' else 'G10-R1 / SIX-GAP COMPARISON')
    files = sorted((out/'rows').glob('*.json'))
    rows = [read(p) for p in files]
    rows = [r for r in rows if 'result' in r and 'tag' in r]
    selection = out/'plot_selection.json'
    if selection.exists():
        tags = set(read(selection)['tags'])
        rows = [r for r in rows if r['tag'] in tags]
    if not rows:
        raise ValueError('No saved solver rows')
    tags = {r['tag'] for r in rows}
    source = {str(p.relative_to(out)): sha(p) for p in files if p.stem in tags}
    base_rows = list(rows)
    rows = expand_rows(out, rows)
    data = []
    for r in rows:
        data.append(flat_row(r, r['result']['positions'], 'final'))
        for t in (TIMES if r['method'] in BASE_METHODS else ()):
            data.append(flat_row(r, at_time(r['result'], t), str(t)))
    csv_write(out/'per_case_metrics.csv', data)
    final = [r for r in data if r['moment'] == 'final']
    mixed = [r for r in final if r['kind'] == 'mixed']
    methods = [m for m in METHODS if any(r['method'] == m for r in final)]
    gaps = sorted({r['gap_db'] for r in mixed})
    summary = []
    for method in methods:
        for kind in ('weak', 'strong', 'mixed'):
            for gap in sorted({r['gap_db'] for r in final if r['kind'] == kind}):
                rr = [r for r in final if r['method'] == method and r['kind'] == kind and r['gap_db'] == gap]
                if not rr:
                    continue
                item = dict(method=method, kind=kind, gap_db=gap, n=len(rr),
                            strong20=describe(rr, 'strong20'), weak20=describe(rr, 'weak20'),
                            both20=describe(rr, 'both20'), strong_rmse_m=describe(rr, 'strong_error', True),
                            weak_rmse_m=describe(rr, 'weak_error', True), gospa_m=describe(rr, 'gospa'),
                            coverage=describe(rr, 'coverage'), mean_seconds=describe(rr, 'seconds'))
                for target in ('strong', 'weak'):
                    ee = [r[target+'_error'] for r in rr if r[target+'_error'] is not None]
                    item[target+'_p95_m'] = float(np.quantile(ee, .95)) if len(ee) == len(rr) else None
                summary.append(item)
    csv_write(out/'summary_table.csv', summary)
    write(out/'comparison_report.json', dict(label=label, rows=len(rows), summary=summary,
          note='Fixed-layout development comparison; conditional within-layout bootstrap. All six gaps paired.',
          source_rows=source, total_solver_seconds=sum(r['result']['seconds'] for r in base_rows),
          base_solver_rows=len(base_rows),derived_output_rows=len(rows)-len(base_rows),
          stops={s: sum(r['result']['stop_reason'] == s for r in rows)
                 for s in sorted({r['result']['stop_reason'] for r in rows})}))
    write(out/'execution_cost_report.json',dict(
        solver_seconds=sum(r['result']['seconds'] for r in base_rows),
        recording_seconds=sum(r['result'].get('recording_seconds',0.) for r in base_rows),
        offline_replay_seconds=sum(r['result'].get('output_replay_wall_seconds',0.) for r in base_rows),
        recovery_setup_seconds=sum(r['result'].get('resume_setup_seconds',0.) for r in base_rows),
        interrupted_attempts={r['tag']:r['result'].get('interrupted_attempts',[]) for r in base_rows if r['result'].get('interrupted_attempts')},
        ledger=read(out/'ledger.json'),
        note='Interrupted-attempt wall cost retained separately; abrupt-kill tail can be unrecorded for up to one 30s ledger interval. Derived outputs do not duplicate solve cost.'))
    md = ['| Method | Input | Gap/dB | N | Both@20m | Weak RMSE/m | Time/s |',
          '|---|---|---:|---:|---:|---:|---:|']
    latex = ['\\begin{tabular}{llrrrrr}', 'Method & Input & Gap & N & Success & RMSE & Time \\\\']
    for r in summary:
        values = [r['method'], r['kind'], f"{r['gap_db']:g}", str(r['n']),
                  f"{r['both20']:.3f}", str(r['weak_rmse_m']), f"{r['mean_seconds']:.3f}"]
        md.append('| '+' | '.join(values)+' |')
        latex.append(' & '.join(values)+' \\\\')
    latex.append('\\end{tabular}')
    (out/'summary_table.md').write_text('\n'.join(md), encoding='utf-8')
    (out/'summary_table.tex').write_text('\n'.join(latex), encoding='utf-8')
    cost = []
    for r in rows:
        cost.append(dict(tag=r['tag'], method=r['method'], kind=r['kind'], seconds=r['result']['seconds'],
                         **r['result']['counts']))
    csv_write(out/'cost_table.csv', cost)
    compact_cost = []
    for method in methods:
        subset = [r for r in rows if r['method'] == method]
        compact_cost.append(dict(method=method, n=len(subset),
                                mean_seconds=float(np.mean([r['result']['seconds'] for r in subset])),
                                p95_seconds=float(np.quantile([r['result']['seconds'] for r in subset], .95)),
                                mean_iterations=float(np.mean([r['result']['counts']['iterations'] for r in subset]))))
    csv_write(out/'cost_summary.csv', compact_cost)
    cm = ['| Method | N | Mean/s | P95/s | Mean iterations |', '|---|---:|---:|---:|---:|']
    ct = ['\\begin{tabular}{lrrrr}', 'Method & N & Mean/s & P95/s & Iterations \\\\']
    for r in compact_cost:
        cells = [r['method'], str(r['n']), f"{r['mean_seconds']:.3f}", f"{r['p95_seconds']:.3f}", f"{r['mean_iterations']:.1f}"]
        cm.append('| '+' | '.join(cells)+' |'); ct.append(' & '.join(cells)+' \\\\')
    ct.append('\\end{tabular}')
    (out/'cost_summary.md').write_text('\n'.join(cm), encoding='utf-8')
    (out/'cost_summary.tex').write_text('\n'.join(ct), encoding='utf-8')
    by_layout = []
    for method in methods:
        for group in sorted({r['group'] for r in mixed}):
            for gap in gaps:
                rr = [r for r in mixed if r['method']==method and r['group']==group and r['gap_db']==gap]
                if rr:
                    by_layout.append(dict(method=method, group=group, gap_db=gap, n=len(rr),
                                          both20=describe(rr,'both20'), weak20=describe(rr,'weak20'),
                                          weak_rmse_m=describe(rr,'weak_error',True)))
    csv_write(out/'per_layout_summary.csv', by_layout)
    stages = [dict(tag=r['tag'], method=r['method'], stage=name, seconds=sec)
              for r in rows for name, sec in r['result']['stages'].items()]
    csv_write(out/'stage_costs.csv', stages)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42})

    def lines(axes, fields, timed=False):
        for ax, (field, ylabel) in zip(np.asarray(axes).ravel(), fields):
            for method in (BASE_METHODS if timed else methods):
                if method not in methods:
                    continue
                i = METHODS.index(method)
                xs = TIMES if timed else gaps
                yy, low, high = [], [], []
                for x in xs:
                    rr = [r for r in (data if timed else mixed) if r['kind'] == 'mixed'
                          and r['method'] == method and (r['moment'] == str(x) if timed else r['gap_db'] == x)]
                    if field.endswith('20'):
                        v, a, b = interval(rr, field)
                    else:
                        v, a, b = describe(rr, field, 'error' in field), np.nan, np.nan
                    yy.append(v)
                    low.append(a)
                    high.append(b)
                ax.plot(xs, yy, color=COLORS[i], ls=STYLES[i], marker=MARKERS[i], ms=4, label=method)
                if field.endswith('20'):
                    ax.fill_between(xs, low, high, color=COLORS[i], alpha=.08)
                    ax.set_ylim(-.03, 1.03)
            ax.set(xlabel='Elapsed algorithm time (s)' if timed else 'Power difference (dB)', ylabel=ylabel)
            ax.grid(alpha=.2)
            if field.endswith('20'):
                counts = sorted({sum(r['method']==m and r['gap_db']==g for r in mixed) for m in methods for g in gaps})
                ax.text(.02, .03, 'n/gap/method='+','.join(map(str,counts)), transform=ax.transAxes, fontsize=7)
        np.asarray(axes).ravel()[0].legend(fontsize=7, ncol=2)

    if mixed:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
        lines(axes, [('strong20', 'Strong success @20 m'), ('weak20', 'Weak success @20 m'), ('both20', 'Both success @20 m')])
        export(out, '01_success_gap', fig, mixed,
               '功率差—成功率；阴影为固定布局内按随机实现重采样的95%区间，配对关系保留；样本数见表。',
               'Success versus power gap. 95% realization-bootstrap intervals conditional on fixed layouts; paired conditions retained.', source, label)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
        lines(axes, [('strong_error', 'Strong RMSE (m)'), ('weak_error', 'Weak RMSE (m)'), ('gospa', 'GOSPA (m)')])
        export(out, '02_error_gap', fig, mixed, '未截断RMSE与GOSPA，缺失输出覆盖率另列。',
               'Untruncated RMSE and GOSPA. Missing-output coverage reported separately.', source, label)
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), constrained_layout=True)
        lines(axes, [('weak20', 'Weak success @20 m'), ('both20', 'Both success @20 m')], True)
        export(out, '03_anytime', fig, [r for r in data if r['kind'] == 'mixed'],
               '运行中已有解的定位能力；采样时刻不会终止算法，无完整输出时记未成功。',
               'Anytime localization; sampling does not stop solvers. No complete output is not counted as success.', source, label)
        for target in ('strong', 'weak'):
            fig, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
            cdf = []
            for ax, gap in zip(axes.ravel(), gaps):
                for method in methods:
                    i = METHODS.index(method)
                    rr = [r for r in mixed if r['gap_db'] == gap and r['method'] == method]
                    ee = sorted(r[target+'_error'] for r in rr if r[target+'_error'] is not None)
                    yy = np.arange(1, len(ee)+1)/max(1, len(rr))
                    ax.step(ee, yy, where='post', color=COLORS[i], ls=STYLES[i], label=method)
                    cdf += [dict(method=method, gap_db=gap, error_m=e, cdf=float(y), n=len(rr)) for e, y in zip(ee, yy)]
                ax.axvline(20, color='gray', lw=.7)
                ax.set(xlabel='Error (m)', ylabel='CDF', title=f'{gap:g} dB', ylim=(0, 1.02))
            for ax in axes.ravel()[len(gaps):]:
                ax.set_visible(False)
            axes.ravel()[0].legend(fontsize=6, ncol=2)
            export(out, '04_cdf_'+target, fig, cdf, '最终误差CDF；缺失输出仍保留在分母。',
                   'Final error CDF; missing estimates remain in the denominator.', source, label)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].boxplot([[r['result']['seconds'] for r in rows if r['method'] == m] for m in methods], tick_labels=methods)
    axes[0].set_ylabel('Complete solver time (s)')
    axes[0].tick_params(axis='x', rotation=35)
    for method in methods:
        rr = [r for r in cost if r['method'] == method]
        i = METHODS.index(method)
        axes[1].scatter([r['frequency_response_points'] for r in rr], [r['seconds'] for r in rr],
                        c=COLORS[i], marker=MARKERS[i], label=method, s=18)
    axes[1].set(xlabel='Frequency-response evaluations', ylabel='Complete time (s)')
    axes[1].legend(fontsize=6)
    export(out, '05_cost', fig, cost, '完整耗时及频率响应计算量；各方法原生矩阵运算成本不同。',
           'Complete runtime and response evaluations; native matrix operations differ across methods.', source, label)
    paired = []
    for r in mixed:
        for target in ('weak', 'strong'):
            single = next((s for s in final if s['method'] == r['method'] and s['kind'] == target
                           and s['group'] == r['group'] and s['realization'] == r['realization']
                           and (target == 'weak' or s['gap_db'] == r['gap_db'])), None)
            if single:
                paired.append(dict(method=r['method'], group=r['group'], realization=r['realization'],
                                   gap_db=r['gap_db'], target=target,
                                   single_success=single['both20'], mixed_success=r[target+'20'],
                                   difference=r[target+'20']-single['both20']))
    if paired:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
        for ax, target in zip(axes, ('strong', 'weak')):
            for method in methods:
                i = METHODS.index(method)
                yy = []
                for g in gaps:
                    values = [r['difference'] for r in paired if r['target']==target and r['method']==method and r['gap_db']==g]
                    yy.append(float(np.mean(values)) if values else np.nan)
                ax.plot(gaps, yy, color=COLORS[i], ls=STYLES[i], marker=MARKERS[i], label=method)
            ax.set(xlabel='Power difference (dB)', ylabel='Mixed minus isolated success', title=target)
            ax.axhline(0, color='gray', lw=.7)
        axes[0].legend(fontsize=6, ncol=2)
        export(out, '06_paired', fig, paired, '同一波形与噪声下混合减孤立成功率；两者均使用正常结束结果。',
               'Paired mixed minus isolated success, same waveform and noise, both at native completion.', source, label)
    for p in sorted((out/'maps').glob('*.json')):
        native = read(p)
        if 'points' not in native:
            continue
        row = next((r for r in rows if r['tag'] == p.stem), None)
        if row is None:
            continue
        pp = np.asarray(native['points'])
        values = np.asarray(native['gamma'] if native.get('output_statistic')=='gamma' else native['power'])
        fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
        im = ax.scatter(pp[:, 0], pp[:, 1], c=values, s=4, cmap='viridis', rasterized=True)
        fig.colorbar(im, ax=ax, label='Native score / power')
        truth, pred = np.asarray(row['truth']), np.asarray(row['result']['positions'])
        ax.scatter(truth[:, 0], truth[:, 1], marker='*', s=90, c='red', label='True')
        ax.scatter(pred[:, 0], pred[:, 1], marker='x', s=55, c='white', label='Estimated')
        ax.scatter(RECEIVERS[:, 0], RECEIVERS[:, 1], marker='^', c='black', label='Receiver')
        ax.set(xlabel='x (m)', ylabel='y (m)', aspect='equal', title=row['method'])
        ax.legend(fontsize=7)
        dd = [dict(x_m=float(x), y_m=float(y), score=float(v)) for (x, y), v in zip(pp, values)]
        meaning = 'SBL gamma spectrum' if native.get('output_statistic')=='gamma' else native.get('meaning','native score')
        export(out, 'map_'+p.stem, fig, dd, '预登记案例；'+meaning,
               'Prespecified case; '+meaning, {**source, str(p.relative_to(out)):sha(p)}, label)
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), constrained_layout=True)
    for i, (ax, positions) in enumerate(zip(axes.ravel(), LAYOUTS)):
        pp = np.asarray(positions)
        hull = np.vstack([RECEIVERS, RECEIVERS[0]])
        ax.plot(hull[:, 0], hull[:, 1], 'k--', lw=.8)
        ax.scatter(RECEIVERS[:, 0], RECEIVERS[:, 1], marker='^', c='black', label='Receiver')
        ax.scatter(pp[0, 0], pp[0, 1], marker='*', s=80, c='#D55E00', label='Scaled source')
        ax.scatter(pp[1, 0], pp[1, 1], marker='o', c='#0072B2', label='Fixed weak source')
        ax.set(xlim=(-1000,1000), ylim=(-1000,1000), xlabel='x (m)', ylabel='y (m)', aspect='equal', title=f'Layout {i}')
    axes.ravel()[0].legend(fontsize=7)
    export(out, '00_geometry', fig, [dict(group=g, source=k, x_m=p[0], y_m=p[1]) for g, pp in enumerate(LAYOUTS) for k,p in enumerate(pp)],
           '四类固定布局；交换同一对内外位置的强弱角色。', 'Four fixed geometries, including role-swapped inner/outer positions.', source, label)
    (out/'运行摘要.md').write_text(f'# G10-R1运行摘要\n\n状态标签：{label}\n\n已保存{len(rows)}条样本—方法结果。完整指标见summary_table.md，运算量见cost_table.csv，阶段耗时见stage_costs.csv。\n\n成功率按20m一对一门槛匹配；RMSE/CDF按未截断平方距离最优匹配。图表可离线重绘，不自动重新定位。\n', encoding='utf-8')
    from .sbl_analysis import generate as sbl_generate
    sbl_generate(out,base_rows,source,label,export,csv_write)
    with (out/'运行摘要.md').open('a',encoding='utf-8') as stream:
        stream.write(f'\n基础求解{len(base_rows)}次，候选派生结果{len(rows)-len(base_rows)}条，不重复求解。'
                     '\nSBL专项见SBL输出分析.md、sbl_comparison_report.json；候选时间曲线仅按保存迭代状态比较，不伪造连续时间轨迹。\n')
