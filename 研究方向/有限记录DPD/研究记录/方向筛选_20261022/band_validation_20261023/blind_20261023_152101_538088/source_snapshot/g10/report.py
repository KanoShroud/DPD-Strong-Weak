"""Replot from saved rows/maps only; publication exports with exact source mapping."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .config import METHODS, CHECKPOINTS
from .runtime import read, write, sha, load_scene
from .methods import at_time
from .metrics import evaluate

COLORS = ['#E69F00', '#56B4E9', '#009E73', '#CC9900', '#0072B2', '#D55E00', '#CC79A7', '#333333']
STYLES = ['-', '--', '-.', ':', '-', '--', '-.', ':']
MARKERS = ['o', 's', '^', 'v', 'D', 'P', 'X', '*']


def csv_write(path, rows):
    if not rows:
        path.write_text('status\nNO_DATA_FOR_THIS_PANEL\n', encoding='utf-8-sig')
        return
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export(out, stem, fig, data, caption_cn, caption_en, source, label):
    folder = out/'figures'
    folder.mkdir(exist_ok=True)
    if 'FORMAL' != label:
        fig.suptitle(label, fontsize=10, color='#8B4513')
    for ext in ('pdf', 'png'):
        fig.savefig(folder/f'{stem}.{ext}', dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    csv_write(folder/f'{stem}.csv', data)
    write(folder/f'{stem}.json', dict(caption_cn=caption_cn, caption_en=caption_en,
        protocol='EVAL-20261011-v2', sources=source, label=label, data_file=stem+'.csv',
        plot_code_sha=sha(__file__)))
    (folder/f'{stem}_caption.md').write_text(f'{caption_cn}\n\n{caption_en}\n', encoding='utf-8')


def flatten(rows):
    out = []
    for row in rows:
        result = row['result']
        for t in CHECKPOINTS:
            if t > result['budget']:
                continue
            m = evaluate(row['truth'], at_time(result, t))
            errors = m['errors_m']
            flags = m['success']['20']
            out.append(dict(tag=row['tag'], group=row['scene']['group'], method=row['method'], kind=row['kind'],
                gap_db=row['scene']['gap_db'], budget=t, strong20=int(flags[0]), weak20=int(flags[-1]),
                both20=int(all(flags)), strong_error=errors[0], weak_error=errors[-1],
                gospa=m['gospa_m'], coverage=m['matched']/m['targets'],
                seconds=result['seconds'], status=result['status'],
                candidate_points=result['counts'].get('candidate_points', 0),
                joint_evaluations=result['counts'].get('joint_evaluations', 0),
                frequency_response_points=result['counts'].get('frequency_response_points', 0)))
    return out


def summarize(values, error=False):
    a = np.asarray([v for v in values if v is not None], float)
    return float(np.sqrt(np.mean(a*a)) if error else np.mean(a)) if len(a) else None


def ci(rows, field):
    if not rows:
        return np.nan, np.nan, np.nan
    groups = sorted(set(r['group'] for r in rows))
    values = np.array([np.mean([r[field] for r in rows if r['group'] == g]) for g in groups])
    rng = np.random.default_rng(2026101302)
    boot = values[rng.integers(len(values), size=(2000, len(values)))].mean(1)
    return float(values.mean()), *np.quantile(boot, [.025, .975]).tolist()


def generate(out, label=None, rows=None, guard=lambda: None):
    files = sorted((out/'rows').glob('*.json'))
    selection_path = out/'plot_selection.json'
    selection = read(selection_path) if selection_path.exists() else None
    if rows is None:
        rows = [read(p) for p in files]
        if selection is not None:
            rows = [r for r in rows if r['tag'] in selection['tags']]
    if label is None:
        label = selection['label'] if selection else read(out/'contract.json')['stage'].upper()
    rows = [r for r in rows if 'result' in r]
    tags = [r['tag'] for r in rows]
    source = {str(p.relative_to(out)): sha(p) for p in files if p.stem in tags}
    write(selection_path, dict(tags=tags, label=label, sources=source))
    data = flatten(rows)
    if not data:
        raise ValueError('No evaluable saved results')
    csv_write(out/'per_case_metrics.csv', data)
    max_budget = max(r['budget'] for r in data if r['kind'] == 'mixed')
    mixed = [r for r in data if r['kind'] == 'mixed' and r['budget'] == max_budget]
    gaps = sorted(set(r['gap_db'] for r in mixed))
    summary = []
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42})
    for method in METHODS:
        for gap in gaps:
            rr = [r for r in mixed if r['method'] == method and r['gap_db'] == gap]
            if not rr:
                continue
            weak, lo, hi = ci(rr, 'weak20')
            summary.append(dict(method=method, gap_db=gap, budget=max_budget, n=len(rr),
                weak20=weak, weak_lo=lo, weak_hi=hi, strong20=summarize([r['strong20'] for r in rr]),
                both20=summarize([r['both20'] for r in rr]),
                weak_rmse=summarize([r['weak_error'] for r in rr], True),
                strong_rmse=summarize([r['strong_error'] for r in rr], True),
                gospa=summarize([r['gospa'] for r in rr]), coverage=summarize([r['coverage'] for r in rr]),
                seconds=summarize([r['seconds'] for r in rr]),
                strong_p95=float(np.quantile([r['strong_error'] for r in rr], .95))
                    if all(r['strong_error'] is not None for r in rr) else None,
                p95=float(np.quantile([r['weak_error'] if r['weak_error'] is not None else np.inf for r in rr], .95))
                    if all(r['weak_error'] is not None for r in rr) else None))
    csv_write(out/'summary_table.csv', summary)
    write(out/'comparison_report.json', dict(label=label, protocol='EVAL-20261011-v2',
        rows=len(rows), summary=summary, total_solver_seconds=sum(r['result']['seconds'] for r in rows),
        statuses={s: sum(r['result']['status'] == s for r in rows) for s in
                  ('FINISHED', 'BUDGET_EXHAUSTED', 'NUMERICAL_FAILURE')},
        sources=source, note='CDF uses squared-distance matching; success uses threshold matching.'))
    table = ['| Method | gap/dB | n | weak@20m | both@20m | weak RMSE/m | mean time/s |',
             '|---|---:|---:|---:|---:|---:|---:|']
    tex = ['\\begin{tabular}{lrrrrr}', 'Method & Gap (dB) & N & Weak success & Both success & Time (s) \\\\']
    for r in summary:
        table.append(f"| {r['method']} | {r['gap_db']:g} | {r['n']} | {r['weak20']:.3f} | {r['both20']:.3f} | {r['weak_rmse']} | {r['seconds']:.3f} |")
        tex.append(f"{r['method']} & {r['gap_db']:g} & {r['n']} & {r['weak20']:.3f} & {r['both20']:.3f} & {r['seconds']:.3f} \\\\")
    tex.append('\\end{tabular}')
    (out/'summary_table.md').write_text('\n'.join(table), encoding='utf-8')
    (out/'summary_table.tex').write_text('\n'.join(tex), encoding='utf-8')
    cost = []
    for method in METHODS:
        for kind in ('mixed', 'weak', 'strong'):
            selected = [r for r in rows if r['method'] == method and r['kind'] == kind]
            if selected:
                item = dict(method=method, kind=kind, n=len(selected),
                            mean_seconds=float(np.mean([r['result']['seconds'] for r in selected])))
                for key in ('candidate_points', 'frequency_response_points', 'joint_evaluations',
                            'covariance_decompositions', 'iterations'):
                    item['mean_'+key] = float(np.mean([r['result']['counts'].get(key, 0) for r in selected]))
                cost.append(item)
    csv_write(out/'cost_table.csv', cost)
    lines = ['| Method | Input | n | time/s | frequency response points | joint evaluations | decompositions |',
             '|---|---|---:|---:|---:|---:|---:|']
    latex = ['\\begin{tabular}{llrrrrr}',
             'Method & Input & N & Time (s) & Responses & Joint & Decomp. \\\\']
    for r in cost:
        values = [r['method'], r['kind'], str(r['n']), f"{r['mean_seconds']:.3f}",
                  f"{r['mean_frequency_response_points']:.1f}", f"{r['mean_joint_evaluations']:.1f}",
                  f"{r['mean_covariance_decompositions']:.1f}"]
        lines.append('| '+' | '.join(values)+' |')
        latex.append(' & '.join(values)+' \\\\')
    latex.append('\\end{tabular}')
    (out/'cost_table.md').write_text('\n'.join(lines), encoding='utf-8')
    (out/'cost_table.tex').write_text('\n'.join(latex), encoding='utf-8')

    def draw_lines(axes, fields, title, time_axis=False):
        for ax, (field, ylabel) in zip(np.asarray(axes).ravel(), fields):
            for i, method in enumerate(METHODS):
                xx = sorted(set(r['budget'] for r in data if r['kind'] == 'mixed')) if time_axis else gaps
                yy, lower, upper = [], [], []
                for x in xx:
                    rr = [r for r in (data if time_axis else mixed) if r['method'] == method
                          and r['kind'] == 'mixed' and r['budget' if time_axis else 'gap_db'] == x]
                    if field.endswith('20'):
                        val, lo, hi = ci(rr, field)
                    elif 'error' in field:
                        val = summarize([r[field] for r in rr], True)
                        lo = hi = np.nan
                    else:
                        val = summarize([r[field] for r in rr])
                        lo = hi = np.nan
                    yy.append(np.nan if val is None else val)
                    lower.append(lo)
                    upper.append(hi)
                ax.plot(xx, yy, color=COLORS[i], linestyle=STYLES[i], marker=MARKERS[i], label=method, ms=4)
                if field.endswith('20'):
                    ax.fill_between(xx, lower, upper, color=COLORS[i], alpha=.08)
                    ax.set_ylim(-.03, 1.03)
            ax.set_xlabel('Time budget (s)' if time_axis else 'Power difference (dB)')
            ax.set_ylabel(ylabel)
            ax.grid(alpha=.2)
            nn = [len(set(r['group'] for r in mixed if r['method'] == method and r['gap_db'] == gap))
                  for method in METHODS for gap in gaps]
            nn = [n for n in nn if n]
            ax.set_title(f'n={min(nn)}–{max(nn)} layouts per gap' if min(nn) != max(nn)
                         else f'n={nn[0]} layouts per gap', fontsize=8)
        np.asarray(axes).ravel()[0].legend(fontsize=7, ncol=2)

    guard()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    draw_lines(axes, [('strong20', 'Strong success @20 m'), ('weak20', 'Weak success @20 m'),
                     ('both20', 'Both success @20 m')], 'success')
    export(out, '01_success_gap', fig, mixed, '功率差与定位成功率；阴影为按布局重采样的95%区间，样本数见CSV。',
           'Success versus power difference. Shading: 95% layout-bootstrap intervals; sample sizes in CSV.', source, label)
    guard()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    draw_lines(axes, [('strong_error', 'Strong RMSE (m)'), ('weak_error', 'Weak RMSE (m)'), ('gospa', 'Mean GOSPA (m)')], 'errors')
    export(out, '02_errors_gap', fig, mixed, '未截断RMSE与GOSPA；缺失输出覆盖率另列，不删除大误差。',
           'Untruncated RMSE and GOSPA. Output coverage is reported separately.', source, label)
    guard()
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), constrained_layout=True)
    draw_lines(axes, [('weak20', 'Weak success @20 m'), ('both20', 'Both success @20 m')], 'time', True)
    export(out, '03_reliability_time', fig, data, '共同时间预算下的可靠性，预处理与传输计入算法时间。',
           'Reliability at common time budgets, including preprocessing and transfers.', source, label)
    guard()
    for target in ('weak', 'strong'):
        fig, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
        cdf_rows = []
        for ax, gap in zip(axes.ravel(), gaps):
            for i, method in enumerate(METHODS):
                rr = [r for r in mixed if r['method'] == method and r['gap_db'] == gap]
                values = sorted(r[target+'_error'] for r in rr if r[target+'_error'] is not None)
                yy = np.arange(1, len(values)+1)/max(len(rr), 1)
                ax.step(values, yy, where='post', color=COLORS[i], linestyle=STYLES[i], label=method)
                cdf_rows += [dict(method=method, gap_db=gap, error_m=x, cdf=float(y), n=len(rr)) for x, y in zip(values, yy)]
            ax.axvline(20, color='gray', lw=.7)
            ax.set(xlabel='Position error (m)', ylabel='CDF', title=f'{gap:g} dB', ylim=(0, 1.02))
            ax.grid(alpha=.2)
        axes.ravel()[0].legend(fontsize=6, ncol=2)
        export(out, f'04_cdf_{target}', fig, cdf_rows, f'{target}源误差CDF；缺失输出保留在分母中。',
               f'{target.capitalize()}-source error CDF. Missing outputs remain in the denominator.', source, label)
    guard()
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    rr = [r for r in rows if r['kind'] == 'mixed']
    times = [[r['result']['seconds'] for r in rr if r['method'] == m] for m in METHODS]
    ax.boxplot([t if t else [np.nan] for t in times], tick_labels=METHODS)
    ax.set_ylabel('Complete solver time (s)')
    ax.tick_params(axis='x', rotation=30)
    export(out, '05_runtime', fig, mixed, '算法完整耗时分布，包括超时运行。',
           'Complete runtime distribution, including budget-exhausted runs.', source, label)
    guard()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    paired = []
    for k, target in enumerate(('weak', 'strong')):
        for i, method in enumerate(METHODS):
            delta = []
            for gap in gaps:
                mm = [r for r in data if r['method'] == method and r['kind'] == 'mixed'
                      and r['gap_db'] == gap and r['budget'] == 30]
                diffs = []
                for r in mm:
                    single = [s for s in data if s['method'] == method and s['kind'] == target
                              and s['group'] == r['group'] and s['budget'] == 30
                              and (target == 'weak' or s['gap_db'] == gap)]
                    if single:
                        d = single[0]['both20']-r[target+'20']
                        diffs.append(d)
                        paired.append(dict(method=method, group=r['group'], target=target, gap_db=gap, success_loss=d))
                delta.append(np.mean(diffs) if diffs else np.nan)
            axes[k].plot(gaps, delta, color=COLORS[i], linestyle=STYLES[i], marker=MARKERS[i], label=method)
        axes[k].set(xlabel='Power difference (dB)', ylabel='Isolated minus mixed success', title=target+' at 30 s')
    axes[0].legend(fontsize=7, ncol=2)
    export(out, '06_isolated_mixed', fig, paired, '相同30秒预算下，单独存在与混合后的配对成功率差。',
           'Paired isolated-minus-mixed success at the same 30 s budget.', source, label)
    render_maps(out, label)
    (out/'运行摘要.md').write_text(f'# G10 {label}\n\n已保存{len(rows)}条方法任务。'
        '\n\n回读 comparison_report.json、summary_table.csv、final_audit_report.json 和 figures/。'
        '\n\n空间图若尚未生成，不代表该环节已通过；正式状态以status.json与最终审计为准。\n', encoding='utf-8')


def spatial_maps(out, cfg, clock):
    from .physics import Model, Spectrum, grid
    points = grid(20.)
    folder = out/'maps'
    folder.mkdir(exist_ok=True)
    # Pre-registered, not selected by observed success/failure.
    for group, gap in ((0, 0), (0, 9), (0, 15)):
        scene_key = f"formal_{group}_0_{cfg['snr']:g}_{gap}_zero"
        scene = load_scene(out, scene_key)
        iq = scene['signal'].sum(0)+scene['noise']
        for method in METHODS:
            path = folder/f'{group}_{gap}_{method}.npz'
            if path.exists():
                continue
            row = read(out/'rows'/f'{group}_{gap}_mixed_{method}.json')
            pred = np.asarray(at_time(row['result'], 60)).reshape(-1, 2)
            model = Model(iq, clock.check)
            if method in ('MUSIC', 'MVDR'):
                settings = cfg['methods'][method]
                value = Spectrum(model, method, 2, settings['segments'], settings['loading']).evaluate(points)
                meaning = 'Native pooled MUSIC/Capon spectrum'
            elif method == 'SBL':
                native = row['result'].get('native_map')
                if native is None:
                    raise RuntimeError('SBL native posterior map missing')
                with path.open('xb') as stream:
                    np.savez_compressed(stream, points=np.asarray(native['points']), values=np.asarray(native['power']),
                                        truth=scene['positions'], predicted=pred)
                write(path.with_suffix('.json'), dict(meaning='SBL native posterior mean coefficient power',
                    input_sha=row['input_sha'], fixed_position=[], scatter=True))
                continue
            else:
                value = model.conditional(points, pred[:1] if len(pred) == 2 and method != 'LS' else ())
                meaning = 'Conditional LS gain; first estimated location fixed' if method != 'LS' and len(pred) == 2 else 'Single-source LS spectrum'
            with path.open('xb') as stream:
                np.savez_compressed(stream, points=points, values=value, truth=scene['positions'], predicted=pred)
            write(path.with_suffix('.json'), dict(meaning=meaning, input_sha=row['input_sha'],
                                                fixed_position=pred[:1].tolist() if method not in ('LS', 'MUSIC', 'MVDR') else []))


def render_maps(out, label):
    for path in sorted((out/'maps').glob('*.npz')) if (out/'maps').exists() else []:
        meta = read(path.with_suffix('.json'))
        with np.load(path, allow_pickle=False) as z:
            points, value, truth, predicted = z['points'], z['values'], z['truth'], z['predicted']
        side = int(np.sqrt(len(points)))
        fig, ax = plt.subplots(figsize=(5.3, 4.5), constrained_layout=True)
        image = (ax.scatter(points[:, 0], points[:, 1], c=value, s=9, cmap='viridis') if meta.get('scatter') else
                 ax.imshow(value.reshape(side, side), extent=(-1000, 1000, -1000, 1000), origin='lower', cmap='viridis'))
        ax.scatter(truth[:, 0], truth[:, 1], marker='*', c='white', edgecolors='black', s=100, label='Truth')
        if len(predicted):
            ax.scatter(predicted[:, 0], predicted[:, 1], marker='x', c='#D55E00', s=70, label='Estimate')
        ax.set(xlabel='x (m)', ylabel='y (m)', title=path.stem)
        ax.legend()
        fig.colorbar(image, ax=ax, label='Native score (relative units)')
        data = [dict(x_m=float(p[0]), y_m=float(p[1]), score=float(v)) for p, v in zip(points, value)]
        fixed = str(meta.get('fixed_position', []))
        caption = meta['meaning']+'; fixed estimated coordinates: '+fixed
        translations = {'Native pooled MUSIC/Capon spectrum': '该方法原生宽带MUSIC/Capon空间谱',
                        'SBL native posterior mean coefficient power': 'SBL原生后验均值系数功率',
                        'Conditional LS gain; first estimated location fixed': '条件LS增益切片，固定第一估计位置',
                        'Single-source LS spectrum': '单源LS空间谱'}
        caption_cn = translations.get(meta['meaning'], meta['meaning'])+'；固定的估计坐标（m）：'+fixed
        export(out, 'map_'+path.stem, fig, data, caption_cn, caption,
               {str(path.relative_to(out)): sha(path), str(path.with_suffix('.json').relative_to(out)): sha(path.with_suffix('.json'))}, label)


def center_figure(out, rows):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    for ax, kind in zip(axes, ('weak', 'mixed')):
        for i, method in enumerate(('B', 'D')):
            values = [np.mean([r['metrics']['success']['20'][-1] for r in rows
                       if r['kind'] == kind and r['method'] == method and r['center'] == center])
                      for center in ('random', 'zero')]
            ax.plot([0, 1], values, marker=MARKERS[i], label=method)
        ax.set(xticks=[0, 1], xticklabels=['Random centre', 'Zero centre'], ylim=(-.03, 1.03),
               ylabel='Weak success @20 m', title=kind)
        ax.legend()
    data = [dict(group=r['group'], center=r['center'], kind=r['kind'], method=r['method'],
                 weak_error_m=r['metrics']['errors_m'][-1], success20=r['metrics']['success']['20'][-1], seconds=r['seconds']) for r in rows]
    source = {str(p.relative_to(out)): sha(p) for p in (out/'rows').glob('*.json')}
    export(out, 'center_pair', fig, data, '原随机中心与零中心配对开发比较；D单源与B相同，不当作额外独立重复。',
           'Paired centre-frequency development comparison; D and B share the single-source solver.', source, 'DEVELOPMENT')
