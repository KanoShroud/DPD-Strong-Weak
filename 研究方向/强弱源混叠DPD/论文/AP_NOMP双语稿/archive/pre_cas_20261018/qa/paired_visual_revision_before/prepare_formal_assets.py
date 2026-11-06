"""Paper-only export of audited R4 results. No IQ, solver, Torch or GPU access."""
from __future__ import annotations

import csv
from itertools import combinations, permutations
import json

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.optimize import linear_sum_assignment

import prepare_assets as art

HERE, ROOT = art.HERE, art.ROOT
RUN = art.BASE / 'r4/formal_recovered_20261017_093236_136920'
TIME = art.BASE / 'r4/paper_evidence_20261017_181729_643730'
CASES = ROOT / 'outputs_e2e/reports/g10_r4_supplement_review_20261017'
CORE = ('AP_1_ALT', 'AP_1_JOINT', 'AP_2_JOINT', 'AP_5_JOINT', 'NOMP', 'NOMP_JOINT')
PAN = ('LS', 'MUSIC', 'MVDR', 'AP_FULL', 'BNOMP_REF', 'NOMP',
       'MULTISTART_FIXED', 'GA', 'SBL', 'AP_5_JOINT')
# Distinct NPG-inspired hues; line/marker encoding also distinguishes variants.
COLORS = {'AP_1_ALT': '#8491B4', 'AP_1_JOINT': '#4DBBD5', 'AP_2_JOINT': '#7E6148',
          'AP_5_JOINT': '#3C5488', 'NOMP': '#E64B35', 'NOMP_JOINT': '#F39B7F',
          'LS': '#666666', 'MUSIC': '#00A087', 'MVDR': '#7E6148',
          'AP_FULL': '#8491B4', 'BNOMP_REF': '#B09C85',
          'MULTISTART_FIXED': '#925E9F', 'GA': '#D59A35', 'SBL': '#4DBBD5'}
LABELS = {'AP_1_ALT': 'AP-1 ALT', 'AP_1_JOINT': 'AP-1 joint', 'AP_2_JOINT': 'AP-2 joint',
          'AP_5_JOINT': 'MCJ-AP (M=5)', 'NOMP': 'VP-BNOMP', 'NOMP_JOINT': 'VP-BNOMP + joint',
          'LS': 'LS peaks', 'MUSIC': 'WB-MUSIC', 'MVDR': 'MVDR-known',
          'AP_FULL': 'Full AP', 'BNOMP_REF': 'BNOMP-REF',
          'MULTISTART_FIXED': 'Multistart ML', 'GA': 'GA-ML', 'SBL': 'WB-SBL'}
ZH = {'AP_1_ALT': '单候选交替AP', 'AP_1_JOINT': '单候选联合AP',
      'AP_2_JOINT': '双候选联合AP', 'AP_5_JOINT': '五候选联合AP',
      'NOMP': 'VP-BNOMP', 'NOMP_JOINT': 'VP-BNOMP＋联合',
      'LS': 'LS谱取峰', 'MUSIC': '宽带MUSIC', 'MVDR': 'MVDR-known',
      'AP_FULL': '充分AP', 'BNOMP_REF': '固定系数BNOMP',
      'MULTISTART_FIXED': '多初值联合ML', 'GA': 'GA联合ML', 'SBL': '宽带SBL'}
CHECKS = {}


def read_json(path):
    return art.read(path)


def read_csv(path):
    path = path.resolve(strict=True)
    if not path.is_relative_to(ROOT / 'outputs_e2e'):
        raise ValueError('CSV outside approved output root')
    art.PROVENANCE[str(path.relative_to(ROOT))] = art.digest(path)
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def matching(truth, predicted):
    d = np.linalg.norm(np.asarray(truth)[:, None] - np.asarray(predicted)[None], axis=-1)
    if d.shape != (2, 2) or not np.isfinite(d).all():
        raise AssertionError('Expected two finite sources')
    ri, ci = linear_sum_assignment(d**2)
    errors = np.empty(2)
    errors[ri] = d[ri, ci]
    choices = [()]
    for size in (1, 2):
        choices += [tuple(zip(r, c)) for r in combinations(range(2), size)
                    for c in permutations(range(2), size)]
    valid = [a for a in choices if all(d[i, j] <= 20 for i, j in a)]
    chosen = min(valid, key=lambda a: (-len(a), sum(d[i, j]**2 for i, j in a), a))
    flags = [any(i == j for j, _ in chosen) for i in range(2)]
    gospa = np.sqrt(min(sum(d[i, j]**2 for i, j in a) + 5000*(4-2*len(a)) for a in choices))
    return errors, flags, float(gospa)


def summarize(rows):
    e = np.asarray([r['errors'] for r in rows])
    return dict(n=len(rows), both_count=sum(all(r['flags']) for r in rows),
                strong_rmse_m=float(np.sqrt(np.mean(e[:, 0]**2))),
                weak_rmse_m=float(np.sqrt(np.mean(e[:, 1]**2))),
                weak_p95_m=float(np.quantile(e[:, 1], .95)),
                mean_seconds=float(np.mean([r['seconds'] for r in rows])),
                p95_seconds=float(np.quantile([r['seconds'] for r in rows], .95)),
                gospa_m=float(np.mean([r['gospa'] for r in rows])))


def load_evidence():
    audit = read_json(RUN / 'final_audit_report.json')
    contract = read_json(RUN / 'contract.json')
    report = read_json(RUN / 'reports/sbl/comparison_report.json')
    if not audit['passed'] or audit['solves'] != 16560 or not audit['metric_replay']:
        raise AssertionError('Incomplete formal audit')
    rows, details, seen = [], {}, {}
    for name in sorted(audit['rows']):
        method = next((m for m in sorted(set(CORE+PAN), key=len, reverse=True)
                       if name.endswith('_'+m+'.json')), None)
        if method is None:
            continue  # PROFILE is deliberately reserved, not promoted here.
        row = art.row_read(RUN / 'rows' / name, audit)
        task = row['job']['task']
        key = tuple(task[:3])
        identity = row['input_identity']
        tag = '_'.join(map(str, key))
        if identity != audit['inputs'][tag]['identity'] or identity != seen.setdefault(key, identity):
            raise AssertionError('Unpaired input identity')
        e, flags, gospa = matching(row['truth'], row['result']['positions'])
        if (not np.allclose(e, row['metrics']['errors_m'], atol=1e-9, rtol=1e-9)
                or flags != row['metrics']['success']['20']
                or not np.isclose(gospa, row['metrics']['gospa_m'])):
            raise AssertionError('Independent position metric replay failed')
        r = row['result']
        flat = dict(key=key, method=method, layout=key[0], realization=key[1], gap_db=key[2],
                    errors=e, flags=flags, gospa=gospa, seconds=r['complete_seconds'],
                    stop=r['stop_reason'], counts=r['counts'], objective=r['joint_objective'])
        rows.append(flat)
        if key[1] in (400, 410):
            details[key, method] = row
    # Formal input IDs and full matching summary are independently reconstructed.
    for scope, methods in [('core/all', CORE), ('panorama/all', PAN)]:
        for method in methods:
            subset = [r for r in rows if r['method'] == method
                      and (scope == 'core/all' or r['realization'] in (400, 410))]
            if len(subset) != (2400 if scope == 'core/all' else 240):
                raise AssertionError('Wrong formal case count')
            expected = next(s for s in report['summaries'] if s['method'] == method and s['scope'] == scope)
            for k, v in summarize(subset).items():
                if not np.isclose(v, expected[k], atol=1e-8, rtol=1e-8):
                    raise AssertionError((scope, method, k))
    CHECKS.update(row_metrics_replayed=len(rows), input_identity_pairs=len(seen),
                  formal_summary_replay=True, original_audit_passed=True,
                  original_solves=16560, no_iq_loaded=True, no_solver_imported=True)
    print(f'正式结果身份及评价重放通过：{len(rows)}行，{len(seen)}条输入。', flush=True)
    return rows, details, contract, report


def export(name, fig, rows, zh, en, selection):
    art.figure_save(name, fig, rows, zh, en, selection)


def tex_table(name, caption, label, header, lines, columns):
    text = '\\begin{table}[htbp]\\centering\\small\n'
    text += '\\caption{'+caption+'}\\label{'+label+'}\n'
    text += '\\begin{tabular}{@{}'+columns+'@{}}\\toprule\n'
    text += ' & '.join(header)+r'\\\midrule'+'\n'
    text += '\n'.join(' & '.join(line)+r'\\' for line in lines)
    text += '\n\\bottomrule\\end{tabular}\\end{table}\n'
    (HERE / 'sections' / (name+'.tex')).write_text(text, encoding='utf-8')


def formal_tables(rows):
    summaries = []
    for scope, methods in [('core', CORE), ('common', PAN)]:
        ss = []
        for method in methods:
            sub = [r for r in rows if r['method'] == method
                   and (scope == 'core' or r['realization'] in (400, 410))]
            ss.append(dict(method=method, scope=scope, **summarize(sub)))
        summaries += ss
        lines = [[ZH[s['method']], f"{s['both_count']}/{s['n']}",
                  f"{100*s['both_count']/s['n']:.2f}", f"{s['strong_rmse_m']:.3f}",
                  f"{s['weak_rmse_m']:.3f}", f"{s['weak_p95_m']:.3f}",
                  f"{s['mean_seconds']:.4f}"] for s in ss]
        tex_table('generated_formal_'+scope,
                  ('全部2400条观测的核心比较。' if scope == 'core' else '同一240条观测上的基线与提出方法比较。')
                  +'成功门槛20\\,m；RMSE和P95单位为m，时间单位为s。',
                  'tab:formal'+scope,
                  ['方法', '双源成功', '成功率/\\%', '强源RMSE', '弱源RMSE', '弱源P95', '时间'],
                  lines, 'lrrrrrr')
    art.table_csv(HERE / 'sources/formal_summary.csv', summaries)
    aux = []
    for method in PAN:
        sub = [r for r in rows if r['method'] == method and r['realization'] in (400, 410)]
        summary = summarize(sub)
        stops = {s: sum(r['stop'] == s for r in sub) for s in sorted({r['stop'] for r in sub})}
        aux.append(dict(method=method, **summary, stops=json.dumps(stops, ensure_ascii=False)))
    art.table_csv(HERE / 'sources/formal_auxiliary_metrics.csv', aux)
    tex_table('generated_formal_auxiliary', '共同240条观测的辅助指标与停止状态。', 'tab:auxiliary',
              ['方法', 'GOSPA/m', '耗时P95/s', '达到迭代上限'],
              [[ZH[s['method']], f"{s['gospa_m']:.3f}", f"{s['p95_seconds']:.3f}",
                str(sum(v for k, v in json.loads(s['stops']).items() if 'LIMIT' in k))] for s in aux], 'lrrr')
    ss = []
    for method in ('AP_1_ALT', 'AP_1_JOINT', 'AP_2_JOINT', 'AP_5_JOINT'):
        sub = [r for r in rows if r['method'] == method]
        s = dict(method=method, **summarize(sub))
        s.update(joint_evaluations=np.mean([r['counts']['joint_evaluations'] for r in sub]),
                 gradient_calls=np.mean([r['counts'].get('gradient_calls', 0) for r in sub]))
        ss.append(s)
    tex_table('generated_formal_ablation', 'AP候选数与精修方式的消融，全部2400条观测。', 'tab:apablation',
              ['配置', '双源成功', '时间/s', '联合目标计算', '梯度计算'],
              [[ZH[s['method']], str(s['both_count']), f"{s['mean_seconds']:.4f}",
                f"{s['joint_evaluations']:.2f}", f"{s['gradient_calls']:.2f}"] for s in ss], 'lrrrr')
    art.table_csv(HERE / 'sources/formal_ablation.csv', ss)
    a = {r['key']: r for r in rows if r['method'] == 'AP_1_JOINT'}
    b = {r['key']: r for r in rows if r['method'] == 'AP_5_JOINT'}
    rescued = [k for k in a if not all(a[k]['flags']) and all(b[k]['flags'])]
    harmed = [k for k in a if all(a[k]['flags']) and not all(b[k]['flags'])]
    CHECKS.update(rescued=len(rescued), harmed=len(harmed),
                  rescued_layouts=sorted({k[0] for k in rescued}))
    art.save_json(HERE / 'sources/candidate_effect.json',
                  dict(rescued=rescued, harmed=harmed, success_threshold_m=20))


def geometry(contract):
    fig, axes = plt.subplots(2, 2, figsize=(6.7, 5.8), layout='constrained')
    stations = np.array([[500, 0], [0, 500], [-500, 0], [0, -500], [500, 0]])
    names = ['Both inside', 'A outside, B inside', 'A inside, B outside', 'Both outside']
    data = []
    for cat, ax in enumerate(axes.flat):
        ax.plot(*stations.T, color='.6', lw=.8)
        ax.scatter(*stations[:4].T, marker='^', c='black', s=25, label='Receiver')
        for layout in [x for x in contract['layouts'] if x['category'] == cat]:
            p = np.array(layout['positions'])
            ax.plot(*p.T, color='.85', lw=.6, zorder=0)
            for k, (marker, color) in enumerate([('o', '#E64B35'), ('s', '#3C5488')]):
                ax.scatter(*p[k], marker=marker, c=color, s=25,
                           label=('Source A' if k == 0 else 'Source B') if layout['id'] % 5 == 0 else None)
                data.append(dict(layout=layout['id'], category=cat, source=k, x_m=p[k, 0], y_m=p[k, 1]))
            offset = {0: (4, -12), 2: (9, 6), 7: (-14, 7), 8: (8, -10)}.get(layout['id'], (4, 4))
            ax.annotate(str(layout['id']+1), p[1], xytext=offset, textcoords='offset points', fontsize=7)
        ax.set(title=f'({chr(97+cat)}) {names[cat]}', xlim=(-1000, 1000), ylim=(-1000, 1000),
               xticks=[-1000, 0, 1000], yticks=[-1000, 0, 1000], aspect='equal')
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    axes[0, 0].legend(loc='upper left', fontsize=6.5)
    export('formal_geometry', fig, data, '20个正式源站布局，每类5个，数字为布局编号。',
           'All 20 formal geometries, five per category; numbers identify layouts.', 'All preregistered layouts.')


def lines(ax, data, methods, field='success', label=True):
    styles, markers = ('-', '--', '-.', ':'), ('o', 's', '^', 'D', 'v', 'P', 'X', '>')
    out = []
    for j, method in enumerate(methods):
        sub = [r for r in data if r['method'] == method]
        ys = []
        for gap in (0, 3, 6, 9, 12, 15):
            s = summarize([r for r in sub if r['gap_db'] == gap])
            y = 100*s['both_count']/s['n'] if field == 'success' else s['weak_rmse_m']
            ys.append(y); out.append(dict(method=method, gap_db=gap, n=s['n'], value=y, field=field))
        ax.plot([0, 3, 6, 9, 12, 15], ys, color=COLORS[method], ls=styles[j % 4],
                marker=markers[j % 8], ms=4, label=LABELS[method] if label else None)
    ax.set_xlabel('Received power gap (dB)'); ax.set_xticks([0, 3, 6, 9, 12, 15])
    return out


def performance(rows):
    methods = ['AP_1_JOINT', 'AP_5_JOINT', 'NOMP', 'NOMP_JOINT']
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.1), layout='constrained')
    data = lines(axes[0], rows, methods)
    data += lines(axes[1], rows, methods, 'rmse')
    axes[0].set(ylabel='Both-source success (%)', ylim=(75, 101))
    axes[1].set(ylabel='Weak-source RMSE (m)', yscale='log')
    axes[0].legend(fontsize=7, loc='lower left')
    export('formal_core', fig, data, '2400条核心观测的功率差扫描，每档400条。',
           'Core power-gap scan: 400 observations per gap, 2400 in total.', 'All core observations; fixed parameters.')
    common = [r for r in rows if r['realization'] in (400, 410)]
    methods = ['LS', 'MUSIC', 'MVDR', 'AP_FULL', 'NOMP', 'MULTISTART_FIXED', 'GA', 'SBL', 'AP_5_JOINT']
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.8), layout='constrained')
    data = []
    for ax, group in zip(axes, [methods[:3]+['AP_5_JOINT'], methods[3:]]):
        data += lines(ax, common, group)
        ax.set(ylabel='Both-source success (%)', ylim=(-3, 105))
        ax.legend(fontsize=6.5, loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=2, frameon=False)
    export('formal_common', fig, data, '共同240条观测上的方法比较，每档40条。',
           'Method comparison on exactly the same 240 observations, 40 per gap.', 'Fixed common subset: realizations 400/410 in every layout.')
    methods = ['AP_1_JOINT', 'AP_5_JOINT', 'NOMP']
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2), layout='constrained')
    data = []
    for i, m in enumerate(methods):
        sub = [r for r in rows if r['method'] == m]
        e = np.sort([r['errors'][1] for r in sub]); yy = np.arange(1, len(e)+1)/len(e)
        axes[0].step(np.maximum(e, 1e-6), yy, where='post', color=COLORS[m], ls=['--', '-', '-.'][i], label=LABELS[m])
        data.extend(dict(panel='cdf', method=m, x=float(x), y=float(y), n=len(e)) for x, y in zip(e, yy))
    axes[0].axvline(20, color='.4', lw=.8, ls=':')
    axes[0].set(xscale='log', xlabel='Weak-source error (m)', ylabel='Empirical CDF', ylim=(0, 1.02))
    axes[0].legend(fontsize=6.5, loc='lower right')
    offsets = {'AP_FULL': (-8, 8), 'SBL': (-4, -14), 'AP_5_JOINT': (-15, 13),
               'NOMP': (-3, -20), 'BNOMP_REF': (4, 5), 'MUSIC': (4, 5),
               'MVDR': (4, -12), 'LS': (4, -12), 'MULTISTART_FIXED': (4, 5), 'GA': (4, 5)}
    for i, m in enumerate(PAN):
        s = summarize([r for r in common if r['method'] == m])
        x, y = s['mean_seconds'], 100*s['both_count']/s['n']
        axes[1].scatter(x, y, color=COLORS[m], marker=['o', 's', '^', 'D', 'v'][i % 5], s=30)
        axes[1].annotate(LABELS[m], (x, y), xytext=offsets[m], textcoords='offset points', fontsize=6,
                         arrowprops=dict(arrowstyle='-', color='.65', lw=.5))
        data.append(dict(panel='cost', method=m, x=x, y=y, n=s['n']))
    axes[1].set(xscale='log', xlabel='Mean complete time (s)', ylabel='Both-source success (%)',
                ylim=(5, 110), xlim=(.14, 110))
    export('formal_cdf_cost', fig, data, '左为核心2400条的误差CDF，右为共同240条的可靠性—完整耗时。',
           'Left: core error CDF (n=2400); right: common-subset reliability versus complete runtime (n=240).',
           'Panels use separate, labeled sample scopes; no cross-scope ranking.')


def layouts_and_ablation(rows):
    methods = ['AP_1_JOINT', 'AP_2_JOINT', 'AP_5_JOINT', 'NOMP']
    matrix, data = [], []
    for m in methods:
        rates = []
        for g in range(20):
            sub = [r for r in rows if r['method'] == m and r['layout'] == g]
            s = summarize(sub); y = 100*s['both_count']/len(sub)
            rates.append(y)
            data.append(dict(method=m, layout=g, success_percent=y, **s))
        matrix.append(rates)
    fig, ax = plt.subplots(figsize=(7.1, 2.3), layout='constrained')
    im = ax.imshow(matrix, aspect='auto', cmap='Blues', vmin=0, vmax=100)
    ax.set_yticks(range(4), [LABELS[m] for m in methods]); ax.set_xticks(range(20), range(1, 21))
    ax.set_xlabel('Geometry index (120 observations per geometry)')
    for j in (4.5, 9.5, 14.5):
        ax.axvline(j, color='white', lw=1)
    for i in range(4):
        for j in range(20):
            if matrix[i][j] < 100:
                ax.text(j, i, f'{matrix[i][j]:.1f}', ha='center', va='center',
                        color='white' if matrix[i][j] >= 65 else 'black', fontsize=7)
    fig.colorbar(im, ax=ax, label='Both-source success (%)', shrink=.8)
    export('formal_layouts', fig, data, '各布局双源成功率，每布局120条；空白数字位置均为100%。',
           'Both-source success by geometry (120 observations each); unnumbered cells are 100%.', 'All 20 geometries; no role-based split.')
    methods = ['AP_1_ALT', 'AP_1_JOINT', 'AP_2_JOINT', 'AP_5_JOINT']
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.9), layout='constrained')
    data = []
    for m in methods:
        sub = [r for r in rows if r['method'] == m]
        data.append(dict(method=m, success=100*np.mean([all(r['flags']) for r in sub]),
                         seconds=np.mean([r['seconds'] for r in sub]),
                         gradients=np.mean([r['counts']['gradient_calls'] for r in sub])))
    for ax, field, ylabel in zip(axes, ['success', 'seconds', 'gradients'],
                                  ['Both-source success (%)', 'Mean complete time (s)', 'Mean gradient evaluations']):
        ax.bar(range(4), [r[field] for r in data], color=[COLORS[m] for m in methods], width=.65)
        ax.set_xticks(range(4), ['1 ALT', '1 joint', '2 joint', '5 joint'], rotation=35, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(r[field] for r in data)*1.16)
    export('formal_ablation', fig, data, '候选数与精修方式对可靠性、耗时和梯度计算的影响。',
           'Candidate count and refinement strategy: reliability, runtime and gradient evaluations.', 'All 2400 observations; five-candidate ALT/joint experiment excluded.')


def time_curves(details):
    source = read_csv(TIME / 'time_curves.csv')
    methods = ['AP_5_JOINT', 'AP_FULL', 'NOMP', 'SBL']
    selected = [r for r in source if r['method'] in methods]
    # Independently replay the published events, never choose by ground truth.
    for record in selected:
        method, t = record['method'], float(record['seconds'])
        results = []
        for (key, m), row in details.items():
            if m != method:
                continue
            available = [e for e in row['result']['report_events'] if e['seconds'] <= t]
            if not available:
                results.append((False, False, None))
            else:
                event = max(available, key=lambda e: e['seconds'])
                errors, flags, _ = matching(row['truth'], event['positions'])
                results.append((True, all(flags), errors))
        if len(results) != 240 or sum(x[1] for x in results) != int(record['success_count']):
            raise AssertionError('Time-success replay mismatch')
        if sum(x[0] for x in results) != int(record['output_count']):
            raise AssertionError('Time-coverage replay mismatch')
        if all(x[0] for x in results):
            rmse = np.sqrt(np.mean(np.asarray([x[2] for x in results])**2))
            if not np.isclose(rmse, float(record['full_coverage_rmse_m'])):
                raise AssertionError('Time-RMSE replay mismatch')
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.8), layout='constrained')
    for j, method in enumerate(methods):
        sub = sorted([r for r in selected if r['method'] == method], key=lambda r: float(r['seconds']))
        for ax, field, ylabel in zip(axes, ['success_rate', 'coverage', 'full_coverage_rmse_m'],
                                    ['Both-source success (%)', 'Complete output (%)', 'Position RMSE (m)']):
            x = [float(r['seconds']) for r in sub]
            y = [float(r[field])*(100 if field != 'full_coverage_rmse_m' else 1) if r[field] else np.nan for r in sub]
            ax.plot(x, y, color=COLORS[method], ls=['-', '--', '-.', ':'][j],
                    marker=['o', 's', '^', 'D'][j], ms=3, label=LABELS[method])
            ax.set(xscale='log', xlabel='Elapsed time (s)', ylabel=ylabel)
            if field != 'full_coverage_rmse_m':
                ax.set_ylim(-3, 105)
            else:
                ax.set_yscale('log')
    axes[0].legend(loc='lower right', fontsize=6)
    export('formal_time', fig, selected, '共同240条观测的已保存输出轨迹重放；无完整输出按未成功计，RMSE仅在全部有输出时显示。',
           'Replay of saved outputs on 240 common observations; missing outputs count as unsuccessful, RMSE requires full coverage.',
           'Fixed time checkpoints, latest available output, no oracle selection or new early-stopping program.')
    CHECKS['time_checkpoints_replayed'] = len(selected)


def candidate_cases(details):
    # Existing CPU-reconstructed spectra only; do not recompute DPD or read IQ.
    audit_path = CASES / 'audit.json'
    art.PROVENANCE[str(audit_path.relative_to(ROOT))] = art.digest(audit_path)
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    if not audit['passed'] or not audit['no_optimizer_called']:
        raise AssertionError('Case reconstruction audit failed')
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 5.2), layout='constrained')
    plotted = []
    for i, realization in enumerate([400, 405]):
        tag = f'3_{realization}_0'
        spectra = read_csv(CASES / (tag+'_spectra.csv'))
        branches = read_csv(CASES / (tag+'_branches.csv'))
        row = details.get(((3, realization, 0), 'AP_5_JOINT'))
        if row is None:
            formal_audit = read_json(RUN / 'final_audit_report.json')
            row = art.row_read(RUN / 'rows' / (tag+'_AP_5_JOINT.json'), formal_audit)
        truth = np.asarray(row['truth']); bb = row['result']['mechanism']['branches']
        for a, b in zip(branches, bb):
            if not np.isclose(float(a['final_objective']), b['objective'], atol=1e-12):
                raise AssertionError('Case branch CSV identity mismatch')
        spectral_map = {(float(s['x_m']), float(s['y_m'])): float(s['conditional_score']) for s in spectra}
        first_j = row['result']['mechanism']['first_optimizer']['after']
        for b in bb:
            point = tuple(b['initial_positions'][1])
            if not np.isclose(first_j-spectral_map[point], b['initial_objective'], atol=2e-12, rtol=0):
                raise AssertionError('Saved spectrum does not match branch conditional gain')
        x = np.array([float(r['x_m']) for r in spectra]).reshape(101, 101)
        y = np.array([float(r['y_m']) for r in spectra]).reshape(101, 101)
        for j, field in enumerate(['single_source_ls', 'conditional_score']):
            z = np.array([float(r[field]) for r in spectra]).reshape(101, 101)
            ax = axes[i, j]
            im = ax.pcolormesh(x, y, z, cmap='viridis', shading='auto',
                               norm=LogNorm(vmin=max(z.min(), 1e-8), vmax=z.max()))
            ax.scatter(*truth[0], c='#E64B35', marker='*', s=65, edgecolors='white', lw=.5)
            ax.scatter(*truth[1], c='white', marker='*', s=65, edgecolors='black', lw=.5)
            if j == 1:
                for b in bb:
                    p = np.asarray(b['initial_positions'])[1]
                    ax.scatter(*p, facecolors='none', edgecolors='white', s=24, marker='o')
                    ax.annotate(str(b['rank']), p, xytext=(4, 3), textcoords='offset points', color='white', fontsize=6,
                                bbox=dict(boxstyle='round,pad=.1', fc='black', ec='none', alpha=.45))
            ax.set(xlim=(-1000, 1000), ylim=(-1000, 1000), aspect='equal',
                   xlabel='x (m)', ylabel='y (m)', title=('Single-source LS' if j == 0 else 'Conditional spectrum'))
            ax.set_xticks([-1000, 0, 1000]); ax.set_yticks([-1000, 0, 1000])
            fig.colorbar(im, ax=ax, fraction=.045, pad=.02).ax.tick_params(labelsize=6)
        ax = axes[i, 2]
        initial = [float(r['initial_objective']) for r in branches]
        final = [float(r['final_objective']) for r in branches]
        ax.plot(range(1, 6), initial, '--o', color='#4DBBD5', ms=3, label='Before refinement')
        ax.plot(range(1, 6), final, '-s', color='#E64B35', ms=3, label='After refinement')
        ax.set(xlabel='Candidate rank', ylabel='Normalized residual J', xticks=range(1, 6),
               title='Recovered' if i == 0 else 'Not recovered')
        for a, b in zip(branches, bb):
            error, _, _ = matching(truth, b['positions'])
            plotted.append(dict(case=tag, rank=b['rank'], initial_objective=float(a['initial_objective']),
                                final_objective=b['objective'], source_b_error_m=float(error[1])))
        if i == 0:
            ax.legend(fontsize=6)
        plotted.extend(dict(case=tag, **r) for r in spectra)
    # CSV has two record types with an explicit union schema.
    keys = list(dict.fromkeys(k for r in plotted for k in r))
    plotted = [{k: r.get(k, '') for k in keys} for r in plotted]
    export('formal_candidates', fig, plotted, '同一困难构型的两个0dB案例：单源谱、条件谱及精修前后候选残差；上行恢复，下行未恢复。',
           'Two equal-power post-hoc cases: single-source/conditional spectra and branch residuals; recovered above, not recovered below.',
           'Post-hoc geometry 3, realizations 400/405, 0 dB; A/B labels at equal received powers. Arrows/paths not inferred.')


def prepare(offline=True):
    art.PROVENANCE.clear(); art.FIGURES.clear()
    art.setup(); art.bibliography(offline); art.parameters()
    plt.rcParams.update({'axes.prop_cycle': plt.cycler(color=list(COLORS.values())), 'lines.linewidth': 1.6})
    rows, details, contract, report = load_evidence()
    formal_tables(rows); geometry(contract); performance(rows); layouts_and_ablation(rows)
    time_curves(details); candidate_cases(details)
    # Preserve supplementary uncertainty and work counts without importing a solver.
    art.save_json(HERE / 'sources/formal_statistical_report.json', report)
    operations = read_csv(RUN / 'reports/sbl/operation_counts.csv')
    art.table_csv(HERE / 'sources/formal_operation_counts.csv', operations)
    art.save_json(HERE / 'sources/source_manifest.json', dict(
        revision='FORMAL_R4_PAPER_20261017',
        restriction='Completed R4 only. Pair experiment and PROFILE results intentionally not included. No IQ/solver/GPU.',
        files=art.PROVENANCE, figures=art.FIGURES, palette=COLORS))
    art.save_json(HERE / 'qa/formal_asset_checks.json', dict(passed=True, **CHECKS,
        figures=len(art.FIGURES), pending=['five_candidate_alt_joint_pair', 'final_profile_version'],
        source_files=len(art.PROVENANCE)))
    print(f'已导出{len(art.FIGURES)}幅正式图；仅读取完成结果，未运行定位。', flush=True)


if __name__ == '__main__':
    prepare()
