"""Build paper assets from completed JSON results only; never import a solver.

Writes solely inside this paper project. No IQ loading, training, localization,
GPU access or modification of experiment evidence is performed.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import ssl
import urllib.parse
import urllib.request

import certifi
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
BASE = ROOT / 'outputs_e2e/unified/e2e_g10'
RUNS = {
    'band': BASE / 'r2/band/independent_20261016_151236',
    'paired': BASE / 'r3/independent_20261016_210849_013753',
    'top5': BASE / 'r3/top5_20261016_215603_777041',
    'mechanism': BASE / 'r4/adaptation_review_20261017_011824_276518',
}
RELEASE = BASE / 'r4/formal_ready_io_20261017_093236_002611/contract.json'
DOIS = {
    'KnappCarter1976': '10.1109/TASSP.1976.1162830',
    'ChanHo1994': '10.1109/78.301830',
    'ChenHudsonYao2002': '10.1109/TSP.2002.800420',
    'WeissAmar2005': '10.1155/ASP.2005.37',
    'Capon1969': '10.1109/PROC.1969.7278',
    'TirerWeiss2016': '10.1109/LSP.2015.2503921',
    'Mamandipoor2016': '10.1109/TSP.2016.2580523',
    'Yang2022': '10.1016/j.jsv.2021.116549',
    'Zeng2025': '10.1109/ACCESS.2025.3645128',
    'GolubPereyra1973': '10.1137/0710036',
    'Byrd1995': '10.1137/0916069',
    'Nannuru2019': '10.1016/j.sigpro.2019.02.003',
    'Rahmathullah2017': '10.23919/ICIF.2017.8009645',
    'Kuhn1955': '10.1002/nav.3800020109',
}
# Muted categorical colors; related AP variants share a blue family.
COLORS = ['#4E718D', '#B0846A', '#6E8B80', '#8E7E9B', '#A49B74']
METHOD_COLORS = {'MUSIC': '#B0846A', 'MVDR': '#8D929B',
                 'AP_1_ALT': '#A6B6C5', 'AP_1_JOINT': '#66859D',
                 'AP_5_JOINT': '#3F5E75', 'NOMP': '#819A8C'}
PROVENANCE: dict[str, str] = {}
FIGURES: dict[str, dict] = {}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    path = path.resolve(strict=True)
    if not path.is_relative_to(BASE):
        raise ValueError(f'Unregistered evidence root: {path}')
    PROVENANCE[str(path.relative_to(ROOT))] = digest(path)
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def table_csv(path: Path, rows: list[dict]):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def row_read(path: Path, audit: dict):
    row = read(path)
    payload = {k: v for k, v in row.items() if k != 'record_sha'}
    seal = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
    if row.get('record_sha') != seal:
        raise ValueError(f'Record seal mismatch: {path}')
    expected = audit.get('rows', {}).get(path.name)
    if expected is not None and expected != digest(path):
        raise ValueError(f'Audit byte identity mismatch: {path}')
    return row


def figure_save(name, fig, rows, caption_zh, caption_en, selection):
    fig.savefig(HERE / 'figures' / f'{name}.png', dpi=600, bbox_inches='tight')
    # Review-only grayscale copy; plotted values and archival CSV stay unchanged.
    with Image.open(HERE / 'figures' / f'{name}.png') as im:
        preview = im.convert('L')
        preview.thumbnail((1800, 1200))
        preview.save(HERE / 'qa' / f'{name}_grayscale.png')
    plt.close(fig)
    table_csv(HERE / 'figures' / f'{name}.csv', rows)
    FIGURES[name] = dict(caption_zh=caption_zh, caption_en=caption_en,
                         selection=selection, evaluation='EVAL-20261011-v2',
                         source_files=list(PROVENANCE))


def setup():
    for name in ('figures', 'data', 'qa'):
        (HERE / name).mkdir(exist_ok=True)
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
        'axes.labelsize': 9, 'legend.fontsize': 8, 'xtick.labelsize': 8,
        'ytick.labelsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'lines.linewidth': 1.5,
        'axes.prop_cycle': plt.cycler(color=COLORS), 'savefig.dpi': 600,
        'hatch.linewidth': .4, 'axes.linewidth': .7,
        'xtick.major.width': .7, 'ytick.major.width': .7,
    })


def bibliography(offline=False):
    path = HERE / 'data/citation_metadata.json'
    cache = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}

    def fetch(item):
        key, doi = item
        if key in cache:
            if cache[key]['DOI'].lower() != doi.lower():
                raise ValueError('Citation DOI changed')
            return key, cache[key]
        if offline:
            raise RuntimeError(f'No cached citation: {key}')
        url = 'https://api.crossref.org/works/' + urllib.parse.quote(doi, safe='')
        request = urllib.request.Request(url, headers={'User-Agent': 'DPD-Manuscript-Metadata/1.0'})
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(request, timeout=40, context=context) as response:
            data = json.load(response)['message']
        if data['DOI'].lower() != doi.lower():
            raise ValueError('Crossref DOI mismatch')
        return key, data

    with ThreadPoolExecutor(max_workers=4) as pool:
        for key, data in pool.map(fetch, DOIS.items()):
            cache[key] = data
    save_json(path, cache)
    entries = []
    for key, data in cache.items():
        authors = ' and '.join(a.get('family', '') + ', ' + a.get('given', '') for a in data['author'])
        if key == 'ChenHudsonYao2002':
            # Crossref stores "Kung Yao" as an unsplit family name. The source
            # paper and author record identify the family name as Yao.
            authors = 'Chen, J. C. and Hudson, R. E. and Yao, Kung'
        year = data.get('published-print', data.get('published', data['issued']))['date-parts'][0][0]
        kind = 'inproceedings' if data['type'] == 'proceedings-article' else 'article'
        fields = {'author': authors, 'title': '{' + data['title'][0] + '}', 'year': str(year),
                  'doi': data['DOI']}
        venue = data.get('container-title', [''])[0]
        fields['booktitle' if kind == 'inproceedings' else 'journal'] = venue
        for name, src in (('volume', 'volume'), ('number', 'issue'), ('pages', 'page')):
            if data.get(src):
                fields[name] = str(data[src]).replace('–', '--').replace('-', '--')
        if data.get('article-number') and 'pages' not in fields:
            fields['pages'] = str(data['article-number'])
        entries.append('@' + kind + '{' + key + ',\n' + ',\n'.join(
            '  ' + k + ' = {' + v.replace('&', r'\&') + '}' for k, v in fields.items()) + '\n}')
    # Publisher full-text page, not a guessed DOI. Metadata verified against it.
    entries.append('''@article{DebAgrawal1995,
  author = {Deb, Kalyanmoy and Agrawal, Ram Bhushan},
  title = {{Simulated Binary Crossover for Continuous Search Space}},
  journal = {Complex Systems}, year = {1995}, volume = {9}, number = {2},
  pages = {115--148},
  url = {https://www.complex-systems.com/abstracts/v09_i02_a02/}
}''')
    # Official recommendation metadata checked against ITU's title page and §2.3.
    entries.append('''@techreport{ITURP5252024,
  author = {{International Telecommunication Union}},
  title = {{Calculation of free-space attenuation}},
  institution = {ITU Radiocommunication Sector},
  type = {Recommendation}, number = {ITU-R P.525-5},
  year = {2024}, month = nov,
  url = {https://www.itu.int/rec/R-REC-P.525-5-202411-I/en}
}''')
    (HERE / 'references/references.bib').write_text('\n\n'.join(entries) + '\n', encoding='utf-8')
    print(f'Citation metadata: {len(entries)} verified records', flush=True)


def parameters():
    contract = read(RELEASE)
    cfg = contract['configurations']
    ap, npcfg, ref = cfg['AP_1_JOINT'], cfg['PROFILE_DIAG'], cfg['BNOMP_REF']
    rows = [
        ('LS谱', '单源LS谱取两个峰', '共同20 m网格；同谱连续精修'),
        ('MUSIC', '噪声子空间宽带汇聚', f"{cfg['MUSIC']['segments']}段；超噪声功率加权；已知K"),
        ('MVDR-known', '加载协方差逆谱', f"{cfg['MVDR']['segments']}段；相对加载{cfg['MVDR']['loading']}"),
        ('充分AP', '反复全域条件更新', f"候选5；循环上限{cfg['AP_FULL']['cycles']}；正常连续精修"),
        ('交替／联合AP', '相同分支，改变精修方式',
         f"候选1/2/5；每次局部上限{ap['local_maxiter']}；共同投影梯度阈值$10^{{-8}}$"),
        ('VP-BNOMP', '残差发现、消元Newton、共同LS',
         f"循环上限{npcfg['bnomp_cycles']}；能量变化阈值$10^{{-8}}$；每次Newton最多{npcfg['bnomp_newton_steps']}步；步长上限50 m；局部小步阈值0.01 m"),
        ('固定系数BNOMP', '原文对应的固定系数更新',
         f"循环上限{ref['bnomp_cycles']}；能量阈值$10^{{-8}}$；局部小步阈值0.01 m"),
        ('多初值联合ML', '同一联合LS目标', f"{cfg['MULTISTART_FIXED']['starts']}个唯一初值：4谱峰+4 Sobol；L-BFGS-B"),
        ('GA联合ML', '同一联合LS目标',
         f"种群{cfg['GA']['population']}；最多{cfg['GA']['generations']}代；连续{cfg['GA']['stagnation']}代停滞停止"),
        ('宽带SBL', r'共享功率、离网字典、单点$\gamma$',
         f"最多{cfg['SBL']['sbl_iterations']}轮；每5轮离网更新；8个活动点；原生停止"),
    ]
    text = ('\\begin{table}[pos=H]\\centering\\footnotesize\n'
            '\\caption{正式比较采用的方法参数。连续位置优化不受20\\,m网格量化。}\\label{tab:parameters}\n'
            '\\begin{tabularx}{\\linewidth}{@{}p{.18\\linewidth}p{.27\\linewidth}X@{}}\\toprule\n'
            '版本 & 核心操作 & 参数\\\\\\midrule\n')
    text += '\n'.join(' & '.join(r) + r'\\' for r in rows)
    text += '\n\\bottomrule\\end{tabularx}\\end{table}\n'
    (HERE / 'tables/generated_parameters.tex').write_text(text, encoding='utf-8')
    table_csv(HERE / 'data/method_parameters.csv', [dict(method=a, operation=b, parameters=c) for a, b, c in rows])
    env = contract['environment']
    environment_text = (f"运行环境为Windows、Python {env['python'].split()[0]}、"
                        f"NumPy {env['numpy']}和PyTorch {env['torch']}，GPU为{env['gpu']}，"
                        f"CPU计算线程设为{env['threads']}。响应与目标使用双精度实数／复数运算。\n")
    (HERE / 'tables/generated_environment.tex').write_text(environment_text, encoding='utf-8')


def geometry():
    layouts = read(RUNS['paired'] / 'contract.json')['layouts']
    receivers = np.array([[500, 0], [0, 500], [-500, 0], [0, -500], [500, 0]])
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 6), sharex=True, sharey=True, layout='constrained')
    labels = ['Both inside', 'Source 1 outside', 'Source 2 outside', 'Both outside']
    rows = []
    for i, (ax, pp) in enumerate(zip(axes.flat, layouts)):
        pp = np.asarray(pp)
        ax.plot(*receivers.T, color='.6', lw=1)
        ax.scatter(*receivers[:4].T, marker='^', color='black', s=30, label='Receiver')
        ax.scatter(*pp[0], color=COLORS[1], marker='o', s=35, label='Source 1')
        ax.scatter(*pp[1], color=COLORS[0], marker='s', s=30, label='Source 2')
        ax.set(title=f'({chr(97+i)}) {labels[i]}', xlim=(-1000, 1000), ylim=(-1000, 1000), aspect='equal')
        ax.set_xticks([-1000, 0, 1000]); ax.set_yticks([-1000, 0, 1000])
        if i >= 2:
            ax.set_xlabel('x (m)')
        if i % 2 == 0:
            ax.set_ylabel('y (m)')
        for j in range(2):
            rows.append(dict(layout=i, source=j + 1, x_m=pp[j, 0], y_m=pp[j, 1]))
    axes[0, 0].legend(loc='upper left', fontsize=7)
    figure_save('geometry', fig, rows, '四类实际源布局。', 'Four source geometries used in paired validation.',
                'All four layouts of completed paired validation; no performance-based selection.')


def power_gap():
    run = RUNS['band']; audit = read(run / 'final_audit_report.json')
    if not audit.get('identity_passed') or not audit.get('complete'):
        raise ValueError('Incomplete band evidence')
    methods = ['MUSIC', 'MVDR', 'AP_1_JOINT', 'NOMP']
    rows = []
    fig, ax = plt.subplots(figsize=(6.1, 3.2), layout='constrained')
    for mi, method in enumerate(methods):
        rates = []
        for gap in (0, 3, 6, 9, 12, 15):
            cases = [row_read(path, audit) for path in sorted((run / 'rows').glob(f'*_{gap}_mixed_{method}_v0.json'))]
            if len(cases) != 20:
                raise ValueError(f'Incomplete subset {method}, {gap}')
            for row in cases:
                a, b, g, kind = row['task']
                good = bool(np.all(row['metrics']['success']['20']))
                rows.append(dict(method=method, layout=a, realization=b, gap_db=g,
                                 both_success=int(good), input_sha=row['input_sha']))
            rates.append(np.mean([np.all(r['metrics']['success']['20']) for r in cases])*100)
        label = {'AP_1_JOINT': 'Single-candidate joint AP', 'NOMP': 'VP-BNOMP', 'MVDR': 'MVDR-known'}.get(method, method)
        ax.plot([0, 3, 6, 9, 12, 15], rates, color=METHOD_COLORS[method], marker=['o', 's', '^', 'D'][mi],
                ls=['-', '--', '-.', ':'][mi], label=label)
    ax.set(xlabel='Average received power gap (dB)', ylabel='Both-source success (%)', ylim=(-3, 105))
    ax.set_xticks([0, 3, 6, 9, 12, 15]); ax.legend(loc='lower left')
    figure_save('power_gap', fig, rows, '不同平均接收功率差下的双源成功率。',
                'Both-source success versus average received power gap (20 observations per gap).',
                'Completed band-limited run, all 120 mixed observations for each of four shown methods.')


def paired():
    run = RUNS['paired']; report = read(run / 'comparison_report.json')
    audit = read(run / 'final_audit_report.json')
    top_report = read(RUNS['top5'] / 'comparison_report.json')
    top_audit = read(RUNS['top5'] / 'final_audit_report.json')
    if not all((audit.get('complete'), audit.get('identity_passed'), top_audit.get('passed'))):
        raise ValueError('Incomplete paired evidence')
    data = {r['method']: r for r in report['summaries']}
    methods = ['AP_1_ALT', 'AP_1_JOINT', 'AP_5_JOINT', 'NOMP']
    labels = ['Single ALT', 'Single joint', 'Five joint', 'VP-BNOMP']
    table = []
    samples = {}
    for method, label in zip(methods, labels):
        top = method == 'AP_5_JOINT'
        folder = RUNS['top5'] if top else run
        mask = '*_AP_1_JOINT_t0.json' if top else f'*_{method}_t0.json'
        rs = [row_read(p, top_audit if top else audit) for p in sorted((folder / 'rows').glob(mask))]
        if len(rs) != 120:
            raise ValueError('Expected 120 paired rows')
        samples[method] = rs
        # Recompute all reported metrics from stored per-case fields.
        e = np.asarray([r['metrics']['errors_m'] for r in rs])
        good = np.array([np.all(r['metrics']['success']['20']) for r in rs])
        wall = np.mean([r['result']['wall_solver_seconds'] for r in rs])
        counts = np.mean([r['result']['counts']['joint_evaluations'] for r in rs])
        grow = dict(method=method, label=label, cases=len(rs), both_count=int(good.sum()),
                    strong_rmse_m=float(np.sqrt(np.mean(e[:, 0]**2))),
                    weak_rmse_m=float(np.sqrt(np.mean(e[:, 1]**2))),
                    gospa_m=float(np.mean([r['metrics']['gospa_m'] for r in rs])),
                    wall_seconds=float(wall), joint_evaluations=float(counts))
        if not top:
            old = data[method]
            for field in ('both_count', 'strong_rmse_m', 'weak_rmse_m', 'gospa_m'):
                if not np.isclose(grow[field], old[field], rtol=1e-9, atol=1e-9):
                    raise ValueError('Summary replay mismatch: ' + field)
        elif grow['both_count'] != top_report['both_count']:
            raise ValueError('Top-five summary mismatch')
        table.append(grow)
    chinese = ['单候选交替AP', '单候选联合AP', '五候选联合AP', 'VP-BNOMP']
    tex = ('\\begin{table}[pos=htbp]\\centering\\small\n'
           '\\caption{相同120条观测的配对结果。RMSE使用全部观测，单位m。}\\label{tab:paired}\n'
           '\\begin{tabular}{@{}lrrrrr@{}}\\toprule\n'
           '方法 & 双源成功 & 强源RMSE & 弱源RMSE & GOSPA & 时间/s\\\\\\midrule\n')
    for label, r in zip(chinese, table):
        tex += f"{label} & {r['both_count']}/120 & {r['strong_rmse_m']:.3f} & {r['weak_rmse_m']:.3f} & {r['gospa_m']:.3f} & {r['wall_seconds']:.4f}" + r'\\' + '\n'
    tex += '\\bottomrule\\end{tabular}\\end{table}\n'
    (HERE / 'tables/generated_paired.tex').write_text(tex, encoding='utf-8')
    table_csv(HERE / 'data/paired_summary.csv', table)
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.7), layout='constrained')
    fields = [('both_count', 'Both-source success (%)', 100/120),
              ('wall_seconds', 'Mean solver wall time (s)', 1),
              ('joint_evaluations', 'Mean objective evaluations', 1)]
    for ax, (field, ylabel, factor) in zip(axes, fields):
        bars = ax.bar(range(4), [r[field]*factor for r in table],
                      color=[METHOD_COLORS[m] for m in methods], edgecolor='.35', lw=.45, width=.68)
        for bar, hatch in zip(bars, ['', '/', '\\', '.']):
            bar.set_hatch(hatch)
        ax.set_xticks(range(4), labels, rotation=40, ha='right'); ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(r[field]*factor for r in table)*1.18)
    figure_save('paired_cost', fig, table, '配对成功率、时间和调用次数。',
                'Paired success, wall time and objective evaluation counts.', 'All 120 paired mixed observations.')
    fig, ax = plt.subplots(figsize=(6.1, 3.1), layout='constrained')
    cdfrows = []
    for i, method in enumerate(methods):
        es = np.sort([r['metrics']['errors_m'][1] for r in samples[method]])
        yy = np.arange(1, 121)/120
        ax.step(es, yy, where='post', color=METHOD_COLORS[method], ls=['-', '--', '-.', ':'][i], label=labels[i])
        cdfrows += [dict(method=method, weak_error_m=float(e), cumulative_probability=float(y)) for e, y in zip(es, yy)]
    ax.axvline(20, color='.5', ls='--', lw=1); ax.set_xscale('log')
    ax.set(xlabel='Weak-source error (m)', ylabel='Empirical CDF', ylim=(0, 1.02))
    ax.legend(loc='lower right')
    figure_save('error_cdf', fig, cdfrows, '弱源误差的经验CDF。', 'Empirical weak-source error CDF.',
                'All 120 observations; no success-only filtering; same paired runs as the main table.')
    case = row_read(RUNS['top5'] / 'rows/1_301_6_AP_1_JOINT_t0.json', top_audit)
    branches = case['result']['branches']; truth = np.asarray(case['truth'])
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.2), layout='constrained')
    ax = axes[0]
    ax.scatter(*truth[1], marker='*', color='black', s=100, label='True weak source')
    case_rows = []
    for i, b in enumerate(branches):
        initial, final = np.asarray(b['initial_positions']), np.asarray(b['positions'])
        weak_error = float(np.linalg.norm(final[1]-truth[1]))
        case_rows.append(dict(rank=b['rank'], initial_x_m=float(initial[1, 0]), initial_y_m=float(initial[1, 1]),
                              final_x_m=float(final[1, 0]), final_y_m=float(final[1, 1]),
                              weak_error_m=weak_error, objective=b['objective']))
        if i < 2:
            ax.scatter(*initial[1], marker='o', facecolors='none', edgecolors=COLORS[i], s=40)
            ax.scatter(*final[1], marker='x', color=COLORS[i], s=40, label=f'Candidate {i+1}')
            ax.annotate('', xy=final[1], xytext=initial[1],
                        arrowprops=dict(arrowstyle='->', color=COLORS[i], lw=1))
            if i == 0:
                ax.add_patch(Circle(truth[1], 20, fill=False, color='black', ls='--', lw=.8))
    ax.set(xlabel='x (m)', ylabel='y (m)', aspect='equal', xlim=(-600, -100), ylim=(-500, -50))
    ax.legend(fontsize=7, loc='lower right')
    bx = axes[1]
    bx.bar(range(1, 6), [b['objective'] for b in branches], color=COLORS, edgecolor='.35', lw=.45, width=.68)
    bx.set(xlabel='Conditional candidate rank', ylabel='Final normalized residual J')
    bx.set_xticks(range(1, 6))
    for r in case_rows:
        bx.text(r['rank'], r['objective']+.001, f"{r['weak_error_m']:.1f} m", ha='center', va='bottom', fontsize=7)
    bx.set_ylim(0, .078)
    figure_save('candidate_case', fig, case_rows, '弱源区域内前两个条件候选的不同精修终点。',
                'Candidate-dependent refinement endpoints and final residuals.',
                'Post-hoc mechanism example: paired layout 1, realization 301, 6 dB; all five branches shown in right panel.')


def mechanism():
    run = RUNS['mechanism']; report = read(run / 'comparison_report.json')
    audit = read(run / 'final_audit_report.json')
    if not audit.get('passed') or audit.get('solves') != 40:
        raise ValueError('Mechanism audit failed')
    for p in sorted((run / 'rows').glob('*.json')):
        if audit['rows'].get(p.name) != digest(p):
            raise ValueError('Mechanism byte identity changed')
        read(p)
    names = ['FIXED_STRICT', 'FIXED_E7', 'FIXED_E6', 'FIXED_NO_INNER_EXIT', 'PROFILE_STRICT']
    labels = ['Fixed strict', 'Fixed relaxed 1', 'Fixed relaxed 2', 'Fixed no exit', 'Profile strict']
    chinese = ['固定系数，严格停止', '固定系数，阈值$10^{-7}$', '固定系数，阈值$10^{-6}$',
               '固定系数，取消小步退出', '消元更新，严格停止']
    rows = [dict(variant=k, cases=report['summaries'][k]['cases'],
                 both_success=report['summaries'][k]['both20'],
                 weak_rmse_m=report['summaries'][k]['weak_rmse'],
                 seconds=report['summaries'][k]['mean_seconds'],
                 derivative_calls=report['summaries'][k]['mean_derivative_calls']) for k in names]
    tex = ('\\begin{table}[pos=htbp]\\centering\\small\n'
           '\\caption{八条观测的曲率机制隔离。各行共享相同输入和外层系数更新时序。}\\label{tab:mechanism}\n'
           '\\begin{tabular}{@{}lrrrr@{}}\\toprule\n'
           '局部更新与停止 & 双源成功 & 弱源RMSE/m & 时间/s & 导数调用\\\\\\midrule\n')
    for name, row in zip(chinese, rows):
        tex += f"{name} & {row['both_success']}/8 & {row['weak_rmse_m']:.3f} & {row['seconds']:.4f} & {row['derivative_calls']:.3f}" + r'\\' + '\n'
    tex += '\\bottomrule\\end{tabular}\\end{table}\n'
    (HERE / 'tables/generated_mechanism.tex').write_text(tex, encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.2), layout='constrained')
    for ax, key, label in zip(axes, ['derivative_calls', 'seconds'],
                              ['Mean local derivative evaluations', 'Mean solver time (s)']):
        bars = ax.bar(range(5), [r[key] for r in rows],
                      color=['#A4A8AC', '#BEC1C4', '#D4D6D8', '#858C92', '#66859D'],
                      edgecolor='.35', lw=.45, width=.68)
        for b, h in zip(bars, ['', '/', '.', '\\', '']):
            b.set_hatch(h)
        ax.set_yscale('log'); ax.set_ylabel(label)
        ax.set_xticks(range(5), labels, rotation=38, ha='right')
    figure_save('curvature_cost', fig, rows, '固定系数与消元更新的受控计算成本。',
                'Controlled computational cost of fixed-coefficient and profiled updates.',
                'Eight existing observations, all five controlled variants; not the enlarged formal comparison.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='Reuse verified local citation metadata')
    args = parser.parse_args()
    # Historical export functions remain for provenance; the current paper uses
    # audited formal R4 evidence exclusively.
    from prepare_formal_assets import prepare
    prepare(offline=args.offline)


if __name__ == '__main__':
    main()
