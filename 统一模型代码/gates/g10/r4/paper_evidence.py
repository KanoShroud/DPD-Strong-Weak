"""Read-only R4 trajectory replay and NOMP comparison; never call a solver."""
from pathlib import Path
from datetime import datetime
import argparse
import csv
import hashlib
import io
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT/'outputs_e2e/unified/e2e_g10/r4'
METHODS = ('AP_5_JOINT', 'AP_FULL', 'SBL', 'BNOMP_REF', 'PROFILE_DIAG', 'NOMP')
NOMPS = METHODS[3:]
TIMES = np.array([.25, .5, .75, 1., 1.5, 2., 3., 5., 10., 15., 30., 60., 120., 300.])
STYLE = Path('C:/Users/Administrator/.codex/skills/scientific-toolkit-skill/references/scientific-skills/scientific-visualization/assets/publication.mplstyle')
COLORS = ('#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00', '#000000')
LINES = ('-', '--', '-.', ':', '--', '-.')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def table(p, rows):
    with Path(p).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def metrics(truth, positions):
    p = np.asarray(positions, float)
    if p.shape != (2, 2) or not np.isfinite(p).all():
        raise ValueError('Expected a finite complete pair')
    d = np.linalg.norm(np.asarray(truth)[:, None]-p[None], axis=-1)
    pairs = np.array([[d[0, 0], d[1, 1]], [d[0, 1], d[1, 0]]])
    errors = pairs[np.argmin(np.sum(pairs**2, axis=1))]
    return bool(np.any(np.all(pairs <= 20, axis=1))), errors


def export(out, name, fig, rows, caption):
    for ext in ('pdf', 'png'):
        fig.savefig(out/(name+'.'+ext), dpi=600, bbox_inches='tight')
    plt.close(fig)
    table(out/(name+'.csv'), rows)
    (out/(name+'_caption.md')).write_text(caption, encoding='utf-8')


def replay(row):
    """Published output events only. No oracle best-error selection/interpolation."""
    result = row['result']
    events = sorted(result['report_events'], key=lambda e: e['seconds'])
    arr = []
    for t in TIMES:
        available = [e for e in events if e['seconds'] <= t]
        if not available:
            arr.append((False, False, np.nan, np.nan))
        else:
            good, error = metrics(row['truth'], available[-1]['positions'])
            arr.append((True, good, *error))
    return np.asarray(arr, float)


def load_iq(source, key, manifest):
    p = source/'inputs'/(key+'.json')
    meta = read(p)
    data = (source/meta['path']).resolve(strict=True).read_bytes()
    if digest(data) != meta['sha256']:
        raise ValueError('IQ payload identity mismatch')
    manifest[str(p)] = digest(p.read_bytes())
    manifest[str((source/meta['path']).resolve())] = digest(data)
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return z['signal'].sum(axis=0)+z['noise']


def joint_evaluator(iq):
    # Independent CPU SVD replay; same FFT, frequency mask and normalized response.
    f = np.fft.fftfreq(8192, 1/50e6)
    keep = abs(f) <= 7e6
    x = (np.fft.fft(iq, axis=-1)/8192)[:, keep].T
    f = f[keep]
    stations = np.stack([500*np.cos(np.arange(4)*np.pi/2),
                         500*np.sin(np.arange(4)*np.pi/2)], axis=1).astype(np.float32).astype(float)
    energy = np.sum(abs(x)**2)

    def objective(p):
        d = np.linalg.norm(np.asarray(p)[None]-stations[:, None], axis=-1).clip(1.)
        a = np.exp(-2j*np.pi*f[:, None, None]*d[None]/299792458.)/d[None]
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        q = np.linalg.pinv(a, rcond=1e-10)@x[..., None]
        return float(np.sum(abs(x-(a@q)[..., 0])**2)/energy)
    return objective


def analyze(out):
    pointer = read(BASE/'formal_run.json')
    source = Path(pointer['out']).resolve(strict=True)
    assert source.is_relative_to(BASE.resolve()) and not out.is_relative_to(source)
    assert digest((source/'contract.json').read_bytes()) == pointer['contract_sha']
    audit = read(source/'final_audit_report.json')
    assert audit['passed'] and read(source/'status.json')['status'] == 'COMPLETED'
    identity = {str(source/'final_audit_report.json'): digest((source/'final_audit_report.json').read_bytes()),
                str(source/'contract.json'): pointer['contract_sha'], str(Path(__file__)): digest(Path(__file__).read_bytes())}
    cases = [f'{g}_{r}_{gap}' for g in range(20) for r in (400, 410) for gap in (0, 3, 6, 9, 12, 15)]
    rows, curves, flat = {}, [], []
    for method in METHODS:
        for key in cases:
            p = source/'rows'/f'{key}_{method}.json'
            b = p.read_bytes()
            assert digest(b) == audit['rows'][p.name]
            identity[str(p)] = digest(b)
            row = json.loads(b)
            rows[key, method] = row
            good, errors = metrics(row['truth'], row['result']['positions'])
            assert good == all(row['metrics']['success']['20'])
            assert np.allclose(errors, row['metrics']['errors_m'])
            result = row['result']
            flat.append(dict(case=key, layout=int(key.split('_')[0]), method=method,
                both20=int(good), strong_error_m=errors[0], weak_error_m=errors[1],
                seconds=result['complete_seconds'], objective=result['joint_objective'],
                gradient=result['projected_gradient'], stop=result['stop_reason'],
                iterations=result['counts'].get('iterations', 0)))
        trace = np.array([replay(rows[k, method]) for k in cases])
        for i, t in enumerate(TIMES):
            visible = trace[:, i, 0].astype(bool)
            # Do not mix changing-coverage conditional RMSE into the main curve.
            errors = trace[visible, i, 2:]
            rmse = float(np.sqrt(np.mean(errors**2))) if len(errors) else None
            curves.append(dict(method=method, seconds=float(t), n=240,
                output_count=int(visible.sum()), success_count=int(trace[:, i, 1].sum()),
                success_rate=float(trace[:, i, 1].mean()), coverage=float(visible.mean()),
                conditional_rmse_m=rmse,
                full_coverage_rmse_m=rmse if visible.all() else None))
        print('轨迹与身份核对完成：'+method, flush=True)
    table(out/'per_case.csv', flat)
    table(out/'time_curves.csv', curves)
    # All methods use the identical ordered sample set.
    for key in cases:
        assert all(rows[key, m]['input_identity'] == rows[key, METHODS[0]]['input_identity'] for m in METHODS)
    plt.style.use(str(STYLE))
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'DejaVu Sans'], 'axes.unicode_minus': False})
    displayed = METHODS[:3]+('NOMP',)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), layout='constrained')
    for method in displayed:
        i = METHODS.index(method)
        data = [v for v in curves if v['method'] == method]
        for ax, field in zip(axes, ('success_rate', 'coverage', 'full_coverage_rmse_m')):
            y = [np.nan if v[field] is None else v[field] for v in data]
            ax.step(TIMES, y, where='post', label=method, color=COLORS[i], linestyle=LINES[i])
            ax.set_xscale('log'); ax.set_xlabel('完整运行时间 (s)')
    axes[0].set_ylabel('双源20 m成功率'); axes[1].set_ylabel('完整双源输出覆盖率')
    axes[2].set_ylabel('未截断总体RMSE (m)'); axes[2].set_yscale('log')
    axes[0].set_ylim(0, 1.03); axes[1].set_ylim(0, 1.03)
    axes[0].legend(fontsize=7)
    export(out, 'time_performance', fig, curves,
        '相同240条观测的描述性曲线，无独立性或显著性假设。使用保存的report_events，包含登记的保守记录开销。'
        '无完整输出计为未成功；RMSE主图仅在该方法240条均有完整输出时显示。停止后保持末态，不按真值择优。'
        '\n\nDescriptive replay on the same 240 observations. Published output events only; conservative recorded overhead included. '
        'Missing outputs count as unsuccessful. RMSE is shown only at full coverage. No oracle selection.\n')
    summaries, comparisons = [], []
    for method in NOMPS:
        part = [v for v in flat if v['method'] == method]
        summaries.append(dict(method=method, n=240, successes=sum(v['both20'] for v in part),
            mean_seconds=float(np.mean([v['seconds'] for v in part])),
            weak_rmse_m=float(np.sqrt(np.mean([v['weak_error_m']**2 for v in part]))),
            median_gradient=float(np.median([v['gradient'] for v in part])),
            p95_gradient=float(np.quantile([v['gradient'] for v in part], .95)),
            mean_iterations=float(np.mean([v['iterations'] for v in part]))))
    for left, right in ((NOMPS[0], NOMPS[1]), (NOMPS[1], NOMPS[2])):
        a = [v for v in flat if v['method'] == left]
        b = [v for v in flat if v['method'] == right]
        delta = np.array([y['objective']-x['objective'] for x, y in zip(a, b)])
        comparisons.append(dict(left=left, right=right, n=240,
            rescued=sum(y['both20'] > x['both20'] for x, y in zip(a, b)),
            harmed=sum(y['both20'] < x['both20'] for x, y in zip(a, b)),
            right_lower_objective_1e10=int((delta < -1e-10).sum()),
            right_higher_objective_1e10=int((delta > 1e-10).sum()),
            median_objective_delta=float(np.median(delta)),
            max_abs_objective_delta=float(max(abs(delta)))))
    table(out/'nomp_summary.csv', summaries); table(out/'nomp_pairs.csv', comparisons)
    layout_rows = []
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), layout='constrained')
    for method in NOMPS+('AP_5_JOINT',):
        i = METHODS.index(method)
        vals = []
        for g in range(20):
            sub = [v for v in flat if v['method'] == method and v['layout'] == g]
            item = dict(method=method, layout=g, n=len(sub), successes=sum(v['both20'] for v in sub),
                        seconds=float(np.mean([v['seconds'] for v in sub])))
            layout_rows.append(item); vals.append(item)
        axes[0].plot(range(20), [v['successes']/v['n'] for v in vals], marker='o', markersize=3,
                     label=method, color=COLORS[i], linestyle=LINES[i])
        axes[1].plot(range(20), [v['seconds'] for v in vals], marker='o', markersize=3,
                     color=COLORS[i], linestyle=LINES[i])
    axes[0].set_ylabel('双源20 m成功率'); axes[0].set_ylim(-.03, 1.03); axes[0].legend(ncol=2)
    axes[0].set_xticks(range(20))
    axes[1].set_ylabel('平均完整耗时 (s)'); axes[1].set_yscale('log'); axes[1].set_xlabel('布局编号')
    axes[1].set_xticks(range(20))
    export(out, 'nomp_by_layout', fig, layout_rows,
        '共同子集每布局12条，保留功率扫描与随机实现关联；逐布局描述性统计，不将12条视为12种几何。'
        '\n\nTwelve paired observations per layout; descriptive statistics, not twelve independent geometries.\n')
    # Internal complete-pair iterates are distinct from the native published events.
    # Replay the CURRENT accepted cycle state, never a truth-selected best state.
    cycle_curves = []
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), layout='constrained')
    for method in NOMPS:
        traces = []
        for key in cases:
            row = rows[key, method]
            r = row['result']
            offset = max(0., r['complete_seconds']-r['seconds'])
            events = [dict(seconds=h['seconds']+offset, positions=h['positions'])
                      for h in r['history'] if len(h['positions']) == 2]
            events.append(dict(seconds=r['complete_seconds'], positions=r['positions']))
            traces.append(replay(dict(truth=row['truth'], result=dict(report_events=events))))
        traces = np.array(traces)
        data = []
        for j, t in enumerate(TIMES):
            visible = traces[:, j, 0].astype(bool)
            v = dict(method=method, seconds=float(t), n=240,
                output_count=int(visible.sum()), success_count=int(traces[:, j, 1].sum()),
                success_rate=float(traces[:, j, 1].mean()), coverage=float(visible.mean()),
                full_coverage_rmse_m=float(np.sqrt(np.mean(traces[:, j, 2:]**2))) if visible.all() else None)
            data.append(v); cycle_curves.append(v)
        i = METHODS.index(method)
        for ax, field in zip(axes, ('success_rate', 'coverage', 'full_coverage_rmse_m')):
            ax.step(TIMES, [np.nan if v[field] is None else v[field] for v in data], where='post',
                    color=COLORS[i], linestyle=LINES[i], label=method)
            ax.set_xscale('log'); ax.set_xlabel('累计计算时间 (s)')
        
    axes[0].set_ylabel('双源20 m成功率'); axes[0].set_ylim(0, 1.03); axes[0].legend()
    axes[1].set_ylabel('完整双源状态覆盖率'); axes[1].set_ylim(0, 1.03)
    axes[2].set_ylabel('未截断总体RMSE (m)'); axes[2].set_yscale('log')
    export(out, 'nomp_internal_time', fig, cycle_curves,
        '共同240条观测、实际接受的双源循环状态离线重放；并非REF/PROFILE原程序的中途返回行为。'
        '时间计入保存的全部记录与包装开销作为保守偏移；停止后保持最终输出，无真值择优。RMSE仅在完整覆盖时显示。'
        '\n\nOffline replay of accepted complete-pair cycle states on the same 240 observations, not a deployed early-return policy. '
        'Conservative recorded overhead included; final output held after termination, no oracle selection.\n')
    # Recompute the SAME profiled joint objective; native REF residual energies are not substituted.
    convergence, check_errors = [], []
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), layout='constrained')
    for ax, key in zip(axes, ('0_400_0', '3_400_0')):
        objective = joint_evaluator(load_iq(source, key, identity))
        for method in NOMPS:
            row = rows[key, method]; result = row['result']
            check_errors.append(abs(objective(result['positions'])-result['joint_objective']))
            history = [h for h in result['history'] if len(h['positions']) == 2]
            indices = sorted(set([0, len(history)-1]+[k-1 for k in (2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000) if k <= len(history)]))
            local = []
            for j in indices:
                h = history[j]; value = objective(h['positions'])
                good, error = metrics(row['truth'], h['positions'])
                v = dict(case=key, method=method, cycle=j+1, native_record_seconds=h['seconds'],
                         joint_objective=value, both20=int(good), weak_error_m=float(error[1]), kind='cycle_state')
                convergence.append(v); local.append(v)
            i = METHODS.index(method)
            ax.plot([v['cycle'] for v in local], [v['joint_objective'] for v in local],
                    label=method, color=COLORS[i], linestyle=LINES[i], marker='o', markersize=3)
        ax.set_xscale('log'); ax.set_xlabel('发现双源后的外层循环次数'); ax.set_title(key)
        ax.set_ylabel('共同联合拟合目标（越低越好）')
    axes[0].legend()
    assert max(check_errors) < 1e-9, check_errors
    export(out, 'nomp_convergence', fig, convergence,
        '预先固定普通案例0_400_0和困难案例3_400_0。在实际保存循环状态上，用独立CPU SVD重算同一联合消元目标，'
        '不是各算法不同的原生损失。只显示预定轮次；内部循环状态不是已部署的提前停止规则。'
        '\n\nFixed ordinary and difficult cases. The same profiled joint objective is recomputed by CPU SVD at saved cycle states. '
        'Predetermined cycle checkpoints; no early-stop algorithm is claimed.\n')
    write(out/'source_manifest.json', identity)
    write(out/'analysis.json', dict(source=str(source), summaries=summaries, pairs=comparisons,
        checks=dict(row_hashes=len(rows), input_pairing=True, final_metric_replay=True,
                    cpu_joint_max_error=max(check_errors)), no_solver_called=True))
    print('离线分析完成：'+str(out), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out')
    args = parser.parse_args()
    out = Path(args.out).resolve() if args.out else BASE/('paper_evidence_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    assert out.is_relative_to(BASE.resolve()) and out != BASE.resolve()
    out.mkdir(exist_ok=False)
    analyze(out)


if __name__ == '__main__':
    main()
