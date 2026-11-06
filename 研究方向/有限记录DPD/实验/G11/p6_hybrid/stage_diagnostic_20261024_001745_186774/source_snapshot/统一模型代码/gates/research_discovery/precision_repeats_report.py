"""Read-only replay of G11-P4 inputs/results; write new derived report only."""
import argparse
import csv
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.research_discovery.precision_repeats import (
    GROUPS, GAPS, METHODS, load, decompose, matched_vectors, digest, write,
)

LABELS = dict(periodic='Periodic', hann_after_band='Hann', finite_fourier='Finite Fourier')
COLORS = ['#0072B2', '#E69F00', '#009E73']


def metrics(rows, method, local=False):
    selected = [r['results'][method] for r in rows]
    if local:
        selected = [r['local_diagnostic'] for r in selected]
    errors = np.asarray([v['error_vectors_m'] for v in selected])
    dist = np.linalg.norm(errors, axis=-1)
    result = decompose(errors)
    result.update(count=len(rows), pooled_rmse_m=float(np.sqrt(np.mean(dist**2))),
        median_m=np.median(dist, axis=0).tolist(), p95_m=np.percentile(dist, 95, axis=0).tolist(),
        max_m=dist.max(axis=0).tolist(), success_count=(dist <= 20).sum(axis=0).tolist(),
        both_success_count=int(np.all(dist <= 20, axis=1).sum()), coverage=1.,
        mean_seconds=float(np.mean([v['seconds'] for v in selected])),
        gradient_pass_count=sum(v['gradient_max'] <= 1e-8 for v in selected),
        optimizer_success_count=sum(v['optimizer_success'] for v in selected))
    return result


def rmse_samples(rows, method, local=False):
    data = [r['results'][method] for r in rows]
    if local:
        data = [v['local_diagnostic'] for v in data]
    return np.sum(np.asarray([v['error_vectors_m'] for v in data])**2, axis=-1)


def save_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    folder = args.run.resolve(strict=True)
    assert folder.is_relative_to(ROOT/'outputs_e2e')
    protocol = load(folder/'protocol.json')
    assert load(folder/'audit.json')['completed']
    rows = load(folder/'results.json')
    identities = load(folder/'input_identities.json')
    assert all(digest(p) == value for p, value in identities.items())
    keys = {(r['group'], r['realization'], r['gap']) for r in rows}
    assert len(keys) == len(rows) == len(GROUPS)*len(protocol['repeats'])*len(GAPS)
    truth_replays = 0
    for row in rows:
        raw = folder/'inputs'/f'{row["group"]}_{row["realization"]}.npz'
        assert row['input_sha'] == digest(raw)
        with np.load(raw) as f:
            np.testing.assert_array_equal(f['positions'], row['truth'])
            iq = (10**(row['gap']/20)*f['signal'][0]+f['signal'][1]+f['noise'])[:, :protocol['n']]
            assert __import__('hashlib').sha256(iq.tobytes()).hexdigest() == row['iq_sha']
        for method in METHODS:
            entry = row['results'][method]
            assert entry['selected'] == int(np.argmin([b['objective'] for b in entry['branches']]))
            for value in (entry, entry['local_diagnostic']):
                reevaluated, vectors = matched_vectors(np.asarray(row['truth']), value['positions'])
                assert reevaluated == value['metrics']
                np.testing.assert_array_equal(vectors, value['error_vectors_m'])
                truth_replays += 1
    out = folder/('report_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir()
    summary = []
    comparisons = []
    raw_table = []
    rng = np.random.default_rng(protocol['bootstrap']['seed'])
    # Same within-layout indices are retained for both power conditions and methods.
    for group in GROUPS:
        n = len(protocol['repeats'])
        boot = rng.integers(0, n, size=(protocol['bootstrap']['count'], n))
        for gap in GAPS:
            selected = sorted([r for r in rows if r['group'] == group and r['gap'] == gap], key=lambda r: r['realization'])
            assert [r['realization'] for r in selected] == protocol['repeats']
            for mode in ('blind', 'truth_local'):
                is_local = mode == 'truth_local'
                for method in METHODS:
                    m = metrics(selected, method, is_local)
                    m.update(group=group, gap=gap, mode=mode, method=method)
                    samples = rmse_samples(selected, method, is_local)
                    estimates = np.sqrt(samples[boot].mean(axis=1))
                    m['rmse_ci95_m'] = np.percentile(estimates, [2.5, 97.5], axis=0).T.tolist()
                    summary.append(m)
                new = rmse_samples(selected, 'finite_fourier', is_local)
                for ref in ('periodic', 'hann_after_band'):
                    old = rmse_samples(selected, ref, is_local)
                    difference = np.sqrt(new[boot].mean(axis=1))-np.sqrt(old[boot].mean(axis=1))
                    comparisons.append(dict(group=group, gap=gap, mode=mode, reference=ref,
                        finite_minus_reference_rmse_m=(np.sqrt(new.mean(axis=0))-np.sqrt(old.mean(axis=0))).tolist(),
                        paired_ci95_m=np.percentile(difference, [2.5, 97.5], axis=0).T.tolist()))
            for row in selected:
                for method in METHODS:
                    entry = row['results'][method]
                    for mode, value in [('blind', entry), ('truth_local', entry['local_diagnostic'])]:
                        raw_table.append(dict(group=group, realization=row['realization'], gap_db=gap,
                            mode=mode, method=method, error_A_m=value['metrics']['errors_m'][0],
                            error_B_m=value['metrics']['errors_m'][1], seconds=value['seconds'],
                            objective=value['objective'], gradient_max=value['gradient_max'],
                            stop=value['message']))
        for mode in ('blind', 'truth_local'):
            for method in METHODS:
                low = sorted([r for r in rows if r['group'] == group and r['gap'] == 0], key=lambda r: r['realization'])
                high = sorted([r for r in rows if r['group'] == group and r['gap'] == 15], key=lambda r: r['realization'])
                low_e = rmse_samples(low, method, mode == 'truth_local')
                high_e = rmse_samples(high, method, mode == 'truth_local')
                delta = np.sqrt(high_e[boot].mean(axis=1))-np.sqrt(low_e[boot].mean(axis=1))
                comparisons.append(dict(group=group, mode=mode, method=method, comparison='15dB_minus_0dB',
                    rmse_change_m=(np.sqrt(high_e.mean(axis=0))-np.sqrt(low_e.mean(axis=0))).tolist(),
                    paired_ci95_m=np.percentile(delta, [2.5, 97.5], axis=0).T.tolist()))
    write(out/'statistics.json', summary)
    write(out/'paired_comparisons.json', comparisons)
    aggregate = []
    solver_audit = {}
    for method in METHODS:
        for gap in GAPS:
            for mode in ('blind', 'truth_local'):
                value = metrics([r for r in rows if r['gap'] == gap], method, mode == 'truth_local')
                value.update(method=method, gap=gap, mode=mode,
                             interpretation='equal-weight mixture of three selected fixed layouts')
                aggregate.append(value)
        values = [r['results'][method] for r in rows]
        solver_audit[method] = dict(
            stop_counts=dict(Counter(v['message'] for v in values)),
            max_selected_gradient=max(v['gradient_max'] for v in values),
            max_blind_local_matched_vector_difference_m=max(float(np.linalg.norm(
                np.asarray(v['error_vectors_m'])-np.asarray(v['local_diagnostic']['error_vectors_m']))) for v in values),
            max_blind_local_objective_difference=max(abs(v['objective']-v['local_diagnostic']['objective']) for v in values),
            all_branch_stop_counts=dict(Counter(b['message'] for v in values for b in v['branches'])),
            finite_certificate_max=max((b['inner_certificate_max'] for v in values for b in v['branches']
                                        if b['inner_certificate_max'] is not None), default=None))
    write(out/'aggregate_statistics.json', aggregate)
    write(out/'solver_audit.json', solver_audit)
    save_csv(out/'per_observation.csv', raw_table)
    table = []
    for row in summary:
        table.append({k: row[k] for k in ('group', 'gap', 'mode', 'method', 'count')} | dict(
            rmse_A_m=row['rmse_m'][0], rmse_B_m=row['rmse_m'][1], pooled_rmse_m=row['pooled_rmse_m'],
            mean_displacement_A_m=row['mean_displacement_m'][0], mean_displacement_B_m=row['mean_displacement_m'][1],
            centered_rms_A_m=row['centered_rms_m'][0], centered_rms_B_m=row['centered_rms_m'][1],
            p95_B_m=row['p95_m'][1], max_B_m=row['max_m'][1],
            both_success_count=row['both_success_count'], mean_seconds=row['mean_seconds']))
    save_csv(out/'statistics.csv', table)
    repetitions = len(protocol['repeats'])
    lines = ['# G11-P4精度复核统计', '', f'RMSE优先；每格{repetitions}次独立波形/噪声，构型为固定开发构型。真值初始仅诊断，不计入盲性能。', '',
             '|模式|构型|功率差dB|方法|A RMSE/m|B RMSE/m|合并RMSE/m|B平均位移/m|B中心波动/m|双源成功|完整/诊断时间s|',
             '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f'|{r["mode"]}|{r["group"]}|{r["gap"]}|{LABELS[r["method"]]}|{r["rmse_m"][0]:.3f}|{r["rmse_m"][1]:.3f}|{r["pooled_rmse_m"]:.3f}|{r["mean_displacement_m"][1]:.3f}|{r["centered_rms_m"][1]:.3f}|{r["both_success_count"]}/{r["count"]}|{r["mean_seconds"]:.3f}|')
    lines += ['', '区间为固定构型内配对bootstrap的探索性、逐项95%区间，不是任意几何的置信保证。平均位移与波动均受有限重复影响；波动包含波形与噪声。',
              '盲流程失败全部保留；没有用真值初始结果替代。运行时间包含共用有限拟合候选前端，周期/Hann精修在GPU、有限Fourier在CPU，不能据此推断算法固有复杂度排序。']
    (out/'统计表.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    captions = {}
    for mode in ('blind', 'truth_local'):
        fig, axes = plt.subplots(2, 3, figsize=(11, 6), layout='constrained')
        for col, group in enumerate(GROUPS):
            for source in range(2):
                ax = axes[source, col]
                for method, color, marker in zip(METHODS, COLORS, ('o', 's', '^')):
                    vals = [next(r for r in summary if (r['group'], r['gap'], r['mode'], r['method']) == (group, gap, mode, method)) for gap in GAPS]
                    y = np.array([r['rmse_m'][source] for r in vals])
                    ci = np.array([r['rmse_ci95_m'][source] for r in vals])
                    ax.plot(GAPS, y, marker=marker, color=color, label=LABELS[method])
                    # Interval endpoints are plotted directly; bootstrap interval need not contain estimate.
                    ax.vlines(GAPS, ci[:, 0], ci[:, 1], color=color, alpha=.55)
                ax.set(title=f'Layout {group} / source {"AB"[source]}', xlabel='Power gap (dB)', ylabel='RMSE (m)', xticks=GAPS)
                ax.set_yscale('log')
                ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=9)
        fig.suptitle('Blind five-branch localization' if mode == 'blind' else 'Truth-initialized local diagnostic (not deployable performance)')
        for ext in ('pdf', 'png'):
            fig.savefig(out/f'{mode}_rmse.{ext}', dpi=300 if ext == 'png' else None)
        plt.close(fig)
        captions[f'{mode}_rmse'] = dict(zh=f'同构型内{repetitions}次随机重复的强弱源RMSE，纵轴对数，竖线为逐项bootstrap 95%区间；真值初始图仅作局部诊断。',
            en=f'Source-wise RMSE on {repetitions} paired realizations per fixed geometry; log y-axis, pointwise bootstrap 95% intervals. Truth-initialized results are local diagnostics only.')
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), layout='constrained')
    for ax, group in zip(axes, GROUPS):
        vals = [next(r for r in summary if (r['group'], r['gap'], r['mode'], r['method']) == (group, 15, 'truth_local', m)) for m in METHODS]
        mean2 = np.array([r['mean_displacement_m'][1]**2 for r in vals])
        centered2 = np.array([r['centered_rms_m'][1]**2 for r in vals])
        ax.bar(range(3), mean2, label='Squared sample mean', color='#0072B2')
        ax.bar(range(3), centered2, bottom=mean2, label='Centered mean square', color='#E69F00')
        ax.set(xticks=range(3), xticklabels=['Periodic', 'Hann', 'Finite'], ylabel='Weak-source MSE (m²)', title=f'Layout {group}, 15 dB')
    axes[0].legend(fontsize=8)
    fig.suptitle('Local diagnostic: exact sample MSE decomposition')
    for ext in ('pdf', 'png'):
        fig.savefig(out/f'local_mse_decomposition.{ext}', dpi=300 if ext == 'png' else None)
    plt.close(fig)
    captions['local_mse_decomposition'] = dict(zh='真值初始局部诊断的弱源MSE分解：样本平均位移平方加中心波动均方。前者不能直接当作真实系统偏差。',
        en='Exact sample MSE decomposition for the weak source in local diagnostics; squared sample mean is not a known population bias.')
    write(out/'figure_captions.json', captions)
    write(out/'provenance.json', dict(source=str(folder), results_sha=digest(folder/'results.json'),
        protocol_sha=digest(folder/'protocol.json'), report_source_sha=digest(__file__),
        filter='all registered groups/repeats/gaps; no outcome-based exclusion',
        plot_data='statistics.csv and per_observation.csv', evaluation=protocol['evaluation']))
    (out/'report_source.py').write_bytes(Path(__file__).read_bytes())
    write(out/'audit.json', dict(passed=True, evaluated_outputs=truth_replays, all_inputs_rehashed=True,
        mse_decomposition_passed=True, matching_replayed=True, branch_selection_replayed=True,
        original_results_unchanged=digest(folder/'results.json') == load(out/'provenance.json')['results_sha']))
    print('报告：', out, flush=True)


if __name__ == '__main__':
    main()
