"""Offline P6 early-stage diagnostics: all saved cases, no tuning/solver calls."""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from .hybrid_runner import read, write, read_row, sha, key
from .hybrid_report import export_csv
from ...g10.r4.paper_checks.assignment_metrics import evaluate
from ...g10.physics import grid
from ...g8.physics import RECEIVERS


def summarize(rows):
    errors = np.asarray([[r['error_A'], r['error_B']] for r in rows])
    return dict(count=len(rows), rmse_A=float(np.sqrt(np.mean(errors[:, 0]**2))),
                rmse_B=float(np.sqrt(np.mean(errors[:, 1]**2))), pooled_rmse=float(np.sqrt(np.mean(errors**2))),
                successes=int(np.sum(np.max(errors, axis=1) <= 20)),
                mean_seconds=float(np.mean([r['seconds'] for r in rows])),
                max_error=float(errors.max()))


def report(folder, supplement=None):
    contract = read(folder/'contract.json')
    if not read(folder/'audit.json')['completed']:
        raise RuntimeError('The bounded experiment is not complete')
    out = folder/('report_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir()
    records, rows, provenance = {}, [], {}
    paths = sorted((folder/'rows').glob('*.json'))
    if supplement:
        assert read(supplement/'audit.json')['completed']
        assert Path(read(supplement/'contract.json')['primary']).resolve() == folder
        paths += sorted((supplement/'rows').glob('*.json'))
    for path in paths:
        row = read_row(path)
        result, task, truth = row['result'], row['task'], row['truth']
        metric = evaluate(truth, result['positions'])
        assert metric == row['metrics']
        assert len(result['branches']) == 5
        selected = min(result['branches'], key=lambda b: (b['objective'], b['rank']))
        assert selected['positions'] == result['positions'] and selected['rank'] == result['selected_rank']
        branch_errors = [evaluate(truth, b['positions'])['errors_m'] for b in result['branches']]
        initial_errors = [evaluate(truth, b['start'])['errors_m'] for b in result['branches']]
        first_old = np.asarray(result['first_original'])[0]
        first_new = np.asarray(result['first'])[0]
        first_label = int(np.argmin(np.linalg.norm(np.asarray(truth)-first_old, axis=1)))
        assert all(b['objective'] <= b['initial_objective']+1e-12 for b in result['branches'])
        np.testing.assert_allclose(result['seconds'], result['prefix_seconds']+result['discovery_seconds']+result['refinement_seconds'], atol=1e-12)
        ident = key(task)
        records[(ident, row['variant'])] = row
        rows.append(dict(case=ident, group=task['group'], gap=task['gap'], variant=row['variant'],
            error_A=metric['errors_m'][0], error_B=metric['errors_m'][1],
            success=max(metric['errors_m']) <= 20, seconds=result['seconds'],
            prefix_seconds=result['prefix_seconds'], discovery_seconds=result['discovery_seconds'],
            refinement_seconds=result['refinement_seconds'], objective=result['objective'],
            selected_rank=result['selected_rank'], stop_reason=result['stop_reason'],
            gradient=result['projected_gradient'],
            first_shift=float(np.linalg.norm(np.asarray(result['first'])-np.asarray(result['first_original']))),
            first_truth_label=first_label,
            first_error_before=float(np.linalg.norm(first_old-np.asarray(truth)[first_label])),
            first_error_after=float(np.linalg.norm(first_new-np.asarray(truth)[first_label])),
            min_initial_B=min(e[1] for e in initial_errors), min_final_B=min(e[1] for e in branch_errors),
            any_branch_success=any(max(e) <= 20 for e in branch_errors)))
        provenance[str(path)] = sha(path)
    expected = len(contract['tasks'])*len(contract['variants'])+len(contract['full_cases'])
    assert len(rows) == expected+(8 if supplement else 0)
    summary = {v: summarize([r for r in rows if r['variant'] == v]) for v in sorted({r['variant'] for r in rows})}
    oracle_rows, gains = [], []
    for task in contract['tasks']:
        ident = key(task)
        path = folder/'oracle'/(ident+'.json')
        oracle = read_row(path)
        oracle_rows.append(dict(case=ident, error_A=oracle['metrics']['errors_m'][0],
            error_B=oracle['metrics']['errors_m'][1], objective=oracle['objective'],
            gradient=oracle['projected_gradient'], stop_reason=oracle['stop_reason']))
        provenance[str(path)] = sha(path)
        for c in oracle['conditional']:
            detail = dict(case=ident, first_kind=c['label'], truth_gain=c['truth_gain'],
                          nearest_grid_gain=c['nearest_grid_gain'],
                          highest_grid_gain=None, truth_gain_rank=None, nearest_grid_rank=None)
            if c['label'] == 'finite_first' and (ident, 'finite_conditional') in records:
                with np.load(folder/'spectra'/(ident+'_finite_conditional.npz')) as z:
                    scores = z['scores']
                detail.update(highest_grid_gain=float(scores.max()),
                              truth_gain_rank=int(np.sum(scores > c['truth_gain']))+1,
                              nearest_grid_rank=int(np.sum(scores > c['nearest_grid_gain']))+1)
            gains.append(detail)
    comparison = []
    for target, reference in (('finite_fit', 'periodic_residual'), ('finite_both', 'finite_fit'),
                              ('finite_position', 'terminal'), ('finite_both', 'terminal'),
                              ('finite_conditional', 'finite_both'),
                              ('finite_position_periodic_residual', 'finite_both')):
        for task in contract['tasks']:
            ident = key(task)
            if (ident, target) not in records:
                continue
            a, b = records[(ident, target)], records[(ident, reference)]
            ea, eb = a['metrics']['errors_m'], b['metrics']['errors_m']
            comparison.append(dict(case=ident, target=target, reference=reference,
                squared_A_change=ea[0]**2-eb[0]**2, squared_B_change=ea[1]**2-eb[1]**2,
                extra_seconds=a['result']['seconds']-b['result']['seconds']))
    export_csv(out/'逐例结果.csv', rows)
    export_csv(out/'配对差异.csv', comparison)
    export_csv(out/'真值初始化诊断.csv', oracle_rows)
    export_csv(out/'真值及最近网格条件收益.csv', gains)
    mechanisms = dict(common_subset=[], residual_control=[], oracle_error_differences=[])
    for task in contract['tasks']:
        ident = key(task)
        chosen = records[(ident, 'finite_both')]
        oracle = read_row(folder/'oracle'/(ident+'.json'))
        mechanisms['oracle_error_differences'].append(dict(case=ident,
            difference_m=float(np.linalg.norm(np.asarray(chosen['metrics']['errors_m'])-np.asarray(oracle['metrics']['errors_m'])))))
        if (ident, 'finite_conditional') in records:
            full = records[(ident, 'finite_conditional')]
            mechanisms['common_subset'].append(dict(case=ident,
                full_seconds=full['result']['seconds'], staged_seconds=chosen['result']['seconds'],
                objective_difference=chosen['result']['objective']-full['result']['objective'],
                positions_difference_m=float(np.linalg.norm(np.asarray(chosen['result']['positions'])-np.asarray(full['result']['positions'])))))
        if supplement:
            control = records[(ident, 'finite_position_periodic_residual')]
            with np.load(folder/'spectra'/(ident+'_finite_both.npz')) as z:
                original_spectrum = z['scores']
            with np.load(supplement/'spectra'/(ident+'_finite_position_periodic_residual.npz')) as z:
                changed_spectrum = z['scores']
            mechanisms['residual_control'].append(dict(case=ident,
                same_candidates=chosen['result']['starts'] == control['result']['starts'],
                same_positions=chosen['result']['positions'] == control['result']['positions'],
                spectrum_relative_difference=float(np.linalg.norm(original_spectrum-changed_spectrum)/np.linalg.norm(original_spectrum))))
    write(out/'机制核对.json', mechanisms)
    write(out/'统计.json', summary)
    provenance[str(Path(__file__).resolve())] = sha(Path(__file__))
    (out/'analysis_source.py').write_bytes(Path(__file__).read_bytes())
    write(out/'来源.json', provenance)
    write(out/'审计.json', dict(complete=True, count=len(rows), metrics_replayed=True,
        all_failures_preserved=True, own_objective_selection=True, timing_sum=True,
        note='finite_conditional is a preselected failure/control subset; do not compare its pooled rate with eight-case cohort'))
    lines = ['# G11-P6 提前介入诊断', '',
             '固定8条既有1024点开发观测。所有版本末端均为相同有限Fourier五分支精修；无参数调整。', '',
             '|流程|样本数|源A RMSE/m|源B RMSE/m|合并RMSE/m|双源20m成功|完整平均秒数|',
             '|---|---:|---:|---:|---:|---:|---:|']
    for method, s in summary.items():
        lines.append(f"|{method}|{s['count']}|{s['rmse_A']:.4f}|{s['rmse_B']:.4f}|{s['pooled_rmse']:.4f}|{s['successes']}/{s['count']}|{s['mean_seconds']:.3f}|")
    lines += ['', '`finite_conditional`只在预定的三条旧失败观测及一条成功对照上运行，不与8条总体直接排名。', '',
              'terminal：Hann条件候选；periodic_residual：原第一源位置、周期拟合残余；finite_fit：只换有限拟合；finite_position：只修正第一源位置、仍Hann条件候选；finite_both：有限修正与拟合、Hann残余候选；finite_conditional：有限修正第一源后执行有限模型精确全域条件搜索。', '',
              '盲结果全部保存后才运行单列真值初始化诊断；不能将这些结果当作实际算法成绩。', '',
              'finite_position_periodic_residual为看到主批开发结果后补的一项归因对照：有限修正第一源位置不变，仅将有限残余拟合换为周期拟合；不计为独立验证。', '',
              '所有算法完整计入第一源搜索、有限基准备、候选形成及五分支精修；诊断与绘图不计入算法时间。现有开发样本用于机制定位，不证明总体泛化或新颖性。', '',
              '原P6输入与结果不改；逐例、分支、条件谱、停止信息和源码快照保存在上级目录。']
    (out/'诊断报告.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    draw(folder, out, records)
    return out


def draw(folder, out, records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42})
    # Predeclared failed example, not selected after this experiment's results.
    ident = '3_28001_15_1024'
    variants = ('terminal', 'finite_fit', 'finite_both', 'finite_conditional')
    points = grid(20.)
    side = int(np.sqrt(len(points)))
    figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for ax, variant in zip(axes.ravel(), variants):
        row = records[(ident, variant)]
        result, truth = row['result'], np.asarray(row['truth'])
        with np.load(folder/'spectra'/(ident+'_'+variant+'.npz')) as z:
            score = z['scores']
        scaled = score/max(float(score.max()), 1e-30)
        ax.pcolormesh(points[:, 0].reshape(side, side), points[:, 1].reshape(side, side),
                      scaled.reshape(side, side), shading='nearest', cmap='viridis', vmin=0, vmax=1, rasterized=True)
        starts = np.asarray(result['starts'])[:, 1]
        sites = np.asarray(RECEIVERS)
        ax.scatter(sites[:, 0], sites[:, 1], c='white', marker='^', edgecolors='black', s=50, label='Stations')
        ax.scatter(truth[:, 0], truth[:, 1], c='red', marker='*', s=100, label='Truth')
        ax.scatter(starts[:, 0], starts[:, 1], c='cyan', marker='D', edgecolors='black', s=35, label='5 candidates')
        final = np.asarray(result['positions'])
        ax.scatter(final[:, 0], final[:, 1], c='orange', marker='x', s=70, label='Selected final')
        ax.set(xlabel='x (m)', ylabel='y (m)', xlim=(-1000, 1000), ylim=(-1000, 1000), aspect='equal',
               title=f"{variant}\nB error {row['metrics']['errors_m'][1]:.2f} m")
    axes[0, 0].legend(loc='upper left', fontsize=8)
    figure.suptitle('Predeclared P6 failure: each score normalized by its own maximum')
    figure.savefig(out/'候选变化.png', dpi=200)
    figure.savefig(out/'候选变化.pdf')
    plt.close(figure)
    write(out/'图注.json', dict(zh='P6预定失败观测的候选变化。各子图分别以自身最大值归一化，不跨模型比较分数绝对值。真值仅事后叠加，末态不代表迭代轨迹。',
        en='Candidate changes for the predeclared failed observation. Each scoring map is normalized separately. Truth is overlaid only after solving; final locations do not show the iteration path.',
        case=ident, sources=[str(folder/'spectra'/(ident+'_'+v+'.npz')) for v in variants]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('--supplement', type=Path)
    args = parser.parse_args()
    print(report(args.folder.resolve(strict=True), args.supplement.resolve(strict=True) if args.supplement else None))
