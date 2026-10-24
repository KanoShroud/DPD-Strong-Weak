"""Saved-result audit and descriptive P6 plots; no optimization or retuning."""
import argparse
import csv
import io
from datetime import datetime
from pathlib import Path

import numpy as np

from .hybrid_runner import BASE, METHODS, key, read, read_row, sha, write
from ...g10.r4.paper_checks.assignment_metrics import evaluate


def flatten(row):
    r, task, metric = row['result'], row['task'], row['metrics']
    return dict(case=key(task), role=task['role'], n=task['n'], group=task['group'], gap=task['gap'],
        method=row['method'], source_A_error_m=metric['errors_m'][0], source_B_error_m=metric['errors_m'][1],
        seconds=r['seconds'], prefix_seconds=r.get('prefix_seconds', 0.),
        refinement_seconds=r.get('refinement_seconds', r['seconds']),
        objective=r['objective'], stop_reason=r['stop_reason'], projected_gradient=r.get('projected_gradient'))


def summary(rows):
    e = np.array([[r['source_A_error_m'], r['source_B_error_m']] for r in rows])
    return dict(count=len(rows), rmse_A=float(np.sqrt(np.mean(e[:, 0]**2))),
        rmse_B=float(np.sqrt(np.mean(e[:, 1]**2))), pooled_rmse=float(np.sqrt(np.mean(e**2))),
        joint_success_count=int(np.sum(np.max(e, axis=1) <= 20)),
        maximum_error_m=float(e.max()), mean_seconds=float(np.mean([r['seconds'] for r in rows])),
        mean_prefix_seconds=float(np.mean([r['prefix_seconds'] for r in rows])),
        mean_refinement_seconds=float(np.mean([r['refinement_seconds'] for r in rows])),
        stop_counts={reason: sum(r['stop_reason'] == reason for r in rows) for reason in sorted({r['stop_reason'] for r in rows})})


def export_csv(path, rows):
    if not rows:
        return
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buffer.getvalue(), encoding='utf-8-sig')


def audit_run(folder):
    contract = read(folder/'contract.json')
    records = [read_row(p) for p in sorted((folder/'rows').glob('*.json'))]
    case_hashes, starts, vp_starts = {}, {}, {}
    branch_audit = []
    for row in records:
        task, method, result = row['task'], row['method'], row['result']
        assert row['metrics'] == evaluate(row['truth'], result['positions'])
        ident = key(task)
        assert ident not in case_hashes or case_hashes[ident] == row['input_sha']
        case_hashes[ident] = row['input_sha']
        if method in METHODS[:4]:
            assert ident not in starts or starts[ident] == result['starts_sha']
            starts[ident] = result['starts_sha']
            assert len(result['branches']) == 5
            best = min(result['branches'], key=lambda b: (b['objective'], b['rank']))
            assert best['rank'] == result['selected_rank'] and best['positions'] == result['positions']
            assert result['seconds'] == result['prefix_seconds']+result['refinement_seconds']
            diagnostics = []
            for b in result['branches']:
                assert b['objective'] <= b['initial_objective']+1e-12
                initial = evaluate(row['truth'], b['start'])
                final = evaluate(row['truth'], b['positions'])
                diagnostics.append(dict(rank=b['rank'], initial_errors=initial['errors_m'], final_errors=final['errors_m']))
            branch_audit.append(dict(case=ident, method=method, selected=best['rank'], diagnostics=diagnostics))
        elif method.startswith('VP_') and method != 'VP_hann':
            assert ident not in vp_starts or vp_starts[ident] == result['prefix_positions_sha']
            vp_starts[ident] = result['prefix_positions_sha']
            assert result['objective'] <= result['initial_objective']+1e-12
    expected = len(contract['tasks'])*len(METHODS)+(1 if contract['mode'] == 'pilot' else 0)
    return records, dict(expected=expected, actual=len(records), complete=len(records) == expected,
                         metrics_replayed=True, common_inputs=True, common_AP_candidates=True,
                         VP_prefix_identity=True, own_objective_selection=True,
                         branch_diagnostics=branch_audit)


def make_report(folders):
    out = BASE/('report_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir()
    rows, provenance, audits = [], {}, []
    for folder in folders:
        records, audit = audit_run(folder)
        audits.append(dict(source=str(folder), **audit))
        rows.extend(flatten(r) for r in records)
        for p in (folder/'rows').glob('*.json'):
            provenance[str(p)] = sha(p)
    assert len({(r['case'], r['method']) for r in rows}) == len(rows), 'Duplicate case/method; select one run explicitly'
    groups = []
    for role in sorted({r['role'] for r in rows}):
        for n in sorted({r['n'] for r in rows if r['role'] == role}):
            for gap in sorted({r['gap'] for r in rows if r['role'] == role and r['n'] == n}):
                for method in sorted({r['method'] for r in rows if r['role'] == role and r['n'] == n and r['gap'] == gap}):
                    selected = [r for r in rows if (r['role'], r['n'], r['gap'], r['method']) == (role, n, gap, method)]
                    groups.append(dict(role=role, n=n, gap=gap, method=method, **summary(selected)))
    pairs = []
    by = {(r['case'], r['method']): r for r in rows}
    for target in ('AP_periodic', 'AP_finite_fourier', 'AP_slepian', 'VP_finite_fourier', 'VP_slepian'):
        reference = 'AP_hann' if target.startswith('AP_') else 'VP_hann'
        for (case, method), row in by.items():
            if method == target and (case, reference) in by:
                ref = by[(case, reference)]
                pairs.append(dict(case=case, role=row['role'], n=row['n'], gap=row['gap'], target=target, reference=reference,
                    source_A_squared_error_change=row['source_A_error_m']**2-ref['source_A_error_m']**2,
                    source_B_squared_error_change=row['source_B_error_m']**2-ref['source_B_error_m']**2,
                    extra_seconds=row['seconds']-ref['seconds']))
    export_csv(out/'per_method.csv', rows)
    export_csv(out/'paired_changes.csv', pairs)
    write(out/'statistics.json', groups)
    write(out/'audit.json', audits)
    write(out/'provenance.json', provenance)
    lines = ['# G11-P6分阶段精修开发短测', '',
             '全部结果来自固定小规模开发协议；RMSE优先，未作总体显著性或新颖性结论。', '',
             '|范围|N|功率差/dB|方法|数量|源A RMSE/m|源B RMSE/m|合并RMSE/m|双源20m成功|完整秒数|',
             '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in groups:
        lines.append(f"|{r['role']}|{r['n']}|{r['gap']}|{r['method']}|{r['count']}|{r['rmse_A']:.4f}|{r['rmse_B']:.4f}|{r['pooled_rmse']:.4f}|{r['joint_success_count']}/{r['count']}|{r['mean_seconds']:.3f}|")
    lines += ['', 'cohort是4个固定构型×1次新实现×0/15 dB；每档只有4例，不是正式总体验证。native_reference与length_cost_probe是预定单例，不能并入主批或当作独立重复。', '',
              'AP四组合共用原始五候选；finite_fourier/slepian精修读取未加窗带限数据。周期精修的前缀仍为Hann，不能称原生周期AP。VP组合保留同一完整原生Hann前缀，再各作一次联合精修。', '',
              '每种组合完整计入自身前缀和基准备；恢复时重建费用单列于结果字段。失败、触顶及数值退出不删除。没有运行正式大批次，也没有根据本批成绩改配置。', '',
              '审计、逐样本平方误差变化、分支诊断和来源见同目录JSON/CSV。']
    (out/'短测报告.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    # Plot only the complete 8-observation cohort; single probes get tables only.
    cohort = [r for r in rows if r['role'] == 'cohort']
    if all(sum(r['method'] == method for r in cohort) == 8 for method in METHODS):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5), constrained_layout=True)
        labels = ['AP: Hann', 'AP: periodic', 'AP: Fourier', 'AP: Slepian', 'VP: Hann', 'VP + Fourier', 'VP + Slepian']
        colors = ['#0072B2', '#777777', '#D55E00', '#009E73', '#56B4E9', '#CC79A7', '#E69F00']
        for method, label, color in zip(METHODS, labels, colors):
            s = summary([r for r in cohort if r['method'] == method])
            for ax, metric in zip(axes, ('rmse_A', 'rmse_B')):
                ax.scatter(s['mean_seconds'], s[metric], marker='^' if method.startswith('AP_') else 'o',
                           s=62, color=color, label=label)
        for ax, title in zip(axes, ('Source A', 'Source B')):
            ax.set(xscale='log', xlabel='Mean complete runtime (s)', ylabel='RMSE (m)', title=title)
            ax.grid(alpha=.22)
        axes[1].legend(fontsize=8, loc='best')
        fig.suptitle('Development only: 4 geometries, one realization each, gaps 0/15 dB', fontsize=10)
        fig.savefig(out/'rmse_cost.png', dpi=300)
        fig.savefig(out/'rmse_cost.pdf')
        plt.close(fig)
        write(out/'figure_caption.json', dict(zh='固定开发短测中源A/源B的RMSE与完整耗时；0/15 dB配对合并，不代表总体置信度。',
            en='Descriptive RMSE versus complete runtime on the paired development cohort; not a population-level conclusion.',
            source='per_method.csv', filter='role=cohort, all seven methods with 8 cases each'))
    print(out)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folders', nargs='+', type=Path)
    args = parser.parse_args()
    make_report([p.resolve(strict=True) for p in args.folders])


if __name__ == '__main__':
    main()
