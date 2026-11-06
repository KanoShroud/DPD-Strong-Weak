"""Read-only localization replay, paired statistics, and reproducible figures."""
import argparse
import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from .p7 import (ROOT, BASE, METHODS, REFERENCE, read, write, sha, read_row,
                 verify, validate_row, get_input, jobs, key)


def csv_file(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary(rows):
    errors = np.asarray([r['metrics']['errors_m'] for r in rows], float)
    if not np.isfinite(errors).all():
        raise ArithmeticError('Incomplete outputs: do not silently drop from RMSE')
    return dict(samples=len(rows), rmse_A=float(np.sqrt(np.mean(errors[:, 0]**2))),
        rmse_B=float(np.sqrt(np.mean(errors[:, 1]**2))),
        rmse_all=float(np.sqrt(np.mean(errors**2))), p95_B=float(np.percentile(errors[:, 1], 95)),
        max_B=float(errors[:, 1].max()),
        both20=sum(r['metrics']['both20'] for r in rows),
        mean_seconds=float(np.mean([r['result']['seconds'] for r in rows])),
        max_seconds=max(r['result']['seconds'] for r in rows),
        gradient_reached=sum(r['result']['stop_reason'] == 'GRADIENT_REACHED' for r in rows))


def intervals(rows, contract):
    """Fixed-layout stratified cluster bootstrap; all lengths/gaps stay paired."""
    if contract['mode'] != 'formal':
        return []
    lookup = {(key(r['task']), r['method']): r for r in rows}
    groups = sorted({t['group'] for t in contract['tasks']})
    reps = sorted({t['realization'] for t in contract['tasks']})
    gaps = sorted({t['gap'] for t in contract['tasks']})
    rng = np.random.default_rng(20261024071)
    indices = rng.integers(0, len(reps), (4000, len(groups), len(reps)))
    result = []
    for n in (1024, 4096):
        for gap in ['all']+gaps:
            use_gaps = gaps if gap == 'all' else [gap]
            for comparator in METHODS[:-1]:
                arrays = {}
                for method in (comparator, 'D_STAGED'):
                    cube = []
                    for g in groups:
                        block = []
                        for rep in reps:
                            values = []
                            for power in use_gaps:
                                ident = key(dict(group=g, realization=rep, gap=power, n=n))
                                if (ident, method) not in lookup:
                                    return []
                                row = lookup[ident, method]
                                a, b = row['metrics']['errors_m']
                                values.append([a*a, b*b, (a*a+b*b)/2, row['result']['seconds']])
                            block.append(np.mean(values, axis=0))
                        cube.append(block)
                    arrays[method] = np.asarray(cube)
                estimates = {}
                for method, cube in arrays.items():
                    sampled = cube[np.arange(len(groups))[None, :, None], indices].mean(axis=(1, 2))
                    sampled[:, :3] = np.sqrt(sampled[:, :3])
                    estimates[method] = sampled
                delta = estimates['D_STAGED']-estimates[comparator]
                for j, metric in enumerate(('RMSE_A_m', 'RMSE_B_m', 'RMSE_all_m', 'mean_seconds')):
                    lo, hi = np.percentile(delta[:, j], [2.5, 97.5])
                    centers = []
                    for method in ('D_STAGED', comparator):
                        v = float(arrays[method][:, :, j].mean())
                        centers.append(np.sqrt(v) if j < 3 else v)
                    result.append(dict(n=n, gap=gap, contrast='D_STAGED-'+comparator,
                        metric=metric, difference=centers[0]-centers[1], low95=float(lo), high95=float(hi),
                        bootstrap=4000, independent_reps_per_fixed_layout=len(reps)))
    return result


def make_report(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    folder = folder.resolve(strict=True)
    contract = read(folder/'contract.json')
    verify(folder, contract)
    out = folder/('report_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir()
    rows, hashes, flat = [], {}, []
    for task, method in jobs(contract):
        path = folder/'rows'/(key(task)+'_'+method+'.json')
        if not path.exists():
            continue
        row = read_row(path)
        iq, truth = get_input(folder, task)
        validate_row(row, iq, truth, task, method)
        if sha(folder/'spectra'/row['spectrum']) != row['spectrum_sha']:
            raise RuntimeError('Spectrum identity mismatch')
        hashes[str(path.relative_to(folder))] = sha(path)
        rows.append(row)
        a, b = row['metrics']['errors_m']
        r = row['result']
        flat.append(dict(**task, method=method, error_A_m=a, error_B_m=b,
            both20=row['metrics']['both20'], seconds=r['seconds'], stop=r['stop_reason'],
            gradient=r['projected_gradient'], selected_rank=r['selected_rank'],
            first_shift_m=float(np.linalg.norm(np.asarray(r['first'])-r['first_original'])),
            **{'time_'+k: v for k, v in r['stage_seconds'].items()},
            fits=r['counts'].get('fits', 0), gradients=r['counts'].get('gradients', 0),
            grid_points=r['counts'].get('grid_points', 0)))
    aggregate = []
    for n in (1024, 4096):
        for method in METHODS+(REFERENCE,):
            for group in ['all']+list(range(4)):
                for gap in ['all']+[0, 3, 6, 9, 12, 15]:
                    selected = [r for r in rows if r['task']['n'] == n and r['method'] == method
                                and (group == 'all' or r['task']['group'] == group)
                                and (gap == 'all' or r['task']['gap'] == gap)]
                    if selected:
                        aggregate.append(dict(n=n, method=method, group=group, gap=gap, **summary(selected)))
    lookup = {(key(r['task']), r['method']): r for r in rows}
    pair_checks = 0
    for task in contract['tasks']:
        ident = key(task)
        available = [lookup[(ident, method)] for method in METHODS if (ident, method) in lookup]
        if not available:
            continue
        assert len({r['input_sha'] for r in available}) == 1
        for row in available[1:]:
            np.testing.assert_array_equal(row['result']['first_original'], available[0]['result']['first_original'])
        if (ident, 'A_HANN') in lookup and (ident, 'B_TERMINAL') in lookup:
            np.testing.assert_array_equal(lookup[(ident, 'A_HANN')]['result']['starts'],
                                          lookup[(ident, 'B_TERMINAL')]['result']['starts'])
        finite = [r for r in available if r['method'] != 'A_HANN']
        if finite:
            assert len({r['result']['data_sha'] for r in finite}) == 1
        pair_checks += 1
    pairs = []
    for task in contract['tasks']:
        ident = key(task)
        if (ident, 'D_STAGED') not in lookup:
            continue
        d = lookup[ident, 'D_STAGED']
        for method in METHODS[:-1]+(REFERENCE,):
            if (ident, method) not in lookup:
                continue
            other = lookup[ident, method]
            p = dict(**task, comparator=method,
                recovered=not other['metrics']['both20'] and d['metrics']['both20'],
                harmed=other['metrics']['both20'] and not d['metrics']['both20'],
                common_success=other['metrics']['both20'] and d['metrics']['both20'],
                delta_error_B=d['metrics']['errors_m'][1]-other['metrics']['errors_m'][1],
                delta_squared_error_B=d['metrics']['errors_m'][1]**2-other['metrics']['errors_m'][1]**2,
                D_seconds=d['result']['seconds'], other_seconds=other['result']['seconds'])
            pairs.append(p)
    mechanism, residual_decomposition = [], []
    from ...g10.r4.paper_checks.assignment_metrics import evaluate
    for row in rows:
        r, truth = row['result'], np.asarray(row['truth'])
        starts = [evaluate(truth, b['start'])['errors_m'] for b in r['branches']]
        # Offline joint-pair coverage avoids assuming that first source always means A.
        mechanism.append(dict(**row['task'], method=row['method'],
            first_nearest_truth_before=float(np.linalg.norm(truth-np.asarray(r['first_original'])[0], axis=1).min()),
            first_nearest_truth_after=float(np.linalg.norm(truth-np.asarray(r['first'])[0], axis=1).min()),
            best_initial_pair_rmse=float(np.min(np.sqrt(np.mean(np.asarray(starts)**2, axis=1)))),
            initial_pair_within20=any(max(e) <= 20 for e in starts),
            final_success=row['metrics']['both20'], selected_rank=r['selected_rank']))
        if row['method'] == 'D_STAGED':
            # Offline attribution only: no new position optimization or true-location fit.
            from .models import build
            iq, _ = get_input(folder, row['task'])
            model = build(iq, 'periodic')
            before = model.fit(r['first_original'])
            after = model.fit(r['first'])
            a1, _ = model.response(r['first'])
            frozen_prediction = np.einsum('flk,fk->lf', a1, before['coefficients'])
            position_part = before['contributions'].sum(axis=0)-frozen_prediction
            coefficient_part = frozen_prediction-after['contributions'].sum(axis=0)
            change = after['residual']-before['residual']
            np.testing.assert_allclose(change, position_part+coefficient_part, atol=1e-12, rtol=1e-10)
            residual_decomposition.append(dict(**row['task'],
                position_part_relative=float(np.linalg.norm(position_part)/np.sqrt(model.energy)),
                coefficient_part_relative=float(np.linalg.norm(coefficient_part)/np.sqrt(model.energy)),
                total_change_relative=float(np.linalg.norm(change)/np.sqrt(model.energy)),
                cross_energy_relative=float(2*np.vdot(position_part, coefficient_part).real/model.energy),
                before_relative=float(np.linalg.norm(before['residual'])/np.sqrt(model.energy)),
                after_relative=float(np.linalg.norm(after['residual'])/np.sqrt(model.energy))))
    ci = intervals(rows, contract)
    csv_file(out/'逐例结果.csv', flat)
    csv_file(out/'分层统计.csv', aggregate)
    csv_file(out/'配对变化.csv', pairs)
    csv_file(out/'候选机制.csv', mechanism)
    csv_file(out/'残余变化分解.csv', residual_decomposition)
    csv_file(out/'配对区间.csv', ci)
    write(out/'统计.json', dict(aggregate=aggregate, paired=pairs, intervals=ci))
    labels = {'A_HANN': 'A: Hann', 'B_TERMINAL': 'B: terminal finite',
              'C_RESIDUAL': 'C: residual', 'D_STAGED': 'D: staged'}
    colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7']
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for i, n in enumerate((1024, 4096)):
        for m, method in enumerate(METHODS):
            selected = sorted([a for a in aggregate if a['n'] == n and a['method'] == method
                               and a['group'] == 'all' and a['gap'] != 'all'], key=lambda a: a['gap'])
            for j, field in enumerate(('rmse_A', 'rmse_B', 'mean_seconds')):
                xx = [m]*len(selected) if contract['mode'] == 'pilot' else [a['gap'] for a in selected]
                axes[i, j].plot(xx, [a[field] for a in selected],
                    marker=['o', 's', '^', 'D'][m], linestyle=['-', '--', '-.', ':'][m],
                    color=colors[m], label=labels[method])
        for j, label in enumerate(('Source A RMSE (m)', 'Source B RMSE (m)', 'Complete time (s)')):
            axes[i, j].set(xlabel='Power difference (dB)', ylabel=label, title=f'N={n}')
            if contract['mode'] == 'pilot':
                axes[i, j].set(xlabel='Method (15 dB pilot)', xticks=range(4), xticklabels=['A', 'B', 'C', 'D'])
            axes[i, j].grid(alpha=.2)
            axes[i, j].set_yscale('log')
    axes[0, 0].legend(fontsize=8)
    title = 'Development pilot (4 realizations; not formal evidence)' if contract['mode'] == 'pilot' else 'Fixed geometries / new realizations'
    fig.suptitle(title)
    fig.savefig(out/'精度与成本.png', dpi=600)
    fig.savefig(out/'精度与成本.pdf')
    plt.close(fig)
    csv_file(out/'精度与成本_plot.csv', [a for a in aggregate if a['group'] == 'all' and a['gap'] != 'all'])
    write(out/'图注.json', dict(zh='固定构型下的定位RMSE和完整运行时间；纵轴对数。开发短测不代表正式验证。',
        en='Localization RMSE and complete runtime at fixed geometries. Logarithmic y axes. Pilot results are developmental.',
        filtering='all available rows; reference subset excluded from main plot',
        source='分层统计.csv', evaluation='EVAL-20261019-v3-SINGLE-SQUARED-ASSIGNMENT'))
    lines = ['# G11-P7结果回读', '', f'阶段：{contract["mode"]}；已保存{len(rows)}/{len(jobs(contract))}次求解。',
        '', '全部有效输出计入RMSE，不按成功筛样本；0 dB的A/B为固定源身份。', '',
        '|长度|方法|样本|A RMSE/m|B RMSE/m|合并RMSE/m|双源20m成功|完整均时/s|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for a in aggregate:
        if a['group'] == 'all' and a['gap'] == 'all':
            lines.append(f'|{a["n"]}|{a["method"]}|{a["samples"]}|{a["rmse_A"]:.4f}|{a["rmse_B"]:.4f}|{a["rmse_all"]:.4f}|{a["both20"]}/{a["samples"]}|{a["mean_seconds"]:.3f}|')
    lines += ['', 'E_FINITE_GRID仅按预定子集解释，与主批总体不直接排名。',
        'C/D仅差第一源有限更新；B/D同时改变候选形成，不能将全部差异归为模型精度。',
        '候选覆盖、恢复/损害、共同成功子集和耗时分解见CSV；新随机实现不是新几何泛化。',
        '正式区间按固定构型内随机实现配对重采样，保持六档功率差和两长度关联；开发短测不计算区间。']
    branch_stops = Counter(b['stop_reason'] for r in rows for b in r['result']['branches'])
    lines += ['', '候选分支停止情况：'+str(dict(branch_stops))+'。非梯度停止分支仍保留在逐例记录中，不改为收敛。']
    (out/'回读报告.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write(out/'来源.json', dict(contract_sha=sha(folder/'contract.json'), records=hashes,
        report_source_sha=sha(Path(__file__)), input_contract=str(folder), generated=datetime.now().isoformat()))
    (out/'report_source.py').write_bytes(Path(__file__).read_bytes())
    write(out/'审计.json', dict(rows=len(rows), complete=len(rows) == len(jobs(contract)),
        metrics_replayed=True, paired_bootstrap=bool(ci), paired_input_prefix_checks=pair_checks,
        stops=dict(Counter(r['result']['stop_reason'] for r in rows)),
        branch_stops=dict(Counter(b['stop_reason'] for r in rows for b in r['result']['branches']))))
    print('报告目录：', out, flush=True)
    return out


def prepare_release(pilot, checks, report):
    contract = read(pilot/'contract.json')
    verify(pilot, contract)
    assert read(pilot/'audit.json')['completed']
    checked = read(checks/'checks.json')
    assert checked['passed'] and checked['sources'] == contract['sources']
    rows = [read_row(p) for p in (pilot/'rows').glob('*.json')]
    costs, component, envelope = [], 0., 0.
    for n in (1024, 4096):
        for method in METHODS:
            sample = [r['result']['seconds'] for r in rows if r['task']['n'] == n and r['method'] == method]
            assert len(sample) == 4
            cost = dict(n=n, method=method, samples=4, gap_db=15,
                        mean_seconds=float(np.mean(sample)), max_seconds=max(sample), formal_jobs=120)
            costs.append(cost)
            component += cost['mean_seconds']*120
            envelope += cost['max_seconds']*120
    reference = [r['result']['seconds'] for r in rows if r['method'] == REFERENCE]
    assert len(reference) == 1
    component += reference[0]*8
    envelope += reference[0]*8
    # Measured bookkeeping per pilot row plus a fixed analysis allowance.
    wall = sum(a['observed_seconds'] for a in read(pilot/'ledger.json')['attempts'])
    io_per_job = max(0., wall-sum(r['result']['seconds'] for r in rows))/len(rows)
    overhead = io_per_job*968+180.
    estimate = (component+overhead)*1.25/3600
    upper = (envelope+overhead)*1.25/3600
    budget = dict(estimate_hours=estimate, max_layout_envelope_hours=upper,
        algorithm_seconds=component, overhead_seconds=overhead, margin=1.25,
        components=costs, reference_seconds=reference[0], reference_sample_count=1,
        limitation='Only 15 dB pilot timings. Six-gap cost is extrapolated, not guaranteed. Four fixed geometries.',
        formal_observations=240, formal_jobs=968)
    write(report/'预算估算.json', budget)
    release = dict(pilot=str(pilot.relative_to(ROOT)), checks=str(checks.relative_to(ROOT)),
        report=str(report.relative_to(ROOT)), source_contract_sha=sha(pilot/'contract.json'),
        checks_sha=sha(checks/'checks.json'), sources=contract['sources'], budget=budget,
        status='READY_FOR_USER_RUN' if upper <= 12 else 'BUDGET_CONFIRMATION_REQUIRED')
    write(BASE/'release.json', release)
    return release


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('--checks', type=Path)
    args = parser.parse_args()
    folder = args.folder.resolve(strict=True)
    report = make_report(folder)
    if args.checks:
        release = prepare_release(folder, args.checks.resolve(strict=True), report)
        print('预算估算（小时）：', release['budget']['estimate_hours'], flush=True)
