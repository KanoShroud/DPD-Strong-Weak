"""Read-only result replay and measured budget, never run a solver."""
import csv
import io
from collections import Counter
import numpy as np
from ..metrics import aggregate
from .runtime import read, write, check_contract, job_key
from .config import METHODS, FORMAL_CASES, FORMAL_SOLVES


def generate(out):
    c = check_contract(out)
    rows = [read(out/'rows'/(job_key(j)+'.json')) for j in c['jobs']]
    summaries, budget = {}, {}
    for method in METHODS:
        group = [r for r in rows if r['job']['method'] == method]
        result = aggregate([r['metrics'] for r in group])
        seconds = np.array([r['result']['wall_solver_seconds'] for r in group])
        total = np.array([r['result']['wall_solver_seconds']+r['result']['certificate_seconds'] for r in group])
        result.update(both20_count=sum(all(r['metrics']['success']['20']) for r in group),
            strong20_count=sum(r['metrics']['success']['20'][0] for r in group),
            weak20_count=sum(r['metrics']['success']['20'][1] for r in group),
            weak_rmse_m=float(np.sqrt(np.mean([r['metrics']['errors_m'][1]**2 for r in group]))),
            mean_wall_seconds=float(seconds.mean()), median_wall_seconds=float(np.median(seconds)),
            p95_wall_seconds=float(np.quantile(seconds, .95)), max_wall_seconds=float(seconds.max()),
            stop_reasons=dict(Counter(r['result']['stop_reason'] for r in group)),
            mean_iterations=float(np.mean([r['result']['counts']['iterations'] for r in group])),
            mean_joint_calls=float(np.mean([r['result']['counts']['joint_evaluations'] for r in group])),
            mean_gradient_calls=float(np.mean([r['result']['counts']['gradient_calls'] for r in group])),
            mean_response_points=float(np.mean([r['result']['counts']['frequency_response_points'] for r in group])),
            gradient_precision_met=sum(r['result']['projected_gradient'] <= r['result']['config']['gtol'] for r in group))
        summaries[method] = result
        budget[method] = dict(pilot_solves=len(group), formal_solves=FORMAL_CASES,
            mean_seconds_per_solve=float(total.mean()), mean_based_seconds=float(total.mean()*FORMAL_CASES),
            p95_based_seconds=float(np.quantile(total, .95)*FORMAL_CASES))
    timing = read(out/'batch_timing.json')
    costs = sum(r['result']['wall_solver_seconds']+r['result']['certificate_seconds'] for r in rows)
    overhead = max(0., timing['seconds']-costs)/len(rows) if timing['newly_computed'] == len(rows) else None
    # Short batch does not generate new observations. One separately measured dev
    # generation/save/read probe supplies this missing cost (not validation IQ).
    io_cost = read(out/'io_cost.json')['seconds']*FORMAL_CASES
    mean_base = sum(x['mean_based_seconds'] for x in budget.values())
    p95_base = sum(x['p95_based_seconds'] for x in budget.values())
    overhead_cost = (overhead or 0.)*FORMAL_SOLVES
    forecast = dict(formal_cases=FORMAL_CASES, formal_solves=FORMAL_SOLVES,
        by_method=budget, observed_job_overhead_seconds=overhead,
        generation_io_seconds=io_cost, recording_save_audit_seconds=overhead_cost,
        warmup_report_plot_reserve_seconds=600., safety_factor=1.25,
        estimate_hours=(1.25*(mean_base+overhead_cost+io_cost)+600)/3600,
        p95_scenario_hours=(1.25*(p95_base+overhead_cost+io_cost)+600)/3600,
        note='16 previously-used cases, only 0/15 dB. P95-based number is a conservative scenario, not a statistical bound. New geometry may cost more. No formal budget is yet authorized.')
    report = dict(scope='development correctness/cost pilot, not independent validation',
                  cases=16, solves=len(rows), summaries=summaries)
    write(out/'comparison_report.json', report)
    write(out/'budget_estimate.json', forecast)
    review = dict(engineering_checks_passed=True, formal_released=False,
        status='HOLD_BNOMP_NATIVE_STOP_REVIEW',
        reason='BNOMP_REF reaches cyclic cap in '+str(summaries['BNOMP_REF']['stop_reasons'].get('ITERATION_LIMIT', 0))+
               '/16 cases. Budget applies only to current finite-cap settings, not a sufficiently converged reference.',
        nominal_budget_hours=forecast['estimate_hours'],
        localization_success_not_release_requirement=True)
    write(out/'readiness.json', review)
    buff = io.StringIO(newline='')
    fields = ['layout', 'realization', 'gap_db', 'method', 'both20', 'weak_error_m',
              'strong_error_m', 'wall_seconds', 'stop_reason', 'objective', 'projected_gradient']
    writer = csv.DictWriter(buff, fieldnames=fields); writer.writeheader()
    for row in rows:
        r, task = row['result'], row['job']['task']
        writer.writerow(dict(zip(fields, [*task[:3], r['method'], all(row['metrics']['success']['20']),
            row['metrics']['errors_m'][1], row['metrics']['errors_m'][0], r['wall_solver_seconds'],
            r['stop_reason'], r['native_objective'], r['projected_gradient']])))
    (out/'per_case.csv').write_text(buff.getvalue(), encoding='utf-8-sig')
    lines = ['# G10-R4短测摘要', '', '仅使用已检查开发观测；正式扩大数据尚未生成。', '',
             '|版本|双源20m成功|弱源RMSE/m|平均完整算法秒|平均循环数|达到完整梯度阈值|',
             '|---|---:|---:|---:|---:|---:|']
    for method, row in summaries.items():
        lines.append(f"|{method}|{row['both20_count']}/16|{row['weak_rmse_m']:.3f}|{row['mean_wall_seconds']:.3f}|{row['mean_iterations']:.1f}|{row['gradient_precision_met']}/16|")
    lines += ['', f"正式2400条、16800次：均值外推并留25%余量约{forecast['estimate_hours']:.2f}小时；",
        f"按各方法短测P95外推约{forecast['p95_scenario_hours']:.2f}小时（不是保证上限）。",
        '估算含输入生成、记录保存、最终证书及10分钟绘图/审计预留。正式新几何尚未计时。',
        '', '数值校验通过不要求16条全部成功；触顶、停滞和定位误差分开记录。',
        '正式放行：HOLD_BNOMP_NATIVE_STOP_REVIEW。当前成本只对应本次上限，BNOMP_REF频繁触顶，不能当充分收敛的强参照。',
        'NOMP_JOINT包含原NOMP全部计算；单次计时可能因设备波动略短，不能据此认定附加精修加速。',
        '正式预算与启动另行确认，本入口不会自动开始正式比较。']
    (out/'运行摘要.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return report, forecast
