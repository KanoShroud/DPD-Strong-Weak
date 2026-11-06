"""Read-only result replay; no optimization, no task-three data generation."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OUT = HERE/'analysis'
UNIT = 299792458/12.5e6


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def dump(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def table(name, rows):
    with (OUT/(name+'.csv')).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def err(p, truth):
    p = np.asarray(p)
    a = np.linalg.norm(p-truth, axis=1)
    b = np.linalg.norm(p[::-1]-truth, axis=1)
    return a if a@a <= b@b else b


def stats(rows):
    e = np.array([r['errors_m'] for r in rows])
    return dict(n=len(rows), rmse_m=float(np.sqrt(np.mean(e**2))),
                source0_rmse_m=float(np.sqrt(np.mean(e[:, 0]**2))),
                source1_rmse_m=float(np.sqrt(np.mean(e[:, 1]**2))),
                p95_m=float(np.quantile(e, .95)), max_m=float(e.max()),
                success20=int(np.all(e <= 20, axis=1).sum()),
                mean_s=float(np.mean([r['seconds'] for r in rows])),
                median_s=float(np.median([r['seconds'] for r in rows])),
                stopped=int(sum(r['reason'] == 'physical_projected_gradient' for r in rows)),
                final_mapping_pass=int(sum(r['mapping'] <= 1e-7 for r in rows)),
                mapping_median=float(np.median([r['mapping'] for r in rows])),
                mapping_max=float(max(r['mapping'] for r in rows)),
                descent_probe=int(sum(r['feasible_descent'] for r in rows)),
                subproblem_unsuccessful=int(sum(not h['subproblem']['success'] for r in rows for h in r['history'])),
                subproblem_fallback=int(sum(h['subproblem']['fallback'] for r in rows for h in r['history'])),
                mean_jacobian_columns=float(np.mean([80*r['counts']['full_jacobians']+
                            40*r['counts']['block_jacobians'] for r in rows])),
                mean_subproblem_iterations=float(np.mean([r['counts']['subproblem_iterations'] for r in rows])),
                mean_residual_calls=float(np.mean([r['counts']['residual_calls'] for r in rows])))


def savefig(fig, name):
    fig.savefig(OUT/(name+'.png'), dpi=300, bbox_inches='tight')
    fig.savefig(OUT/(name+'.pdf'), bbox_inches='tight')
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    origins = {k: Path(read(HERE/(k+'_latest.json'))['path'])
               for k in ['task1_short', 'task1_full', 'task2_short', 'task2_full']}
    all_data = {k: read(p/'result.json') for k, p in origins.items()}
    assert all(all(d['checks'].values()) for d in all_data.values())
    d1, d2 = all_data['task1_full'], all_data['task2_full']
    assert len(d1['rows']) == 24 and len(d2['rows']) == 96 and d2['comparisons'] == 48
    theory, sensitivity, remainder = [], [], []
    for r in d1['rows']:
        p = r['prediction']
        slopes = np.array(p['slopes'])[:, :4].reshape(2, 2, 2)*UNIT
        for origin in range(2):
            for target in range(2):
                v = slopes[origin, target]
                sensitivity.append(dict(tag=r['tag'], origin=origin, target=target,
                                        dx_m=float(v[0]), dy_m=float(v[1]), norm_m=float(np.linalg.norm(v))))
        for a in r['rows'][1:]:
            theory.append(dict(tag=r['tag'], alpha=a['alpha'],
                actual_norm_m=float(np.linalg.norm(a['delta_m'])),
                exact_absolute_m=a['exact_error']['absolute_m'], exact_relative=a['exact_error']['relative'],
                exact_cosine=a['exact_error']['cosine'], gn_relative=a['gn_error']['relative'],
                gradient=a['fit']['gradient'], nfev=a['fit']['nfev'], fit_status=a['fit']['status']))
        remainder.append(r['rows'][2]['exact_error']['absolute_m']/r['rows'][1]['exact_error']['absolute_m'])
    table('task1_predictions', theory)
    table('task1_source_coupling', sensitivity)
    tsummary = []
    for a in [.05, .1, 1.]:
        rs = [r for r in theory if r['alpha'] == a]
        tsummary.append(dict(alpha=a, n=len(rs),
            exact_relative_median=float(np.median([r['exact_relative'] for r in rs])),
            exact_relative_max=float(max(r['exact_relative'] for r in rs)),
            gn_relative_median=float(np.median([r['gn_relative'] for r in rs])),
            cosine_median=float(np.median([r['exact_cosine'] for r in rs])),
            cosine_min=float(min(r['exact_cosine'] for r in rs))))
    table('task1_summary', tsummary)
    pairs = {}
    truths = {}
    oldrows = []
    for row in d2['rows']:
        pairs.setdefault(row['tag'], {})[row['method']] = row
        tag = row['tag'].rsplit('_', 1)[0]
        if tag not in truths:
            directory = BASE/('independent_backend_20261029_222111_026721' if row['phase'] == 'development'
                              else 'consensus_validation_20261029_223520_495993')
            truths[tag] = np.load(directory/(tag+'_input.npz'))['p']
        assert np.allclose(err(row['p'], truths[tag]), row['errors_m'], rtol=1e-12, atol=1e-12)
        if row['method'] == 'JOINT':
            folder = BASE/('consensus_development_20261029_223235_334946' if row['phase'] == 'development'
                          else 'consensus_validation_20261029_223520_495993')
            old = read(folder/(row['tag']+'.json'))
            for stage in ['geometry', 'los', 'mp']:
                oldrows.append(dict(tag=row['tag'], phase=row['phase'], observation=row['observation'],
                                     stage=stage, errors_m=old[stage+'_errors_m']))
    summaries = []
    for phase in ['all', 'development', 'validation']:
        for obs in ['all', 'LOS', 'MP']:
            for method in ['JOINT', 'BLOCK']:
                rows = [r for r in d2['rows'] if (phase == 'all' or phase == r['phase']) and
                        (obs == 'all' or obs == r['observation']) and method == r['method']]
                summaries.append(dict(phase=phase, observation=obs, method=method, **stats(rows)))
    table('task2_summary', summaries)
    bylayout = []
    for layout in sorted(set(r['layout'] for r in d2['rows'])):
        for method in ['JOINT', 'BLOCK']:
            bylayout.append(dict(layout=layout, method=method,
                            **stats([r for r in d2['rows'] if r['layout'] == layout and r['method'] == method])))
    table('task2_layouts', bylayout)
    paired, checkpoints, targets = [], [], []
    for tag, pair in pairs.items():
        j, b = pair['JOINT'], pair['BLOCK']
        assert j['initial_signature'] == b['initial_signature']
        ee = err(j['p'], np.array(b['p']))
        paired.append(dict(tag=tag, phase=j['phase'], observation=j['observation'],
            position_difference_max_m=float(ee.max()), objective_joint_minus_block=j['cost']-b['cost'],
            rmse_joint_m=float(np.sqrt(np.mean(np.array(j['errors_m'])**2))),
            rmse_block_m=float(np.sqrt(np.mean(np.array(b['errors_m'])**2))),
            joint_s=j['seconds'], block_s=b['seconds'], ratio_joint_block=j['seconds']/b['seconds'],
            joint_mapping=j['mapping'], block_mapping=b['mapping'],
            joint_reason=j['reason'], block_reason=b['reason']))
        for method, r in pair.items():
            # Retrospective, objective-only target; not a deployable stopping rule.
            for tolerance in [1e-6, 1e-8]:
                target = min(j['cost'], b['cost'])+tolerance
                hits = [e for e in r['events'] if e['cost'] <= target]
                targets.append(dict(tag=tag, method=method, tolerance=tolerance, target=target,
                                    reached=bool(hits), first_time_s=hits[0]['seconds'] if hits else None))
            for t in [.5, 1., 2., 3., 5., 8., 10.]:
                events = [e for e in r['events'] if e['seconds'] <= t]
                # Before first recorded score, the common initial position is already available.
                e = events[-1] if events else r['events'][0]
                errors = err(e['p'], truths[tag.rsplit('_', 1)[0]])
                checkpoints.append(dict(tag=tag, phase=r['phase'], observation=r['observation'],
                    method=method, time_s=t, squared_error_mean=float(np.mean(errors**2)),
                    error0_m=float(errors[0]), error1_m=float(errors[1]), cost=e['cost'],
                    before_first_event=not bool(events), ended=r['seconds'] <= t))
    table('task2_pairs', paired)
    table('task2_saved_time_states', checkpoints)
    table('task2_retrospective_objective_targets', targets)
    oldsummary = []
    for phase in ['development', 'validation']:
        for obs in ['LOS', 'MP']:
            for stage in ['geometry', 'los', 'mp']:
                es = np.array([r['errors_m'] for r in oldrows if r['phase'] == phase and
                               r['observation'] == obs and r['stage'] == stage])
                oldsummary.append(dict(phase=phase, observation=obs, stage=stage, n=len(es),
                                       rmse_m=float(np.sqrt(np.mean(es**2)))))
    table('saved_pipeline_context', oldsummary)
    relative_residual = np.array([r['prediction']['base_residual_norm']/np.sqrt(2) for r in d1['rows']])
    audit = dict(sources={k:str(v) for k,v in origins.items()},
                 checks={k:dict(n=len(v['checks']), passed=all(v['checks'].values()), seconds=v['seconds'])
                         for k,v in all_data.items()},
                 source_sha256={str(p/'result.json'):hashlib.sha256((p/'result.json').read_bytes()).hexdigest()
                                for p in origins.values()},
                 task1_summary=tsummary, remainder_ratio_median=float(np.median(remainder)),
                 remainder_ratio_range=[min(remainder),max(remainder)],
                 base_raw_relative_residual_range=[float(relative_residual.min()),float(relative_residual.max())],
                 max_hessian_fd_relative=max(r['prediction']['hessian_step_relative'] for r in d1['rows']),
                 max_final_task1_gradient=max(t['gradient'] for t in theory),
                 task1_all_regular=all(r['prediction']['implicit_regular'] for r in d1['rows']),
                 task1_summary_scope='NOISELESS_LOCAL_ORACLE_DIAGNOSTICS',
                 task2_summary=summaries,
                 task2_pair_max_position_difference_m=max(r['position_difference_max_m'] for r in paired),
                 task2_pair_max_absolute_objective_difference=max(abs(r['objective_joint_minus_block']) for r in paired),
                 task2_joint_faster=sum(r['joint_s'] < r['block_s'] for r in paired),
                 task2_joint_lower_cost=sum(r['objective_joint_minus_block'] < -1e-10 for r in paired),
                 task2_block_lower_cost=sum(r['objective_joint_minus_block'] > 1e-10 for r in paired),
                 task3_started=False)
    audit['common_objective_targets'] = []
    for tolerance in [1e-6, 1e-8]:
        rs = [r for r in targets if r['tolerance'] == tolerance]
        matched = {tag: {r['method']:r for r in rs if r['tag'] == tag} for tag in pairs}
        both = [v for v in matched.values() if all(r['reached'] for r in v.values())]
        audit['common_objective_targets'].append(dict(tolerance=tolerance, both_reached=len(both),
            joint_reached=sum(r['reached'] for r in rs if r['method']=='JOINT'),
            block_reached=sum(r['reached'] for r in rs if r['method']=='BLOCK'),
            joint_faster_in_both=sum(v['JOINT']['first_time_s'] < v['BLOCK']['first_time_s'] for v in both),
            joint_median_s=float(np.median([v['JOINT']['first_time_s'] for v in both])) if both else None,
            block_median_s=float(np.median([v['BLOCK']['first_time_s'] for v in both])) if both else None))
    dump(OUT/'metrics.json', audit)
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                         'pdf.fonttype':42, 'font.family':'DejaVu Sans'})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for i,a in enumerate([.05,.1,1.]):
        rs=[r for r in theory if r['alpha']==a]
        vals=np.array([r['exact_relative'] for r in rs])
        axes[0].scatter(i+np.linspace(-.13,.13,len(vals)), vals, s=18, alpha=.65, color='#0072B2')
        axes[0].plot([i-.2,i+.2], [np.median(vals)]*2, color='black', linewidth=2)
    axes[0].set(xticks=range(3), xticklabels=['0.05','0.1','1 (original)'],
                xlabel='Reflection amplitude multiplier', ylabel='Relative vector prediction error',
                title='Local sensitivity: 24 fixed cases')
    means=np.array([[np.mean([r['norm_m'] for r in sensitivity if r['origin']==a and r['target']==b])
                     for b in range(2)] for a in range(2)])
    axes[1].bar(np.arange(4),means.ravel(),color=['#0072B2','#E69F00','#E69F00','#0072B2'])
    axes[1].set(xticks=range(4),xticklabels=['0 to 0','0 to 1','1 to 0','1 to 1'],
                xlabel='Perturbed source to affected source',ylabel='Mean derivative norm (m / alpha)',
                title='Self and cross-source influence')
    savefig(fig,'task1_mechanism')
    fig, axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    for method,color,style in [('JOINT','#0072B2','-o'),('BLOCK','#D55E00','--s')]:
        ts=sorted(set(r['time_s'] for r in checkpoints))
        vals=[np.sqrt(np.mean([r['squared_error_mean'] for r in checkpoints
                              if r['method']==method and r['time_s']==t])) for t in ts]
        axes[0].plot(ts,vals,style,color=color,label=method,markersize=4)
    axes[0].set(xlabel='Backend time (s), common seed already available',ylabel='Position RMSE (m)',
                title='Saved-state replay; not a new stopping policy')
    axes[0].legend()
    axes[1].scatter([r['block_s'] for r in paired],[r['joint_s'] for r in paired],s=25,color='#0072B2')
    hi=max(max(r['joint_s'],r['block_s']) for r in paired)*1.05
    axes[1].plot([0,hi],[0,hi],'--',color='gray')
    axes[1].set(xlabel='BLOCK backend time (s)',ylabel='JOINT backend time (s)',
                title='48 paired observations; interleaved order',xlim=(0,hi),ylim=(0,hi))
    savefig(fig,'task2_backend')
    (OUT/'图注与来源.md').write_text('''# 图注与来源

## task1_mechanism
中文：无噪声局部敏感性核验。左图为24组固定波形/信道在三个反射幅度缩放下的四维位置差预测误差；横线为中位数，不是置信区间。右图为不同源的反射扰动对两个位置的平均导数范数。局部起点使用真值，不能解释为盲定位成绩。来源task1_predictions.csv、task1_source_coupling.csv；原始文件及SHA见metrics.json。
English: Noiseless local sensitivity diagnostics. Prediction error is the relative norm of the four-dimensional position displacement error. Horizontal bars show medians. Derivative norms separate self-source and cross-source effects. Oracle local diagnostics, not blind-estimation performance.

## task2_backend
中文：48条相同观测、相同已保存盲初始化、相同多径模型与可行域的后端对照。左图仅离线重放固定后端时间点前已保存的位置，不是已验证的新停止器；完成后沿用终态。初始位置在后端计时开始前已经获得，公共前端/LOS拟合/信道初始化成本不计入横轴。右图为每条观测的完整后端耗时，包含末态检查及四个梯度方向探测；虚线表示等时。来源task2_saved_time_states.csv、task2_pairs.csv；不得把60循环触顶写成充分收敛。
English: Paired backend comparison using identical observations and common saved blind seeds. Time-performance curves replay saved states at fixed elapsed times; common initialization is excluded from the backend clock. Scatter compares full backend times including terminal checks. Iteration-limited results are not claims of full convergence.

PNG为300 dpi诊断图，PDF为矢量图；本轮不是正式论文图版排版。源匹配使用最小总平方距离的两源交换匹配，RMSE纳入全部输出，未截断；源标签仅用于评价。

补充CSV task2_retrospective_objective_targets：离线以双方较好终态目标加固定1e-6/1e-8作为共同目标，分别记录首次达到时刻。无真值参与目标设定；该回顾性目标依赖双方已完成结果，不能作为在线停止规则。未达到的样本必须保留，不能只报告两者皆达到子集的时间。
''', encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
