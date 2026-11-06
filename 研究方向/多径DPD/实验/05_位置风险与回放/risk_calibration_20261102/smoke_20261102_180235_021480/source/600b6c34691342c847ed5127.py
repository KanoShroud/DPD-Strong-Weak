"""Paired descriptive calibration reporting, including all failures/denominators."""
import csv
import html
import io
from pathlib import Path

import numpy as np
from scipy.stats import t
from storage import load,write,cost


def interval(values):
    a=np.asarray(values,float);n=len(a)
    if not n:return dict(n=0,mean=None,se=None,interval95=None)
    mean=float(a.mean())
    if n<2:return dict(n=n,mean=mean,se=None,interval95=None)
    se=float(a.std(ddof=1)/np.sqrt(n));radius=float(t.ppf(.975,n-1)*se)
    return dict(n=n,mean=mean,se=se,interval95=[mean-radius,mean+radius],
                interpretation='descriptive_paired_t_interval_no_multiplicity_claim')


def stats(rows,expected=8):
    result=dict(completed=len(rows),expected=expected,risk_applicable=sum(r['risk']['applicable'] for r in rows),methods={})
    for method in ('LOS','WIDE'):
        finite=[r['evaluation'][method] for r in rows if method in r['evaluation']]
        if not finite:
            result['methods'][method]=dict(n=0,missing=len(rows),rmse_finite_m=None);continue
        e=np.array([r['position_error'] for r in finite]);mean=e.mean(axis=0)
        mse=float(np.mean(np.sum(e**2,axis=(1,2))/2))
        bias=float(np.sum(mean**2)/2)
        variation=float(np.mean(np.sum((e-mean)**2,axis=(1,2))/2))
        errors=np.sqrt(np.sum(e**2,axis=2))
        result['methods'][method]=dict(n=len(finite),missing=len(rows)-len(finite),
            rmse_finite_m=float(np.sqrt(mse)),mse_finite_m2=mse,
            mean_error_vector_m=mean.tolist(),mean_error_squared_m2=bias,
            centered_variation_m2=variation,variance_denominator=len(finite),
            decomposition_error=abs(mse-bias-variation),p95_source_error_m=float(np.quantile(errors,.95)),
            stationary=sum(r['stationary'] for r in finite))
    eligible=[r for r in rows if r['risk']['applicable'] and all(k in r['evaluation'] for k in ('LOS','WIDE'))]
    result['risk_coverage_denominator']=len(rows)
    result['joint_subset_n']=len(eligible)
    result['subset_warning']=len(eligible)!=len(rows) or len(rows)<expected
    result['paired_calibration']={}
    for method in ('LOS','WIDE'):
        score=[r['risk'][method] for r in eligible]
        loss=[r['evaluation'][method]['loss_m2'] for r in eligible]
        result['paired_calibration'][method]=dict(mean_estimated_risk=interval(score)['mean'],
            mean_actual_loss=interval(loss)['mean'],risk_minus_loss=interval(np.array(score)-np.array(loss)))
    difference=[r['risk']['difference']-(r['evaluation']['LOS']['loss_m2']-r['evaluation']['WIDE']['loss_m2']) for r in eligible]
    result['risk_difference_minus_loss_difference']=interval(difference)
    result['solver_exceptions']=sum(r['solver_status']!='returned' for r in rows)
    result['stop_reasons']={}
    for r in rows:
        reason=r.get('stop') or r['solver_status']
        result['stop_reasons'][reason]=result['stop_reasons'].get(reason,0)+1
    result['risk_inapplicable_reasons']={}
    for r in rows:
        if not r['risk']['applicable']:
            reason=r['risk']['reason']
            result['risk_inapplicable_reasons'][reason]=result['risk_inapplicable_reasons'].get(reason,0)+1
    return result


def num(v):return '—' if v is None else f'{v:.6f}'


def summarize(out):
    out=Path(out);contract=load(out/'contract.json');manifest=contract['cases']
    rows=[]
    for case in manifest:
        p=out/'cases'/case['id']/'record.json'
        if p.exists():
            r=load(p)
            if r['case']!=case:raise RuntimeError('汇总样本合同不一致。')
            rows.append(r)
    conditions={}
    for name in dict.fromkeys(r['condition'] for r in manifest):
        chosen=[r for r in rows if r['case']['condition']==name]
        expected=sum(r['condition']==name for r in manifest)
        conditions[name]=stats(chosen,expected)
    summary=dict(complete=len(rows)==len(manifest),smoke=contract['smoke'],completed=len(rows),
                 expected=len(manifest),conditions=conditions,cost=cost(out),
                 risk_calibration_proven=False,selection_algorithm_tested=False,
                 scope='one_fixed_geometry_four_conditional_signal_channel_settings',
                 missing_ids=[c['id'] for c in manifest if c['id'] not in {r['case']['id'] for r in rows}])
    write(out/'summary.json',summary)
    buffer=io.StringIO(newline='');writer=csv.writer(buffer)
    writer.writerow(['case','condition','repeat','stop','risk_applicable','LOS_loss_m2','WIDE_loss_m2','LOS_risk_m2','WIDE_risk_m2'])
    for r in rows:
        writer.writerow([r['case']['id'],r['case']['condition'],r['case']['repeat'],r['stop'],r['risk']['applicable'],
            r['evaluation'].get('LOS',{}).get('loss_m2'),r['evaluation'].get('WIDE',{}).get('loss_m2'),r['risk'].get('LOS'),r['risk'].get('WIDE')])
    (out/'records.csv').write_text(buffer.getvalue(),encoding='utf-8-sig')
    lines=['# 位置风险32条条件校准报告','',
        '**开发联调：不属于正式32条。**' if contract['smoke'] else '**正式条件校准；不等同独立场景泛化或风险选模算法验证。**','',
        f'完成 {len(rows)}/{len(manifest)} 条。'+('全部记录完成（包括失败记录）。' if summary['complete'] else '尚未完成，当前仅为阶段结果。'),
        f'已登记执行耗时 {summary["cost"]["seconds"]/3600:.3f} h；重复/中断成本 {summary["cost"]["repeated_or_aborted_seconds"]:.2f} s。'+('存在强制中断耗时下界，未计未观察的末尾时间。' if summary['cost']['has_lower_bound'] else ''),'',
        '## 定位结果：有限输出与覆盖率一起报告','',
        '|条件|完成/计划|LOS有效输出数|LOS RMSE / m|多径有效输出数|多径RMSE / m|风险适用数|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,s in conditions.items():
        a,b=s['methods']['LOS'],s['methods']['WIDE']
        lines.append(f'|{name}|{s["completed"]}/{s["expected"]}|{a["n"]}|{num(a["rmse_finite_m"])}|{b["n"]}|{num(b["rmse_finite_m"])}|{s["risk_applicable"]}/{s["completed"]}|')
    lines += ['', '缺失输出不填零、不删记录。RMSE按全部有限输出计算，数值异常终态仍保留并另报停止原因；有缺失时不是全样本RMSE。', '',
        '## 风险差校准：只使用双方适用的共同子集','',
        '比较 (预测LOS风险−预测多径风险) − (实际LOS损失−实际多径损失)，平均应接近0才支持校准；单条符号不作为有效性判断。', '',
        '|条件|共同子集数|平均风险差校准误差 / m²|描述性95%区间|', '|---|---:|---:|---|']
    for name,s in conditions.items():
        a=s['risk_difference_minus_loss_difference'];ci=a['interval95']
        interval_text='不足2条，不计算' if ci is None else f'[{ci[0]:.6f}, {ci[1]:.6f}]'
        lines.append(f'|{name}|{a["n"]}|{num(a["mean"])}|{interval_text}|')
    lines += ['', '每条件只有8个噪声实现；区间是配对描述性t区间，未作多重比较的整体控制。风险不适用可能有选择性，不能把共同子集结论外推到所有样本；区间含零不等于证明无偏。', '',
        '## 均方误差分解及失败原因','']
    for name,s in conditions.items():
        lines.append(f'### {name}')
        for method,m in s['methods'].items():
            if m['n']:
                lines.append(f'- {method}: MSE={m["mse_finite_m2"]:.6f} = 平均偏移平方 {m["mean_error_squared_m2"]:.6f} + 中心波动 {m["centered_variation_m2"]:.6f} m²；中心波动使用分母n，确保样本恒等分解，不冒称无偏总体方差。')
        lines += [f'- 停止原因：{s["stop_reasons"]}',f'- 风险不适用：{s["risk_inapplicable_reasons"]}','']
    lines += ['## 解释边界','',
        '- 同一L20几何；每条件固定原波形、信道，只改变噪声。L3使用原未见阶数批的波形/信道，不把跨条件差异全归于路径数。',
        '- 未改变初始化、候选数、约束、停止阈值或求解预算；真实位置只用于末端评价。',
        '- 风险评分不参与本批增径或输出选择，没有检验完整风险选择器，也没有证明新理论或比强近邻更优。',
        '- 完整参照偏差、同数据选支持及非线性影响均可能造成校准误差；结果须回读判断，不自动判定创新通过。',
        '- 未运行640条。所有异常、阶段产物、尝试和耗时在cases与events.jsonl保留。','']
    (out/'条件校准报告.md').write_text('\n'.join(lines),encoding='utf-8')
    table='<table><tr><th>条件</th><th>完成</th><th>LOS RMSE</th><th>多径RMSE</th><th>风险覆盖</th></tr>'
    for name,s in conditions.items():
        table+=f'<tr><td>{name}</td><td>{s["completed"]}/{s["expected"]}</td><td>{num(s["methods"]["LOS"]["rmse_finite_m"])}</td><td>{num(s["methods"]["WIDE"]["rmse_finite_m"])}</td><td>{s["risk_applicable"]}/{s["completed"]}</td></tr>'
    table+='</table>'
    text=html.escape('\n'.join(lines))
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>位置风险条件校准</title><style>body{{background:#f5f3ed;color:#183137;font:16px/1.8 "Microsoft YaHei",sans-serif;margin:0}}main{{max-width:1000px;margin:auto;padding:30px 22px}}section{{background:white;border:1px solid #d5dfdc;border-radius:12px;padding:22px;margin:20px 0}}.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;white-space:nowrap}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #ddd}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.8 "Microsoft YaHei",sans-serif}}a{{color:#087f79}}</style><main><h1>{'开发联调' if contract['smoke'] else '32条位置风险条件校准'}</h1><p>已完成 {len(rows)}/{len(manifest)}。{'全部记录完成，请回读判断。' if summary['complete'] else '阶段报告，尚未完成。'} 记录完成不等于风险校准通过。</p><section><h2>定位误差与风险覆盖</h2><div class="scroll">{table}</div><p>RMSE单位米；缺失输出见完整报告。仅一个构型的条件重复，不声称泛化。</p></section><p><a href="条件校准报告.md">完整报告</a> · <a href="summary.json">统计JSON</a> · <a href="records.csv">逐例CSV</a></p><details open><summary>校准、偏移—波动与失败记录</summary><pre>{text}</pre></details></main></html>'''
    (out/'条件校准报告.html').write_text(page,encoding='utf-8')
    return summary
