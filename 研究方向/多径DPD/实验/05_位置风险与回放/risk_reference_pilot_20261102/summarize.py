"""Render saved pilot evidence; no estimator calls or performance selection."""
import html
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def main():
    out=Path(read(HERE/'latest.json')['path'])
    d=read(out/'result.json')
    states=[dict(tag=r['tag'],**s) for r in d['audit'] for s in r['states']]
    unique=[s for s in states if s['kind']!='final']
    bad=[s for s in unique if not s['stationary']]
    boundary=[s for s in unique if not s['interior']]
    ds=[s for r in d['diagnostics'] for s in r['models'].values()]
    local=[r for r in d['local'] if not r['skipped']]
    checks={
        'source_identity':d['identity_unchanged'],
        'all_audited_states_finite_feasible_dimension':all(s['finite'] and s['feasible'] and s['dimension_ok'] for s in states),
        'fixed_linear_identity':d['linear']['passed'],
        'predeclared_eight_states_applicable':len(ds)==8 and all(s['applicable'] for s in ds),
        'four_local_direction_tests':len(local)==4 and all(s.get('local_response_pass',False) for s in local),
        'twelve_or_fewer_fixed_support_solves':d['fixed_support_local_solves']<=12,
        'no_new_blind_trials':d['new_blind_localization_runs']==0,
    }
    diag_seconds=sum(s['seconds'] for s in ds)
    max_mixed=max(s['mixed_relative_error'] for s in ds if 'mixed_relative_error' in s)
    max_hstep=max(s['hessian_relative_change'] for s in ds if 'hessian_relative_change' in s)
    table=[]
    for r in d['diagnostics']:
        s=r.get('plugin_scores_m2',{})
        e=r['evaluation_only_saved_losses_m2']
        table.append([r['tag'],s.get('LOS'),s.get('WIDE'),e['LOS'],e['WIDE']])
    history_costs={}
    old=HERE.parent/'unknown_order_20261030'
    for key in ('feasible_repair','feasible_unseen'):
        root=Path(read(old/(key+'_latest.json'))['path'])
        for r in read(root/'result.json')['rows']:
            if r['tag'] in ('L20_L0','L20_L2','L20_L3','L20_HIGH'):
                history_costs[r['tag']]=dict(seconds=r['seconds'],basis=r['timing_basis'])
    # Planning only: four fixed existing conditions, eight fresh noise records each.
    raw_estimate=8*(sum(v['seconds'] for v in history_costs.values())+diag_seconds)
    budget=dict(proposed_records=32,conditions=4,noise_repeats_per_condition=8,
                historical_full_pipeline_costs=history_costs,
                measured_eight_state_derivative_seconds=diag_seconds,
                central_hours=raw_estimate*1.25/3600,allowance_fraction=.25,
                caution='One timing per condition, two reused older timings; not a runtime guarantee. 32 not executed.')
    refit_seconds=sum(r['base_fit']['seconds']+r.get('plus',{}).get('seconds',0)+r.get('minus',{}).get('seconds',0) for r in local)
    summary=dict(checks=checks,checks_pass=all(checks.values()),audited_records=len(d['audit']),
                 audited_state_records=len(states),distinct_model_states=len(unique),
                 nonstationary=bad,boundary=boundary,max_mixed_error=max_mixed,max_hessian_step_change=max_hstep,
                 full_pilot_seconds=d['seconds'],local_refit_seconds=refit_seconds,budget=budget,
                 risk_calibrated=False,selection_algorithm_implemented=False,new_blind_trials=0)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=[
        '# 位置风险参照核验与短测结果', '',
        '2026-11-02。结论：本次预定内部点上的非零残差敏感性计算和局部响应短测通过；实际定位风险校准尚未完成，未接入风险选模。', '',
        '## 1. 实际完成范围', '',
        f'- 回读{len(d["audit"])}条已有观测、{len(states)}条状态记录；终态重复登记后，实际为{len(unique)}个不同阶段状态。',
        '- 预先指定L20的无反射、两反射、未见三反射及总反射功率超过直达四条，每条LOS/完整模型，共8个状态。不是按真值挑选。',
        f'- 固定支持局部扰动求解{d["fixed_support_local_solves"]}次：无反射和三反射，各LOS/完整模型，分别基点与正负小扰动。不是新盲定位或独立性能样本。',
        f'- 完整短测墙钟{d["seconds"]:.2f}秒；8状态导数核验{diag_seconds:.2f}秒；12次局部求解自身合计{refit_seconds:.2f}秒。另有一次前缀冒烟，单独保留，不计独立样本。',
        '- 源码/输入哈希不变，初始化、主求解器、阈值、300次预算未修改；未启动重复校准或640条。', '',
        '## 2. 正确性和边界', '',
        f'- 观测—梯度交叉导数最大相对误差{max_mixed:.3g}；两步长Hessian最大相对变化{max_hstep:.3g}；8状态均通过预定条件。',
        '- 使用实际非零残差目标Hessian，不用零残差GN替代。零残差构造另行核验GN极限。通用隐函数/风险恒等式不列原创。',
        f'- 全轨迹中{len(bad)}个状态重算驻点不满足原门槛、{len(boundary)}个中间状态触及约束。未删除、未用内部点公式覆盖。',
    ]
    for s in bad:
        lines.append(f'  - {s["tag"]}，{s["kind"]}，阶数{s["order"]}，投影梯度{s["projected_gradient"]:.8g}。这是原LOS阶段状态重算，不是完整多径终态失败。')
    lines += ['  - 边界状态集中在L20_HIGH增项11—23，均可行且满足驻点门槛；边界最优解不是数值错误，但内部点敏感性公式不适用。', '',
              '|局部短测|预测与中心位置响应的相对差|原基点重拟合漂移 / m|', '|---|---:|---:|']
    for r in local:
        lines.append(f'|{r["tag"]} / {r["model"]}|{100*r.get("response_relative_error",0):.4f}%|{r["center_drift_m"]:.3g}|')
    lines += ['', '## 3. 插件风险分数：只作诊断，不作校准结论', '',
              '所有数值为两个源平均平方位置误差尺度（m²），不是RMSE。右列是单次保存误差，不是重复实验MSE。', '',
              '|观测|LOS估计风险|完整模型估计风险|LOS单次平方误差|完整模型单次平方误差|', '|---|---:|---:|---:|---:|']
    for row in table:
        lines.append('|'+row[0]+'|'+'|'.join('不适用' if v is None else f'{v:.6f}' for v in row[1:])+'|')
    lines += ['',
              '多径三个例子的分数方向与当次误差改善一致；无反射例分数略偏向完整模型，但当次误差略偏向LOS。单次相反既不能证明风险估计失败，也不能以其余三例宣布风险有效。',
              '完整模型分数仍主要是局部方差项，并未证明其偏差为零。该分数可计算及导数正确，不意味着它能区分多种多径复杂度，更不意味着支持同数据最小风险选模。', '',
              '## 4. 理论—算法增量判断', '',
              '本轮推进的是实际RF非零残差估计器与已有风险思想之间的数值衔接，不是新定理或独立论文贡献。新增证据是：在预定8个内部终态上可稳定计算位置敏感性，且4个局部响应试验吻合。风险是否校准、宽模型偏差、选择后乐观性仍是核心缺口。',
              '不能把全过程所有模型纳入一个无条件风险选择器：已有13个边界状态需要合法的约束敏感性处理；本轮按批准不开发KKT或交叉拟合模块。', '',
              '## 5. 下一步与预算（尚未执行）', '',
              '建议先做同一个既有L20构型下的四个已固定条件，各8个新噪声实现，共32条条件校准观测。每条件固定原波形和信道，条件间及重复间使用独立噪声种子；算法仍每条从IQ盲初始化，LOS与完整模型在同一观测上配对。噪声由生成器产生仅供生成，算法噪声尺度仍由观测估计。只评价LOS与完整模型两个端点的风险差，先检验完整参照偏差，不能称为完整模型选择验证。',
              '统计独立单位是噪声实现（条件于固定波形/信道）；不宣称新构型泛化。共同初值与同一观测配对，保留边界/非驻点/异常，报告公式覆盖率，不替补。8次只能发现较大失准，不能给出高精度偏差或小收益结论。',
              f'预算中心约{budget["central_hours"]:.2f}小时：四条件历史完整求解成本之和×8，加本轮两端点导数成本×8，再加25%余量。每条件仅有一个历史计时，且两项为旧版本等价复用；建议预留4—6小时并逐例检查预算，不承诺按中心值完成。局部暖启动耗时不得用于估计盲全流程。',
              '通过此关后，才评估中间复杂度的风险排序及选择偏差，并安排新波形/信道复核。若参照偏差导致排序失真，则停止当前插件风险构造，不通过改场景、挑候选、调阈值挽救。等待用户确认后运行，不自动扩大。', '',
              '## 6. 复现与证据', '',
              f'- 原始输出：[{out.name}/result.json]({out.name}/result.json)',
              f'- 审计汇总：[{out.name}/summary.json]({out.name}/summary.json)',
              '- [事前协议](执行协议.md)；[诊断源码](pilot.py)。源码快照和输入身份在运行目录preflight.json。',
              '- 本轮使用已读FIC/GSURE来源及上轮公式核验，不宣称新增全文精读或全面近邻检索。',
              '- Python语法编译、Ruff F/E9检查通过。新输出未覆盖原实验。', '']
    report='\n'.join(lines)
    (HERE/'短测结果与下一步.md').write_text(report,encoding='utf-8')
    # Compact standalone reader, exact report preserved below in readable paragraphs.
    table_html='<table><thead><tr><th>观测</th><th>LOS风险</th><th>完整模型风险</th><th>LOS单次误差²</th><th>完整模型单次误差²</th></tr></thead><tbody>'
    for row in table:
        table_html+='<tr><td>'+html.escape(row[0])+'</td>'+''.join('<td>'+('不适用' if v is None else f'{v:.4f}')+'</td>' for v in row[1:])+'</tr>'
    table_html+='</tbody></table>'
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>位置风险：正确性短测</title>
<style>body{{margin:0;background:#f5f3ed;color:#183137;font:16px/1.8 "Microsoft YaHei",sans-serif}}main{{max-width:960px;margin:auto;padding:36px 22px}}h1{{line-height:1.4}}.panel{{background:white;border:1px solid #d5dfdc;border-radius:12px;padding:22px;margin:22px 0}}.lead{{background:#e5f2ef;border-left:4px solid #087f79;padding:20px}}a{{color:#087f79}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #ddd}}.scroll{{overflow:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.9 "Microsoft YaHei",sans-serif}}small{{color:#52666b}}</style><main>
<small>2026-11-02 · 冻结定位器 · 未启动正式校准</small><h1>风险计算接上了实际定位器，<br>但还不能说风险判断已经可靠。</h1>
<div class="lead">8个预定内部终态的非零残差导数核验通过；4组固定支持局部响应通过。新增盲定位样本为0，局部求解12次。完整短测约{d['seconds']/60:.1f}分钟。</div>
<div class="panel"><h2>完成了什么？</h2><p>核对14条旧观测、{len(unique)}个不同阶段状态；明确找出1个非驻点LOS状态及13个约束边界状态。未删除这些结果，也未把边界最优解误写成算法失败。</p><p>当前可用的是内部点公式；完整风险选模还要处理边界、参照偏差和同数据选择。初值、峰数、约束、阈值均未改。</p></div>
<div class="panel"><h2>分数与当次误差：只能看线索</h2><div class="scroll">{table_html}</div><p><small>单位m²；右侧是单次损失，不是平均风险。无反射例方向相反，三个多径例方向一致，都不足以判断校准是否成立。</small></p></div>
<div class="panel"><h2>下一步需确认</h2><p>建议同一L20构型的四个固定条件各8次新噪声，共32条盲全流程条件校准。预算中心约{budget['central_hours']:.2f}小时，建议预留4—6小时；样本少，不能替代独立泛化或正式论文统计。</p><p>先判断完整模型能否作为风险参照，不开发新搜索器，也不声称端到端加速。640条仍未启动。</p></div>
<p><a href="短测结果与下一步.md">完整报告</a> · <a href="{out.name}/summary.json">审计汇总</a> · <a href="{out.name}/result.json">原始结果</a> · <a href="执行协议.md">事前协议</a></p>
<details><summary>展开过程与全部限定</summary><pre>{html.escape(report)}</pre></details></main></html>'''
    (HERE/'位置风险短测报告.html').write_text(page,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
