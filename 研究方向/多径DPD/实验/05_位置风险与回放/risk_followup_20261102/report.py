"""Saved-result report; never reruns or changes a solver."""
from pathlib import Path
import html
import json
import numpy as np

HERE=Path(__file__).resolve().parent


def read(p):
    d=json.loads(Path(p).read_text(encoding='utf-8'))
    return d.get('payload',d) if isinstance(d,dict) else d


def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+
                     ['|'+'|'.join(map(str,r))+'|' for r in rows])


def curve(case):
    rows=case['scores'];last=max(r['order'] for r in rows);mx=last or 1
    top=max(max(r.get('risk',0),r['actual_loss_m2']) for r in rows)*1.08 or 1
    def points(key):
        return ' '.join(f'{45+610*r["order"]/mx:.2f},{205-170*r[key]/top:.2f}'
                        for r in rows if key in r)
    markers=''.join(f'<circle cx="{45+610*r["order"]/mx:.2f}" cy="{205-170*r[k]/top:.2f}" r="3.5" fill="{color}"/>'
                    for k,color in [('risk','#087d83'),('actual_loss_m2','#bf622b')]
                    for r in rows if k in r)
    return f'''<svg viewBox="0 0 700 245" role="img" aria-label="{case['case']}风险与实际平方误差">
<path d="M45 25V205H655" stroke="#9caeae" fill="none"/>
<polyline points="{points('risk')}" fill="none" stroke="#087d83" stroke-width="2.5"/>
<polyline points="{points('actual_loss_m2')}" fill="none" stroke="#bf622b" stroke-width="2.5"/>
{markers}
<text x="45" y="18" font-size="13">m² · 纵轴上限 {top:.3f}</text>
<text x="45" y="227" font-size="13">0</text><text x="580" y="227" font-size="13">终态 K={last}</text>
<text x="245" y="240" font-size="13">全站双源累计新增分量数 K</text></svg>'''


def main():
    audit=Path(read(HERE/'latest.json')['path']);clean=Path(read(HERE/'latest_clean.json')['path'])
    selector=Path(read(HERE/'latest_selector.json')['path'])
    a,c,s=[read(p/'result.json') for p in (audit,clean,selector)]
    formal=HERE.parent/'risk_calibration_20261102/formal_20261102_181736_622970'
    rows=a['noise']
    ratios=np.array([r['sigma_ratio'] for r in rows])
    leakage=max(r['signal_oob_fraction'] for r in rows)
    cont=[r for r in a['continuations'] if not r['skipped']]
    high=[]
    for r in a['continuations']:
        ident=r['case'];f=formal/'cases'/ident;sol=read(f/'solve.json');w=read(f/'WIDE_risk.json')
        if not w['applicable']:
            continue
        if r.get('accepted'):
            with np.load(audit/(ident+'_continued_influence.npz'),allow_pickle=False) as z:l=z['L']
            # Source contract uses UNIT=c/B; retrieve exact unit from frozen context via input positions.
            unit=299792458/12500000
            p=np.array(r['x'][:4]).reshape(2,2)*unit
        else:
            old=read(f/'LOS_risk.json')
            if not old['applicable']:
                continue
            l=np.array(old['L']);p=np.array(sol['los']['p'])
        lw=np.array(w['L']);sigma=w['sigma2_est']
        risk=np.sum((p-np.array(sol['final']['p']))**2)/2+sigma*np.sum(l*lw)-sigma*np.sum(lw*lw)/2
        e=read(f/'record.json')['evaluation']
        high.append(dict(case=ident,LOS_risk=float(risk),WIDE_risk=float(sigma*np.sum(lw*lw)/2),
                         original_LOS_loss=e['LOS']['loss_m2'],WIDE_loss=e['WIDE']['loss_m2']))
    # Original LOS losses here are explicitly kept as a sensitivity reference, not substituted outputs.
    (HERE/'HIGH六条诊断风险.json').write_text(json.dumps(high,ensure_ascii=False,indent=2),encoding='utf-8')
    geometry=table(['记录','最近候选误差/米','盲选得分优势','解释'],[
        [r['case'],', '.join(f'{v:.3f}' for v in r['coverage_m']),f"{r['reward_margin']:.6f}",
         '正确附近候选存在，排序选错' if r['case'] in ('HIGH_R00','HIGH_R05') else '选中正确区域']
        for r in a['geometry']])
    selection=table(['记录','可用/全部模型','风险选择K','原终态K','选中位置误差/米','原终态误差/米'],[
        [r['case'],f"{r['applicable_count']}/{r['candidate_count']}",r['selected_order'],
         r['scores'][-1]['order'],f"{np.sqrt(r['selected_loss_m2']):.6f}",f"{np.sqrt(r['final_loss_m2']):.6f}"]
        for r in s['rows']])
    changed=[r for r in s['rows'] if r['selected_order']!=r['scores'][-1]['order']]
    worsened=[r['case'] for r in changed if r['selected_loss_m2']>r['final_loss_m2']]
    improved=[r['case'] for r in changed if r['selected_loss_m2']<r['final_loss_m2']]
    decision=f'三条探针中{len(changed)}条改变输出：改善{len(improved)}条、损害{len(worsened)}条。这是探索性描述，不是独立统计验证。'
    decomposition=[]
    for case in s['rows']:
        folder=formal/'cases'/case['case']
        end=read(folder/'solve.json')['final']
        pw=np.array(end['p'])
        error=np.array(read(folder/'record.json')['evaluation']['WIDE']['position_error'])
        contract=read(formal/'contract.json')['contract']
        condition=case['case'].split('_')[0]
        with np.load(contract['conditions'][condition]['bank'],allow_pickle=False) as z:truth=z['p']
        if np.sum((pw[::-1]-truth)**2)<np.sum((pw-truth)**2):error=error[::-1]
        rw=case['scores'][-1]['risk']
        chosen=next(row for row in case['scores'] if row['order']==case['selected_order'])
        delta=np.array(chosen['position'])-pw
        disagreement=float(np.sum(delta*delta)/2)
        predicted_cross=chosen['risk']-rw-disagreement
        actual_cross=float(np.sum(delta*error))
        identity=disagreement+actual_cross-(chosen['actual_loss_m2']-case['final_loss_m2'])
        assert abs(identity)<1e-8
        decomposition.append(dict(case=case['case'],disagreement_m2=disagreement,
            predicted_covariance_cross_m2=predicted_cross,
            actual_single_record_cross_m2=actual_cross,identity_error=identity))
    (HERE/'选模差值恒等核验.json').write_text(json.dumps(decomposition,ensure_ascii=False,indent=2),encoding='utf-8')
    decomp_table=table(['记录','位置差平方/2','估计交叉项','实际单次交叉项'],[
        [d['case'],f"{d['disagreement_m2']:.6f}",f"{d['predicted_covariance_cross_m2']:.6f}",
         f"{d['actual_single_record_cross_m2']:.6f}"] for d in decomposition])
    text=f'''# 风险尺度、困难例与中间模型核验

## 结论摘要

正式32条结果保持不变。未发现约2倍的FFT/复实坐标方差错误；HIGH五条近门槛LOS在剩余预算内收敛，两条严重失败已定位到前端候选排序。完成预定三条保存轨迹风险探针。{decision}

## 1. 本轮做了什么

全部32条噪声尺度与输入身份检查；HIGH全部8条冻结几何前端重放；5次原剩余预算LOS继续；1次仅用于诊断的无噪中心固定支持拟合；L0/L2/L3各R00的所有已接纳中间模型风险评分。无新盲定位样本，未改原定位器、候选数、门槛或增径规则，未运行640条。阶段耗时分别{a['seconds']:.2f}、{c['seconds']:.2f}、{s['seconds']:.2f}秒，合计{(a['seconds']+c['seconds']+s['seconds'])/60:.2f}分钟；含原终态导数复用，不当作新算法完整时间。

## 2. 风险高估：没有发现倍数错误，但绝对校准仍未通过

32条观测带外方差估计/生成方差为{ratios.min():.6f}—{ratios.max():.6f}；最大带外信号泄漏仅占噪声方差{leakage*100:.4f}%。正交FFT能量与sqrt(2)实坐标能量恒等均通过。模拟真噪声只用于诊断，没有交给定位器或风险评分。

L0共同无噪中心的局部预测MSE为{c['expected_linear_mse']:.6f} m²；原8份噪声的局部预测平均损失为{c['realized_linear_mse']:.6f} m²；实际盲LOS平均损失为{c['observed_mse']:.6f} m²。固定该四维线性响应，100000组“8重复”高斯响应平均损失的95%模拟范围为[{c['fixed_linear_eight_sample_interval95'][0]:.6f}, {c['fixed_linear_eight_sample_interval95'][1]:.6f}] m²；实际均值处于该参考分布的{c['empirical_percentile_actual']*100:.2f}百分位附近。它处于低侧，小样本波动能解释部分差异，但这不是实际非线性自适应算法无偏的证明。

带噪终态的局部方差约0.0387 m²，无噪中心约0.0331 m²，说明导数评价位置也有影响；有限噪声下局部预测与实际响应仍有差异。不乘经验系数，不把误差一概归于样本少，绝对风险校准仍待独立证据。

## 3. HIGH：数值精度和初始化是不同问题

5条LOS各只增加2次函数评价，全部达到原1e-7投影梯度。累计函数评价最大{max(r['total_nfev'] for r in cont)}，小于300；位置最大变化{max(r['position_change_m'] for r in cont)*1e6:.3f}微米。与原多径端点结合后，风险可比较覆盖由1/8增为6/8；这只是单列诊断，没有替换正式结果。不能把两条错误区域失败放宽成通过。

{geometry}

两条严重失败的正确附近候选已经存在，但原全站峰覆盖评分更偏好错误组合。继续提高搜索网格精度不是本轮证据指出的首要修复。没有据此开发新的初始化或扩大候选数。

## 4. 风险是否真的会选择不同模型

评分只用观测；决策先落盘，再读取真值。完整参照代价必须承担，降低选中K不等于节约已有参照构建成本。K是全5站×2源累计的有效反射分量数，不是每条链路的物理反射数。

{selection}

{decision} 选择规则为既有插件风险最小，精确同分才取更低K；没有按真值修改规则。不适用状态不伪造评分，覆盖率同时报告。真值最优模型只作评价，不能作为算法选择。

### 为什么分数更低，单条实际误差仍可能更高？

记 d=p_M−p_W，e_W=p_W−p_true，两源平均平方误差差严格等于 ||d||²/2+dᵀe_W。当前风险差用局部敏感性给出的协方差交叉项代替未知的dᵀe_W。这个代替针对期望风险，不保证单条噪声实现的优劣排序；若参照有偏，期望中还存在均值交叉项，不能自动忽略。

{decomp_table}

以上交叉项只在评分后用于解释，禁止交给选择器。逐条恒等式已数值核对；这是误差分解，不是新的理论贡献或“风险公式必错”的证明。下一步若只依据这几个样本给协方差项乘系数或加阈值，就会成为针对样本调参，本轮没有这样做。

## 5. 当前边界与下一决策

数值一致性、定位效果、风险校准与创新性分别判断。三条保存轨迹不能确认稳定收益，HIGH错误分支风险仍未覆盖；本轮不自动把条件子集结论外推到全体，也不开展32条全轨迹或640条。应根据风险排序是否存在具体可解释的系统问题决定有限下一步，而非重新调参或增加一个初始化模块。

当前需要决策的是验证投入，而不是调整参数：是否给现有冻结风险规则一次32条保存轨迹的完整配对回放，以平均位置平方误差、失败覆盖和完整评分成本判断是否继续。建议若批准则只做这一项，不换公式、不换前端、不生成新场景；先预算，超过1小时交一键脚本。三条探针中的单次损害不足以否定期望风险，因此不据此直接改成新评分或关闭整条研究线；同样，一条毫米级改善不足以放行扩大实验或宣称创新完成。

## 来源

- 阶段1—2：[{audit.name}/result.json]({audit.name}/result.json)
- 无噪诊断：[{clean.name}/result.json]({clean.name}/result.json)
- 轨迹探针：[{selector.name}/result.json]({selector.name}/result.json)
- [执行协议](执行协议.md)
- [原32条正式报告](../risk_calibration_20261102/formal_20261102_181736_622970/条件校准报告.html)
'''
    (HERE/'核验结果与下一决策.md').write_text(text,encoding='utf-8')
    cards=''.join(f'<section><h2>{r["case"]}：风险与实际位置损失</h2>{curve(r)}<p>青色：观测风险；橙色：真值评价损失。选择 K={r["selected_order"]}，原终态 K={r["scores"][-1]["order"]}。</p></section>' for r in s['rows'])
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>位置风险后续核验</title>
<style>body{{margin:0;background:#f4f2ed;color:#203b40;font:16px/1.8 "Microsoft YaHei",sans-serif}}main{{max-width:1000px;padding:28px 20px;margin:auto}}section{{background:white;border:1px solid #d7e0db;border-radius:14px;padding:22px;margin:20px 0}}h1{{line-height:1.35}}h2{{font-size:21px}}svg{{width:100%;height:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.85 "Microsoft YaHei",sans-serif}}a{{color:#087d83}}.lead{{border-left:5px solid #087d83}}.note{{color:#596e72}}</style>
<main><h1>位置风险：这一轮查清了什么？</h1><section class="lead"><p>没有发现方差倍数错误；5条近门槛数值问题查清；2条严重失败来自候选排序。</p><p>{html.escape(decision)}</p><p class="note">原正式结果不改，640条未启动。新结果是诊断和有限离线选择探针。</p></section>{cards}
<section><h2>完整证据与解释</h2><pre>{html.escape(text)}</pre></section>
<p><a href="核验结果与下一决策.md">Markdown报告</a> · <a href="{selector.name}/result.json">轨迹原始结果</a> · <a href="{audit.name}/result.json">尺度及困难例结果</a> · <a href="{clean.name}/result.json">无噪中心诊断</a></p></main></html>'''
    (HERE/'位置风险后续核验.html').write_text(page,encoding='utf-8')
    print(selection);print(decision)


if __name__=='__main__':
    main()
