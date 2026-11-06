"""Report only saved diagnostics; does not run localization."""
from pathlib import Path
import json
import numpy as np
from structural_probe import ROOT, write, sha

A=ROOT/'structure_20261027_202324_816706'
D=ROOT/'alternatives_20261027_202706_051461'
G=ROOT/'gradient_20261027_203059_256888'


def load(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    out=ROOT/'review_20261027';out.mkdir(exist_ok=True)
    a=load(A/'result.json');d=load(D/'result.json');g=load(G/'result.json')
    assert a['passed'] and d['passed'] and g['passed']
    z=dict(np.load(A/'inputs.npz'));checks={};min_bound=1-np.sqrt(2*10**(-.6))
    for layout in [0,1]:
        gg=z[f'layout{layout}_g'];delta=z[f'layout{layout}_delta']
        ratio=np.sum(abs(gg[:,:,1:])**2,axis=2)/abs(gg[:,:,0])**2
        checks[f'L{layout}/DRR']=bool(np.allclose(ratio,10**(-.6),rtol=1e-12))
        checks[f'L{layout}/separation']=bool(np.min(np.diff(delta,axis=2))>=3)
    for r in a['rows']:
        checks[r['case']+str(r['nf'])+'/rank_stability']=len({s['nuisance_rank'] for s in r['spectra']})==1
    for r in g['rows']:
        checks[r['case']+'/acceptance']=r['final_objective']<=r['first']+1e-12 and r['best_relative_residual']**2<=r['first']+1e-12 and r['valid']
    assert all(checks.values())
    cap=sum('LIMIT' in r['message'] for r in g['rows']);normal=sum(r['success'] for r in g['rows'])
    gtol=sum(r['projected_gradient_inf']<=1e-7 for r in g['rows'])
    residuals=[r['best_relative_residual'] for r in g['rows']]
    smallest=min(min(r['spectra'][1]['singular']) for r in a['rows'])
    nullmax=max(r['free_LOS_delay_remaining'] for r in a['rows'])
    data=dict(status='LOCAL_STRUCTURE_SUPPORTED_GLOBAL_DIAGNOSTIC_INCONCLUSIVE_NO_IQ_RELEASE',
        ideal_operator_cases=len(a['rows']),parameter_models=18,structural_checks=len(a['checks']),
        minimum_normalized_singular=smallest,free_LOS_delay_max_relative=nullmax,
        wrong_position_trials=len(g['rows']),normal_returns=normal,iteration_caps=cap,projected_gradient_passes=gtol,
        relative_residual_range=[min(residuals),max(residuals)],
        warning='Residuals are obtained-fit values, not certified minima or lower bounds. No RMSE or blind success-rate experiment.',
        comparison_seconds=[a['seconds'],d['seconds'],g['seconds']],DRR6_two_reflection_amplitude_lower_bound=min_bound,
        checks=checks,source_hashes={str(p):sha(p) for p in [A/'result.json',A/'inputs.npz',D/'result.json',G/'result.json',Path(__file__)]})
    write(out/'audit_summary.json',data)
    rows=''
    for layout in [0,1]:
        for paths in [1,2,3]:
            r=next(x for x in a['rows'] if x['case']==f'L{layout}_Q01_P{paths}' and x['nf']==96)
            rows+=f'<tr><td>{layout}</td><td>{paths-1}</td><td>4 / 4</td><td>{min(r["spectra"][1]["singular"]):.4f}</td></tr>'
    text=f'''# 位置辨识结构核验：已有局部依据，尚未放行性能比较

## 执行范围

用户批准多轮、沿主线推进。本轮执行的是理论模型的理想频率算子检查及固定错误位置解释搜索，不是18条有限时间IQ记录实验，更不是90条有噪定位统计。采用已登记五站、两个预定构型、独立BPSK有限符号串的RRC频谱（B=12.5MHz、滚降0.25）、最多两反射，生成DRR=6dB。同源路径间BΔ≥3；未知波形在诊断中逐频自由消元，没有利用真实波形辅助求解。真值用于局部导数及有利初值的诊断，不把这种结果称为盲定位。

## 第一轮：位置变化是否会完全被吸收

18个参数模型（两构型×A/B/双源×0/1/2反射）各用48/96/192个非均匀频率点核验，共54项。非均匀点用于理想响应采样，不是从有限IQ抽频或论文运行配置。扣除未知波形、复增益和反射时延的一阶变化后，单源/双源位置导数在全部已检点具有满列秩。归一化最小奇异值最小{smallest:.6f}，不是RMSE、CRB或成功率。三个SVD阈值下辅助参数秩一致；标量/向量前向、差分梯度、直接/顺序投影等{len(a['checks'])}项检查通过。

允许LOS时延脱离几何自由调整后，位置导数可被辅助参数完全解释，剩余相对量最大{nullmax:.3e}。这是受控的非物理消融，用于说明几何联系的必要性，不是当前物理模型的失败。

## 第二轮：波形/路径不唯一是否必然破坏定位

代数例：若各站共享一个源的公共滤波因子c(f)，H_m(f)=D_m(p,f)c(f)，则H_mS=D_m(p,f)[c(f)S]。把该公共因子归入源波形或信道可产生同一观测，源位置不变。两例独立数值重建一致。该特殊公共因子例仅说明无需把全部信道/波形唯一恢复作为目标，不计入主随机信道模型，也不证明不存在其他位置歧义。

三角不等式还给出条件界：若反射分量功率和为rho|g0|²、反射数≤L，则|H(f)|≥|g0|(1-√(L rho))（右侧为负时不提供正下界）。本轮L=2、DRR=6dB时正下界为{min_bound:.4f}|g0|；而用户一般条件仅为DRR>0，不能推广为全频无凹陷。此界约束单源链路，不排除两个源混合后的抵消；只是常规不等式，不是新颖性结论。

## 第三轮：固定错误位置能否得到等价解释

两构型、分别移动A的x或B的y坐标20/200m、每项两个初始化，共16项。路径增益满足LOS强于反射和；每相邻路径的B时延间隔限定[3,20]，范围比生成区间宽，但仍是有限的参数域。波形按当前信道重新最小二乘估计；不把有限频率系数自动视为满足所有源独立统计条件的物理波形。

250轮有限差分控制全部触顶，不能作为反例不存在的依据。随后仅做一次解析梯度复核：76维梯度在全部16个初值上通过中央差分检查，目标与原实现一致；固定上限5000轮，得到{normal}项正常返回、{cap}项触顶，其中只有{gtol}项达到登记的投影梯度1e-7门槛，其余正常返回基于目标变化停止。已获得的相对拟合残差为{min(residuals):.4f}—{max(residuals):.4f}，不能写成最小可达残差或可辨识证明。进一步全局搜索亦未执行。

三批数值运行墙钟分别约{a['seconds']:.2f}/{d['seconds']:.2f}/{g['seconds']:.2f}秒。两次非线性诊断总32项，不是32个独立统计样本。未用增加随机样本或参数扫描掩盖收敛问题。

## 近邻条件核对

- Mulleti等，IEEE TSP2020，DOI 10.1109/TSP.2020.3011804：本轮核对p.4627、pp.4629—4630，§II的公共单源卷积、尺度/时移歧义及A1—A5。库内ZIR6J5Y9/U68LCSG6。其单源稀疏FIR理论不直接证明本项目双源连续时延的位置可辨识。
- Fabrizio与Farina，GEMS2014，DOI 10.1049/iet-rsn.2014.0058：本轮取得出版社12页PDF并核对§2—3相关内容，重点p.1260 §3.2、式(10)要求节点内模式波前矩阵满列秩且模式数M_p<N_p。单天线节点N_p=1不能直接满足；全网五站也不能未经推导直接当作同一个小阵列来满足条件。此前“可考虑其思想”不升级为可直接迁移。未声称本轮读完全部12页或完成新颖性检索。

## 当前判断与待审事项

值得保留的依据是局部位置敏感性及几何的作用，不是已经发现新算法。尚不能判断较远位置歧义是否排除，也没有一套通过充分求解检查的常规多径盲定位参照。

这里存在推进顺序取舍：继续完善高维常规拟合，还是先进一步约化理论模型。建议先只核验常规多径参照的求解可靠性，不再自动提高本轮迭代上限、不启动90条统计、不先设计新的评分或分离模块；若需要改变先验或目标，则先审批。请用户明确是否接受将下一步收敛为这一项，避免再次进入未经验证的基线性能比较。

本轮没有生成新定位RMSE或成功率，未修改历史Xia结果、论文、Zotero。
'''
    (out/'核验结论与待审问题.md').write_text(text,encoding='utf-8')
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>位置辨识：三轮核验</title><style>
    body{{background:#f5f6f2;color:#193740;font:17px/1.8 'Segoe UI','Microsoft YaHei',sans-serif;margin:0}}main{{max-width:1040px;margin:40px auto;padding:0 24px}}h1{{font-size:36px;line-height:1.35}}h2{{margin-top:36px}}article,details,.lead{{padding:22px;background:white;border:1px solid #d9e3e2;border-radius:16px;margin:18px 0}}.lead{{background:#e8f3ee;border-left:6px solid #16847b}}.warn{{background:#fff4e5}}table{{border-collapse:collapse;width:100%;font-size:16px}}td,th{{padding:10px;border-bottom:1px solid #dce3e1;text-align:left}}small{{color:#5c7177}}a{{color:#08787a}}summary{{font-weight:bold;cursor:pointer}}footer{{margin:35px 0}}</style><main>
    <small>2026-10-27 · 理论结构与数值诊断，不是定位性能报告</small><h1>几何约束确实留下了位置信息，<br>但还不能跳到大规模定位比较</h1>
    <div class="lead"><b>目前可以继续研究，但不能宣布算法已成立。</b><p>在所查模型的真实位置附近，波形和多径无法完全吸收位置变化。较远错误位置的拟合检查仍有收敛问题，因此暂不启动90条有噪统计。</p></div>
    <h2>① 看位置变化能不能被“调信道”掩盖</h2><article><p>五站、两个预定构型、单源/双源、0/1/2条反射，检查18个参数模型、三组频率取样，共54项。全部保留了所需的位置导数维数。</p><table><tr><th>构型</th><th>每链路反射数</th><th>双源位置有效维数</th><th>归一化最小奇异值（96频点）</th></tr>{rows}</table><p>这不是“54次定位全成功”，而是理想模型的局部结构检查；没有运行盲搜索。</p></article>
    <h2>② 分清哪些不唯一其实不影响位置</h2><article><p>同一个公共滤波效应可以记在波形中，也可以记在各站信道中，观测和位置都不变。因此，恢复全部波形与路径不应成为定位的前置目标。</p><p>另一个控制中，把LOS时延也允许自由调整，位置变化便被完全吸收。说明“直达时延由位置决定”是有效信息，不能被参数化时悄悄放开。</p></article>
    <h2>③ 主动尝试错误位置解释</h2><article class="warn"><p>固定偏移20m/200m的位置，重新拟合合法信道与未知波形，共16项。第一版全部触顶；核对解析梯度后仅追加了一次固定预算复核。</p><p><b>{normal}/16正常返回，{cap}/16仍触顶；仅{gtol}/16达到投影梯度门槛。</b>观察到的拟合残差不是最小残差的证明，不能用它宣称错误位置已被排除。</p><p>没有把残差当定位误差，也没有通过降低停止标准或删除困难项将它包装成通过。</p></article>
    <h2>文献带来的实际约束</h2><article><p><a href="https://doi.org/10.1109/TSP.2020.3011804">盲多通道辨识TSP论文</a>有助于区分无关歧义，但主要是单源卷积理论。</p><p><a href="https://doi.org/10.1049/iet-rsn.2014.0058">GEMS论文</a>虽提及分布式单天线节点，其分离条件却依赖节点内足够多的接收通道；不能直接搬成五个分布式单天线站的DPD算法。</p></article>
    <h2>现在需要决定什么？</h2><article><b>建议先把“常规多径参照能否可靠求解”这一关做清楚。</b><p>不启动90条统计，不再自动增加5000轮上限，不先拼新模块。先确认数值困难来自参数冗余、条件性还是其他求解因素，再决定最小处理；这一步不是创新验证。</p><p>若宁愿先进一步约化理论模型，也可以，但不应把两条路线同时铺开。本轮停在这个推进顺序取舍处。</p></article>
    <details><summary>当前没有完成的内容</summary><p>18条有限记录IQ控制、90条有噪实验、盲定位成功率与RMSE、全局唯一性证明、新方法对强基线的收益，均未完成。本轮频域算子不能直接替代有限观测窗模型。</p></details>
    <footer><a href="核验结论与待审问题.md">完整结论与理论解释</a> · <a href="audit_summary.json">独立汇总审计</a> · <a href="../structure_20261027_202324_816706/result.json">局部结构原始结果</a> · <a href="../gradient_20261027_203059_256888/result.json">非线性诊断原始结果</a></footer></main></html>'''
    (out/'位置辨识核验报告.html').write_text(html,encoding='utf-8')
    print(data);print(out/'位置辨识核验报告.html')


if __name__=='__main__':main()
