"""Source-linked offline report of the objective test and route decision."""
import json
from pathlib import Path
from html import escape
import re

HERE=Path(__file__).resolve().parent


def main():
    a=json.loads((HERE/'analysis.json').read_text(encoding='utf-8'))
    old=(HERE.parent/'regularization_gate_20261105/统一正则化验证报告.html').read_text(encoding='utf-8')
    style=re.search(r'<style>(.*?)</style>',old,re.S).group(1)
    def f(v):return f'{v:.3f}'
    def table(head,rows):
        return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+escape(str(c))+'</th>' for c in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(c))+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    conditions=table(['原条件 / 每档6条','LOS RMSE/m','MP RMSE/m','MP单例RMSE中位数/m','MP最大单例RMSE/m'],
        [[r['condition'],f(r['los_rmse']),f(r['mp_rmse']),f(r['mp_median_case_rmse']),f(r['mp_max_case_rmse'])] for r in a['old_all_conditions']])
    pairs=table(['同观测、同后端控制','几何分支RMSE/m','近真分支RMSE/m','近真分支正常停止'],
        [[r['condition'],f(r['geometry_rmse']),f(r['near_rmse']),str(r['near_normal'])+'/6'] for r in a['same_observation_initialization']])
    soft=table(['起点来源','固定权重/anchor','条数','无正则参考RMSE/m','软目标RMSE/m','误差下降条数'],
        [['旧盲状态' if r['arm']=='BLIND' else '真值辅助选支诊断',r['factor'],r['count'],f(r['reference_rmse']),f(r['updated_rmse']),str(r['improved'])+'/'+str(r['count'])] for r in a['soft_control']])
    s='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>多径DPD：继续、收缩，还是换架构？</title><style>'''+style+'''</style><main>
    <div class="eyebrow">2026.11.05 · 修订目标执行与研究取舍</div><h1>多径DPD：<br>继续、收缩，还是换架构？</h1>
    <p class="lead">建议保留多径定位问题与已有精修能力，停止把“自由增径＋高维局部拟合＋复杂度惩罚”当作唯一解法。先缩小场景难度，集中解决双源与路径的错误关联，比继续修补当前整套流程更值得投入。</p>
    <nav><a href="#executed">本轮执行</a><a href="#difficulty">环境是否太难</a><a href="#theory">原理解释</a><a href="#choice">具体取舍</a><a href="#records">过程记录</a></nav>
    <div class="note"><b>这不是否定DPD，也不是证明复杂场景不可解。</b><p>同一批双反射观测，只换成真值辅助选出的近真起点，已有后端就能达到约0.31米RMSE。信息并没有普遍消失；当前主要难点之一是仅凭观测找到并识别正确的双源解释。</p></div>
    <section id="executed"><h2>本轮按批准执行了什么？</h2><div class="cards"><div class="card"><span class="number">192 / 192</span>预定LOS位置扰动不再与中心同分<br><small>拟合项修复了上一准则的平台退化</small></div><div class="card"><span class="number">24</span>旧状态全部纳入登记<br><small>23个可做受控更新，1个条件不可行</small></div><div class="card"><span class="number">24 / 24</span>适用状态的小扰动位移检查符合容差<br><small>其中18个扰动终态达严格驻点门</small></div></div>
    <ol><li>采用“拟合误差＋有限权重的接收反射分量惩罚”，一次位置更新内固定权重。</li><li>检查四个位置方向的导数，以及原来192个同分位置是否重新可区分。</li><li>固定保存的增益、反射时延和路径清单，只更新位置与未知波形，检查目标怎样推动位置。这是公开的机制控制，不是简化后冒充完整新算法。</li><li>将观察半径导出的anchor权重冻结。0.005与0.01倍仅核对局部响应，不挑选最优档；1倍核验原定标的实际影响。</li><li>最后才读取真实位置，完整保存改善、退化和未驻定状态。</li></ol>
    '''+soft+'''
    <p><b>结论：目标形式修正有效，但没有形成定位精度收益。</b>小权重只产生小幅移动，anchor权重下盲子集只3/12改善、总体RMSE略升；近真诊断子集也没有整体改善。因此没有追加权重扫描或建设完整搜索器。</p>
    <details><summary>实现、时间与公平性细节</summary><p>首次单元检查发现新位置包装器提前结束导数枚举，漏掉第二个源的位置导数。保留v1失败，修复为跳过非位置变量后在v2重新运行，所有方向核验通过；旧算法不受影响。</p><p>response共'''+f(a['response_all_seconds'])+'''秒，包含参考位置求解、条件权重、Hessian和三档受控计算，不是完整算法耗时。12个起点来自旧盲状态，另12个来自历史真值辅助近真选支，严格分开。后者1个不满足条件内点，未强行给权重。两组都不是新的独立测试数据。</p><p>场景、噪声、四站、未知波形、原始IQ、逐径幅度约束和时延范围都没改。没有运行新的简单场景、独立重复、780/3120或新完整盲算法。</p></details></section>
    <section id="difficulty"><h2>是不是反射太多、太强，所以根本做不好？</h2><p>有可能增加难度，但目前不能把它当作全部原因。先看原24条跨构型开发结果，均使用完整终态、没有丢掉大误差：</p>'''+conditions+'''
    <div class="split"><div class="panel"><h3>单反射也已有问题</h3><p>单反射、总反射/直达比0.75时，旧MP的RMSE约30.93米，最差单例约59.35米。并不是只有三反射或反射总功率大于直达才失败。</p><p>双反射总比也是0.75，仍有大误差；但其中位单例RMSE只有约0.42米，说明大误差与普通精度表现必须分开看。</p></div><div class="panel"><h3>强多径可能放大瓶颈</h3><p>三反射/总比1.5确实更差，近真起点下仍有后端问题。它可以暂时不作为第一版必须解决的条件。</p><p>不过L2→L3同时改变条数与总强度；每构型只有一个已见实现，尚不能估计两因素的独立因果效应或普遍可解边界。</p></div></div>
    <h3>更关键：保持观测不变，只换位置分支</h3>'''+pairs+'''
    <p>这是另一个历史配对批，不与上表混算。双反射六条的近真分支全部正常，最大单例RMSE约0.51米；它强烈提示当前盲关联/初值问题值得重视。近真分支靠真值选取，因此它是诊断证据，绝不是可部署算法成绩。</p>
    <div class="note warning">不能据此说“已有算法已经解决双反射”，也不能说“只要减弱多径就能发表论文”。需要解决的是如何在不知道答案时区分源与路径，而非只展示进入正确区域后的精修精度。</div></section>
    <section id="theory"><h2>为什么一个合理惩罚项仍未能救回定位？</h2><p>惩罚项偏好“更小的反射解释”，并不知道真实位置在哪里。位置、波形与信道可以相互补偿，减小反射范数时，位置既可能变好，也可能被拉偏。</p>
    <div class="equation">Q<sub>λ</sub>(p) = min<sub>S</sub> { ½‖Y−H(p)S‖² + λR }<br>p<sub>λ</sub> − p₀ ≈ −λG⁻¹∇r(p₀)</div>
    <p>G是局部拟合目标的完整Hessian，r还包括最小二乘波形随位置变化的影响。该公式描述移动方向，<b>不保证它朝向真值</b>。本轮12个适用参考状态的小权重位移符合这一局部解释；它属于常规敏感性分析，不是新理论贡献认证。</p>
    <p>所以，本轮没有找到“换一个整体目标，当前盲搜索就自动变好”的捷径。接收分量惩罚仍可作为工具，但不宜继续承担解决所有关联、阶数、局部极值和时延问题的任务。</p>
    <details><summary>相关文献支持什么，不支持什么？</summary><p><a href="https://epubs.siam.org/doi/abs/10.1137/1.9781611972344.ch5">SIAM正则化章节</a>提供一般正则化与选参背景，本轮不把标准工具作为原创。<a href="https://www.mdpi.com/1424-8220/18/3/892">Du等多径DPD原文</a>明确使用已知转发体位置与接收阵列，其成功不能直接迁移为无地图单天线双源问题的保证。<a href="https://link.springer.com/article/10.1155/ASP.2005.37">Weiss–Amar多源DPD</a>强调跨站共同位置与源关联，但阵列/AOA条件不同，也不是当前问题的现成解法。</p><p>本轮是定向原理和假设核对，不是新的全面全文综述；没有因近邻条件不匹配而自动提供真实路径信息。</p></details></section>
    <section id="choice"><h2>建议怎样收敛，而不是继续修补？</h2><div class="split"><div class="panel"><h3>保留</h3><ul><li>双源混叠、单天线分布式站、未知波形这个核心问题。</li><li>同源路径共享波形与跨站几何一致性。</li><li>已有多径DPD作为局部精修工具和高成本参照。</li><li>跨构型数据、近邻实现及失败记录。</li></ul></div><div class="panel"><h3>停止作为唯一主线</h3><ul><li>错误源对进入后端，再期待自由增径把它修正。</li><li>通过反复改评分、权重或停止细节寻找局部提升。</li><li>第一版同时要求未知阶数、强三反射、全构型和3秒全部解决。</li></ul></div></div>
    <h3>我倾向先做一个边界清楚的较简单问题</h3>'''+table(['项目','建议（尚未执行）'],[
        ['主要场景','每链路0/1反射；总反射/直达功率比0、0.5、0.75'],
        ['保留的难点','双源同频混叠、未知数据波形、四站、多个全内构型'],
        ['信息条件','算法不获得逐样本真实路径条数、时延、功率或真实起点'],
        ['构型与重复','保留6种构型；方法冻结后按已批准10次独立实现核验'],
        ['主要算法问题','区分双源与各自路径，减少错误源对；精修只承担必要局部工作'],
        ['扩展条件','双反射在主场景通过后加入；三反射/高总反射强度暂作后续压力'],
        ['论文证据','相容多径强近邻、RMSE与完整成本，不能只胜LOS模型']])+'''
    <p>0.75不是几乎没有多径的环境；旧算法已在这个条件出现明确失败。缩小范围是为了把真正的机制讲清楚，而非给容易数据重新命名成创新。生成器只用0/1反射，也不等于可以暗中让算法知道恰好一条反射；如果采用已知至多一条的模型类上限，必须另行明确并让基线享有同样信息。</p>
    <div class="note"><b>现在需要的决策：</b>是否同意把主问题先收缩到上述低阶多径条件，并把算法重点转向双源—路径关联，而不继续将当前高维自由增径与正则化作为唯一推进主线？本轮没有替你改变场景或启动该新方向。</div></section>
    <section id="records"><h2>完整记录</h2><button id="expand">展开明细</button><div class="links" style="margin-top:18px">'''
    for path,label in [('protocol.md','执行前协议'),('理论与路线判断.md','完整推导与研究判断'),('implementation_correction.md','失败与实现修正'),('math_v1/summary.json','保留的数学门失败'),('math_v2/summary.json','有效数学门'),('response_v2/summary.json','受控求解汇总'),('response_v2/decisions.json','所有决定与轨迹'),('response_v2/evaluation.json','后置位置评价'),('analysis.json','分条件与配对数据'),('soft_v2.py','软目标实现'),('run_v2.py','实验入口')]:s+='<a href="'+path+'">'+label+'</a>'
    s+='''</div></section><div class="footer muted">数据来自已见开发记录；未生成新的独立场景。本文所有优化与位置结论分别标注条件控制、真值辅助诊断和完整流程边界。离线可读。</div></main><script>document.querySelector('#expand').onclick=()=>{const ds=[...document.querySelectorAll('details')],v=ds.some(d=>!d.open);ds.forEach(d=>d.open=v);document.querySelector('#expand').textContent=v?'收起明细':'展开明细'}</script></html>'''
    s=s.replace('</nav>','</nav><div class="note warning"><b>用户最新范围修订：先保留0、1、2条反射。</b><p>以下历史报告中的“先缩至0/1”建议已被替代：只有0/1/2范围仍无法推进时，才讨论只保留0/1。强度等其他已批准参数不自动降低。下一步先核对近邻论文的场景与信息条件，未启动新实验。</p><a href="范围修订与近邻场景对齐.md">查看修订范围与Xia原文场景核对</a></div>',1)
    (HERE/'软目标验证与路线判断.html').write_text(s,encoding='utf-8')


if __name__=='__main__':main()
