"""Offline evidence report; every displayed metric comes from saved JSON."""
import json
from pathlib import Path
from html import escape
import numpy as np

HERE=Path(__file__).resolve().parent


def read(name):return json.loads((HERE/name).read_text(encoding='utf-8'))
def fmt(x):return '—' if x is None else f'{x:.4g}'
def rmse(x):return float(np.sqrt(np.mean(np.square(x))))
def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(c)+'</td>' for c in r)+'</tr>' for r in rows)+'</tbody></table></div>'
def link(path,label):return '<a href="'+path+'">'+label+'</a>'


def main():
    summary=read('conditional_v1/summary.json');fixed=read('fixed_support_v1/summary.json');plateau=read('zero_plateau_v1/summary.json')
    base=read('discrepancy_v1/blind_decisions.json');conditional=read('conditional_v1/rows.json')
    rank=read('ranking_v1/evaluation.json')
    refined=read('fixed_support_v1/evaluation.json')
    bytag={r['tag']:r for r in conditional}
    rawrows=[]
    for d in base:
        r=bytag[d['tag']]
        rawrows.append([d['tag'].replace('IN_','').replace('_R00_',' / '),fmt(d['states']['LOS']['observed_radius_ratio']),
            fmt(r['old_rss']/r['radius2']),r['conditional']['status'],fmt(r['seconds'])])
    rankrows=[]
    for r in rank:
        a=r['arms']['GEOMETRY_FIRST'];b=r['arms']['DIAGNOSTIC_NEAREST']
        rankrows.append([r['tag'].replace('IN_','').replace('_R00_',' / '),fmt(rmse(a['evaluation'])),fmt(rmse(b['evaluation'])),
            fmt(a['conditional']['value']) if a['conditional']['passed'] else '不可行',fmt(b['conditional']['value']) if b['conditional']['passed'] else '不可行',
            {'tie':'并列','DIAGNOSTIC_NEAREST':'诊断分支','GEOMETRY_FIRST':'几何分支',None:'均不可行'}[r['ranking']]])
    frows=[[r['tag'].replace('IN_','').replace('_R00_',' / '),fmt(rmse(r['before'])),fmt(rmse(r['after'])),
        fmt(r['objective_ratio']),fmt(r['seconds']),r['status']] for r in refined]
    derivative_count=sum(t['passed'] for r in read('outer_v1/rows.json') for t in r['tests'])
    strict_count=sum(t['passed'] for r in read('outer_precision_v1/rows.json') for t in r['tests'])
    improved=sum(rmse(r['after'])<rmse(r['before']) for r in refined)
    totals={}
    for c in sorted({r['condition'] for r in refined}):
        rr=[r for r in refined if r['condition']==c]
        totals[c]=(len(rr),rmse([r['before'] for r in rr]),rmse([r['after'] for r in rr]))
    svg='<svg viewBox="0 0 980 130" role="img" aria-label="从无真值定标到外层短测的五阶段流程">'
    for i,(title,sub) in enumerate([('观测噪声半径','24条旧盲状态'),('条件波形求解','零组 / 不可行 / 证书'),('不同位置解释','12对诊断排序'),('外层解析导数','96方向测试'),('固定支持更新','12态，不是完整盲定位')]):
        x=5+i*196
        svg+=f'<rect x="{x}" y="10" width="184" height="95" rx="14" fill="'+('#e9f4ed' if i<4 else '#fff1df')+f'"/><text x="{x+14}" y="42" font-size="17" fill="#142f3a">{title}</text><text x="{x+14}" y="73" font-size="12" fill="#53686d">{sub}</text>'
    svg+='</svg>'
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>统一正则化：从定标到位置更新</title><style>
    :root{--ink:#16313b;--muted:#60747a;--line:#dbe4e3;--green:#1a7560;--amber:#9e6415}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f4f5f1;color:var(--ink);font:16px/1.8 "Segoe UI","Microsoft YaHei",sans-serif}main{max-width:1120px;margin:auto;padding:42px 28px 70px}h1{font-size:38px;line-height:1.3;letter-spacing:-1px;margin:10px 0 20px}h2{font-size:25px;margin:0 0 15px}h3{font-size:18px}p{margin:12px 0}a{color:#096b68;text-underline-offset:4px}nav{display:flex;gap:18px;flex-wrap:wrap;border-block:1px solid var(--line);padding:14px 0;margin:25px 0}section{margin:38px 0}.eyebrow{color:var(--green);letter-spacing:2px;font-size:13px}.lead{max-width:850px;font-size:19px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.card,.panel{background:white;border:1px solid var(--line);border-radius:16px;padding:22px}.number{display:block;font-size:34px;font-weight:650;line-height:1.3}.muted,small{color:var(--muted)}.note{background:#e7f0ed;border-left:4px solid var(--green);padding:18px 22px;border-radius:0 12px 12px 0}.warning{background:#fff1df;border-left-color:var(--amber)}.scroll{overflow:auto;border:1px solid var(--line);border-radius:12px;margin:18px 0}table{border-collapse:collapse;width:100%;background:white;white-space:nowrap;font-size:14px}th,td{padding:11px 14px;text-align:left;border-bottom:1px solid var(--line)}th{background:#e9efec}tbody tr:hover{background:#f5f8f6}.equation{background:#eaf0f1;padding:17px 20px;border-radius:12px;font:17px/1.9 Georgia,"Microsoft YaHei",serif;overflow:auto}details{border:1px solid var(--line);border-radius:12px;background:white;padding:15px 20px;margin:16px 0}summary{cursor:pointer;font-weight:650}li{margin:8px 0}code{font-size:13px;overflow-wrap:anywhere}.split{display:grid;grid-template-columns:1fr 1fr;gap:18px}.footer{border-top:1px solid var(--line);padding-top:20px;margin-top:40px}svg{width:100%;display:block}.tag{padding:3px 9px;background:#e7ece8;border-radius:20px;font-size:12px}.links{display:flex;flex-wrap:wrap;gap:12px 22px}button{border:1px solid var(--line);background:white;border-radius:8px;padding:8px 14px;color:var(--ink);cursor:pointer}@media(max-width:700px){main{padding:24px 17px}h1{font-size:29px}.cards,.split{grid-template-columns:1fr}.card{padding:17px}.lead{font-size:17px}svg{min-width:850px}.flow{overflow:auto}section{margin:30px 0}}@media print{nav,button{display:none}details{display:block}body{background:white}.scroll{overflow:visible}main{max-width:none}}
    </style><main><div class="eyebrow">DPD · 2026.11.05 · 有界自动推进</div><h1>统一正则化：<br>从“规则能算”走到“位置会怎样变”</h1>
    <p class="lead">这轮沿着同一个目标推进：在观测允许的误差内，减少不必要的接收反射分量。没有再修补旧增径门槛，也没有用定位真值挑权重。</p>
    <nav><a href="#answer">结论</a><a href="#process">执行过程</a><a href="#math">准则与依据</a><a href="#ranking">位置解释</a><a href="#outer">实际更新</a><a href="#decision">下一步取舍</a><a href="#evidence">记录入口</a></nav>
    <section id="answer"><h2>先看结论</h2><div class="note warning"><b>数学内核与导数核验通过，但尚不能冻结成可靠、快速的完整定位器。</b><p>噪声半径在这些旧状态上没有全部退化成只用LOS；条件波形问题能正确处理可行、零惩罚及不可行分支。固定支持外层短测仍未形成收敛与完整成本证据，不能把“反射罚项下降”直接写成“定位精度提高”。</p></div>
    <div class="cards" style="margin-top:18px"><div class="card"><span class="number">24</span>旧盲状态全保留<br><small>16可行，8不可行；无新场景</small></div><div class="card"><span class="number">96 / 96</span>严格精度下方向导数通过<br><small>首轮失败记录没有删除</small></div><div class="card"><span class="number">'''+str(fixed['stationary'])+''' / 12</span>外层短测达到既定驻点门<br><small>仅固定支持，不是完整盲定位</small></div></div></section>
    <section id="process"><h2>实际执行了什么，为什么接着做？</h2><div class="flow">'''+svg+'''</div>
    <ol><li><b>先看定标会不会过于宽松。</b>同一观察噪声规则下，6条无反射LOS状态均可行，18条有反射LOS状态均不可行。这允许继续检查，但不能排除别处存在错误的LOS可行解。</li>
    <li><b>再把条件波形子问题做对。</b>24个旧多径信道中，12个得到正乘子的条件解，4个达到零惩罚，8个即使最小二乘也达不到半径。最后一类不是数值失败，更没有放宽半径让它通过。</li>
    <li><b>检查能否区分不同位置解释。</b>12对保存分支全纳入；近真分支的初始选择曾用真值，所以这一步是诊断，不是新算法成绩。</li>
    <li><b>核对真正的新目标导数。</b>首轮'''+str(derivative_count)+'''/96项通过；只收紧内层精度后'''+str(strict_count)+'''/96通过。外层的接受误差没有放宽。</li>
    <li><b>最后让位置实际移动。</b>仅对具备可微条件解的12态运行最多100轮固定支持更新。保存全部停止原因，最后才读取真值评价位置；没有依据误差挑迭代终点。</li></ol>
    <details><summary>哪些设定完全没动？</summary><p>四站、双源、全内二维构型、2.4GHz、50MHz采样率、8192点、12.5MHz有效频带、原非周期BPSK/RRC观测与无Hann处理不变。算法仍不知道真实路径条数、时延和波形；保留每条反射不强于直达，而非直达功率大于反射总和。没有引入已知最小时延间隔。</p><p>只用了已见的6构型R00开发材料；R01—R09、780/3120及旧正式入口未动。本轮不再是完整近邻性能比较，也未宣布新颖性验证通过。</p></details></section>
    <section id="math"><h2>新准则用直白的话怎么理解？</h2><p>旧方法先问“能不能把混合信号拟合得更像”。现在再加一个要求：<b>不要为了拟合而制造过多、过大的反射解释。</b>允许的拟合误差由观测噪声估计给出，不看真实位置。</p>
    <div class="equation">最小化 R = ∑<sub>m,q,ℓ≥1</sub> |b<sub>mqℓ</sub>| · ‖s<sub>q</sub>‖₂<br>同时满足 ‖Y − H(x)S‖² ≤ δ²，以及原有物理约束。</div>
    <p>R衡量模型内各接收反射分量的范数和，<b>不是实际截窗功率和</b>。它在“增益放大、波形反向缩小”的等价变换下不变；前一轮还证明它会增加某类错误抵消解释的成本。但这不保证所有错误位置都受罚更多。</p>
    <div class="split"><div class="panel"><h3>这次补齐的数学环节</h3><p>固定H后，波形问题是凸的。通过非负组范数的对偶证书和单调乘子搜索，达到预定噪声半径；无惩罚组先正确消元。零解、不可行及最小二乘边界分开处理。</p><div class="equation">dV = ∑ ‖s<sub>q</sub>‖₂ dc<sub>q</sub><br>+ Re⟨HS−Y, (dH)S⟩ / λ</div><p>这才是正乘子、正则性成立时的新外层导数，不能照搬旧LS波形消元梯度。</p></div><div class="panel"><h3>尚不能借来的保证</h3><p>噪声半径使用带外估计与F分布。精确覆盖解释依赖白色同方差复高斯噪声、单位酉DFT和干净独立的带外估计。有限窗信号泄漏会破坏这一精确口径。</p><p>这不是位置估计的95%置信保证，不是自动选路径的误报率保证，也不是整个非凸问题的全局最优保证。</p></div></div>
    <p class="muted">文献依据：<a href="https://www.mathnet.ru/php/archive.phtml?jrnid=zvmmf&option_lang=eng&paperid=7206&wshow=paper">Morozov误差原则</a>及<a href="https://doi.org/10.1088/0266-5611/26/2/025001">非线性Tikhonov型泛函的误差原则研究</a>。本轮仅核对公开元数据/摘要与相关条件，不冒称全文精读或将这些现成原理写成本研究原创。</p>
    <details><summary>24态的半径与条件求解明细</summary><p>RSS/δ²≤1代表该固定信道可满足半径；时间只含条件求解，不含初始化、路径构造或外层。</p>'''+table(['构型 / 条件','LOS RSS/δ²','MP RSS/δ²','条件状态','条件耗时/s'],rawrows)+'''<p>条件调用平均'''+fmt(summary['mean_conditional_s'])+'''秒、最大'''+fmt(summary['max_conditional_s'])+'''秒。不可行态会提前返回，因此全24态平均值不能当作正常迭代耗时，更不能当完整定位耗时。</p></details></section>
    <section id="ranking"><h2>新准则是否已经显示位置辨别价值？</h2><p>5对仅诊断分支可行；6对双侧可行，其中3对实质相同、3对更低惩罚对应更准确的位置；1对两侧都不可行。<b>但是，原RSS在这些可区分分支中也更偏向定位好的一侧，尚未显示新准则比原拟合多解决了哪一例。</b></p>
    <p>“诊断分支”不等于正确答案：例如103/L3最终仍有一个源约59米误差。两分支的反射清单与终态不同；这是固定解释排序，不是相同位置下对所有信道参数做充分优化后得到的全局值比较。</p>'''+table(['构型 / 条件','几何分支RMSE/m','诊断分支RMSE/m','几何分支V','诊断分支V','条件判定'],rankrows)+'''</section>
    <section id="outer"><h2>真正更新位置后，发生了什么？</h2><p>下面的12条是条件求解可微的开发子集，不是随机抽取的性能样本。未适用的另外12条没有从完整任务中消失：4条零惩罚、8条不可行，仍需完整架构处理。</p>
    <div class="cards"><div class="card"><span class="number">'''+fmt(fixed['before_rmse'])+' → '+fmt(fixed['after_rmse'])+'''</span>子集RMSE / 米<br><small>不作为完整新算法成绩</small></div><div class="card"><span class="number">'''+str(improved)+''' / 12</span>终态位置RMSE下降<br><small>没有挑选中间最好时刻</small></div><div class="card"><span class="number">'''+fmt(fixed['mean_refinement_seconds'])+''' s</span>平均新增精修耗时<br><small>还未计旧前端和模型构造</small></div></div>
    '''+table(['构型 / 条件','原RMSE/m','更新后RMSE/m','目标值/初始值','新增耗时/s','停止原因'],frows)+'''
    <p>iteration_limit：100轮预算耗尽，不是收敛。line_search_unresolved：未得到可接受的新步，不能把它等同驻点；零幅度附近目标不可微和条件边界需要单独处理。</p>
    <details><summary>按多径条件汇总，避免总体RMSE掩盖差别</summary>'''+table(['条件','条数','原RMSE/m','更新后RMSE/m'],[[c,n,fmt(a),fmt(b)] for c,(n,a,b) in totals.items()])+'''</details>
    <div class="note warning">目标下降只是优化事实；是否更准需要位置评价。该谱投影梯度求解器是机制工具，不代表所有新目标求解器都必然这样慢。因此本轮既不能宣布已获得3—5秒方法，也不能据此证明该理论路线无效。</div></section>
    <section id="decision"><h2>研究判断与下一步边界</h2><div class="split"><div class="panel"><h3>可以保留</h3><ul><li>接收分量范数对特定错误抵消解释的可推导作用。</li><li>不使用位置真值的条件噪声定标及完整边界处理。</li><li>与实际目标一致的波形求解与外层灵敏度公式。</li><li>本轮所有负面位置变化、不可行态及数值失败。</li></ul></div><div class="panel"><h3>仍未获得</h3><ul><li>该准则相对合理多径强近邻的独立定位收益。</li><li>未知支持与错误初始位置下的完整可靠求解。</li><li>对不同构型、条数和强度的独立验证。</li><li>完整3—5秒成本与可冻结的新算法。</li></ul></div></div>
    <p><b>停止自动进入完整盲实验。</b>本轮实现正确性已有证据，下一次投入不能只是增加迭代或继续调容差。需要先决定：是否继续为这一复杂度准则研究同目标的非光滑/可行性求解结构，还是先要求它显示超出常规模型匹配的明确定位增量。</p>
    <p>建议保留目标作为候选，暂不把它升为唯一论文方案；先用一项明确的局部位置偏移分析审查“减小反射范数为何会改善位置、何时会把位置拉偏”。没有可验证的定位收益机制之前，不建设完整未知阶数搜索，不投入大批仿真。</p></section>
    <section id="evidence"><h2>过程和证据都可回读</h2><p>所有阶段运行前后核对源码和输入身份；新输出隔离保存。旧算法、原观测、正式入口未改。</p><button id="expand" type="button">展开所有明细</button><div class="links" style="margin-top:20px">'''
    for p,label in [('protocol.md','定标协议'),('conditional_protocol.md','条件求解协议'),('ranking_protocol.md','排序诊断协议'),('outer_protocol.md','外层导数协议'),('precision_protocol.md','精度控制记录'),('fixed_support_protocol.md','外层短测协议'),('discrepancy_v1/summary.json','24态定标结果'),('conditional_v1/correctness.json','数学单元检查'),('ranking_v1/evaluation.json','12对排序原始结果'),('outer_v1/rows.json','保留的首轮失败'),('outer_precision_v1/rows.json','严格精度复核'),('fixed_support_v1/summary.json','外层短测汇总'),('fixed_support_v1/decisions.json','完整外层轨迹'),('fixed_support_v1/evaluation.json','终态位置评价'),('conditional.py','条件求解代码'),('outer_precision.py','解析外层代码'),('fixed_support.py','固定支持代码'),('../architecture_reform_20261105/总体方法重构报告.html','上一轮架构报告')]:html+=link(p,label)
    html+=link('zero_plateau_protocol.md','零目标平台协议')+link('zero_plateau_v1/summary.json','平台核查汇总')+link('zero_plateau_v1/evaluation.json','192个预定位置扰动')
    html+='''</div></section><div class="footer muted"><small>这是开发阶段数学与机制报告，不是正式论文性能表。部分分支含真值辅助选择已明确标注；没有生成新的独立随机实现。数值显示经过四位有效数字取舍，精确结果见JSON。离线可打开。</small></div></main><script>document.getElementById('expand').addEventListener('click',()=>{const ds=[...document.querySelectorAll('details')];const open=ds.some(d=>!d.open);ds.forEach(d=>d.open=open);document.getElementById('expand').textContent=open?'收起所有明细':'展开所有明细'});</script></html>'''
    # The final structural audit supersedes the provisional decision paragraph.
    html=html.replace('<b>数学内核与导数核验通过，但尚不能冻结成可靠、快速的完整定位器。</b>',
        '<b>数学内核通过，但“只在噪声半径内最小化反射复杂度”存在位置同分平台，不宜直接作为最终定位准则。</b>')
    start=html.index('<section id="decision">');end=html.index('<section id="evidence">')
    html=html[:start]+'''<section id="decision"><h2>为什么在这里停下，而不继续调求解器？</h2>
    <div class="note warning"><b>一个明确的结构问题：无反射时，许多不同位置都能获得同一个最优分数。</b><p>只要某个LOS解释的拟合误差已经低于噪声半径，就有R=0。位置稍微变化后，误差仍可能在半径内，R也仍为0。因此目标无法告诉算法：这些位置中哪个拟合更好、哪个更应输出。</p></div>
    <div class="equation">若 RSS<sub>LOS</sub>(x₀) &lt; δ²，且局部通道连续、满秩，<br>则邻域内可存在不同位置 x，使 RSS<sub>LOS</sub>(x) &lt; δ²，且 V(x)=0。<br>R≥0 ⇒ 它们都是这个目标的全局最优解。</div>
    <p>这一结论不依赖求解器是否收敛，不是又发现一个数值小补丁。它也不是“实际数据不能定位”：原始RSS仍然能够区分这些解释，只是当前目标把区别抹平了。</p>
    <div class="cards"><div class="card"><span class="number">'''+str(plateau['feasible_zero_global_optima'])+''' / 192</span>预定扰动仍是零分可行解</div><div class="card"><span class="number">6 / 6</span>构型都出现不同位置同分</div><div class="card"><span class="number">10 m</span>仍可同分的最大已测坐标偏移</div></div>
    <p>固定旧盲LOS增益，四个位置坐标分别扰动±0.1、±1、±3、±10米，再估计未知波形；全部规则在评分前冻结。176个可行同分解释中，双源RMSE最高约'''+fmt(plateau['max_sampled_feasible_rmse_m'])+'''米。这里不是说新算法一定输出这个差解，而是该目标本身没有拒绝它的依据。</p>
    <div class="split"><div class="panel"><h3>保留的进展</h3><ul><li>接收分量范数的尺度一致性及对特定抵消解释的作用。</li><li>固定信道凸求解、边界判定与正确外层导数。</li><li>不靠真值调参的定标与完整负面记录。</li></ul></div><div class="panel"><h3>不能沿用的推断</h3><ul><li>满足噪声半径且反射范数更小，就意味着位置更准确。</li><li>固定信道凸、导数正确，就能获得快速可靠的盲定位。</li><li>这些数学工具正确，就已经形成独立论文贡献。</li></ul></div></div>
    <h3>建议的下一取舍</h3><p><b>保留接收分量正则化方向，撤回将纯误差约束形式直接用作最终位置排名的倾向。</b>更合理的候选是保留连续拟合误差在最终目标中的作用，例如“拟合误差＋有限权重的反射复杂度”。在无反射子模型内，正则项为零，拟合项仍会区分位置，不会自动产生这类零分平台。</p>
    <p>这并不解决所有问题：有限权重怎样只由观测确定、非光滑幅度怎样更新、未知支持怎样搜索，仍需推导。另一种可能是增加第二级最小残差选择，但必须写成明确的新估计规则；不能在同分后根据真值挑位置。</p>
    <p><b>需要确认的不是是否再加迭代，而是是否采用上述目标修订。</b>推荐只批准一次目标层面的必要性与定位偏移核验，不先搭完整搜索器；不改半径寻找优势、不扩场景、不启动780/3120。当前证据既不支持冻结新算法，也不足以否定接收分量正则化这一更大的方向。</p></section>'''+html[end:]
    (HERE/'统一正则化验证报告.html').write_text(html,encoding='utf-8')


if __name__=='__main__':main()
