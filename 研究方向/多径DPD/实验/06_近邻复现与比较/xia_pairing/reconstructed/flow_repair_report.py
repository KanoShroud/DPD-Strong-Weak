"""Independent paired-error audit and report for an explicit flow correction."""
from itertools import permutations
from pathlib import Path
import numpy as np
from pilot import ROOT, sha, write
from trace_audit import read

OUT=ROOT/'repair_20261027_184715_930592'


def main():
    s=read(OUT/'summary.json');protocol=read(OUT/'protocol.json');source=Path(protocol['input'])
    assert s['full_run'] and s['n']==84
    checks={};movement=[];pairs=[]
    for p in sorted(OUT.glob('r*_G*.json')):
        r=read(p);z=dict(np.load(source/f'input_{r["realization"]:02d}.npz'))
        checks[p.stem+'/observation_identity']=r['input_sha']==sha(source/f'input_{r["realization"]:02d}.npz')
        new=r['outputs']['repair'];old=r['outputs']['original']
        if new['positions'] is None:
            checks[p.stem+'/same_failure']=old['positions'] is None and new['steps']==old['steps']
            continue
        old_valid=[step for step in old['steps'] if step['output']['positions'] is not None][-1]
        values={}
        for name,pos in [('original',old_valid['output']['positions']),('repair',new['positions'])]:
            pos=np.array(pos);options=[np.sum((pos[list(order)]-z['truth'])**2,axis=1) for order in permutations([0,1])]
            e=np.sqrt(min(options,key=lambda a:a.sum()))
            expected=r['original_retained' if name=='original' else 'repair_terminal']['error_m']
            checks[p.stem+'/'+name+'_matching']=np.allclose(e,expected,rtol=1e-12,atol=1e-12)
            values[name]=e
        pairs.append(values)
        movement.extend(np.linalg.norm(np.array(new['positions'])-np.array(new['steps'][0]['output']['positions']),axis=1).tolist())
    for name in ['original','repair']:
        e=np.array([r[name] for r in pairs])
        checks[name+'/aggregate']=np.isclose(np.sqrt(np.mean(e**2)),s['overall'][name]['joint_RMSE_m'])
    for name,h in protocol['input_hashes'].items():checks['unchanged/'+name]=sha(source/name)==h
    for name,h in protocol['solver_hashes'].items():checks['solver/'+name]=sha(Path(name))==h
    assert all(checks.values())
    extra=dict(passed=True,count=len(checks),checks={k:bool(v) for k,v in checks.items()},
        displacement_median_m=float(np.median(movement)),displacement_max_m=float(max(movement)),
        source_sha=sha(__file__))
    write(OUT/'report_audit.json',extra);(OUT/'flow_repair_report.py').write_bytes(Path(__file__).read_bytes())
    a=s['overall']['original'];b=s['overall']['repair']
    change=100*(1-b['joint_RMSE_m']/a['joint_RMSE_m'])
    table=''.join(f'<tr><td>{r["G"]}</td><td>{r["SNR"]:g}</td><td>{r["SIR"]:+g}</td><td>{r["common_n"]}/{r["n"]}</td><td>{r["original_joint_RMSE_m"]:.2f}</td><td>{r["repair_joint_RMSE_m"]:.2f}</td></tr>' for r in s['groups'])
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Xia流程缺陷修正验证</title><style>body{{max-width:1060px;margin:auto;padding:32px 22px;background:#f4f4ee;color:#223f49;font:16px/1.85 "Microsoft YaHei",system-ui}}section{{background:white;border:1px solid #dce5e6;border-radius:15px;padding:24px;margin:20px 0}}h1{{font-size:32px;line-height:1.45}}h2{{font-size:22px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #dce5e6}}th{{background:#edf4f2}}.scroll{{overflow:auto}}.cards{{display:flex;gap:14px;flex-wrap:wrap}}.cards>div{{flex:1;min-width:220px;background:#eef5f2;padding:16px;border-radius:10px}}.note{{background:#fff1db;padding:14px;border-left:4px solid #bd8b35}}a{{color:#007b73}}summary{{cursor:pointer;font-weight:bold}}.muted{{color:#6a7d83}}</style>
<p class="muted">2026-10-27 · 明确的方法改动 · 正方形84例配对 · 不作为论文原版复现</p><h1>保住了两个输出，<br>但没有解决找错位置</h1>
<section><div class="cards"><div><b>流程稳定性</b><p>原终态0/84完整；修正版37/84完整。所有首轮已有位置对都保留到最后。</p></div><div><b>定位精度</b><p>同37例双源RMSE：{a['joint_RMSE_m']:.2f} → {b['joint_RMSE_m']:.2f}m，降幅约{change:.3f}%。</p></div><div><b>研究判断</b><p>后轮误删问题可以修正，但这项修改尚未产生值得主张的总体定位精度收益。</p></div></div></section>
<section><h2>只改了什么？</h2><table><tr><th>环节</th><th>原重建版</th><th>流程修正版</th></tr><tr><td>初次1000点全域搜索</td><td colspan="2">原评分、原阈值、原分组，完全相同</td></tr><tr><td>后两轮各1000点</td><td colspan="2">两个旧位置周围各500点，标准差10m；相同随机数流</td></tr><tr><td>后轮如何更新</td><td>重新计算全局阈值、重新分组</td><td>按重采样来源保留两组，在各组内取原评分最高点</td></tr><tr><td>评分/频点/迭代数</td><td colspan="2">不变；未删频点、加窗、正则化或增加迭代</td></tr></table><p>没有把真实位置告诉求解器，没有注入旧中心作为保底候选，没有新加两位置排斥或防重合规则。这是公开标注的<b>流程修正</b>，不是仅补论文未说明细节，也不冒充作者原程序。</p></section>
<section><h2>37例有输出，究竟是什么意思？</h2><p>首次搜索中，47例本来就没有得到完整位置对，修正版也原样停止；其余37例已经有两个候选，原版后来丢失，修正版全部完成三轮。</p><p class="note">37/84不是定位成功率，也不是新找回37个正确目标。保组规则本来就保证有限且非空组各有一个输出；真正需要验证的是位置准不准。双源20m内仍为0/84。</p><p>原版最后没有完整输出，不能直接算终态双源RMSE。公平的精度比较使用同一37例：原版取“最近一次完整输出”（不看真值挑轮），修正版取最终输出。其余47例没有伪造误差，也没有静默删除后声称全体RMSE。</p></section>
<section><h2>相同37例的定位精度</h2><table><tr><th>指标</th><th>原版最近完整输出</th><th>修正版最终输出</th></tr><tr><td>双源整体RMSE/m</td><td>{a['joint_RMSE_m']:.2f}</td><td>{b['joint_RMSE_m']:.2f}</td></tr><tr><td>源A匹配RMSE/m</td><td>{a['A_RMSE_m']:.2f}</td><td>{b['A_RMSE_m']:.2f}</td></tr><tr><td>源B匹配RMSE/m</td><td>{a['B_RMSE_m']:.2f}</td><td>{b['B_RMSE_m']:.2f}</td></tr><tr><td>双源均20m内</td><td>0/37</td><td>0/37</td></tr></table><p>逐例整体平方误差：25例降低、12例增加。整体RMSE只减少约1.71m，而原误差约2km，不能据此主张已经恢复论文性能。</p><details><summary>分条件查看，避免只看平均值</summary><div class="scroll"><table><tr><th>路径G</th><th>SNR/dB</th><th>SIR/dB</th><th>共同有输出/全部</th><th>原版RMSE/m</th><th>修正版RMSE/m</th></tr>{table}</table></div></details><p>输出身份未知，双源误差继续使用既定最小平方距离一对一匹配。JSON另存第一输出对A的误差，但修正版保留初始分支身份，不把它直接冒充论文终态全局最高峰的同义指标。</p></section>
<section><h2>为什么只是很小的精度变化？</h2><p>两轮局部更新后，每个分支相对首轮位置的移动中位数{np.median(movement):.2f}m，最大{max(movement):.2f}m。原来的许多候选已经错到数百米或数公里外；在旧位置附近移动几十米，很难弥补这种错误区域选择。</p><p>这说明保组解决了“结果被后轮删掉”，没有解决“首次位置本来就选错”。它不证明原评分是唯一原因；初始覆盖、评分和首轮分组仍需分别判断。</p></section>
<section><h2>计算代价与公平性</h2><p>同进程两线程、原版/修正版交错顺序运行。平均完整求解时间{a['mean_seconds']:.3f}s → {b['mean_seconds']:.3f}s；评分位置总数{a['score_positions']:,} → {b['score_positions']:,}。修正版继续完成了原版提前失败后未做的工作，不能把增加的计算隐去。</p><p>两版首次搜索逐点相同；进入第二轮时粒子与评分逐点相同；原版重跑轨迹与保存轨迹完全一致。后续修正版的组内选峰由独立切片实现核验，匹配误差由枚举两种配对复算。</p><p>84条来自12个基础随机实现、7个条件，并非84个独立构型。本轮是机制修正验证，不宣称统计显著性或新颖性。</p></section>
<section><h2>到这里应该怎样收尾？</h2><p><b>保留这个明确区分的流程修正版，但不继续给它增加迭代、调分组参数来追求成功。</b>本轮已回答：后轮删组缺陷可修，修复后的总体精度仍差。</p><p>这项修正本身目前不足以形成论文核心贡献。若继续核验Xia性能，下一步应回到首轮评分/候选有效性，并保持与常规可靠定位参照的同输入比较；不能把“修复一个流程问题”写成“已解决双源多径定位”。本轮未继续修改评分、未启动新大批。</p></section>
<section><h2>证据与复现</h2><p>原版84次重放＋修正版84次，共168次完整求解调用，总运行及保存{s['seconds']:.2f}s；此前2例和3例工程前缀不计入本批。{s['checks']}项运行检查、{len(checks)}项独立结果审计通过，原输入和求解代码身份不变。</p><p><a href="protocol.json">运行前冻结协议</a> · <a href="audit.json">运行检查</a> · <a href="report_audit.json">配对复算</a> · <a href="summary.json">汇总</a> · <a href="summary.csv">分条件CSV</a> · <a href="result.json">逐例结果</a> · <a href="flow_repair.py">修正版与配对入口</a> · <a href="flow_repair_report.py">报告生成</a></p></section></html>'''
    (OUT/'流程缺陷修正验证.html').write_text(page,encoding='utf-8')
    md=f'''# Xia流程缺陷修正验证

用户已批准“缺陷修正验证”。版本XIA_FLOW_REPAIR_GROUPWISE_V1明确为方法改动，不是作者实现补全或新颖性结论。首轮原评分/阈值/分组不变；后两轮按重采样来源各500点、标准差10m，分别取原评分最高点。不重新全局筛选或径向分组、不加迭代/保底旧点/排斥规则，不改评分、频点、IQ、源站设置。

输入square_20261027_182013_211372的84例、12基础实现、7条件；两版本交错顺序运行，各84次，共168次。原重建版与保存轨迹完全一致，首轮两版本相同，第二轮候选和分数相同。2例工程前缀repair_20261027_184646_781380、3例覆盖修正分支的前缀repair_20261027_184706_137961均不混入本批。当前完整结果repair_20261027_184715_930592。

终态完整输出0/84→37/84；首轮已有两候选的37例全部完成，首轮失败47例不变。输出可用不等于定位成功。配对误差比较：原版最近完整输出对修正版终态，恰为同37例，不以真值选轮。双源匹配RMSE{a['joint_RMSE_m']:.9f}→{b['joint_RMSE_m']:.9f}m，降幅{change:.6f}%；A {a['A_RMSE_m']:.9f}→{b['A_RMSE_m']:.9f}m；B {a['B_RMSE_m']:.9f}→{b['B_RMSE_m']:.9f}m。25例平方误差下降、12例上升，双源20m均0/37（全84仍0/84）。缺失47例无虚构误差，不能称37例RMSE为全84总体RMSE。

初始至最终分支移动中位{np.median(movement):.6f}m、最大{max(movement):.6f}m；最终两分支最小间距{s['final_pair_separation_min_m']:.6f}m。本批未发生两分支重合，但没有加入防重合条件。第一输出对A另存，修正版分支身份固定，不能把其语义等同原文终态全局最高峰。

平均完整求解{a['mean_seconds']:.6f}→{b['mean_seconds']:.6f}s；评分位置{a['score_positions']}→{b['score_positions']}，修正版完成原版失败后未做的工作；总运行含保存{s['seconds']:.3f}s。{s['checks']}项运行检查、{len(checks)}项独立配对/身份检查通过，编译/Ruff通过。没有新正式扩测或参数调优。

结论：明确修复了后轮误删导致的输出中断，但未产生实质总体精度改善，不足以支撑论文核心贡献；停止在此输出支线加轮次/调参数。若继续，应回到首轮评分/候选有效性，另行设计与可靠常规参照的限定同输入核验。新核验本轮未执行。[HTML](流程缺陷修正验证.html)。
'''
    (OUT/'流程修正结论.md').write_text(md,encoding='utf-8')
    previous=ROOT/'score_20261027_182953_721266/评分与筛选机制报告.html'
    text=previous.read_text(encoding='utf-8')
    if 'id="repair-latest"' not in text:
        text=text.replace('<section>',f'<section id="repair-latest"><h2>新增：缺陷修正验证已完成</h2><p>修正版保住37例完整输出，但同样本双源RMSE仅由2021.83降至2020.11m。</p><p><a href="../{OUT.name}/流程缺陷修正验证.html">查看严格配对结果 →</a></p></section><section>',1)
        previous.write_text(text,encoding='utf-8')
    print('REPORT',OUT/'流程缺陷修正验证.html','checks',len(checks))


if __name__=='__main__': main()
