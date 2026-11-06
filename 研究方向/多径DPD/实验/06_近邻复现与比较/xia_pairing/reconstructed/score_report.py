"""Report the bounded spectral/selection audit; never relabel it a new solver."""
import csv
from pathlib import Path
import numpy as np
from pilot import ROOT, sha, write, xia
from trace_audit import read

OUT=ROOT/'score_20261027_182953_721266'


def main():
    s=read(OUT/'summary.json'); protocol=read(OUT/'protocol.json'); source=Path(protocol['input'])
    assert all(sha(source/n)==h for n,h in protocol['input_hashes'].items())
    checks={}
    low_high=np.r_[np.ones(500),np.ones(500)*2]
    t=xia.threshold(low_high)
    checks['two_equal_groups_T_1_2']=np.isclose(t['threshold'],1.2)
    checks['low_group_all_deleted']=not t['keep'][:500].any() and t['keep'][500:].all()
    dominant_high=xia.threshold(np.r_[np.ones(499),np.ones(501)*2])
    checks['high_modal_bin_all_deleted']=dominant_high['threshold']==3 and not dominant_high['keep'].any()
    spectral=list(csv.DictReader((OUT/'spectral.csv').open(encoding='utf-8-sig')))
    sensitivity=list(csv.DictReader((OUT/'sensitivity.csv').open(encoding='utf-8-sig')))
    for a in s['sensitivity']:
        sub=[r for r in sensitivity if float(r['removed'])==a['removed']]
        checks[f'RMSE/{a["removed"]}']=np.isclose(np.sqrt(np.mean([float(r['first_A_m'])**2 for r in sub])),a['first_A_RMSE'])
        checks[f'count/{a["removed"]}']=sum(float(r['first_nearest_m'])<=100 for r in sub)==a['nearest100']
    assert all(checks.values())
    extra=dict(passed=True,checks={k:bool(v) for k,v in checks.items()},grouped=[],script_sha=sha(__file__))
    for snr in [-10,5,20]:
        sub=[r for r in spectral if float(r['SNR'])==snr and int(r['G'])==3]
        extra['grouped'].append(dict(SNR=snr,n=len(sub),
            low5_ratio_energy_share_median=float(np.median([float(r['low5_ratio_energy_share']) for r in sub])),
            low5_noise_dominated_fraction_median=float(np.median([float(r['low5_noise_dominated_fraction']) for r in sub])),
            low5_cancellation_fraction_median=float(np.median([float(r['low5_strong_cancellation_fraction']) for r in sub]))))
    outer=[]
    for f in OUT.glob('*_spectral.npz'):
        z=np.load(f);outer.append(float(np.mean(abs(z['frequency'][z['low_power_order'][:100]])>=200000)))
    extra['low5_outside_rectangular_BPSK_first_null_fraction_median']=float(np.median(outer))
    write(OUT/'report_audit.json',extra);(OUT/'score_report.py').write_bytes(Path(__file__).read_bytes())
    table=''.join(f'<tr><td>{r["removed"]:.0%}</td><td>{r["winner_changed"]}/84</td><td>{r["nearest100"]}/84</td><td>{r["recovered_nearest100"]} / {r["damaged_nearest100"]}</td><td>{r["first_A_RMSE"]:.1f}</td></tr>' for r in s['sensitivity'])
    bysnr=''.join(f'<tr><td>{r["SNR"]}</td><td>{r["low5_ratio_energy_share_median"]:.1%}</td><td>{r["low5_noise_dominated_fraction_median"]:.1%}</td><td>{r["low5_cancellation_fraction_median"]:.1%}</td></tr>' for r in extra['grouped'])
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原始评分与筛选机制核查</title><style>body{{max-width:1080px;margin:auto;padding:32px 22px;background:#f5f4ef;color:#203c47;font:16px/1.85 "Microsoft YaHei",system-ui}}section{{background:white;border:1px solid #dbe5e7;border-radius:15px;padding:24px;margin:20px 0}}h1{{font-size:31px;line-height:1.5}}h2{{font-size:22px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #dee6e7}}th{{background:#edf5f3}}.scroll{{overflow:auto}}.note{{background:#fff1d8;padding:14px;border-left:4px solid #bd8830}}.equation{{font:17px/1.8 Cambria,serif;background:#eef3f5;padding:16px;overflow:auto}}.cards{{display:flex;gap:15px;flex-wrap:wrap}}.cards>div{{flex:1;min-width:220px;background:#edf4f1;padding:18px;border-radius:10px}}a{{color:#007b73}}summary{{cursor:pointer;font-weight:bold}}.muted{{color:#677d86}}</style>
<p class="muted">2026-10-27 · 正方形五站 · 同84条已保存观测 · 新定位求解0次</p>
<h1>小分母会扰乱评分，<br>重复筛选又会删掉整组候选</h1>
<section><h2>本轮结论</h2><div class="cards"><div><b>评分确实敏感</b><p>参考功率最低的5%频点，占频谱比值平方和的中位数70.6%。但这不是“贡献了70.6%的最终定位评分”。</p></div><div><b>删频点不是完整解法</b><p>排除5%后，靠近任一源100m内的第一峰只从35增至38次，并有3次变差。仍存在大量错误峰。</p></div><div><b>后轮删组已能解释</b><p>30次删掉整个第二候选群，7次全部筛空；原阈值与重采样后分布的配合存在明确问题。</p></div></div></section>
<section><h2>1．原评分为什么容易受小参考值影响？</h2><p>原文式(9)对每个频点计算的比值可以直接写成：</p><div class="equation">R₁*(f) Rₘ(f) / |R₁(f)|² = Rₘ(f) / R₁(f)</div><p>当参考站某个频点的值很小，其他站该频点的信号或噪声都会被放大。随后再按候选位置补偿相位、跨频率求和，最后把各站的结果相乘。</p><p>所有84例中，最低1%参考功率频点占比值平方和的中位数53.6%，最低5%占70.6%。这些最低5%频点中，参考站“纯信号功率低于噪声期望功率”的比例中位数为88.5%；两源强抵消的比例中位数仅6%。所以本批小参考值不宜主要解释为两源相互抵消。</p><p class="note">“纯信号”和噪声期望功率由保存的生成分量事后计算，定位器没有获得这些真值。强抵消定义为 |A+B|²/(|A|²+|B|²)&lt;0.1。频谱比值平方和只是数值集中度；最终评分有复数相消及跨站乘积，不能把70.6%直接当定位误差或评分贡献。最低5%频点引起的站间复数和变化范数比，其中位数约29.4%。</p><details><summary>按SNR查看，而非只看混合统计</summary><div class="scroll"><table><tr><th>平均接收SNR/dB</th><th>最低5%占比值平方和</th><th>其中信号低于噪声的比例</th><th>其中强抵消的比例</th></tr>{bysnr}</table></div><p>每行24例，G=3，包含两档SIR；均为逐例统计的中位数。</p></details><p>这些低参考功率频点有80%的中位比例位于当前矩形BPSK首零点±200kHz以外。但矩形脉冲在首零点外仍有信号旁瓣，不应直接把那里全部称为“纯带外噪声”。本轮没有据此新增滤波器。</p></section>
<section><h2>2．临时拿掉这些频点，原候选排序会怎样？</h2><p>只对原来的1000个第一轮候选重新评分；按观测参考功率选最低1%或5%频点，不看真实位置。没有重跑筛选/分组/精修，没有改FFT的2000点归一化，也没有挑一个最有利比例作为新参数。</p><div class="scroll"><table><tr><th>诊断时排除</th><th>最高分候选改变</th><th>第一峰距任一源≤100m</th><th>变好 / 变差例数</th><th>固定目标A第一峰RMSE/m</th></tr>{table}</table></div><p>这里“变好/变差”只指是否跨过100m诊断范围，不是论文20m成功率。暂时删频点能影响排序，却不能恢复全部正确峰；1%和5%的效果也不单调。上述RMSE是冻结候选上的首峰诊断，<b>不是修复后的完整定位器性能</b>。</p><p>仍未区分完的因素包括：源间混合、参考站信道和未知脉冲形状补全、粗粒子覆盖等。现有证据支持“小分母敏感性”，不支持“全部失败都由它造成”。</p></section>
<section><h2>3．为什么第二轮会把整组候选删掉？</h2><p>第一轮粒子铺满全区域，有大量低分背景点。重采样后，1000个粒子集中在两个旧位置附近，每组500个，标准差只有10m；这时评分常常成为两个很窄的群。</p><p>实际评分最小值/最大值的中位数：第一轮约0.000104，第二轮升至0.262。候选分布已经不同，原文步骤v却要求再次执行同一个全局直方图阈值。</p><div class="equation">T = L + (b/5)(H−L)<br>L：最低分；H：最高分；b：出现最多评分的箱编号（1～10）</div><p><b>一个不用调参的简单算例：</b>500点得1分、500点得2分。按当前并列时选较低箱的约定，阈值是1.2，得1分的整组全部被删；若高分箱成为众数，则阈值是3，连2分的点也全部被删。不是浮点误差，而是直接按公式得到的结果。</p><p>这解释了原规则在“低方差的两组评分”上为什么可能无法保留两组。算例只说明筛选结构，不是假装仿真得到两个正确源；它也不证明作者实际代码如何处理未明示边界。</p><div class="scroll"><table><tr><th>已保存轮次</th><th>进入</th><th>第二组全删、第一组有保留</th><th>阈值高于最高分</th><th>完整输出</th></tr><tr><td>第二轮</td><td>37</td><td>30</td><td>6</td><td>1</td></tr><tr><td>第三轮</td><td>1</td><td>0</td><td>1</td><td>0</td></tr></table></div><p>原文式(11)—(13)的阈值、步骤v重复阈值，以及式(16)双群重采样均已重新核对。实现中的≤0终止解释仍是此前已公开的约定，不冒充作者代码。</p></section>
<section><h2>4．接下来怎样做，才不继续盲目调参？</h2><p><b>先处理最明确的输出链路问题，评分暂不同时改动。</b>建议单列一个最小适配对照：第一次全域搜索保持原阈值；得到两个候选后，后续按已有两组分别局部更新，避免重复全局阈值把其中一组删掉。只检验“能否稳定保留两个输出、位置有没有随之改善”，不要求全部成功。</p><p>这属于待验证的常规输出适配，不是原论文已经采用的步骤，也不是已经确立的创新。若输出恢复而误差仍大，再单独处理评分稳定化；不能把两项一起修改后将全部收益归给其中之一。也不直接采用本轮1%或5%删频点作正式方法。</p><p>本轮没有实施上述适配；原始求解器、输入和84例最终结果均未修改。暂不扩展场景，不跑1000次重复。</p></section>
<section><h2>证据与可重放文件</h2><p>分析耗时{s['seconds']:.2f}s，{s['checks']}项身份/评分等价/分量重建/解码检查通过；另完成{len(checks)}项阈值算例与汇总复算。不存在新定位性能批次。</p><p><a href="protocol.json">执行前协议</a> · <a href="audit.json">审计</a> · <a href="report_audit.json">算例核验</a> · <a href="summary.json">汇总</a> · <a href="spectral.csv">逐例频谱</a> · <a href="sensitivity.csv">固定候选敏感性</a> · <a href="filter.csv">逐轮筛选</a> · <a href="score_checks.py">诊断源码</a> · <a href="score_report.py">报告源码</a></p><p>原文：Xia等，IEEE Systems Journal 2021，pp.2856–2857，式(9)—(16)。当前重建的BPSK脉冲、噪声标定等补全约定不等于作者原脚本。本轮没有取得作者代码，也没有把复现差距当成论文创新证据。</p></section></html>'''
    (OUT/'评分与筛选机制报告.html').write_text(page,encoding='utf-8')
    md=f'''# 正方形固定输入：原评分与筛选机制核验

输入`{source.name}`全部84例，12个基础实现；新定位求解0次。冻结候选、IQ、算法文件均不变。原式(9)比值与R_m/R_1等价；按原式复算评分、分解源A/源B/噪声分子，固定共同的含噪参考分母。真值仅离线诊断。逐例频谱量与评分保存为NPZ，CSV保留分条件。

最低参考功率1%/5%频点占频谱比值平方和中位数53.5956%/70.6122%；最低5%中真实参考信号低于噪声期望功率的比例中位88.5%，两源强抵消(|A+B|²/(|A|²+|B|²)<0.1)比例中位6%。它们在旧最高分点的复数和变化相对范数中位29.4285%。比值平方和不是最终乘积评分的可加分量，不能称70.6%的定位评分由噪声贡献。

按原1000粒子做只读敏感性：排除观测参考功率最低0/1/5%，首峰距任一源100m内为35/37/38例；1%改变46例最高分候选，跨100m变好5/变差3；5%改变49例，变好6/变差3。固定A第一峰RMSE2480.293836/2169.306244/2273.044875m。不是完整算法性能，不调整正式成功门槛，不挑1%当最优参数。删频点未重跑筛选或精修。

原始第一轮84例，第二轮37例，第三轮1例；第一轮min/max中位0.000103626，第二轮0.262093。第二轮30例删掉整个第二旧候选群，6例阈值高于最高分，余1例第三轮阈值高于最高分。T=L+(b/5)(H-L)，b≥6时必高于最高分。两个等量常值群L=1/H=2的当前规则给T=1.2，低分群全删；高分群占众数时T=3，全删。此算例为代数结构检查，不是新的RF性能试验。原文步骤v确实要求重算式9—13。

结论：低参考功率敏感性有数据支持，但不是完整根因；重复全局阈值和集中双群重采样之间存在明确失效机制。建议下一步只验证最小输出适配（初次全域阈值不变，后续保留两组分别局部更新），原评分先不改。该对照未执行；不将适配等同作者代码或理论创新，不扩大1000次重复。

分析{s['seconds']:.2f}s、{s['checks']}项检查、{len(checks)}项算例/汇总复算通过。相关原文pp2856—2857式9—16已回读。报告[HTML](评分与筛选机制报告.html)，详细结果见summary.json、spectral.csv、sensitivity.csv、filter.csv；输入哈希前后不变。
'''
    (OUT/'评分筛选核验结论.md').write_text(md,encoding='utf-8')
    previous=source/'正方形构型对照.html';p=previous.read_text(encoding='utf-8')
    if 'id="score-latest"' not in p:
        p=p.replace('<section>',f'<section id="score-latest"><h2>新增：原始评分与筛选机制检查</h2><p>低参考功率频点敏感性存在，但删频点尚不能解决；后轮阈值删组已得到直接解释。</p><p><a href="../{OUT.name}/评分与筛选机制报告.html">查看机制核查与下一步 →</a></p></section><section>',1)
        previous.write_text(p,encoding='utf-8')
    print('REPORT',OUT/'评分与筛选机制报告.html')


if __name__=='__main__':main()
