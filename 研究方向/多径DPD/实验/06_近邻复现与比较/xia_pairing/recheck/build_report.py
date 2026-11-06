"""Read saved diagnostics, audit provenance, and write the current report."""
import json
from pathlib import Path
from itertools import permutations
from html import escape
import numpy as np
from run_exact import sha, write

ROOT = Path(__file__).resolve().parent
EXACT = ROOT/'exact_20261027_153508_733933'
ATTR = ROOT/'attribution_20261027_153749_427073'
GRID = ROOT/'score_grid_20261027_154015_320980'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def main():
    exact, attr, grid = [read(d/'result.json') for d in [EXACT, ATTR, GRID]]
    checks = dict(exact['checks'])
    for directory, key in [(EXACT, 'sources'), (ATTR, 'sources'), (GRID, 'source')]:
        manifest = read(directory/'preflight.json')
        for original, digest in manifest[key].items():
            checks[f'{directory.name}/snapshot/{Path(original).name}'] = sha(directory/Path(original).name) == digest
    data = np.load(EXACT/'components.npz')
    checks['exact_component_identity'] = sha(EXACT/'components.npz') == exact['components_sha']
    for directory in [ATTR, GRID]:
        checks[directory.name+'/input_identity'] = sha(EXACT/'components.npz') == read(directory/'preflight.json')['input_sha']
    truth = data['truth']
    for row in grid['rows']:
        p = np.asarray(row['positions'])
        rms = min(np.sqrt(np.mean(np.sum((p-truth[list(order)])**2, axis=1)))
                  for order in permutations(range(2)))
        checks['independent_matching/'+row['channel']] = abs(rms-row['evaluation']['rmse_m']) < 1e-9
    for row in attr['rows']:
        checks[f'true_threshold/{row["channel"]}/{row["seed"]}'] = bool(
            np.array_equal(np.array(row['true_scores']) < row['threshold'], row['true_below_threshold']))
    # Formula (4), rebuilt with an explicit path loop independent of broadcasting.
    independent = np.zeros_like(data['direct'])
    f = data['frequency']
    for q in range(2):
        for m in range(5):
            tau = (np.linalg.norm(truth[q]-data['stations'][m])-np.linalg.norm(truth[q]-data['stations'][0]))/299792458.
            independent[q, m] = data['gain'][q, m]*data['source'][q]*np.exp(-2j*np.pi*f*tau)
            for j, dt in enumerate(np.array([3, 8, 11, 13, 21])*1e-6):
                independent[q, m] += data['extra_gain'][q, m, j]*data['source'][q]*np.exp(-2j*np.pi*f*(tau+dt))
    err = np.linalg.norm(independent-data['direct']-data['reflection'])/np.linalg.norm(independent)
    checks['independent_eq4_rebuild'] = bool(err < 1e-12)
    assert all(checks.values()), checks
    write(ROOT/'审计.json', {'checks': checks, 'all_pass': all(checks.values()), 'count': len(checks),
                             'eq4_relative_error': float(err),
                             'note': 'Staged protocol snapshots differ intentionally; solver sources are unchanged.'})
    rows = ''.join(f'<tr><td>{escape(r["case"])}</td><td>{escape(r["status"])}</td><td>'
                   + ('未输出，不计算误差' if not r['evaluation']['complete'] else f'{r["evaluation"]["rmse_m"]:.3f} m')
                   + '</td></tr>' for r in exact['rows'])
    pairrows = ''.join(f'<tr><td>{"直达" if r["channel"]=="los" else "多径"}</td><td>{r["seed"]}</td>'
                      f'<td>{r["true_scores"][1]:.6g}</td><td>{r["threshold"]:.6g}</td><td>第二源被删</td></tr>'
                      for r in attr['rows'])
    rankrows = ''.join(f'<tr><td>{"直达" if r["channel"]=="los" else "多径"}</td>'
                      f'<td>{r["nearest_local_peak_posthoc"][0]["nearest_peak_rank"]}</td>'
                      f'<td>{r["nearest_local_peak_posthoc"][1]["nearest_peak_rank"]}</td></tr>' for r in grid['rows'])
    correction_error = attr['rounding_replay']['evaluation']['rmse_m']
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Xia复现复查：实现、适配与方法局限</title><style>
:root{{color-scheme:light;--ink:#173c45;--accent:#14786e}}*{{box-sizing:border-box}}body{{margin:auto;max-width:1100px;padding:42px 22px;background:#f5f4ee;color:var(--ink);font:16px/1.85 "Microsoft YaHei",system-ui}}h1{{font-size:34px;line-height:1.4}}h2{{font-size:23px}}h3{{font-size:18px}}section{{background:white;border:1px solid #dce5df;border-radius:16px;margin:20px 0;padding:22px 28px}}a{{color:var(--accent)}}.muted{{color:#667c80}}.badge{{font-size:13px;background:#e2eee7;padding:6px 12px;border-radius:20px}}.cards,.flow{{display:flex;flex-wrap:wrap;gap:14px}}.cards div,.flow div{{flex:1;min-width:220px;background:#edf4ef;padding:18px;border-radius:12px}}.note{{background:#fff1db;padding:14px 18px;border-left:4px solid #c79840}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:10px;border-bottom:1px solid #dfe7e4;text-align:left}}.scroll{{overflow-x:auto}}summary{{cursor:pointer;font-weight:bold}}code{{font-size:14px}}.formula{{font-size:19px;text-align:center;background:#f0f5f1;padding:16px}}footer{{margin-top:28px;font-size:14px}}</style>
<section class="note"><b>最新任务已转为严格原文性能复现。</b>本页是已披露设置下的诊断，不代表作者实验性能。<a href="../paper_protocol/原文性能复现说明.html">查看原文设定、作者原图及尚待澄清的关键细节 →</a></section>
<p class="badge">2026-10-27 · 精确模型诊断 · 不加Hann · 不是性能排行榜</p>
<h1>这次分清了：<br>程序错误修好了，原评分与筛选仍有不同层次的问题</h1>
<p>按批准顺序完成：数值修正 → 原参数尺度的精确频域控制 → 换粒子种子 → 固定评分的局部峰检查。没有为获得成功结果调整信道或阈值，也没有把旧失败并入本轮统计。</p>
<section><div class="cards"><div><b>已修复：实现错误</b><p>数学上等于最高分的阈值，不再因一位浮点舍入清空候选。</p></div><div><b>已确认：筛选局限</b><p>精确、无噪、无多径的双源输入，也可能把第二源整个区域筛掉。</p></div><div><b>新增线索：评分干扰</b><p>加入表I多径后，第二源附近局部峰从第2降至第29；单纯取消阈值不够。</p></div></div></section>
<section><h2>1. 怎样排除适配不足造成的假失败？</h2><div class="flow"><div>原尺度与原公式<br><small>1MHz / 2000点 / 200kbps BPSK<br>6km域、1000粒子、三轮</small></div><div>精确构造观测<br><small>按式(4)生成频域延迟副本<br>不加噪声，不加Hann，不筛频</small></div><div>固定原输出规则<br><small>保留≤0约定和原直方图规则<br>不保底Top2，不输入真值</small></div></div>
<p>单源、双源均检查；反射延迟和幅度来自表I。波形数据、路径相位和具体坐标仍是已披露的补全，不是作者原始仿真数据。独立路径循环重建相对误差为 <b>{err:.2e}</b>，因此本批问题不能用“观测窗截断造成式(4)不成立”解释。</p>
<p class="note">这里的精确频域数据只是公式控制。它不替代当前真实非周期记录，也不等于复现论文的1000次Monte Carlo曲线。</p></section>
<section><h2>2. 修正了什么，没动什么？</h2><p>众数落在第五格时，阈值代数上恰等于最高分；代码直接使用这个端点。旧保存状态重放后，孤立A恢复为约 <b>{correction_error:.3f}m</b> 的输出。这是程序修正，不是算法创新。</p>
<p>没有把所有过高阈值截到最高分，没有加入正则化，没有调粒子数、功率或反射强度。重采样按式(16)的高斯分布生成，不再额外裁剪公共区域；本次未出现域外粒子。</p>
<details><summary>六项原尺度精确控制的实际状态</summary><div class="scroll"><table><tr><th>观测</th><th>输出状态</th><th>误差</th></tr>{rows}</table></div><p>孤立源是显式K=1扩展；真正对应原双源任务的是dual两项。缺失输出不捏造距离，也不删除后只计算成功样本的RMSE。</p></details></section>
<section><h2>3. 原阈值为什么会删掉真实源？</h2><p>设十格直方图的众数在第 b 格，最低和最高分分别为 L、H。原式可整理为：</p><div class="formula">T = L + (b / 5) × (H − L)</div><p>b=5时T=H；b≥6时T&gt;H，会清空所有点。若众数在第一格、最低分接近零，T约为最高分的20%。较弱的第二个峰可能低于这条全局门槛。</p>
<p>下面是在定位完成后才查询的真实第二源评分。三个种子只改变粒子，不改变波形、信道或观测。真实位置与其±100m诊断网格的最高分均低于门槛：</p><div class="scroll"><table><tr><th>信道</th><th>粒子种子</th><th>真实第二源分数</th><th>筛选门槛</th><th>结果</th></tr>{pairrows}</table></div>
<p class="muted">真值查询仅用于归因，不参与算法。三种子不是三个独立信道实现；本表证明一个可复现的机制反例，不估计普遍失败概率。</p></section>
<section><h2>4. 不用阈值，问题是否就消失？</h2><p>另算一张固定100m错位网格谱，使用普通3×3局部极大值，不连续精修、不改变评分。两个真值都不在网格上。结果：</p><table><tr><th>信道</th><th>第一源附近峰排名</th><th>第二源附近峰排名</th></tr>{rankrows}</table>
<p>直达时，前两个局部峰分别在两个源附近。多径时，第二高峰落在第一源一侧的错误区域，第二源附近峰降至第29。由此可以区分：<b>直达例已有阈值误删；多径例还有评分竞争问题。</b></p>
<p class="note">这是3600点固定谱图诊断，既不是原1000粒子求解器，也不是公平计时比较。最近峰距真值约79m；没有精修，不能当成最终RMSE。一个案例仍不足以证明新方法必要或论文创新成立。</p></section>
<section><h2>5. 当前场景接回了吗？</h2><p><b>还没有进入当前场景的方法性能比较。</b>原参照的筛选局限已能解释，但完整原流程尚未稳定输出双源。不能把“找到原因”写成“原文已经跑通”。</p>
<p>接回时配置已经明确：二维五站、50MHz/8192点、独立BPSK/RRC、12.5MHz带宽、原参数化多径及噪声；<b>不加Hann，仅保留±7MHz有效频点与原FFT归一化</b>。不额外恢复窗口研究分支。</p>
<h3>后续只围绕一个问题</h3><p>多径如何改变Xia跨谱比值和站间连乘，使错误位置超过真实第二源？先在原模型上分解各站评分和参考站频谱比值；与普通多源联合拟合在同一输入下核验。若常规方法已解决，就采用强基线，不将修阈值包装为贡献。只有场景相关的剩余机制和针对性收益，才值得形成新算法。</p></section>
<footer><p>本轮新增10次原定位器控制调用（6项控制＋4次预定种子复核），2张固定网格评分；{len(checks)}项数学、身份、输出重放检查通过。旧失败未纳入本轮汇总。未删除历史目录，未改论文或Zotero。</p>
<a href="执行协议.md">执行协议</a> · <a href="核验结论.md">完整结论</a> · <a href="审计.json">审计</a> · <a href="{EXACT.name}/result.json">精确输入控制</a> · <a href="{ATTR.name}/result.json">阈值归因</a> · <a href="{GRID.name}/result.json">局部峰诊断</a> · <a href="https://doi.org/10.1109/JSYST.2020.3005918">原论文 §IV—V</a></footer></html>'''
    (ROOT/'分层核验报告.html').write_text(html, encoding='utf-8')
    md = f'''# Xia原模型分层核验结论

## 结果与界限

新增10次原求解器调用、2张固定评分网格，{len(checks)}项审计通过。原参数尺度精确频域模型无噪、无Hann，排除了本批有限窗输入失配。修正第五直方图格的浮点端点，并保留≤0解释；没有给真超限阈值加上限或保底输出。旧失败不参加本轮汇总。

原求解器六项控制中，仅B_LOS给出完整终态，误差78.937706m；其余为阈值清空或缺第二簇。单源是扩展控制，不能等同原双源任务。第一轮双源在三种子、LOS/MP共六种调用中均缺第二簇；这是同一数据实现的算法种子复核，不是六次独立Monte Carlo。

## 归因

1. 实现问题：b=5阈值精确等于max，修复浮点边界；保存状态重放单源A为{correction_error:.9f}m。不把修复计作新定位实验或创新。
2. 确认的输出局限：T=L+(b/5)(H-L)，b≥6真实超最高分。b=1且L近零时门槛约0.2max；精确双源控制的真实B及±100m、25m步长诊断网格最高分均低于三个粒子种子的首轮门槛。原模型独立重建相对误差{err:.3e}。
3. 评分线索：3600点、100m错位全域网格，普通3×3局部极大值且不设阈值，A/B附近局部峰在LOS中排名1/2，MP中1/29。两种输入波形、直达与增益相同，仅加入表I反射；这是真实模型内的多径影响证据，不是Hann导致。网格粗细、单布局及随机实现限制结论，未证明所有强基线都困难。

## 参数与公平性

原尺度fs1MHz、N2000、200kbps BPSK、五站、±3000m、参考站原点、1000粒子、三轮、Δh400m、Γ=100m²；源频谱继承已登记参考站BPSK/RRC补全，COST231及5dB阴影常复增益；反射延迟3/8/11/13/21us、相对幅度-7/-15/-22/-24/-19dB，新增相位种子270271。各站源数、延迟、增益在生成器，求解只接观测和公共参数。全FFT、不加噪声、不加窗。不是原文图线复现。

式(16)不截断高斯，本实现撤去前版的额外边界拒绝；实际无域外点，不能把改善或失败归于边界处理。三轮指初次评分加两次重采样评分。使用带符号FFT轴，与生成式(4)一致。

普通局部峰诊断与原粒子预算不同，不作耗时/性能排名。原求解缺失输出不填巨大误差；谱图诊断的79m粗峰偏差不充作最终RMSE。

## 执行状态

原文完整定位尚未稳定跑通，不能宣布适配完成。因而未启动当前50MHz场景性能比较。已按用户要求冻结下一步无Hann、FFT/N、±7MHz的输入口径，不把精确控制替换成真实非周期记录的最终结论。

建议下一步只做原模型多径评分归因及同输入常规多源联合拟合参照，区分阈值局限与跨谱比值/连乘的源间干扰。简单峰提取在本例MP不能直接恢复，不等于已有常规方法都无法解决；不据此宣布论文突破或开始大规模调参。

## 证据

- [精确控制]({EXACT.name}/result.json)
- [种子与阈值归因]({ATTR.name}/result.json)
- [局部峰评分]({GRID.name}/result.json)
- [审计](审计.json)
- [原文](https://doi.org/10.1109/JSYST.2020.3005918)，§IV—V、pp.2856—2857。
'''
    (ROOT/'核验结论.md').write_text(md, encoding='utf-8')
    print(f'报告完成：{ROOT}; {len(checks)} checks; exact residual={err:.3e}')


if __name__ == '__main__':
    main()
