"""Build result-derived HTML/summary; no manually entered performance numbers."""
from pathlib import Path
import json
import html
import numpy as np

HERE=Path(__file__).resolve().parent
RAW=HERE.parent/'运行数据'


def main():
    rows=json.loads((RAW/'iq18/rows.json').read_text(encoding='utf-8'))
    done=json.loads((RAW/'iq18/completion.json').read_text(encoding='utf-8'))
    control=json.loads((RAW/'support18/summary.json').read_text(encoding='utf-8'))
    summary=[];lines=[];mechanism=[]
    for cond in ('L0','L1_P075','L2_P075'):
        for arm in ('FLAT','OBSERVED_COMMON'):
            a=[r['report'][arm] for r in rows if r['condition']==cond]
            phase=[v for x in a for v in x['oracle_phase_error_rad'] if v is not None]
            item=dict(condition=cond,arm=arm,direct_hits=sum(x['direct_hits'] for x in a),direct_total=72,
                components_hits=sum(x['matched'] for x in a),components_total=sum(x['total'] for x in a),
                oracle_phase_median_deg=float(np.rad2deg(np.median(phase))))
            summary.append(item)
            lines.append(f"<tr><td>{cond}</td><td>{arm}</td><td>{item['direct_hits']}/72</td><td>{item['components_hits']}/{item['components_total']}</td><td>{item['oracle_phase_median_deg']:.1f}°</td></tr>")
        pp=[x for r in control['rows'] if r['condition']==cond for x in r['pairs']]
        near=[v for x in pp for v in x['direct_nearest_same_source_B'] if v is not None]
        item=dict(condition=cond,near_count=len(near),near_below_quarter=sum(v<.25 for v in near),
            condition_median=float(np.median([x['condition'] for x in pp])),condition_max=max(x['condition'] for x in pp),
            noise_relative_rms_median=float(np.median([x['one_percent_noise_predicted_relative_coefficient_rms'] for x in pp])),
            noise_relative_rms_max=max(x['one_percent_noise_predicted_relative_coefficient_rms'] for x in pp),
            noiseless_error_max=max(x['noiseless_relative_coefficient_error'] for x in pp))
        mechanism.append(item)
    output=dict(interface=summary,conditioning=mechanism,iq_stage_seconds=done['seconds'],
        support_control_seconds=control['seconds'],identity_verified=done['identity_verified'],
        gate='NOT_READY_FOR_IDEAL_GRAPH',new_localization=False,actual_iq_cases=18)
    (HERE/'summary.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    mech=''.join(f"<tr><td>{x['condition']}</td><td>{x['near_below_quarter']}/{x['near_count'] if x['near_count'] else '—'}</td><td>{x['condition_median']:.1f}</td><td>{x['condition_max']:.0f}</td><td>{100*x['noise_relative_rms_median']:.2f}%</td></tr>" for x in mechanism)
    details=''.join(f"<tr><td>{html.escape(r['tag'])}</td><td>{r['report']['FLAT']['direct_hits']}/12</td><td>{r['report']['OBSERVED_COMMON']['direct_hits']}/12</td><td>{r['seconds']:.3f}</td></tr>" for r in rows)
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>真实IQ分量接口核验</title>
<style>body{{margin:0;background:#f3f5f1;color:#16303c;font:16px/1.8 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1120px;margin:30px auto;padding:0 24px}}header{{background:#173f46;color:white;border-radius:20px;padding:32px}}h1{{font-size:30px}}h2{{font-size:23px}}section{{background:white;border-radius:18px;padding:26px;margin:22px 0}}.note{{border-left:5px solid #c78e2c;background:#fff1d8;padding:17px;margin:20px 0}}a{{color:#007e88}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #d7dfdc}}th{{background:#e4eeeb}}.flow{{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}}.flow div{{padding:18px;background:#e7efec;border-radius:12px}}.formula{{font:20px/2 monospace;padding:15px;background:#edf2ee;overflow:auto}}small{{color:#557078}}@media(max-width:750px){{.flow{{grid-template-columns:1fr}}section{{overflow:auto}}}}</style>
<main><header><small style="color:#d1e4df">多径DPD · 第17轮 · 2026-11-07</small><h1>站内路径分得开，不代表站对分量也分得开</h1><p>真实IQ前端已测试；理想源径图的输入条件尚不成立。没有新增定位RMSE。</p></header>
<p><a href="机制分析.md">完整机制与边界</a> · <a href="执行合同.md">执行合同</a> · <a href="summary.json">汇总数据</a> · <a href="../../../16_联合源径图结构核验/20261107/成果/联合源径图候选报告.html">上一轮条件结构</a></p>
<div class="note"><b>结论：</b>不继续调图阈值，也不修改信道使它更容易。下一步若继续，必须允许一个观测分量包含多个路径对的叠加。</div>
<section><h2>这轮实际完成了什么</h2><div class="flow"><div><b>实际IQ原型</b><br>完整复跨谱 → 两种固定公共谱核 → 自适应增量分量拟合。算法没有路径真值。</div><div><b>保存后诊断</b><br>读取真实路径，检查直达分量是否保留、复系数是否可靠；真实支持拟合另标ORACLE。</div><div><b>条件数控制</b><br>进一步给出准确时差和已知核，只加1%合成跨谱扰动，检查分量系数稳定性。</div></div><p>复用原6构型×3条件的18条开发观测；不是18个独立新样本，也不是正式统计验证。四站、全内双源、未知波形、0/1/2反射和20微秒公开范围不变。</p></section>
<section><h2>为什么会卡在这里？</h2><p>两个站的反射都比直达晚，但互相关比较的是它们的<b>时差</b>。两个相似的“晚到时间”相减，可能几乎抵消。</p><div class="formula">反射—反射时差 = 直达时差 + δ<sub>n</sub> − δ<sub>m</sub></div><p>原生成器中同序号反射具有相近的额外时延。因此，反射相关分量会靠近直达相关分量，幅度和相位相干叠加。把这个合成峰直接当成一条路径的复系数，会破坏上一轮图方法依赖的关系。</p><p><b>不是完全相同才有问题：</b>只要两列跨谱响应很相似，分离系数就会放大扰动。不能用增加峰数量解决这一点。</p></section>
<section><h2>实际IQ接口结果</h2><table><tr><th>条件</th><th>前端核</th><th>直达路径对匹配</th><th>全部路径对匹配</th><th>真实支持拟合后的<br>闭合相位偏差中位数</th></tr>{''.join(lines)}</table><p><small>每条件6条记录×6站对×2源＝72个直达路径对。匹配尺度0.25/B，是分量诊断而非定位成功率。最后一列用真实时差支持，只说明即使知道时差，当前公共核拟合也未稳定得到理想复系数；不是盲算法成绩。</small></p><p>观测公共谱核并未稳定优于平坦核。因此没有继续调平滑尺度或BIC权重。两种原型也没有代表所有可能的分量恢复方法。</p></section>
<section><h2>更有利的控制仍暴露稳定性问题</h2><table><tr><th>条件</th><th>直达附近有同源分量<br>间距小于0.25/B</th><th>支持矩阵条件数<br>中位数</th><th>最大条件数</th><th>1%跨谱扰动引起的<br>系数相对RMS，中位数</th></tr>{mech}</table><p>本表是<b>真实支持＋平坦已知核的合成控制</b>，不是把原IQ噪声改成1%，不是位置误差。无噪声系数恢复误差最大为{max(x['noiseless_error_max'] for x in mechanism):.2e}，支持符号、复数方向和线性求解的数值正确性。</p><p>结果揭示稳定性障碍，但没有证明该障碍解释了全部前端损失，也没有证明位置不可辨识。</p></section>
<section><h2>现在如何判断路线</h2><p><b>保留：</b>同源共享波形、跨站一致性、联合比较源—路径解释。<br><b>暂不保留为必要前提：</b>先准确分离所有站对路径分量，再按唯一边建图。</p><p>若继续，需要直接保留重叠分量的叠加关系，在完整复跨谱或原始IQ上比较整体解释。这不是把阈值放宽，也不是已有快速方法已经成功；其可计算目标与更新方式仍需推导。</p><p>本轮尚不满足进入图关联定位或大规模实验的条件。暂超3–5秒的原型授权继续有效，但本次障碍主要是表示稳定性，并非算得不够久。</p></section>
<section><h2>执行与复现</h2><p>实际IQ阶段（含输出、诊断等）{done['seconds']:.3f}秒；已知支持控制{control['seconds']:.3f}秒。均不含完整定位，不据此声称达成3–5秒目标。源码与输入前后身份核验通过；Python编译、Ruff和精确网格复数系数单元检查通过。没有改公共MP源码。</p><details><summary>展开18条结果</summary><table><tr><th>记录</th><th>FLAT直达匹配</th><th>公共核直达匹配</th><th>两臂前端合计秒</th></tr>{details}</table></details></section></main></html>'''
    (HERE/'真实IQ分量接口报告.html').write_text(page,encoding='utf-8')


if __name__=='__main__':main()
