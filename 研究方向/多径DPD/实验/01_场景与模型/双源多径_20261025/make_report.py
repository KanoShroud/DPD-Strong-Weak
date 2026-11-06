"""Saved-result-only plots and self-contained Chinese report."""
import argparse
import html
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiment import ROOT, LAYOUTS, sha, write
from model import STATIONS


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--audit',required=True);parser.add_argument('--oracle',required=True);parser.add_argument('--two-path',required=True);parser.add_argument('--sensitivity',required=True)
    args=parser.parse_args()
    auditfile=Path(args.audit).resolve(strict=True)/'result.json'
    oraclefile=Path(args.oracle).resolve(strict=True)/'result.json'
    tapfile=Path(args.two_path).resolve(strict=True)/'result.json'
    tap=json.loads(tapfile.read_text(encoding='utf-8'))
    sensitivity_file=Path(args.sensitivity).resolve(strict=True)/'result.json'
    sens=json.loads(sensitivity_file.read_text(encoding='utf-8'))
    audit=json.loads(auditfile.read_text(encoding='utf-8'));oracle=json.loads(oraclefile.read_text(encoding='utf-8'))
    if not audit['all_checks_pass']:
        raise RuntimeError('Do not publish a passed report from failed audit')
    out=ROOT/('report_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir(exist_ok=False)
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    summary=audit['summary'];rows=audit['rows']
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'font.size':11,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for ax,stage,title in zip(axes,['periodic','nonperiodic'],['周期数值对照 · 每条件4例','非周期记录复核 · 每条件4例']):
        labels=['A单源','B单源','双源'];x=np.arange(3)
        for offset,suffix,color,label in [(-.18,'los','#277c80','纯直达'),(.18,'mp','#bd733a','直达＋远反射')]:
            group=[next(s for s in summary if s['stage']==stage and s['condition']==prefix+'_'+suffix) for prefix in ['A','B','dual']]
            vals=[s['rmse_m'] for s in group]
            bars=ax.bar(x+offset,vals,.34,color=color,label=label)
            ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
        ax.set_xticks(x,labels);ax.set_ylabel('全部定位结果的RMSE / m');ax.set_title(title);ax.legend();ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(out/'paired_rmse.png',dpi=180);fig.savefig(out/'paired_rmse.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
    names=[];vals=[];colors=[]
    for prefix,lab in [('A','A单源'),('B','B单源'),('dual','双源')]:
        for suffix,title,col in [('ground','地面反射分量','#bd733a'),('power_only','同功率纯直达','#277c80')]:
            r=next(s for s in summary if s['stage']=='ground' and s['condition']==prefix+'_'+suffix)
            names.append(lab+'\n'+title);vals.append(r['rmse_m']);colors.append(col)
    bars=ax.bar(np.arange(6),vals,color=colors);ax.bar_label(bars,fmt='%.2f',padding=3)
    ax.set_xticks(np.arange(6),names);ax.set_ylabel('RMSE / m');ax.set_title('地面近延迟反射：每条件2例，不能当作稳定成功率');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(out/'ground_power.png',dpi=180);fig.savefig(out/'ground_power.svg');plt.close(fig)
    fig=plt.figure(figsize=(11,4.7),layout='constrained')
    for i in range(2):
        ax=fig.add_subplot(1,2,i+1,projection='3d')
        ax.scatter(*STATIONS.T,marker='^',s=55,color='#277c80',label='5个单天线站')
        ax.scatter(*LAYOUTS[i].T,marker='*',s=100,color='#bd733a',label='两个真实目标')
        for j,p in enumerate(LAYOUTS[i]):
            ax.text(*p,'  '+['A','B'][j])
        for j,s in enumerate(STATIONS):
            ax.text(*s,' S'+str(j))
        ax.set(xlabel='x / m',ylabel='y / m',zlabel='高度 / m',zlim=(0,160),title=f'预定构型{i}（不按成绩选位置）')
        ax.legend(loc='upper left',fontsize=8)
    fig.savefig(out/'geometry.png',dpi=180);fig.savefig(out/'geometry.svg');plt.close(fig)
    table=['|条件|样本数|RMSE / m|水平RMSE|高度RMSE|全部源20m内|平均求解耗时 / s（不含FFT）|','|---|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        table.append(f'|{s["stage"]} / {s["condition"]}|{s["count"]}|{s["rmse_m"]:.3f}|{s["horizontal_rmse_m"]:.3f}|{s["vertical_rmse_m"]:.3f}|{s["all_success"]}/{s["count"]}|{s["mean_seconds"]:.2f}|')
    oracle_table=['|条件|布局/实现|盲RMSE / m|真值初值RMSE / m|目标差：oracle−盲|','|---|---|---:|---:|---:|']
    for r in oracle['rows']:
        oracle_table.append(f'|{r["condition"]}|{r["layout"]}/{r["seed"]}|{r["blind_evaluation"]["rmse_m"]:.3f}|{r["evaluation"]["rmse_m"]:.3f}|{r["fit"]["loss"]-r["blind_loss"]:.3g}|')
    detail='\n'.join(table)+'\n\n## 只用于归因的真值初始化\n\n'+'\n'.join(oracle_table)
    (out/'统计表.md').write_text(detail,encoding='utf-8')
    write(out/'绘图数据.json',{'summary':summary,'rows':rows,'oracle':oracle['rows']})
    identity={'audit':str(auditfile),'audit_sha256':sha(auditfile),'oracle':str(oraclefile),'oracle_sha256':sha(oraclefile),'two_path':str(tapfile),'two_path_sha256':sha(tapfile),'sensitivity':str(sensitivity_file),'sensitivity_sha256':sha(sensitivity_file),'script_sha256':sha(Path(__file__))}
    write(out/'来源.json',identity)
    def link(p,label):
        return f'<a href="{html.escape(Path(p).as_uri())}">{html.escape(label)}</a>'
    tr=''.join(f'<tr><td>{html.escape(s["stage"]+" / "+s["condition"])}</td><td>{s["count"]}</td><td>{s["rmse_m"]:.2f}</td><td>{s["vertical_rmse_m"]:.2f}</td><td>{s["all_success"]}/{s["count"]}</td><td>{s["mean_seconds"]:.1f}</td></tr>' for s in summary)
    orows=''.join(f'<tr><td>{html.escape(r["condition"])}</td><td>{r["layout"]}/{r["seed"]}</td><td>{r["blind_evaluation"]["rmse_m"]:.2f}</td><td>{r["evaluation"]["rmse_m"]:.2f}</td><td>{r["fit"]["loss"]-r["blind_loss"]:.2g}</td></tr>' for r in oracle['rows'])
    taprows=''.join(f'<tr><td>{r["layout"]}/{r["seed"]}</td><td>{r["direct_evaluation"]["rmse_m"]:.2f}</td><td>{r["blind_evaluation"]["rmse_m"]:.2f}</td><td>{r["oracle_evaluation"]["rmse_m"]:.2f}</td><td>{r["blind"]["evaluations"]} / {r["oracle"]["evaluations"]}</td><td>{html.escape(r["blind"]["message"])}</td></tr>' for r in tap['rows'])
    sensrows=''.join(f'<tr><td>{i}</td><td>{html.escape(e["label"])}</td><td>{e["evaluation"]["rmse_m"]:.2f}</td><td>{html.escape(e["fit"]["message"])}</td></tr>' for i,r in enumerate(sens['rows']) for e in r['experiments'])
    page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>重新聚焦：双源、同频、多径DPD</title><style>
    :root{{--ink:#193b42;--teal:#277c80;--amber:#bd733a}}*{{box-sizing:border-box}}body{{margin:0;background:#f5f4ef;color:var(--ink);font:16px/1.8 'Microsoft YaHei',sans-serif}}main{{max-width:1080px;margin:auto;padding:36px 24px 70px}}h1{{font-size:clamp(27px,4vw,40px);line-height:1.4;margin:10px 0}}h2{{font-size:24px;margin:0 0 12px}}p{{margin:10px 0}}a{{color:var(--teal)}}.eyebrow{{letter-spacing:.14em;color:var(--teal);font-size:13px}}.lead{{font-size:19px}}.box,section{{background:white;border:1px solid #dce4e1;border-radius:16px;padding:24px;margin:20px 0}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.cards .box{{margin:0}}.cards strong{{display:block;font-size:21px}}.small{{font-size:14px;color:#5b6d71}}.note{{border-left:4px solid var(--amber);background:#f7f0e6;padding:16px 20px;border-radius:0 10px 10px 0}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:9px 11px;text-align:left;border-bottom:1px solid #e0e7e3;white-space:nowrap}}th{{background:#eef4f1}}.scroll{{overflow-x:auto}}summary{{cursor:pointer;color:var(--teal);font-weight:bold}}li{{margin:7px 0}}.flow{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.flow div{{background:#edf4f1;border-radius:12px;padding:18px}}footer{{color:#60767b;font-size:13px}}@media(max-width:680px){{main{{padding:24px 16px}}section{{padding:18px}}.cards,.flow{{grid-template-columns:1fr}}}}
    </style></head><body><main><div class="eyebrow">2026-10-25 · 有限配对核验 · 不是正式论文排名</div><h1>把问题收回到定位本身</h1><p class="lead">五个单天线站，两个未知同频信号，少量反射分量。<br>先弄清哪里出错，不再同时重建墙面。</p>
    <div class="cards"><div class="box"><strong>不估墙面</strong><span class="small">反射面只在生成器里生成反射分量</span></div><div class="box"><strong>同输入配对</strong><span class="small">删掉另一源或反射分量，噪声不变</span></div><div class="box"><strong>先核常规方法</strong><span class="small">本轮没有认定新算法贡献</span></div></div>
    <section><h2>先看结论：问题收缩成功，算法尚未放行</h2><p>普通直达DPD可以处理本批无多径双源；加入反射分量后，主要损失体现在高度精度。地面近延迟反射的一组困难案例不能只用接收功率下降解释，但另一组功率对照也失败，不能把全部损失都归为双源多径特有问题。</p><p>简单增加自由双径参数没有稳定解决。局部诊断显示：反射路径参数可以补偿部分位置变化，使高度信息更弱；微小窗口误差与噪声也会被放大。目前只支持继续有针对性的机制研究，不支持新算法的大规模性能宣称。</p><div class="note"><b>准入状态：</b>数据配对与常规直达参照可扩展用于问题诊断；双径候选未通过稳定性/充分求解门槛。没有启动正式大批次，没有为了放行继续调反射强度、增路径或延长迭代。</div></section>
    <section><h2>这轮只问三个问题</h2><div class="flow"><div><b>两源，没有反射分量</b><br>定位器本身是否正常？</div><div><b>单源，有反射分量</b><br>反射分量独自造成什么损失？</div><div><b>两源，都有反射分量</b><br>是否出现额外混叠损失？</div></div><p>五站高度为4/18/8/28/12m；两源位置不按结果选择。先检查全部直达存在的基础条件，过去的部分直达缺失尚未解决。</p><img src="geometry.png" alt="两组事先固定的三维源站坐标"></section>
    <section><h2>先修基础，再看多径</h2><p>最初60m网格跳过有效区域；统一20m后仍因求解尺度与迭代上限留下误差。保持目标不变，改用有完整Jacobian的标准信赖域最小二乘，无噪双源控制达到约8×10⁻⁷m。旧失败全部保留。</p><p>12项数值检查覆盖LS消元、梯度、残差Jacobian、谱与目标关系、输出交换、FFT功率及配对。它们证明公式与实现一致，不证明所有场景都能定位。</p></section>
    <section><h2>远处反射分量：看RMSE，而不是只看成功率</h2><img src="paired_rmse.png" alt="单源双源在纯直达和远反射下的配对RMSE"><p>左图是排除窗口近似的周期数值对照；右图是新随机实现、非周期记录加Hann的复核。不同长度和实现之间不作单因素排名。每条件只有两个布局×两次实现，不据此声称普遍规律。</p></section>
    <section><h2>低空地面反射：别把相消降SNR当作新难题</h2><img src="ground_power.png" alt="地面反射与逐源逐站同功率直达对照"><p>地面与远墙分开比较；反射系数保持−0.7，没有加到算法失败为止。功率对照由生成端保留各源各站的接收能量，但去掉反射分量的频率依赖；算法不接收这些真值尺度。</p><div class="note">同功率对照不是一种实际定位算法。它只帮助判断：误差能否仅由信号变弱解释。</div></section>
    <section><h2>完整结果，不删除失败</h2><div class="scroll"><table><thead><tr><th>条件</th><th>例数</th><th>RMSE / m</th><th>高度RMSE / m</th><th>全部源≤20m</th><th>平均求解时间 / s</th></tr></thead><tbody>{tr}</tbody></table></div><p class="small">RMSE汇总所有匹配位置误差平方；成功率为辅助指标。时间包含候选网格与精修，但本轮计时从频域输入开始，不含外部FFT/Hann预处理、保存和测试，不能称完整IQ端到端耗时。当前CPU两线程、密网格原型不是最终算法性能目标，本轮不主张速度优势。</p></section>
    <section><h2>是不是搜索没找到？</h2><p>另外从真实位置与真实直达增益开始优化，作为有额外信息的诊断，不参与盲算法排名。仍允许位置改变。若它与盲输出落到近似同一目标和位置，支持当前错误不是简单增加初值就能解决；不构成全局最优证明。</p><details><summary>展开真值初始化诊断</summary><div class="scroll"><table><tr><th>条件</th><th>布局/实现</th><th>盲RMSE</th><th>诊断RMSE</th><th>目标差</th></tr>{orows}</table></div></details></section>
    <section><h2>不估墙面，常规双径模型能否处理？</h2><p>再给模型一条未知反射路径：其延迟、幅度和相位由数据估计，仍不知道地面在哪。本基础诊断明确假定直达存在且单条反射弱于直达（幅度比≤0.9）。这是标准参数化拟合参照，不是新的创新算法。</p><div class="scroll"><table><tr><th>布局/实现</th><th>直达模型RMSE</th><th>双径盲RMSE</th><th>双径真值初值RMSE</th><th>评价次数：盲/诊断</th><th>盲停止原因</th></tr>{taprows}</table></div><p>真值初值包含真实位置和路径参数，只作容量/局部求解诊断，不是可部署结果。触顶不当作已收敛，不根据结果延长预算。</p><p>{link(tapfile,'双径原始结果')} · {link(ROOT/'常规双径检查合同.md','双径检查信息权限及合同')}</p></section>
    <section><h2>最后一轮：噪声、窗口和参数补偿</h2><p>把信号严格投影到双径模型、再去掉噪声时，真位置可以保持不动；保留实际有限记录，或重新加入原噪声，就出现偏移与预算触顶。说明不是多加一个反射路径参数就自动变准，也不能把这些触顶值当作最优估计器的能力上限。</p><p>在真参数处，消去未知增益与反射路径参数方向后，六维位置灵敏度的最小奇异值较直达模型低约两个数量级。这支持位置与通道参数耦合很强的线索；不是CRB、全局不可辨识证明或新理论贡献。</p><details><summary>展开仅用于归因的结果</summary><div class="scroll"><table><tr><th>布局</th><th>条件</th><th>RMSE / m</th><th>停止</th></tr>{sensrows}</table></div><p>{link(sensitivity_file,'噪声与窗口诊断原始结果')} · {link(ROOT/'噪声与窗口归因合同.md','诊断合同')}</p></details></section>
    <section><h2>证据与适用范围</h2><p>主表估计器仅使用混合IQ、站点、频率和搜索域，联合估计常复增益与位置；不使用已知1/d增益、源波形、地图或真实路径。它是常规直达DPD的适配参照，不是已充分适配的多径算法，不能据此宣布多径DPD领域失效。</p><p>非周期Hann后的逐频模型有近似；模型来源、信息权限及已知环境文献的区别单独登记。</p><ul><li>{link(ROOT/'执行合同.md','原执行合同')} · {link(ROOT/'初始化修订_v2.md','网格修订')} · {link(ROOT/'求解修订_v3.md','求解修订')}</li><li>{link(ROOT/'后续条件诊断.md','地面反射与功率归因合同')} · {link(ROOT/'模型与文献边界.md','模型与论文对应')}</li><li>{link(auditfile,'逐例独立重放审计')} · {link(oraclefile,'真值初始化原始结果')} · <a href="统计表.md">完整统计表</a> · <a href="绘图数据.json">绘图数据</a></li></ul></section>
    <footer>来源：本目录冻结输入、源码、原始结果与哈希。结论另见上层阶段总结；本文图表只读取保存结果，不重跑定位。</footer></main></body></html>'''
    (out/'双源多径核验报告.html').write_text(page,encoding='utf-8')
    print(out)


if __name__=='__main__':
    main()
