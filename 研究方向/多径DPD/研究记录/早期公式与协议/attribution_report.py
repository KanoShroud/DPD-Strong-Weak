"""Read saved component interventions, plot and document their evidence limits."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from structural_probe import sha, write


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    out=parser.parse_args().directory.resolve(strict=True)
    d=json.loads((out/'result.json').read_text(encoding='utf-8'))
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    assert d['passed'] and len(d['rows'])==12
    assert all(sha(p)==h for p,h in protocol['hashes'].items())
    r=next(t for t in d['rows'] if t['case']=='L1_P2' and t['first']==0)
    maps=np.load(out/'L1_P2_first0_maps.npz')
    truth=maps['truth'];first=maps['first']
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10})
    fig,axes=plt.subplots(2,2,figsize=(11,9),constrained_layout=True)
    titles=['实际扣除后的评分','仅保留扣除后的第二源（诊断）',
            '实际残差去掉首源反射残留（诊断）','恢复被扣掉的第二源分量（诊断）']
    for ax,idx,title in zip(axes.ravel(),[0,2,3,5],titles):
        v=r['score_variants'][idx]
        score=maps['scores'][idx]/v['max_grid_score']
        im=ax.imshow(score.reshape(300,300),origin='lower',extent=[-3000,3000,-3000,3000],
                     cmap='viridis',vmin=.8,vmax=1,interpolation='nearest')
        cand=np.array(v['candidates'])
        ax.scatter(*truth[1],marker='*',s=160,c='#e84855',edgecolors='white',linewidths=.8,label='第二源真位置')
        ax.scatter(cand[:,0],cand[:,1],marker='^',s=55,facecolors='none',edgecolors='white',linewidths=1.2,label='保留的3个候选')
        ax.scatter(*first,marker='x',s=60,c='#ffb347',label='首源粗位置')
        for i,p in enumerate(cand):ax.annotate(str(i+1),p,xytext=(5,-14 if p[1]>2900 else 4),textcoords='offset points',color='white')
        ax.set(xlim=(1800,3000),ylim=(1500,3000),xlabel='x / m',ylabel='y / m',title=title)
        ax.set_aspect('equal')
    handles,labels=axes.flat[0].get_legend_handles_labels()
    handles[1]=Line2D([],[],marker='^',linestyle='none',markerfacecolor='none',markeredgecolor='#60757b',markersize=7)
    fig.legend(handles,labels,loc='outside lower center',ncol=3)
    fig.colorbar(im,ax=axes,shrink=.75,label='各图评分 / 该图允许搜索区域最大值')
    fig.suptitle('同一首源拟合算子下的分量诊断：L1，两反射，首候选0\n各图独立归一化，只比较位置排序；右/下图不是可执行算法',fontsize=12)
    fig.savefig(out/'候选评分分量诊断.png',dpi=180)
    fig.savefig(out/'候选评分分量诊断.svg')
    plt.close(fig)
    diag=np.array(r['score_breakdown']['actual_top1']['diagonal'])-r['score_breakdown']['truth']['diagonal']
    cross=float(np.sum(r['score_breakdown']['actual_top1']['cross'])-np.sum(r['score_breakdown']['truth']['cross']))
    write(out/'figure_source.json',dict(case='L1_P2_first0',source='L1_P2_first0_maps.npz',
        sha=sha(out/'L1_P2_first0_maps.npz'),variants=[0,2,3,5],crop=[1800,3000,1500,3000],
        normalization='each-map max on allowed grid',score_difference_components=diag,cross_difference=cross))
    table=['| 固定拟合算子下的输入 | 三候选最近误差/m | 真位置20m内最佳网格点的原始分数排名 |',
           '|---|---:|---:|']
    labels=['实际残差','只保留完整第二源','只保留扣除后的第二源','实际残差去掉首源反射残留','只保留扣除后的第二源LOS','恢复被扣掉的第二源']
    htmlrows=[]
    for label,v in zip(labels,r['score_variants']):
        table.append(f'| {label} | {v["min_candidate_error_m"]:.2f} | {v["near20_point_rank"]} |')
        htmlrows.append(f'<tr><td>{label}</td><td>{v["min_candidate_error_m"]:.2f}</td><td>{v["near20_point_rank"]}</td></tr>')
    md='''# 首源扣除与第二源候选：固定算子的分量归因

2026-10-28。结论：不能把困难简单归为“第二源被扣掉太多”或“首源反射留下假峰”。在保存的困难案例中，扣除后的第二源分量本身仍能指出真位置附近；首源拟合残留使候选排序恶化，其中首源LOS残留不可忽略。当前粗网格和选峰也有独立缺口，尚非已确认的多径理论瓶颈。

## 1. 本轮范围

固定上一轮四条理想频率观测、每条三个首源粗位置，复算当时的秩一拟合增益，共12个固定算子。没有位置优化，没有改初值，没有新增噪声/样本，没有更换频率点。每个算子下只计算6种预先登记的分量反事实，共72张原网格评分图。计算约104.99秒；不是算法运行时间或新定位成绩。

生成器真值仅用于拆出A直达、A反射和、B直达、B反射和。通过原真实响应对已保存无噪观测求取源频谱，并验证重建与原观测一致；该操作是诊断权限，不进入盲求解器。首源标签按粗候选离哪个真源近事后分配，不提供给算法。

## 2. 为什么可以拆，哪些东西不能直接相加

给定原观测产生的首源位置和常复增益，频率f的拟合向量为h_f，令P_f=h_f h_fᴴ/(h_fᴴ h_f)。首源拟合为P_f y_f，残差为(I−P_f)y_f。

虽然原始h由混合观测估计得到，但在本轮干预中将其固定不再重估，因此P是一个固定线性正交投影。可以分别计算每个真分量被移除和被保留的部分，并精确相加回原残差。不能将这些反事实解释成“重新生成某个场景、重新运行算法”的完整效果。

各分量能量不能直接相加：分量之间有相干交叉项。定位评分又是相关矩阵最大特征值，更不能把各分量最大特征值相加。本轮在实际总残差给出的候选特征向量下分解评分，保留4个自项和6个交叉项，核对其和等于真实评分。

## 3. 不能单看“误删了多少能量”

首候选0下，普通构型第二源总分量被扣掉的能量比例，无反射为75.02%、两反射为76.59%，但三候选均包含距真位置10m的点。困难构型对应81.46%和84.90%，其中两反射原残差三候选距第二源最近281.60m。

所以能量损失确实存在，但比例大并不自动导致候选失效。需要看剩余跨站关系和其他残留怎样改变位置排序。未知增益秩一投影吸收的能量也不能全部解释为已正确估计了首源。

## 4. 困难构型首候选0的固定算子干预

首源粗位置(2090,1710)m，距A真位置14.14m；待找的B为(2500,2150)m。所有干预保留原评分、全域20m错位网格、100m去重及三个名额，不根据结果改设置。

TABLE

这里的排名是允许搜索区域内按原始分数排序的网格点名次，不是去重后的峰排名；不能简单将第177名与“三个候选”作同一种排名解释。

最直接的证据是：只保留扣除后的B时，最近候选10m、正确附近网格得分第一；加回A残留后，最近候选281.60m。只删A反射残留仍为281.60m，说明A的LOS拟合残留及其与其余分量的交叉关系不可忽略。这个固定P结论不等于“反射没有影响”，因为反射也已经影响了最初估计出来的P。

实际错误第一候选(2510,2910)m相对B真位置的评分高0.92486。在各自实际评分特征向量下，自项差分别为：A_LOS +1.63880、A反射 +0.42410、B_LOS −1.59631、B反射 +0.10955，交叉项合计 +0.34873。A_LOS残留的支持足以抵消B_LOS对真位置的支持；这是评分差的精确账目，不是各项可独立变化的全局因果定理。

## 5. 更基础的粗搜索问题也暴露出来

困难构型中，即使只保留完整B的无反射分量，原网格三个候选距真位置最近仍64.03m；但精确真位置的评分比允许网格最高评分高约5.60%。两反射的完整B对应3.39%。真位置本身不存在于固定错位网格，直接在真位置评分只用于事后核验，不提供给算法。

这表明在当前96点理想频率、20m网格和去重规则下，正确位置可以有高评分，但粗搜索并未稳定保留它。不能把64m偏离全归双源或多径，也不能据此断言更细网格必然足够。频率容量、峰形与连续初始化尚需区分，本轮没有更改它们。

另须区分最终失败分支：最终选中的首候选1其实先从B附近的(2550,2110)m开始，初始就偏64.03m；随后第二源A候选已在14.14m内。此前最终B约50m误差不是“所有分支都漏了第二个源”，而是在该分支中先找到的B没有被修回来。这个角色交换已经在12个算子中逐一登记。

## 6. 下一步应怎样收敛

不继续调整优化器、DRR或候选数量。优先补一个常规初始化对照：首源粗候选先作正常连续精修，再扣除并产生第二源候选；其余输入、网格、名额和后端固定。它检验“粗位置下过早扣除”能解释多少，而不是先宣称需要新多径算法。当前固定P分量诊断不能保证这一改动必然有效。

如果常规连续初始化即可恢复，按适配缺口处理，不包装为创新；若仍不足，再依据评分中保留下来的结构性误差研究必要改动。残差观测与候选响应的一致性值得核对，但不在本轮偷偷改评分或同时增加多个模块。

本轮未确认新的论文贡献，也未否定多径定位研究；只是把“是什么在妨碍观测驱动起点”缩小到可直接检验的环节。两构型的固定无噪结果不作统计推广。

## 7. 证据与质量检查

5项基础检查、66项完整检查通过：投影线性/幂等/能量恒等、批量评分与原函数一致、交叉项重建、真分量重建和12组原候选完全复现，输入/源码身份保持不变。所有反事实均标记为oracle诊断，不加入方法排名。py_compile和Ruff F/E9通过。图按保存的地图绘制，各图独立归一化，不据颜色比较绝对能量。

[运行协议](protocol.json) · [全部12组结果](result.json) · [图源与口径](figure_source.json) · [图像](候选评分分量诊断.png)。未修改原算法、论文或Zotero，未启动扩大实验。
'''.replace('TABLE','\n'.join(table))
    (out/'首源扣除机制报告.md').write_text(md,encoding='utf-8')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>首源扣除机制</title><style>body{background:#f5f5f0;color:#19343d;font:17px/1.8 system-ui,"Microsoft YaHei",sans-serif;margin:0}main{max-width:1050px;margin:auto;padding:38px 24px}h1{font-size:33px;line-height:1.4}h2{font-size:23px}section,.lead{padding:24px;border:1px solid #dbe3df;border-radius:16px;background:white;margin:22px 0}.lead{background:#e7f1ed}small{color:#60757b}table{width:100%;border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #dbe3df;text-align:left}.scroll{overflow:auto}img{width:100%;height:auto}summary{cursor:pointer;font-weight:600}a{color:#007980}</style><main><small>2026-10-28 · 固定四条观测、12个首源拟合 · 只做分量归因，没有新算法成绩</small><h1>第二源不是被彻底删没了，<br>首源残留和粗搜索改变了候选排序</h1><div class="lead"><strong>本轮最重要的发现</strong><p>困难案例中，扣除后单独剩下的第二源，仍能指向真位置附近。加回首源残留后，候选偏离；只去掉首源反射残留并不足以恢复。不能只怪“反射假峰”或“误删能量太多”。</p></div>
<section><h2>用同一个扣除算子，把影响拆开看</h2><p>固定原观测产生的首源粗位置和增益，不在干预后重新拟合。生成真分量只供事后拆解，不提供给盲算法。</p><div class="scroll"><table><tr><th>评分输入</th><th>三个候选距目标最近/m</th><th>20m内最佳点的原始排名</th></tr>ROWS</table></div><small>困难构型首候选0。原始排名不是去重后的峰排名；除“实际残差”外均是诊断反事实，不是可执行方法。</small></section>
<section><h2>评分图：看候选是否覆盖红色星形</h2><img src="候选评分分量诊断.png" alt="四种固定算子干预的评分图、真位置和三候选"><p>扣除后的第二源本身仍保留可用位置线索。A的LOS残留与其余成分共同改变了排序；不能直接把分量评分相加，完整报告已保留交叉项。</p></section>
<section><h2>还发现一个更基础的问题</h2><p>困难构型中，完整B单独存在、没有反射时，原粗网格三候选最近仍偏64m；但精确真位置的评分比网格最高点高约5.6%。因此，不能把粗选峰损失全部解释为多径或双源困难。</p><details><summary>为什么最终偏50m的源不一定是“第二个找到的源”？</summary><p>最终选中的是另一条分支：先从偏64m的B附近开始，再找到准确的A。A已在候选里，B却没修回来。源标签与算法发现次序不是同一件事。</p></details></section>
<section><h2>下一步只验证一个常规缺口</h2><p>建议先让首源粗候选完成正常连续精修，再扣除和生成第二源候选，其他条件固定。检验粗位置下过早扣除的影响，不增加候选或继续拉长优化。</p><p>若常规处理即可恢复，就按适配缺口收尾；若仍不足，再找有证据支持的算法改动。</p><small>本轮理想频率无噪诊断约105秒；没有扩大样本，也没有确认创新贡献。</small></section>
<footer><a href="首源扣除机制报告.md">完整理论、数据与解释</a> · <a href="result.json">原始结果</a> · <a href="protocol.json">冻结范围</a></footer></main></html>'''.replace('ROWS','\n'.join(htmlrows))
    (out/'首源扣除机制报告.html').write_text(page,encoding='utf-8')
    for f in ['首源扣除机制报告.md','result.json','protocol.json','候选评分分量诊断.png']:
        assert (out/f).exists()
    (out/'source'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(out/'首源扣除机制报告.html')


if __name__=='__main__':main()
