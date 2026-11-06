"""Source-linked route decision; separates valid controls and failed optimizers."""
from pathlib import Path
from html import escape
import json
import numpy as np
from pilot import ROOT, write, sha
from trace_audit import read

SCREEN=ROOT/'screen_20261027_190544_337983'
OUT=ROOT/'numerical_20261027_191234_666421'


def main():
    a=read(SCREEN/'result.json');b=read(OUT/'acceptance_audit.json');raw=read(OUT/'result.json')
    assert read(SCREEN/'audit.json')['passed'] and read(OUT/'audit.json')['passed'] and b['passed']
    metrics=[s for s in a['summary'] if s['scope']=='all' and s['version'].endswith('groups')]+b['summary']
    names={'ratio_groups':'原比值评分＋保组修正','ccf_groups':'普通互相关评分＋保组修正',
           'ratio_accepted':'原比值评分＋六初值选峰（离线接受）','ccf_accepted':'普通互相关＋六初值选峰（离线接受）'}
    table=''.join(f"<tr><td>{names[s['version']]}</td><td>{s['complete']}/28</td><td>{s['conditional_RMSE']:.1f}</td><td>{s['both20']}/28</td></tr>" for s in metrics)
    # Paired available-set comparison, never mix denominators.
    by={(r['case'],r['version']):r for r in a['rows']+b['rows']}
    pairs=[]
    for left,right in [('ratio_groups','ccf_groups'),('ratio_groups','ratio_accepted'),('ratio_accepted','ccf_accepted')]:
        cases=sorted(c for c,v in by if v==left and by[c,v]['evaluation']['complete'] and by[c,right]['evaluation']['complete'])
        vals=[float(np.sqrt(np.mean([np.array(by[c,v]['evaluation']['error_m'])**2 for c in cases]))) for v in [left,right]]
        pairs.append(dict(left=left,right=right,n=len(cases),rmse=vals,cases=cases))
    write(OUT/'paired_summary.json',dict(pairs=pairs,metrics=metrics,source_hashes={str(p):sha(p) for p in [SCREEN/'result.json',OUT/'result.json',OUT/'acceptance_audit.json']}))
    pairtable=''.join(f"<tr><td>{names[p['left']]} → {names[p['right']]}</td><td>{p['n']}</td><td>{p['rmse'][0]:.1f} → {p['rmse'][1]:.1f}</td></tr>" for p in pairs)
    refs=[
      ('10.1109/LWC.2020.2992078','YC2XSZJZ','4US29C8J','2020：空间稀疏/互相关','式(8)—(10)、4页全文','可借鉴普通互相关，不能把本轮控制叫原论文复现。原文还有多普勒、时序运动与核函数；未解决我们当前未知多径双源。'),
      ('10.1109/LCOMM.2023.3347620','YWUTRNXK','R55J7VF5','2024：循环互谱','pp.303—305，模型、式(1)及仿真设置','用不同符号率对应的循环频率分离源。我们的两个源同符号率，不能直接套用；不能为跑通它而偷偷改变场景。'),
      ('10.1109/JIOT.2022.3228810','JXFE2X2D','F2QKDL58','2023：PARAFAC/Multi-TDOA','pp.7405—7407，§II观测模型','多TDOA不等于多发射源；该模型为单源、多接收站和多分段。可借鉴分段结构，但混合双源与路径关联需要重新推导。'),
      ('10.3390/s22197245','XWJMZUKE','BJJDPD4Z','2022：多径互谱子空间','§2—3，pp.3—5相关模型','单源且参考站无多径。可借鉴跨谱平均及相干路径处理，不能假装我们已有干净参考站。')]
    literature=[];cards=''
    for doi,item,att,label,scope,note in refs:
        meta=read(OUT/('crossref_'+doi.split('/')[-1]+'.json'))['message']
        pdfs=list((Path('F:/Zotero/storage')/att).glob('*.pdf'))
        entry=dict(doi=doi,title=meta['title'][0],item=item,attachment=att,read_scope=scope,interpretation=note,
                   endpoint='https://api.crossref.org/works/'+doi.replace('/','%2F'),pdf=str(pdfs[0]) if pdfs else None,
                   pdf_sha=sha(pdfs[0]) if pdfs else None)
        literature.append(entry)
        cards+=f'<article><h3>{label}</h3><p><a href="https://doi.org/{doi}">{escape(meta["title"][0])}</a></p><p>{note}</p><small>本轮核对：{scope}；Zotero {item}。已在库，无需重复导入。</small></article>'
    write(OUT/'literature_check.json',dict(database='Crossref DOI metadata + local Zotero PDFs',papers=literature,
        distinction='Metadata is not evidence of assumptions; assumptions come from stated local PDF sections. Not all four papers reread in full this turn.'))
    md=f'''# Xia路线有界筛查：暂停主线，不把有限失败当作不可能证明

本轮仅用预定4个基础实现（0、3、6、9）及7条件，共28观测；不改变输入，不提供真实波形、增益、路径。正方形五站、原公开信息重建模型；不是当前50MHz场景的最终比较，也不是作者代码复现。四个基础实现不足以建立泛化结论。

## 执行与质量纠正

第一轮评分×输出2×2共112次调用，输入/评分/梯度/冻结流程345项检查通过。事后发现L-BFGS-B的336个分支中111个试探到接收站坐标而被不可微保护中止；两列连续精修不能作为充分求解参照，screen_pass不能单独支撑路线否定。

第二轮保持评分及六初值，另用Powell，56次调用、336分支正常返回。不过原比值139/168、普通互相关28/168终点比初值评分更差；不能凭优化器success标志称精修充分。另存离线接受检查：仅按原生目标保留初值或终点的较优者，未看真值。该检查不新增优化、不保证局部/全局最优，不用于宣称充分优化后的Xia仍失败。

离线接受后原比值完整28/28、RMSE {b['summary'][0]['conditional_RMSE']:.3f}m；普通互相关完整14/28、条件RMSE {b['summary'][1]['conditional_RMSE']:.3f}m；双源20m均0/28。未形成可信创新收益。不同完整子集RMSE不能直接比较，配对列表见paired_summary.json。第一轮约{a['seconds']:.2f}s，第二轮约{raw['seconds']:.2f}s；审计/文献/文档成本另计，不作为算法计时。

## 当前决策

暂停Xia阈值、聚类、局部重采样的继续改进；保留原重建与流程修正作历史参照。不声称Xia理论失效、全域无正确峰、排除全部搜索问题或发现可投稿贡献。退出依据是有限修改未产生可用收益，且继续完善求解已接近另起算法，机会成本偏高。

## 下一项只做一个问题

核对“未知同符号率双源在多径下，哪些跨站、跨分段信息能区分源与各自路径”。先写统一观测模型及最近邻条件表，再构造一个最小辨识/机制测试；不继续调Xia阈值，不直接启动新大型算法。独立分段可用于平均，但相同二阶源谱下未必能分离，需要证明或承认可辨识不足。不得引入已知数据、不同符号率或无多径参考站来换取通过。

## 文献

'''+''.join(f'- [{r["title"]}](https://doi.org/{r["doi"]})：{r["interpretation"]} 核对范围：{r["read_scope"]}。\n' for r in literature)
    (OUT/'路线筛查说明.md').write_text(md,encoding='utf-8')
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Xia路线：有界筛查与下一步</title><style>
    body{{margin:0;background:#f4f5f2;color:#19313a;font:17px/1.8 'Segoe UI','Microsoft YaHei',sans-serif}}main{{max-width:1080px;margin:45px auto;padding:0 25px}}h1{{font-size:36px;line-height:1.3}}h2{{margin-top:40px}}.lead,article,details{{background:white;border:1px solid #dce5e5;border-radius:16px;padding:22px;margin:16px 0}}.lead{{border-left:6px solid #137f85;background:#e9f4f1}}small{{color:#536973}}a{{color:#087681}}table{{border-collapse:collapse;width:100%;background:white;font-size:15px}}td,th{{padding:12px;border-bottom:1px solid #dce5e5;text-align:left}}.scroll{{overflow-x:auto}}summary{{cursor:pointer;font-weight:bold}}.warn{{color:#964600}}footer{{margin:40px 0}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}@media(max-width:700px){{.grid{{display:block}}h1{{font-size:28px}}}}
    </style><main><small>2026-10-27 · 小规模路线筛查 · 非正式性能复现</small><h1>停止围绕Xia反复修补，<br>把研究重点转向源与路径的区分</h1>
    <div class="lead"><strong>本轮没有找到值得扩测的改进。</strong><br>输出能恢复，准确定位仍没有建立。这个结论足以决定暂缓投入，但不是证明Xia在所有场景都不行，也不是新算法已形成。</div>
    <h2>这次具体试了什么？</h2><div class="grid"><article><h3>改评分</h3><p>保留原比值评分；另用普通互相关作常规对照，避免逐频除以很小的参考频谱。其余输入和初始候选不变。</p></article><article><h3>改输出</h3><p>一条用已核验的保组修正；另一条从同一批点挑六个初值，连续选峰。不是增加随机试次后挑最好结果。</p></article></div>
    <p>固定4个基础实现 × 7个条件＝28观测。沿用正方形五站与已登记重建模型，没有Hann、真值提示或新的信道参数。每种评分共享相同初始1000点。</p>
    <h2>结果：能给出两个位置，不等于定位准确</h2><div class="scroll"><table><tr><th>版本</th><th>完整输出</th><th>有输出样本RMSE / m</th><th>双源≤20m</th></tr>{table}</table></div>
    <p class="warn">各行完整样本不同，不能直接用945.7比1900小来宣布改进。缺失结果没有填成零误差，也没有从成功率分母删除。底部两行是基于已存初值/终点的离线接受结果，不是充分局部优化或新的一键算法。</p>
    <details><summary>只比较双方都有输出的同一批观测</summary><div class="scroll"><table><tr><th>配对</th><th>共同样本</th><th>RMSE / m</th></tr>{pairtable}</table></div><p>这些是选定子集的描述性结果，不做总体显著性或泛化结论。</p></details>
    <details open><summary>质量复核发现了什么？为什么不能直接宣布“框架失败”？</summary><p>第一轮L-BFGS-B有111/336分支在接收站坐标处被不可微保护终止，因此该两列不作为充分精修参照。</p><p>补充Powell核验336分支均正常返回，但167个终点评分反而低于初值。我们没有把“success”当作正确性证明：另做<strong>仅依据原评分保留更优初值或终点</strong>的离线审计，结果仍不理想。</p><p>因此本轮支持的是“所试有限修改没有显示继续投入的价值”，而不是“全局评分一定错误”或“所有搜索改进都无效”。初始覆盖与充分优化尚未被排除；继续把它们做全将是额外研究投入。</p></details>
    <h2>从邻近论文学什么？哪些条件不能偷换？</h2>{cards}
    <h2>下一步只聚焦一个问题</h2><article><strong>同一组混合观测中，怎样区分两个源，以及每个源的直达与反射分量？</strong><p>先核对跨站、跨分段观测能提供什么独立信息，再判断最近邻方法缺了哪一步。只有明确区分依据，才设计一个核心改动。</p><p>保持二维、单天线分布式、未知同频双源、参数化多径。不增加墙面、不提供真实波形、不把两源改成不同符号率、不假设干净参考站。</p><p>停止线：若常规方法已经满足要求，或现有观测根本不能区分所需参数，就直接记录结论；不靠更多阈值、候选数量或新场景继续碰运气。</p></article>
    <footer><a href="路线筛查说明.md">完整说明</a> · <a href="paired_summary.json">配对统计</a> · <a href="acceptance_audit.json">接受审计</a> · <a href="literature_check.json">原文条件与Crossref端点</a> · <a href="protocol.json">补充核验协议</a><p><small>本轮文献均已在Zotero；未修改库或论文。Crossref原始JSON保存在同目录。新的完整定位研究尚未启动。</small></p></footer></main></html>'''
    (OUT/'Xia路线筛查.html').write_text(html,encoding='utf-8')
    print(json.dumps(pairs,ensure_ascii=False));print(OUT/'Xia路线筛查.html')


if __name__=='__main__':main()
