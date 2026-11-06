"""Independent saved-state counting and reader-facing completion; no solvers."""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from pilot import ROOT, sha, write

OUT = ROOT/'trace_20261027_180535_576489'
RUN = ROOT/'pilot_20261027_170421_805496'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def main():
    protocol = read(OUT/'protocol.json')
    assert all(sha(RUN/n)==h for n,h in protocol['input_hashes'].items())
    rows = list(csv.DictReader((OUT/'rounds.csv').open(encoding='utf-8-sig')))
    initial = [r for r in rows if r['round']=='1']
    checks = []
    second = []
    first_empty_radius = []
    for file in sorted(RUN.glob('r*_G*.json')):
        rec=read(file)
        for step in rec['output']['steps']:
            p=np.array(step['points']); v=np.array(step['scores']); result=step['output']
            hist,edges=np.histogram(v,bins=10)
            t=2*edges[int(hist.argmax())+1]-v.min()
            checks.append(bool(np.isclose(t,result['threshold'],rtol=1e-14,atol=0)))
            mask=v>=result['threshold']
            checks.append(int(mask.sum())==result['retained'])
            if mask.any():
                first=p[mask][v[mask].argmax()]
                saved_first=result['positions'][0] if result.get('positions') is not None else result['first_position']
                checks.append(bool(np.array_equal(first,saved_first)))
                d=np.linalg.norm(p[mask]-first,axis=1)
                for h,count,inc in result['trace']:
                    checks.append(int(np.sum(d<=h))==count)
                    checks.append(int(np.sum(d<=h)-np.sum(d<=h-400))==inc if h>400 else int(np.sum(d<=h))==inc)
                if result['status']=='EMPTY_SECOND_CLUSTER':
                    checks.append(bool(np.max(d)<=result['trace'][-1][0]))
                    if step['round']==1: first_empty_radius.append(result['trace'][-1][0])
            if step['round']==2:
                # Frozen sampler alternates the two parents; this is ancestry,
                # not a newly inferred or truth-based clustering.
                second.append(dict(case=file.stem,status=result['status'],
                    mode=result['modal_bin'], threshold=result['threshold'], maximum=float(v.max()),
                    parent1_survivors=int(mask[::2].sum()),parent2_survivors=int(mask[1::2].sum()),
                    old_center_separation_m=float(np.linalg.norm(np.subtract(*rec['output']['steps'][0]['output']['positions'])))))
    assert all(checks)
    nearest = {str(rad):dict(A=sum(float(r['first_A_error_m'])<=rad for r in initial),
                            B=sum(float(r['first_B_error_m'])<=rad for r in initial),
                            either=sum(min(float(r['first_A_error_m']),float(r['first_B_error_m']))<=rad for r in initial)) for rad in [20,100,400]}
    counts=Counter((r['parent1_survivors'],r['parent2_survivors']) for r in second)
    extra=dict(independent_checks=len(checks),passed=True,first_peak_near_source=nearest,
               second_round_parent_retained={str(k):v for k,v in counts.items()},
               second_round=second,
               first_failure_cluster_radius=dict(n=len(first_empty_radius),
                   minimum=min(first_empty_radius),median=float(np.median(first_empty_radius)),maximum=max(first_empty_radius)),
               second_empty_threshold_above_max=sum(r['threshold']>r['maximum'] for r in second),
               sources_unchanged=True,finalizer_sha=sha(__file__))
    write(OUT/'supplement.json',extra)
    (OUT/'trace_finalize.py').write_bytes(Path(__file__).read_bytes())
    s=read(OUT/'summary.json')
    html=(OUT/'逐轮链路核查.html').read_text(encoding='utf-8')
    html=html.replace('<h1>不是只丢了第二个输出：<br>第一峰的定位也没有跑通</h1>',
                      '<h1>目标在哪一步丢失？<br>筛选、分组和第一峰都需要分开看</h1>')
    tag='<!-- EXTRA_TRACE_BEGIN -->'
    if tag not in html:
        details=f'''{tag}<section><h2>最明确的两个失效环节</h2>
<p><b>第一轮：59次分组没有找到两个分开的区域。</b>保留点之间没有出现规则要求的空白距离环，分组半径一直扩大，最后把所有保留点都收进第一组。半径中位数{np.median(first_empty_radius):.0f}m，范围{min(first_empty_radius):.0f}—{max(first_empty_radius):.0f}m。这不意味着算法真的把两个真实源都找到了后合并；很多保留点本身就在错误位置。</p>
<p><b>第二轮：25次已有位置对全部丢失。</b>其中18次只留下一个旧中心附近的全部500点，另一个中心附近500点全被阈值删掉；另7次众数落在第9或第10箱，使阈值真正超过最大评分，1000点全部被删。新一轮的候选分布变了，原阈值却继续按相同公式计算，这是后轮失效的直接数值过程。</p>
<p>这些是本重建实现中已经查明的处理环节，不等于证明作者实际程序一定采用完全相同的未明示细节。</p>
<h3>不是仅仅把目标身份搞反</h3><p>第一峰有41/84次更接近源B，但“更近”不代表准确。在100m这个诊断范围内，17次靠近A、8次靠近B，合计只有25/84次靠近任一真实源；剩余59次离两个源都超过100m。以400m查看，仍有39/84次离两个源都超过400m。</p>
<p>因此，第一峰的固定A误差里既可能有源身份混淆，也包含位置本身偏离。这里的“靠近任一源”只用于区分原因，不能替换论文的固定目标A RMSE。20/100/400m均已输出，100m不是新的成功门槛。</p>
<p><a href="supplement.json">独立计数与重采样来源审计：{len(checks)}项通过</a> · <a href="trace_finalize.py">生成脚本</a></p></section><!-- EXTRA_TRACE_END -->'''
        html=html.replace('<section><h2>3．覆盖',details+'<section><h2>3．覆盖')
    (OUT/'逐轮链路核查.html').write_text(html,encoding='utf-8')
    md=f'''# Xia保存结果逐轮链路核查

## 范围

源目录`{RUN.name}`。84次原求解、12个基础随机实现；本轮重算109个保存轮次的原评分、原阈值及原解码，没有新增定位求解、改配置、换随机实现或修算法。真值只用于离线误差及固定49点/源的100m局部评分探针，未反馈给算法。20/100/400m为诊断覆盖尺度，成功门槛未改。

## 直接结果

- 第一轮：59次EMPTY_SECOND_CLUSTER、25次完整；第二轮：18次EMPTY_SECOND_CLUSTER、7次NO_RETAINED_PARTICLE；没有样本进入第三轮。
- 第一轮59次分组半径扩至包含全部保留点；半径中位数{np.median(first_empty_radius):.0f}m，范围{min(first_empty_radius):.0f}—{max(first_empty_radius):.0f}m。
- 第二轮18次仅保留旧第一中心附近的500点；旧第二中心附近500点全部被删。其余7次众数箱为9/10，阈值超过最大分值。明确发生于阈值/分组链路，不是保存文件丢失。
- 第一轮第一峰固定源A RMSE {s['first_initial']['rmse_m']:.6f}m，84/84可评价；终轮第一峰77/84可评价，条件RMSE {s['first_terminal']['rmse_m']:.6f}m。缺失7例不能用条件RMSE宣称改善。
- 第一峰更接近B：41/84。距A100m内17次、距B100m内8次；距任一源100m内25次。59次离两个源都超过100m，39次都超过400m，因此并非只有目标身份交换。
- 第一轮A/B附近100m内有候选49/56次；其中全部被筛掉9/26次。真值点评分通过阈值51/50次；100m局部49点至少一个通过54/55次；局部最佳探针相对1000粒子进入前10仅35/27次。探针不是连续空间全局峰搜索，既不能保证找到了局部峰，也不能证明其余样本没有更好的近邻位置。

## 判断与下一步

不能只修第二目标输出就宣布复现成功。当前实现同时存在覆盖、原始评分和阈值/分组失效，后两者的直接数值事件已经保存。第一峰误差并非双源配对RMSE；本轮补齐了之前因缺少完整位置对而未统计的第一峰。

下一步建议固定同一观测，核对式(9)参考频谱除法的各频点贡献及原文统计假设；特别关注参考功率低的频点是否主导错误峰。此处是待验证线索，不是已确认根因。不改阈值、加正则、换模型或启动1000次重复。若需要适配，另列版本，不冒充原作者代码。

原文依据：Xia等2021，pp.2856–2857，式(9)—(16)、(20)。本重建补全约定不等于作者真实脚本。原报告目标为固定j=1，本页没有把输出事后改配为最近源。

## 产物与审计

[HTML](逐轮链路核查.html)、[逐例](cases.csv)、[逐轮](rounds.csv)、[各邻域](neighborhoods.csv)、[精确汇总](summary.json)、[补充计数](supplement.json)。原始证据身份前后不变；{s['checks']}项重放/身份检查和{len(checks)}项独立直方图/计数核查通过。离线评分核对耗时{s['seconds']:.2f}s；未产生新算法性能批次。
'''
    (OUT/'链路核查结论.md').write_text(md,encoding='utf-8')
    previous = ROOT/'公开信息重建报告.html'
    old = previous.read_text(encoding='utf-8')
    if 'id="latest-trace"' not in old:
        banner = f'<section id="latest-trace"><h2>新增：目标在哪一轮、哪一步丢失？</h2><p>已完成原84次结果的逐轮核查：59次首轮分组失败；25次进入第二轮后，18次删掉整个第二候选群，7次筛空全部候选。第一目标的独立评价也已补齐。</p><p><a href="{OUT.name}/逐轮链路核查.html">查看逐轮核查、示意图与下一步判断 →</a></p></section>'
        old = old.replace('<section>',banner+'<section>',1)
        previous.write_text(old,encoding='utf-8')
    print(json.dumps(extra,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
