"""Summarize the paired scan; truth matching is evaluation only."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE=Path(__file__).resolve().parent
RAW=HERE.parent/'运行数据'


def main():
    rows=[];runtime=0.;pairs_total=0;details=[];max_position_difference=0.
    for name in ('pilot3','remaining15'):
        folder=RAW/name
        done=json.loads((folder/'completion.json').read_text(encoding='utf-8'))
        assert done['identity_verified'];runtime+=done['seconds']
        for r in json.loads((folder/'rows.json').read_text(encoding='utf-8')):
            tag=r['tag'];i=tag.split('_')[1]
            with np.load(folder/'evaluation'/f'{i}_00.npz',allow_pickle=False) as z:truth=z['p']
            with np.load(folder/f'{tag}_scan.npz',allow_pickle=False) as z:pool=z['pool'];scores=z['scores']
            _,ids=linear_sum_assignment(np.linalg.norm(truth[:,None]-pool[None],axis=2))
            nearerr=np.linalg.norm(pool[ids]-truth,axis=1)
            # Both label assignments inspected only in this truth-based diagnostic.
            nearcost=min(scores[tuple(ids)],scores[tuple(ids[::-1])])
            d=json.loads((folder/f'{tag}_decision.json').read_text(encoding='utf-8'))
            diff=float(np.max(abs(np.array(d['answers']['CONTINUE']['p'])-np.array(d['answers']['JOINT_POSITION']['p']))))
            max_position_difference=max(max_position_difference,diff)
            r['evaluation_only']=dict(near_errors_m=nearerr.tolist(),near_conditional_cost=float(nearcost),candidate_count=len(pool),pairs=int(scores.size))
            pairs_total+=scores.size;rows.append(r)
            errs={a:float(np.sqrt(np.mean(np.array(v['errors_m'])**2))) for a,v in r['arms'].items()}
            details.append(f"<tr><td>{tag}</td><td>{len(pool)} / {scores.size}</td><td>{max(nearerr):.2f}</td><td>{errs['PARENT']:.3f}</td><td>{errs['CONTINUE']:.3f}</td><td>{errs['JOINT_POSITION']:.3f}</td></tr>")
    assert len(rows)==18
    groups=[]
    for c in ('L0','L1_P075','L2_P075'):
        subset=[r for r in rows if r['condition']==c]
        out=dict(condition=c)
        for a in ('PARENT','CONTINUE','JOINT_POSITION'):
            out[a]=float(np.sqrt(np.mean([err**2 for r in subset for err in r['arms'][a]['errors_m']])))
        groups.append(out)
    result=dict(count=18,parent_selected=sum(r['scan']['winner']==r['scan']['parent_pair'] for r in rows),
        paired_total_seconds=runtime,total_scored_pairs=pairs_total,
        scan_seconds_mean=float(np.mean([r['scan']['seconds'] for r in rows])),
        scan_seconds_max=max(r['scan']['seconds'] for r in rows),
        max_score_audit_error=max(r['scan']['audit_error'] for r in rows),
        max_paired_position_difference_m=max_position_difference,groups=groups,rows=rows,
        gate='NO_INCREMENTAL_BENEFIT_OVER_CONTINUATION',scope='SAVED_DEVELOPMENT_PAIRED_ONE_STEP_NOT_FORMAL')
    (HERE/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    table=''.join(f"<tr><td>{g['condition']}</td><td>{g['PARENT']:.4f}</td><td>{g['CONTINUE']:.4f}</td><td>{g['JOINT_POSITION']:.4f}</td></tr>" for g in groups)
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>整组位置更新核验</title><style>body{{margin:0;background:#f1f5f1;color:#193942;font:16px/1.85 system-ui,"Microsoft YaHei"}}main{{max-width:1100px;margin:26px auto;padding:0 22px}}header{{background:#183e47;color:white;border-radius:20px;padding:30px}}h1{{font-size:30px}}h2{{font-size:23px}}section{{margin:22px 0;background:white;border-radius:16px;padding:26px}}.note{{padding:18px;background:#fff1d7;border-left:5px solid #c68f31;margin:20px 0}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:12px;text-align:left;border-bottom:1px solid #dce4de}}th{{background:#e5eee8}}a{{color:#007c87}}.eq{{padding:18px;background:#e7efeb}}small{{color:#5c7275}}details{{overflow:auto}}</style><main>
<header><p>多径DPD · 第19轮 · 2026-11-07</p><h1>把位置候选全部重比，仍然选回原位置</h1><p>固定当前多径形状的联合位置块更新，没有超出继续原精修的收益。</p></header>
<p><a href="方法推导与判断.md">方法与判断</a> · <a href="执行合同.md">固定执行合同</a> · <a href="summary.json">结果数据</a> · <a href="../../../18_重叠整体目标核验/20261107/成果/重叠整体目标报告.html">上一轮目标核验</a></p>
<div class="note"><b>结论：</b>18/18次联合扫描选回父位置。该步骤不加入最终算法，不增加扫描轮次或调权重。没有获得新精度收益。</div>
<section><h2>这次终于落实了怎样的更新？</h2><p>保留当前盲估计的全部多径形状，把两个源的位置一起换到候选池中的每个有序组合，重新消元两路未知波形，再按原IQ目标选优。选后进行原同阶数联合精修。</p><div class="eq">完整观测候选池 × 完整观测候选池 → 完整多径评分 → 同阶数联合精修</div><p>不按LOS评分截断候选，不逐源扣除。两臂K、约束、停止精度和精修上限相同；另外一臂只从同一父状态继续精修，用来排除“多算一次”的影响。</p><p><b>边界：</b>扫描时多径形状固定，尚未同时重构路径归属。因此这只是整体架构的一个可检验步骤，不是完整新算法。</p></section>
<section><h2>配对定位结果</h2><table><tr><th>条件</th><th>原父状态RMSE/m</th><th>只继续原精修/m</th><th>联合扫描＋精修/m</th></tr>{table}</table><p>每条件6条旧开发记录，全部保留。大误差没有被成功门槛截掉。两臂位置最大差异仅{max_position_difference:.2e}米。</p><p>IN101单反射案例由约2.214米改善到0.437米，但两臂都改善。因此不能把它归功于新扫描。单反射两臂6/6正常，双反射5/6正常；不把非正常样本剔除。</p></section>
<section><h2>这不是“扫描漏了原位置”的结果</h2><p>候选池主动加入当前两位置。实际共计算{pairs_total:,}个有序位置对；每个记录均核对父状态、胜者和12个固定随机对，批量评分与原IQ目标最大差异为{result['max_score_audit_error']:.2e}。</p><p>36个代数单元对还包含严格秩一情形，近共线时回退原SVD秩规则。未抽掉频点，也未提供真实时延、反射数或波形。</p><p>后验真值匹配仅检查候选池覆盖，不参与选择；完整逐例数据见下表。近真候选存在，不意味着它能与当前已适应错误位置的多径形状配合。</p></section>
<section><h2>耗时与推进判断</h2><p>两阶段合计{runtime:.2f}秒（含两臂精修、扫描、保存及评价，不含前期代码准备）。新增候选扫描平均{result['scan_seconds_mean']:.3f}秒，最大{result['scan_seconds_max']:.3f}秒。这个时间不含此前生成父状态的完整盲定位，不能报告为端到端3–5秒达标。</p><p><b>现在更明确的缺口：</b>不能只拿当前解释里的多径形状，去判断另一个位置是否值得保留。下一种提案需要让位置和它自己的路径形状共同产生，而不是先选错再固定形状全池扫描。</p><p>这不是所有联合求解方法无效的证明，也不是要求改场景。当前尚无可冻结的低成本整组提案规则，因此没有自动增加多启动、改变路径数或扩大实验。</p></section>
<section><details><summary>展开18条明细（近真覆盖仅为评价）</summary><table><tr><th>记录</th><th>候选数/有序对数</th><th>近真匹配最大误差/m</th><th>父RMSE/m</th><th>继续/m</th><th>联合/m</th></tr>{''.join(details)}</table></details><p><small>输入与源码前后身份核验通过；Python编译、Ruff及投影单元核验通过。公共算法源码未改，正式扩量未启动。</small></p></section></main></html>'''
    (HERE/'整组位置更新报告.html').write_text(page,encoding='utf-8')


if __name__=='__main__':main()
