"""Descriptive paired replay report: truth used only by post-decision evaluator."""
import csv
import html
import io
from pathlib import Path

import numpy as np
from scipy.stats import t
from storage import load, write, cost


def interval(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return dict(n=0, mean=None, ci95=None)
    mean = float(a.mean())
    ci = None
    if len(a)>1:
        width = float(t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a)))
        ci = [mean-width,mean+width]
    return dict(n=len(a),mean=mean,ci95=ci)


def group(rows):
    eligible = [r for r in rows if not r['decision']['fallback']]
    full = [r for r in rows if r['decision']['full_trajectory_coverage']]
    models = {}
    for name in ('LOS','WIDE','SELECTED_OR_FALLBACK'):
        losses = [r['losses'][name] for r in rows]
        error = [v for r in rows for v in r['source_errors'][name]]
        models[name] = dict(mse=float(np.mean(losses)) if losses else None,
                            rmse=float(np.sqrt(np.mean(losses))) if losses else None,
                            p95_source_error=float(np.quantile(error,.95)) if error else None,
                            max_source_error=max(error) if error else None)
    differences = [r['losses']['SELECTED_OR_FALLBACK']-r['losses']['WIDE'] for r in rows]
    return dict(n=len(rows),risk_output_count=len(eligible),fallback_count=len(rows)-len(eligible),
                full_coverage_count=len(full),
                multi_model_choice_count=sum(r['decision']['applicable_count']>=2 for r in rows),
                models=models, paired_loss_difference_all=interval(differences),
                paired_loss_difference_eligible=interval([r['losses']['SELECTED_OR_FALLBACK']-r['losses']['WIDE'] for r in eligible]),
                paired_loss_difference_full_coverage=interval([r['losses']['SELECTED_OR_FALLBACK']-r['losses']['WIDE'] for r in full]),
                # No post-hoc tolerance: exact stored-output equality is distinguished.
                improved=sum(v<0 for v in differences),harmed=sum(v>0 for v in differences),unchanged=sum(v==0 for v in differences),
                original_localization_seconds=sum(r['baseline_solve_seconds'] for r in rows),
                equivalent_risk_seconds=sum(r['risk_equivalent_seconds'] for r in rows))


def fmt(value):
    return '—' if value is None else f'{value:.6g}'


def summarize(out, contract):
    out = Path(out)
    rows = [load(p) for p in sorted((out/'cases').glob('*/record.json'))]
    groups = {name:group([r for r in rows if r['condition']==name]) for name in ('L0','L2','L3','HIGH')}
    overall = group(rows)
    data = dict(completed=len(rows),expected=32,complete=len(rows)==32,conditions=groups,overall=overall,
                replay_cost=cost(out),independent_validation=False,new_localization_runs=0,
                risk_selection_validated=False,requires_scientific_review=True,
                inference='conditional descriptive paired intervals; fixed geometry and waveforms; no automatic route verdict',
                contract_identity_checked_at_run=True)
    write(out/'summary.json',data)
    lines = ['# 32条保存轨迹位置风险完整回放', '',f'完成 {len(rows)}/32；'+('正式回放已完成。' if len(rows)==32 else '阶段结果，不能作最终结论。'),'',
        '这是旧观测的冻结规则离线回放，不是新盲定位或独立验证。没有改变公式、初始化、停止条件或多径模型。', '',
        '|条件|条数|LOS RMSE/m|完整多径 RMSE/m|风险选择＋原终态回退 RMSE/m|风险输出/完整覆盖|',
        '|---|---:|---:|---:|---:|---:|']
    for name,g in groups.items():
        m=g['models']
        lines.append(f'|{name}|{g["n"]}|{fmt(m["LOS"]["rmse"])}|{fmt(m["WIDE"]["rmse"])}|{fmt(m["SELECTED_OR_FALLBACK"]["rmse"])}|{g["risk_output_count"]}/{g["full_coverage_count"]}|')
    lines += ['', '## 如何解释', '',
        '- 两条HIGH容量触顶没有可靠宽模型，不评分，原终态回退仍保留严重失败。',
        '- 原LOS不满足风险适用条件时不替换成后续继续求解状态；可用子集上的选择单列覆盖，不能称全部模型可比较。',
        '- 没有风险输出时保留原完整终态。回退的零增益不是风险方法成功。只有一个可用模型也不是有效模型选择。',
        '- 评分先落盘，然后读取真值评价。匹配仅发生在评价；报告中的事后最优仅用于归因。',
        '- 负风险、损害、边界、非驻点和不适用状态均保留。不根据真值筛选。',
        '- 配对差为“选择－完整终态”的每源平均平方位置误差，小于零为改善。区间为条件内8次噪声重复的描述性t区间，非跨构型推广或多重检验保证。',
        '- 总体均值会受HIGH失败主导，必须同时看条件内与风险覆盖子集，不能用有利子集代替全体。', '',
        '## 配对差与成本', '']
    for name,g in groups.items():
        z=g['paired_loss_difference_all']
        lines.append(f'- {name}：平均平方误差差 {fmt(z["mean"])} m²，95%区间 {z["ci95"]}；改善/损害/不变 {g["improved"]}/{g["harmed"]}/{g["unchanged"]}。')
    lines += ['',f'本次回放账本 {cost(out)["seconds"]/3600:.3f} h；原定位求解合计 {overall["original_localization_seconds"]/3600:.3f} h；风险诊断等价计算 {overall["equivalent_risk_seconds"]/3600:.3f} h。',
        '等价诊断时间包含复用端点/探针的历史诊断时间与新诊断时间，跨批计时只作描述。完整算法成本必须包含原定位及所有必要评分，不能把缓存命中称为新算法加速。中断成本下界见summary.json。', '',
        '## 结果后怎么办', '',
        '不自动宣布论文贡献，不自动调整公式或扩大实验。回读时综合评估条件内RMSE、长尾、覆盖和成本；若无清楚净收益，客观评估其原因属于局部近似、参照偏差、同数据选择还是路线本身价值不足，再决定停止或进行有理论依据的有限修订。640条未授权。']
    md='\n'.join(lines)+'\n'
    (out/'完整风险回放报告.md').write_text(md,encoding='utf-8')
    stream=io.StringIO(newline='')
    fields=['case','condition','fallback','full_coverage','selected_order','candidate_count','applicable_count','LOS_mse','WIDE_mse','SELECTED_mse','paired_difference','baseline_seconds','risk_equivalent_seconds']
    writer=csv.writer(stream,lineterminator='\n');writer.writerow(fields)
    for r in rows:
        d=r['decision'];m=r['losses']
        writer.writerow([r['case'],r['condition'],d['fallback'],d['full_trajectory_coverage'],d['selected_order'],d['candidate_count'],d['applicable_count'],m['LOS'],m['WIDE'],m['SELECTED_OR_FALLBACK'],m['SELECTED_OR_FALLBACK']-m['WIDE'],r['baseline_solve_seconds'],r['risk_equivalent_seconds']])
    (out/'records.csv').write_text(stream.getvalue(),encoding='utf-8-sig',newline='')
    esc=html.escape
    table=''.join(f'<tr><td>{name}</td><td>{g["n"]}</td><td>{fmt(g["models"]["LOS"]["rmse"])}</td><td>{fmt(g["models"]["WIDE"]["rmse"])}</td><td>{fmt(g["models"]["SELECTED_OR_FALLBACK"]["rmse"])}</td><td>{g["risk_output_count"]}/{g["full_coverage_count"]}</td></tr>' for name,g in groups.items())
    cards=''.join(f'<article><h3>{name}</h3><p>选择－终态平均MSE：<strong>{fmt(g["paired_loss_difference_all"]["mean"])} m²</strong></p><p>改善 / 损害 / 不变：{g["improved"]} / {g["harmed"]} / {g["unchanged"]}</p><p>95%区间：{esc(str(g["paired_loss_difference_all"]["ci95"]))}</p><p>可用至少两个模型：{g["multi_model_choice_count"]}/{g["n"]}</p></article>' for name,g in groups.items())
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>位置风险完整回放</title><style>
body{{margin:0;background:#f4f6f5;color:#183239;font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1080px;margin:auto;padding:40px 22px}}h1{{line-height:1.3}}.tag{{color:#28776f}}.notice,article{{background:white;padding:22px;border:1px solid #d8e2df;border-radius:14px;margin:18px 0}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:16px}}.grid article{{margin:0}}.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;white-space:nowrap}}td,th{{text-align:left;border-bottom:1px solid #d8e2df;padding:12px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}}a{{color:#067e78}}small{{color:#526970}}</style><main>
<p class="tag">冻结评分 · 保存轨迹 · 不新增定位求解</p><h1>位置风险选模：完整回放</h1><p>完成 {len(rows)}/32 · {'已完成，等待科学回读' if len(rows)==32 else '阶段结果，请等待全部完成'}</p>
<div class="notice">本页回答：同一个冻结评分，能否比直接使用完整多径终态得到更好的定位结果？不是独立验证，也不把选中较小模型等同于计算加速。</div>
<h2>定位结果与覆盖</h2><div class="scroll"><table><thead><tr><th>条件</th><th>条数</th><th>LOS RMSE/m</th><th>多径终态 RMSE/m</th><th>选择＋回退 RMSE/m</th><th>风险输出/全覆盖</th></tr></thead><tbody>{table}</tbody></table></div><p><small>无可靠宽模型时回退原终态，失败仍在总体统计中；部分模型可评分不等于全轨迹适用。</small></p>
<h2>每个条件的得失</h2><div class="grid">{cards}</div>
<h2>成本必须完整计算</h2><p>回放账本 {cost(out)['seconds']/3600:.3f} h；原定位 {overall['original_localization_seconds']/3600:.3f} h；诊断等价计算 {overall['equivalent_risk_seconds']/3600:.3f} h。缓存复用减少本次等待，不是算法加速证据。</p>
<details><summary>协议、解释边界与后续判断</summary><pre>{esc(md)}</pre></details>
<p><a href="summary.json">统计JSON</a> · <a href="records.csv">逐例CSV</a> · <a href="完整风险回放报告.md">文字报告</a> · <a href="contract.json">冻结合同</a></p></main></html>'''
    (out/'完整风险回放报告.html').write_text(page,encoding='utf-8')
    return data
