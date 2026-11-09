"""Descriptive audit summaries; no significance or generalization claims."""
import csv
import html
import json
from pathlib import Path
import numpy as np
from run_gate import cfg, sha, read, OLD_C


HERE = Path(__file__).resolve().parent
DATA = HERE.parent/'运行数据'


def rmse(v):
    return float(np.sqrt(np.mean(np.square(v)))) if v else None


def main():
    anchor = read(DATA/'anchor24/rows.json')
    common = read(DATA/'common18/rows.json')
    support = read(DATA/'support12/rows.json')
    cold = read(DATA/'cold12/rows.json')
    assert len(anchor) == 24 and len(common) == 18 and len(support) == len(cold) == 12
    # Recheck original receipt identity and both copied and source observation files.
    checks = []
    for row in common:
        src = OLD_C/'cases'/row['tag']
        assert sha(src/'iq.npz') == row['input_sha'] == sha(DATA/'common18/inputs'/row['tag']/'iq.npz')
        receipt = read(src/'receipt.json')
        assert sha(src/'iq.npz') == receipt['iq.npz']
        truth = Path(row['truth_evaluation_file'])
        assert sha(truth) == row['truth_file_sha'] == read(truth.with_suffix('.json'))['sha']
        checks.append(row['tag'])
    for phase in ('anchor24', 'common18', 'support12', 'cold12'):
        pre = read(DATA/phase/'preflight.json')
        assert all(sha(p) == h for p, h in pre['source_hashes'].items())
        complete = read(DATA/phase/'completion.json')
        assert complete['source_unchanged']
        for p, h in complete.get('source_hashes', {}).items():
            assert sha(p) == h
    metrics = []
    for condition in ('L0', 'L1_P075', 'L2_P075'):
        for method in ('GEO', 'LOS', 'MP', 'FLOOD_LP_K2', 'XIA_ADAPTED'):
            values = [r['methods'].get(method) for r in common if r['condition'] == condition]
            returned = [v for v in values if v and v['errors_m'] is not None]
            times = [v['seconds'] for v in values if v]
            conditional = rmse([v['errors_m'] for v in returned])
            metrics.append(dict(condition=condition, method=method, total=len(values),
                returned=len(returned), normal=sum(bool(v and v['normal']) for v in values),
                all_case_rmse_m=conditional if len(returned) == len(values) else None,
                returned_subset_rmse_m=conditional, mean_seconds=float(np.mean(times)),
                median_seconds=float(np.median(times)), p90_seconds=float(np.percentile(times, 90)),
                maximum_seconds=float(max(times))))
    controls = []
    for r in support:
        c = next(z for z in cold if z['tag'] == r['tag'])
        v = dict(tag=r['tag'], original_rmse_m=rmse(r['original_errors_m']),
            original_cost=r['original_cost'], original_pg=r['original_pg'],
            original_paths=len(r['paths']), oracle_pool_rmse_m=rmse(r['oracle_pool_errors_m']),
            cold_rmse_m=rmse(c['errors_m']), cold_cost=c['cost'], cold_normal=c['normal'],
            cold_pg=c['pg'], cold_paths=c['paths'], cold_seconds=c['seconds'])
        for arm, name in [('RESTART', 'restart'), ('ORACLE_POOL_START', 'warm_oracle')]:
            a = r['arms'][arm]
            v.update({name+'_rmse_m': rmse(a['errors_m']), name+'_cost': a['cost'],
                name+'_pg': a['pg'], name+'_stationary': a['stationary'], name+'_seconds': a['seconds']})
        controls.append(v)
    for name, rows in [('共同场景指标.csv', metrics), ('固定支持与重建支持控制.csv', controls)]:
        with (HERE/name).open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = dict(anchor_complete=sum(r['errors_m'] is not None for r in anchor), anchor_count=len(anchor),
        anchor_old_terminal_reproduced=all(r['old_terminal_reproduced'] for r in anchor),
        maximum_kernel_error=max(r['kernel_error'] for r in anchor), metrics=metrics, controls=controls,
        cold_normal=sum(r['normal'] for r in cold), cold_rmse=rmse([r['errors_m'] for r in cold]),
        restart_stationary=sum(r['arms']['RESTART']['stationary'] for r in support),
        warm_oracle_stationary=sum(r['arms']['ORACLE_POOL_START']['stationary'] for r in support),
        postflight=dict(source_checks=True, original_and_copy_inputs=checks, truth_bank_receipts=True),
        scope='descriptive_saved_development_diagnostics_not_independent_algorithm_validation',
        statistical_notes=['six_layouts_one_realization_each', 'paired_conditions_not_independent',
            'no_pvalue_or_population_claim', 'missing_outputs_not_imputed', 'oracle_arms_excluded_from_method_ranking'])
    cfg.write(HERE/'summary.json', summary)
    build_report(summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('metrics', 'controls', 'postflight')}, ensure_ascii=False))
    print(json.dumps(controls, ensure_ascii=False))


def build_report(s):
    lines = ['# 文献后核验：问题不只是多径强，也不只是没有收敛', '',
        '2026-11-07。本轮沿批准范围连续完成比较合同、原尺度锚定、共同场景复测、固定支持控制和重建支持控制。没有更改现用算法，没有扩大正式样本，没有用真值挑选正式算法输出。', '',
        '## Material Passport', '', '- Origin Skill: academic-research-suite / experiment-agent',
        '- Origin Mode: run', '- Origin Date: 2026-11-07',
        '- Verification Status: VERIFIED（本轮数值与身份检查；非原文性能复现或算法贡献认证）',
        '- Version Label: post_literature_v1', '',
        '## 结论先行', '',
        '继续研究双源多径定位有依据，但现有完整盲流程不能冻结。候选池包含有用位置，并不意味着当前关联能选对；错误分支的路径支持也不能通过简单移动位置可靠复用。另一方面，确有单次局部求解未充分的问题，但多迭代不是全部解法。', '',
        '本轮没有获得新的可部署算法或理论创新。得到的是更明确的设计要求：两个位置及各自多径解释应共同评判；局部驻定、模型完整停止、真实定位精度必须分开。后续不把“多重启一次”或“增加候选数”直接写成新贡献。', '',
        '## 1. 本轮保持了什么', '',
        '主场景始终为四单天线站、二维同频双未知源全内；2.4GHz、50MHz、12.5MHz、8192点、双源平均直达带内SNR6dB，物理功率与自由空间距离衰减均未改。公共处理为原FFT有效频带，无新Hann处理。', '',
        '当前主研究范围是0/1/2反射。本轮复用六个全内构型R00的L0、L1_P075、L2_P075共18条已见观测，反射总功率比为0/.75；未声称覆盖全部已批准强度。没有读取R01—R09来设计方法，没有启动旧780/3120批次。五站原尺度仅用于Xia原文锚定。', '',
        '## 2. 完整流程共同观测结果', '',
        '每个条件6条。全样本双源RMSE包含所有完整有限输出，包括未正常停止的结果；正常停止数另列，绝不通过删除失败尾部降低RMSE。Xia缺失输出时全样本RMSE不可计算，条件RMSE只列于CSV。GEO、LOS是MP前端保存状态，不是独立完整多径方法。', '',
        '| 条件 | 方法 | 完整输出 | 正常停止 | 全样本RMSE/m | 平均耗时/s | 最大耗时/s |',
        '|---|---|---:|---:|---:|---:|---:|']
    for m in s['metrics']:
        value = '不可计算' if m['all_case_rmse_m'] is None else f"{m['all_case_rmse_m']:.3f}"
        lines.append(f"|{m['condition']}|{m['method']}|{m['returned']}/6|{m['normal']}/6|{value}|{m['mean_seconds']:.3f}|{m['maximum_seconds']:.3f}|")
    lines += ['', 'MP当前单反射平均3.38s、双反射平均5.80s，并没有全面达成3—5秒目标。双反射RMSE主要受IN102大误差影响，不能只看总体均值；逐例见控制表。时间在同机、两BLAS线程、方法轮换顺序下实测，仍不代表跨机器复杂度保证。', '',
        '## 3. Xia锚定没有证明作者方法无效', '',
        f"原尺度24条的严格终态完整输出为{s['anchor_complete']}/24，与旧记录逐项一致。加速核与直接有限和最大相对差{s['maximum_kernel_error']:.3g}。没有靠改变成形、相位、SNR定义、阈值或粒子数追作者曲线。", '',
        '包括12条无反射控制也未完整输出，因此不能把原尺度失败归因于我们反射功率更高。明确仍有公开补全约定（接收SNR、随机相位、矩形BPSK等）；这些算式一致性只验证本重建实现，不证明等同作者代码。当前不能宣布作者曲线复现成功，也不能用0/24当作本方法优越性证据。', '',
        '共同场景Xia采用预先登记的宽带覆盖规则，未削减粒子数；输出缺失仍在。Flood列是零熵LP＋已知K=2适配，不冒称原文熵算法；完整近邻原流程与同前端工具比较的界限不变。', '',
        '## 4. 固定支持控制：求得更充分，不一定定位更准', '',
        '12条多径观测全部做了RESTART和ORACLE_POOL_START两组，共24次连续求解。RESTART不改变路径列表和初值，只再调用一次原求解器；ORACLE_POOL_START用真值诊断选池中近真位置，仍继承原路径列表、增益和时延。后者绝不是盲定位成绩。', '',
        f"重启后{s['restart_stationary']}/12达到原PG，近真起点固定支持组{s['warm_oracle_stationary']}/12达到原PG。PG通过只表示固定模型的局部数值条件。", '',
        '| 构型/条件 | 原MP双源RMSE/m | 同支持重启/m | 近真起点＋旧支持/m | 近真起点＋重新增径/m | 后者正常 |',
        '|---|---:|---:|---:|---:|---|']
    for r in s['controls']:
        lines.append(f"|{r['tag']}|{r['original_rmse_m']:.3f}|{r['restart_rmse_m']:.3f}|{r['warm_oracle_rmse_m']:.3f}|{r['cold_rmse_m']:.3f}|{'是' if r['cold_normal'] else '否'}|")
    lines += ['',
        'IN100单反射从59.350m变为60.227m，但目标0.066020→0.054963且达到PG：这是局部拟合改善却位置未改善的直接实例。IN101单反射重启从2.214m到0.437m，说明数值未充分也确实存在，不能一概归于关联。', '',
        'IN105单反射近真位置继承旧支持时目标0.067094，比错误分支的0.046671更高；重新增径后目标0.045472、误差0.326m。因固定支持组不一定包含适合新位置的解释，不能从前者直接推导“拟合目标本质上偏爱错误位置”。重建后维数不同，后一个目标差也不能当作同复杂度选模证明。', '',
        '## 5. 重建支持：有可利用信息，但盲选择仍未解决', '',
        f"12条从诊断近真候选对重新开始原LOS与未知阶数增径流程，{s['cold_normal']}/12正常停止，合并双源RMSE为{s['cold_rmse']:.3f}m（包含未正常停止的那条）。", '',
        '最明显的是IN102双反射：原盲流程1931.160m，诊断分支重新建模0.187m。此例说明不能把失败全部归因于四站或场景没有定位信息；但真值帮助选择了分支，尚不知道如何在线得到同样结果。', '',
        'IN101单反射仍未正常停止。正确起点不是后端可靠性的充分保证；3—5秒也尚未全面满足。诊断只运行一个指定候选对，不能拿它与原两分支完整流程的时间直接声称加速。', '',
        '## 6. 文献后的贡献与下一步边界', '',
        '| 文献线索 | 已有内容 | 本轮允许得出的启示 |',
        '|---|---|---|',
        '| Xia N02 | 未知同频多源多径的直接空间谱与聚类 | 场景不是空白；保留完整流程和复现局限 |',
        '| Flood N08 / 环一致性N10 | 无标签TDOA关联、几何一致性筛选 | 几何一致不等于真实源身份；不能把普通三站候选作为独创 |',
        '| Friedlander N01 / EM与SAGE N37—38 | 未知波形消元、分量更新与似然优化 | 后端组件可借用；通用优化器不是新增理论 |',
        '| OFDM-SAGE N58及未知阶数系列 | 区分模型内收敛、初始化显著性和外层停止 | 不能把求解没完成留下的残差自动解释为缺路径 |',
        '| N69及盲解混系列 | 通道秩、结构约束和条件辨识 | 其单输入/公共时延或已知子空间条件不满足当前模型，不能直接套定理 |', '',
        '建议的核心对象是“观测驱动的双源—路径联合关联与位置解释”，现用MP保留为条件精修和高成本参照。它不是马上再加一个启发式筛选器：先定义可观测判别量、其对未知波形/增益歧义的不变性及适用条件，再用有限配对否证。尚未获得这样的新规则，不能宣称已经有SP贡献。', '',
        '本轮停止自动推进：完整盲流程门未通过，Xia作者性能锚定未通过；因此不扩大独立实验，不自动放行原780/3120，不调整场景或先验。需要用户明确下一方法定位：是否接受将现有递增MP流程降为精修/参照，正式转入联合源—路径关联架构研究，而不是继续以保留现有主算法为前提修补。新架构仍遵守原场景和信息条件。', '',
        '## 7. 可复核材料与范围', '',
        '执行合同.md、固定支持控制.md、重建支持控制.md均在对应运行前登记；summary.json与两张CSV由保存结果自动生成。运行数据下anchor24/common18/support12/cold12保留决策、输入副本与身份记录。所有本轮源码与输入前后检查通过；原观测与真值银行匹配旧收据。', '',
        '这是已见开发观测的描述性配对诊断，不是独立泛化验证；每构型一实现，不作显著性检验、不宣称人口层面的成功率，不把相邻条件当独立样本。负面结果和缺失输出全部保留。未修改现用算法、原数据、文献分级或Git提交。']
    (HERE/'本轮结论.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    # Small standalone HTML renderer for the saved report, no external dependencies.
    parts = []
    in_table = False
    for line in lines:
        if line.startswith('|'):
            if line.startswith('|---'):
                continue
            if not in_table:
                parts.append('<div class="table"><table>')
                in_table = True
                cell = 'th'
            else:
                cell = 'td'
            parts.append('<tr>'+''.join(f'<{cell}>{html.escape(x.strip())}</{cell}>' for x in line.strip('|').split('|'))+'</tr>')
            continue
        if in_table:
            parts.append('</table></div>')
            in_table = False
        if line.startswith('# '):
            parts.append('<h1>'+html.escape(line[2:])+'</h1>')
        elif line.startswith('## '):
            parts.append('<h2>'+html.escape(line[3:])+'</h2>')
        elif line:
            parts.append('<p>'+html.escape(line)+'</p>')
    if in_table:
        parts.append('</table></div>')
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>文献后机制核验</title><style>body{background:#f5f6f3;color:#173345;font:16px/1.8 system-ui,"Microsoft YaHei",sans-serif;margin:0}main{max-width:1120px;margin:auto;padding:36px 26px}h1{font-size:30px;line-height:1.4}h2{margin-top:40px;border-left:5px solid #278777;padding-left:12px}p{max-width:1000px}.table{overflow:auto;background:white;border:1px solid #dce3e5;border-radius:10px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px 12px;text-align:left;border-bottom:1px solid #e5ebed}th{background:#173f50;color:white}tr:nth-child(even){background:#f3f8f7}a{color:#147c71}</style><main>'+''.join(parts)+'<p><a href="本轮结论.md">Markdown</a> · <a href="共同场景指标.csv">完整指标CSV</a> · <a href="固定支持与重建支持控制.csv">逐例控制CSV</a> · <a href="summary.json">汇总JSON</a></p></main></html>'
    (HERE/'文献后机制核验报告.html').write_text(page, encoding='utf-8')


if __name__ == '__main__':
    main()
