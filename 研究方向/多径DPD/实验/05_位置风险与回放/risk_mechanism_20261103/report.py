"""Render exact saved-data diagnostic, with no selector or solver execution."""
import json
from pathlib import Path

import audit


HERE = Path(__file__).resolve().parent


def main():
    out = Path(json.loads((HERE/'latest.json').read_text(encoding='utf-8'))['output'])
    result = json.loads((out/'analysis.json').read_text(encoding='utf-8'))
    identity = json.loads((out/'identity.json').read_text(encoding='utf-8'))
    assert identity['script_sha256'] == audit.sha(HERE/'audit.py')
    assert identity['frozen_pre'] == identity['frozen_post']
    g = result['groups']
    lines = ['# 冻结风险评分的损失差分解', '',
             '日期：2026-11-03。类型：已有保存结果离线评价，不是新算法试验。', '',
             '## 结论', '',
             '风险选择的损失差额已经精确分解，但当前8次条件重复不足以在因果上分开参照偏差、数据驱动支持选择和局部敏感性近似。没有新增充分依据可直接设计修正系数或置信门槛。', '',
             'L2中插件交叉项与经验交叉项异号；L3中两者同号但幅度不准。L3的经验均值交叉项很小，差额仍存在，因此不能把问题统一归结为“完整参照平均位置偏了”，也不支持现在只减去一个经验参照偏差。', '',
             '下一阶段仍应优先完成同信息近邻；风险路线保留，但不自动开始重新标定、更多重复或新阈值。', '',
             '## 实际执行与证据身份', '',
             f'- 输出：`{out.name}`。',
             f"- {result['coverage']['cases']}条、{result['coverage']['states']}状态、{result['coverage']['applicable_states']}可评分状态；2条回退不删除。",
             f"- {result['checks']}项逐文件/恒等式/汇总检查；耗时{result['seconds']:.3f}s；新定位次数0。",
             f"- 冻结身份{len(identity['frozen_pre'])}项前后相同，实际读入{len(identity['read_artifacts'])}个文件前后SHA相同。",
             '- 真位置只用于本次评价；选择及每状态评分读取原冻结决定，并核对独立state与record完全一致。',
             f"- 所有状态使用完整终态确定的同一个源标签排列；与逐状态最优排列评价不一致记录{len(result['source_label_warnings'])}项。",
             '- 修复被中断脚本的缺省字段读取：不适用状态没有risk键，改为get并保留null，未补造评分；本轮第一版有效审计目录也保留，不覆盖。', '',
             '## 精确恒等式与单位', '',
             '两源位置按(x1,y1,x2,y2)组成向量。d=p_selected−p_W，e_W=p_W−p_true；损失为两源平均平方位置误差，单位m²。', '',
             '`Delta = ||d||²/2 + dᵀe_W`。', '',
             '在同条件n次重复中，严格有', '',
             '`mean(Delta)=mean(||d||²)/2 + mean(d)ᵀmean(e_W) + trace(Cov_n(d,e_W))`。', '',
             '这里Cov_n使用除以n的经验协方差。sample-mean项不是总体bias贡献的无偏估计，经验协方差也不是独立已知的真实方差。', '',
             '当前评分与实际损失差都具有相同的位移项；差额全部可以数值归入交叉项的估计差额。但“归入代数项”不等于识别物理或统计根因。', '',
             '## 同条件分解（全组口径，含回退）', '',
             '| 条件 | n/回退 | 位移项 | 经验均值交叉项 | 经验协方差迹 | 实际损失差 |',
             '|---|---:|---:|---:|---:|---:|']
    for c in ['L0', 'L2', 'L3', 'HIGH', 'HIGH_eligible6_diagnostic']:
        v = g[c]
        lines.append(f"| {c} | {v['n']}/{v['fallback']} | {v['displacement']:.8g} | {v['sample_mean_cross']:.8g} | {v['empirical_covariance_n_trace']:.8g} | {v['empirical_loss_difference']:.8g} |")
    lines += ['', 'HIGH全8条含2条严重失败，其均值项与协方差项数值很大且相互抵消；不能把这一分解读作正常局部估计的偏差机制。HIGH可评分6条仅是条件诊断，不替代全8条成绩。混合条件ALL32只存JSON作账本，不作机制推断。', '',
              '## 评分差与实际差（严格相同可评分子集）', '',
              '负数表示相对完整终态改善。预测误差=实际差−估计差。', '',
              '| 条件 | 可评分n | 估计损失差 | 实际损失差 | 插件交叉项 | 实际交叉项 |',
              '|---|---:|---:|---:|---:|---:|']
    for c in ['L0', 'L2', 'L3', 'HIGH_eligible6_diagnostic']:
        v = g[c]
        lines.append(f"| {c} | {v['scored_n']} | {v['mean_predicted_difference']:.8g} | {v['scored_mean_empirical_difference']:.8g} | {v['mean_plugin_cross']:.8g} | {v['mean_realized_cross']:.8g} |")
    lines += ['', '## 当前评分分辨能力的描述性检查', '',
              '| 条件 | 平均估计改善 | 对实际损失差的预测误差RMS | 误差绝对值大于估计差绝对值的条数 |',
              '|---|---:|---:|---:|']
    for c in ['L0', 'L2', 'L3', 'HIGH_eligible6_diagnostic']:
        v = g[c]
        lines.append(f"| {c} | {v['mean_predicted_improvement']:.8g} | {v['prediction_error_rms']:.8g} | {v['absolute_error_exceeds_absolute_risk_gap']}/{v['scored_n']} |")
    lines += ['', '这里RMS是在同一批已选模型上、对实现损失差计算的误差，不是风险估计器的标准误或已校准置信区间；实现噪声与期望风险不同，因此不能单凭此表证明风险估计有偏。它说明小幅预测改善缺少足够可靠的实际排序证据，不允许把本表换算成新的经验门槛。', '',
              '## 固定增径序号不等于固定支持', '',
              '| 条件 | 八条共同拥有的最大K | 站—源路径数分配出现不同模式的K个数 |',
              '|---|---:|---:|']
    for c, table in result['common_order_descriptive'].items():
        lines.append(f"| {c} | {max(v['order'] for v in table)} | {sum(v['unique_station_source_count_patterns'] > 1 for v in table)} |")
    lines += ['', '即使每站每源的数量相同，各条路径时延及波形估计仍由同一噪声观测选择。此表不构成固定支持、独立选择或交叉拟合消融。所有K表原样保存在JSON，不选取最有利的K重新发布算法成绩。', '',
              '## 11项解释风险核查', '',
              '1. 事后最优是保存候选内经验下包络，不是CRLB、理论极限或运行时可用答案。',
              '2. 8次条件重复的平均误差不是总体偏差真值，其平方也不是平方偏差的无偏估计。',
              '3. K固定不等于路径支持固定；同站源数量分配相同也不等于时延支持相同。',
              '4. 对实现损失预测不准，不单独证明对期望风险有偏；本轮不提供已校准置信结论。',
              '5. 相同数据生成与选择的依赖可能重要，但本轮未做独立选择反事实，不能宣称已确认选择偏差为根因。',
              '6. 差额定位到交叉项是精确代数分解，不等于参照偏差、选择效应、局部近似三者的因果分离。',
              '7. 两条HIGH失败仍在全组口径；6条诊断不得替换总体或声称全部处理成功。',
              '8. 插件均值与实际均值必须使用同一可评分集合；全32实际差与可评分30估计差不直接配对。',
              '9. 当前已使用实际非零残差局部曲率和模型相关项，不能把补Hessian/补相关性当作新修订。',
              '10. 真值仅在离线诊断使用，不反馈评分；不拟合经验系数，不按32条最优结果挑阈值。',
              '11. 已做完整定位再选较少分量并不节省定位成本；本次秒级离线分解不代表实际风险选模秒级。', '',
              '## 阶段结论与最小下一决策', '',
              '有限机制核验已完成；没有获得足以唯一决定下一种风险修订的证据。维持当前评分冻结、停止扩大该评分正式批。若要再次启动，需先给出明确处理数据驱动选择或参照误差的理论规则，以及不依赖本32条真值拟合的可否证短测；本协议不自动批准开发它。先推进已经批准的同信息强近邻比较。', '',
              '## Material Passport', '',
              '- Origin Skill: scientific-toolkit-skill；承接已批准academic-research-suite执行协议。',
              '- Origin Mode: saved-result audit / exact algebraic decomposition。',
              '- Origin Date: 2026-11-03。',
              '- Verification Status: ANALYZED。',
              '- Version Label: frozen-risk-mechanism-v1。',
              '- Provenance: analysis.json、identity.json；5项单元测试仅证明计算恒等式及fallback分母处理。',
              '- Uncertainty: 无新独立数据、无新定位、无评分修改；理论新颖性与因果根因未确认。', '']
    (out/'机制分解报告.md').write_text('\n'.join(lines), encoding='utf-8')
    print(out/'机制分解报告.md')


if __name__ == '__main__':
    main()
