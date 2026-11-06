"""Read-only replay and reporting of bounded path-structure diagnostics."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
GENERAL = ROOT/'run_20261025_151228_347508'
VERTICAL = ROOT/'vertical_20261025_151450_154591'
PRIOR = ROOT.parent/'step1_deployment_20261025/run_20261025_142854_848860'


def main():
    general = json.loads((GENERAL/'result.json').read_text(encoding='utf-8'))
    vertical = json.loads((VERTICAL/'result.json').read_text(encoding='utf-8'))
    witness = json.loads((PRIOR/'result.json').read_text(encoding='utf-8'))['witness']
    # The addendum was appended after the first run. Recover, label and hash-check
    # its exact earlier prefix; do not pretend this copy was saved before that run.
    contract = (ROOT/'执行合同.md').read_text(encoding='utf-8')
    prefix = contract.split('\n\n## 共同垂直墙的追加诊断')[0]+'\n'
    contract_key = str((ROOT/'执行合同.md').resolve())
    expected = general['identity'][contract_key]
    recovered_ok = hashlib.sha256(prefix.encode('utf-8')).hexdigest() == expected
    if not recovered_ok:
        raise RuntimeError('Cannot recover original contract identity')
    (GENERAL/'原合同回溯副本.md').write_text(prefix, encoding='utf-8')
    frozen_ok = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
                    for p,h in general['identity'].items() if p != contract_key)
    current_ok = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p,h in vertical['identity'].items())
    s = np.array(witness['stations_m'])
    evidence = {}
    max_errors = []
    for name in ['original', 'alternate']:
        src = np.array([witness[name+'_position_m'], witness['unchanged_second_source_m']])
        missing = next(i for i in range(5) if i not in witness[name+'_los_support'])
        receivers = s[[missing, 4]]
        unique = []
        for c in vertical['results'][name]['candidates']:
            n, b = np.array(c['plane_normal']), c['plane_offset_m']
            if not np.all((src@n-b)*(receivers@n-b) > 0):
                continue
            if not all(r['path_error_m'] < 1e-6 and r['reflection_error'] < 1e-8
                       and 0 < r['line_fraction'] < 1 for r in c['rays']):
                continue
            if any(min(np.linalg.norm(n-n0)+abs(b-b0), np.linalg.norm(n+n0)+abs(b+b0))
                   < 1e-4 for n0,b0,_ in unique):
                continue
            # Explicitly test all declared LOS line segments. An infinite opaque
            # plane cannot intersect these segments; no finite wall is assumed.
            crossing = []
            for q in range(2):
                los = witness[name+'_los_support'] if q == 0 else [0, 1, 2, 3]
                for station in los:
                    if (src[q]@n-b)*(s[station]@n-b) <= 0:
                        crossing.append([q, station])
            unique.append((n,b,{'normal': n.tolist(), 'offset_m': b,
                                'crossed_los_links': crossing, 'rays': c['rays']}))
            max_errors.extend(r['path_error_m'] for r in c['rays'])
        evidence[name] = {'pairwise_reflection_planes': len(unique),
                          'no_los_crossing_planes': sum(not c[2]['crossed_los_links'] for c in unique),
                          'planes': [c[2] for c in unique]}
    # Independent image-distance check of all reported valid general-plane trials.
    general_error = 0.
    with np.load(PRIOR/'iq_witness.npz') as iq:
        for name in ['original','alternate']:
            src = np.array([witness[name+'_position_m'], witness['unchanged_second_source_m']])
            absent = next(i for i in range(5) if i not in witness[name+'_los_support'])
            receivers = s[[absent,4]]
            lengths = iq[name+'_lengths_m'][[0,1],[absent,4]]
            for trial in general['common_plane_delay_fits'][name]['trials']:
                if not trial['valid']:
                    continue
                n,b = np.array(trial['plane_normal']),trial['plane_offset_m']
                image = src-2*(src@n-b)[:,None]*n
                general_error = max(general_error,float(np.max(abs(np.linalg.norm(image-receivers,axis=1)-lengths))))
    result = {'general_contract_recovered_after_addendum': recovered_ok,
              'unchanged_scientific_inputs': frozen_ok and current_ok,
              'vertical_distinct_planes': evidence, 'general_image_error_m': general_error,
              'vertical_max_path_error_m': max(max_errors),
              'pass': bool(recovered_ok and frozen_ok and current_ok and general_error < 1e-6
                           and general['all_checks_pass'] and vertical['all_checks_pass'])}
    (ROOT/'独立核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if not result['pass']:
        raise RuntimeError('Evidence audit failed')
    out = general['changed_support_candidates_in_box']
    excess = [x['nlos_excess_m'] for x in out]
    rows = []
    for mode,label in [('flat','同高'),('slope','斜平面'),('roof','屋顶高差')]:
        a = [c for c in out if c['mode']==mode]
        rows.append(f"| {label} | {len(set(x['case'] for x in a))} | {len(a)} | {min(x['nlos_excess_m'] for x in a):.4f}—{max(x['nlos_excess_m'] for x in a):.4f}m |")
    text = f'''# 哪些物理约束真的能排除假位置？

2026-10-25｜小规模机制核验完成。没有运行定位性能批次，没有冻结新增先验。

## 一句话结论

**最值得进一步检查的是“反射解释与直达可见性是否同时符合一个环境”，而不是把最强、最早或最短绕行当成直达标签。** 本例中，仅共享平面、甚至仅共享垂直墙，都不足以排除假位置；要求同一不透明无限墙面不能截断已有直达线段，才排除了该替代解释。

这说明可见性结构可能提供真实信息，但尚未证明有限建筑立面下也能消歧，更没有确认新的算法贡献。

## 1 实际完成的三项检查

| 检查 | 结果 | 能说明什么 |
|---|---|---|
| 假位置是否仅来自5.7cm绕行 | 保存结果中还有米级替代绕行 | 不能只删掉那个厘米级案例就认定问题消失 |
| 两个源的反射共享一个未知平面 | 真、假解释均找到合法同侧平面 | “不独立拟合每条反射”还不一定足够 |
| 共享垂直平面，再核对全部直达线段 | 仅反射长度：真、假各4个不同平面；不穿越无限不透明墙：真剩1个、假剩0个 | 排除来自新增可见性约束，不是“垂直”两个字，也不是优化器更好 |

保留每源4LOS、总5站、未知波形与路径常复增益。另一源位置与两份已保存的IQ解释都不改。对照中允许真假解释各估计一个环境，因为接收端没有地图；没有把真实墙面直接给算法。

## 2 为什么最强、最早不能直接解决

**最早：**直达存在且可分辨时，它是该源到该站最短的几何路径。但直达缺失时，最先收到的也只是反射/绕射；多个未知源同时出现时，直接比较混合信号的峰时刻更不能给出逐源LOS标签。上轮反例每个源—站只保留一条到达，所以“选第一条”根本没有新增信息。[NJIT研究组的说明](https://research.njit.edu/cwcspr/high-precision-localization-rf-sources-multipath-channels)也明确指出直达遮挡和旁瓣混淆会使首到达判断出错。

**最强：**接收功率同时受距离、遮挡、天线和反射影响。不能保证不同站之间所有直达都强于所有反射。[NIST的室内定位实测项目总结](https://www.nist.gov/ctl/smart-connected-systems-division/indoor-localization)说明多径妨碍直达隔离；这是定性旁证，不是低空场景的定量功率先验。

**已知自由空间相对幅度：**能够增加约束，但当前新方向允许路径系数未知。旧反例也不是按严格1/d生成的，不能事后对它施加1/d后声称物理模型已验证。是否恢复已标定幅度模型是研究范围选择，本轮没有自动变更。

## 3 短绕行究竟有多重要

复用上轮540条条件，全量保留工作域内不同支持的合法替代根：

| 部署 | 涉及的条件数 | 替代根数 | 替代NLOS超额长度范围 |
|---|---:|---:|---|
{chr(10).join(rows)}

合计{len(out)}个根、{len(set(x['case'] for x in out))}条条件，范围{min(excess):.4f}—{max(excess):.4f}m。它们不是独立随机样本，不估计出现概率。

几何反射可以连续接近直线长度，因此“反射至少绕行若干米”不是一般物理定律。具体墙体、站点安全距离与可见性可能给出下界，但需要先给环境依据。本轮没有选择一个恰好过滤这些根的阈值。

## 4 共同墙面核验：把收益来源拆清楚

对源p、接收站r和平面nᵀx=b（n为单位法向量），镜像法给出：

`L² = ||p−r||² + 4(nᵀp−b)(nᵀr−b)。`

两条反射共用n、b，就不再各自任意调延迟。一般平面仍有三个几何自由度，本例两条长度约束可同时拟合真假解释；数值求解32个预定初值中，真解释{general['common_plane_delay_fits']['original']['valid_starts']}次、假解释{general['common_plane_delay_fits']['alternate']['valid_starts']}次得到满足条件的平面。这不是定位成功率，也不是自由度计数代替证明，明确平面证据都已保存。

再限制n的竖直分量为0，用半角代换求多项式根，处理分母零和角度极点，并回代原长度与反射定律。去掉重复法向表示后：

| 条件逐步增加 | 原位置解释 | 约75m外替代解释 |
|---|---:|---:|
| 两条反射共享垂直平面，各自满足镜面条件 | {evidence['original']['pairwise_reflection_planes']}个不同平面 | {evidence['alternate']['pairwise_reflection_planes']}个不同平面 |
| 再要求已有LOS线段均不穿过该无限不透明面 | {evidence['original']['no_los_crossing_planes']}个 | {evidence['alternate']['no_los_crossing_planes']}个 |

**关键限制：这是“无限不透明面”的可见性检查。** 有限墙面的边缘允许直达线段从墙外通过；它还可能产生其他可见反射，需要一并生成。原来缺LOS的两条链路也仍需有遮挡原因。当前结果没有验证这些完整环境条件，不能直接宣布“共同垂直墙已解决真实场景”。

本轮只排除了这一个固定替代位置，没有证明真实位置是结构化模型的全部解，也没有运行新的全域盲定位。

## 5 论文依据给出的约束

- Briso等，[TVT 2020](https://doi.org/10.1109/TVT.2020.2968136)，PDF第2—4页、§II—IV：低空遮挡与高度、具体建筑和工作频率有关；该试验的低频遮挡段以绕射为主，高频多径更显著。它支持物理环境生成的必要性，不能据此把所有低空NLOS都设成一条镜面反射。原文为连续波接收功率测量，不提供本任务逐条路径的幅度排序或最小延迟阈值。
- Garcia等，[2015预印本](https://arxiv.org/abs/1505.03193)，PDF第6页Assumptions 2、3和Theorem 1：保证需要路径数量与几何唯一性条件；不能用“已发表方法有保证”省略本模型检查。
- Wang等，[TWC正式版本](https://doi.org/10.1109/TWC.2025.3601543)，PDF第3页、式(1)：阵列与已知导频反卷积后的CSI提供了当前盲单天线任务没有的信息。不免费借入其方向信息或干净信道。
- 多径虚拟点/共同反射几何已有丰富研究。此次公式和镜像检查不是创新；潜在问题只能是明确场景下仍存在的关联困难，以及有别于常规共同几何拟合的必要处理。

## 6 下一步建议：先闭合生成模型，不再追加经验门槛

目前不宜直接开定位性能批次，也不需要再增加一串候选算法。若继续，最小有意义的工作是**用一个有限主立面与明确遮挡规则生成完整可见路径，再判断可辨识范围**：

1. 生成端知道环境，估计端仍无地图、无波形、无真LOS标签；只声明真实可用的结构假设。
2. 同一环境同时决定直达是否被挡、反射点是否落在墙面内、其他反射是否可见，不再独立指定支持和延迟。
3. 不要求所有构型都唯一；但多解条件应被识别并记录，不能删掉难例后宣称全局定位可靠。
4. 在这一模型能成立后，再测试普通关联/联合拟合是否留下值得研究的困难。

“单主立面”“相应遮挡机制”和是否纳入绕射尚不是用户冻结的正式场景。该建议涉及实质建模收敛，因此本轮到此停止，没有自动引入建筑地图、墙面真值或新的主算法。

<details><summary>复现、原文阅读范围与验证记录</summary>

- [执行合同与追加诊断登记](执行合同.md)、[一般平面源码](check_constraints.py)、[垂直墙枚举源码](check_vertical.py)、[独立汇总源码](summarize.py)。
- [一般平面原始结果]({GENERAL.name}/result.json)、[垂直墙原始结果]({VERTICAL.name}/result.json)、[独立审计](独立核验.json)。原始枚举计数含正负法向和极点重复，报告中的不同平面数量已去重。
- 原IQ和结果输入SHA保持不变；一般平面阶段之后合同追加了垂直墙条款。先前合同前缀已回溯恢复并与当时SHA核对一致，副本明确不是事前另存：[原合同回溯副本]({GENERAL.name}/原合同回溯副本.md)。
- 已知垂直平面x=-900m的正对照通过。一般平面有效解镜像长度独立误差最大{general_error:.3e}m；垂直反射回代误差最大{max(max_errors):.3e}m。
- 一般平面脚本内部{general['wall_seconds']:.3f}s，垂直枚举{vertical['wall_seconds']:.3f}s。不是DPD定位耗时；没有训练、性能统计或参数调优。
- 本轮重读既有PDF相关页；公开查询只补充NIST/NJIT及候选论文元数据。IEEE文档10509551网页受验证页阻挡，没有把检索摘要写成全文结论，也不因此宣称查重新颖性已完成。

</details>
'''
    (ROOT/'物理约束核验报告.md').write_text(text,encoding='utf-8')
    print(json.dumps({'audit_pass':result['pass'],'distinct_planes':{k:(v['pairwise_reflection_planes'],v['no_los_crossing_planes']) for k,v in evidence.items()}},ensure_ascii=False))


if __name__ == '__main__':
    main()
