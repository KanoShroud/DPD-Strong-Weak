"""Replay the saved four-LOS IQ witness and write an evidence-grounded report."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RUN = ROOT/'run_20261025_140742_623340'


def main():
    result = json.loads((RUN/'result.json').read_text(encoding='utf-8'))
    d = np.load(RUN/'iq_witness.npz')
    signals = []
    for coeff, length in [(d['coefficient'], d['original_lengths_m']), (d['shifted_coefficient'], d['alternate_lengths_m'])]:
        H = np.zeros((5, len(d['frequency_hz'])), complex)
        for q in range(2):
            H += d['gain'][q, :, None]*coeff[q, None, :]*np.exp(-2j*np.pi*length[q, :, None]/299792458.0*d['frequency_hz'][None, :])
        signals.append(H@np.exp(2j*np.pi*d['frequency_hz'][:, None]*d['t_s'][None, :]))
    errors = [float(np.linalg.norm(signals[i]-d[k])/np.linalg.norm(d[k])) for i, k in enumerate(['original', 'alternate'])]
    error = float(np.linalg.norm(signals[0]-signals[1])/np.linalg.norm(signals[0]))
    identity = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in result['identity']['inputs_sha256'].items())
    check = {'original_replay_error': errors[0], 'alternate_replay_error': errors[1],
             'independent_iq_pair_difference': error, 'input_identity_unchanged': identity,
             'pass': max(*errors, error) < 1e-10 and identity}
    (RUN/'independent_replay.json').write_text(json.dumps(check, indent=2), encoding='utf-8')
    if not check['pass']:
        raise RuntimeError('Independent replay failed')
    w = result['iq_witness']
    p = w['original_sources_m'][w['source']]
    a = w['alternate_sources_m'][w['source']]
    rows = []
    for key, label in [('domain_20_120', '20—120m'), ('domain_20_150', '20—150m'), ('domain_0_300', '0—300m')]:
        s = result['summary'][key]
        rows.append(f'| {label} | {s["known_support_with_alternative"]}/6 | {s["unknown_support_with_alternative"]}/6 | {s["with_changed_support_alternative"]}/6 |')
    table = '\n'.join(rows)
    text = f'''# 四条直达之后：连续歧义消失，离散双解仍需处理

2026-10-25｜5站、每源至少4LOS的模型续核验。未运行定位性能批次。

## 结论

**增加到每源4条直达，解决了上一轮的连续欠约束，但没有普遍解决全局唯一性。** 在沿用的6个源—构型组合中，有一例原位置和替代位置都落在20—120m高度范围内，位置相距{w['changed_source_separation_m']:.2f}m，5站双源混合IQ仍完全等价至数值精度。

本轮没有发现“换一组直达站”在所查高度域内额外增加假解。当前明确问题首先是四LOS几何本身的离散双解，不宜直接归因为路径关联算法差。

## 1 实际做了什么

- 总接收站保持5个，三个构型和每构型两个源沿用上一轮，不根据本轮结果选点。
- 每源4站直达、余下一站只有一条弱反射，反射幅度为直达0.2。无地图、未知波形、未知常复增益、未知反射延迟的权限不变。
- 每个源先给定真实直达集合，枚举四站原方程的全部合法实根；再枚举全部5种四站子集，检查未知支持可能产生的额外解。
- 每个候选均回代未平方的距离差方程，并用独立非线性残差再次核验；剩余反射必须满足正超额长度。
- 对固定遍历顺序中第一例域内假解构造完整混合IQ；另一源保持不变。不按误差最大挑案例。
- 五LOS已知支持只作正对照，不把主场景偷偷改成所有站无遮挡。

这些是oracle信息更充分的几何诊断，不是已实现的盲多源分离算法，更不是定位成功率实验。

## 2 一个明确反例

| 解释 | 第二个源位置（m） |
|---|---|
| 原位置 | ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}) |
| 替代位置 | ({a[0]:.4f}, {a[1]:.4f}, {a[2]:.4f}) |

两种解释中，第一个源不变，第二个源的四个直达站也完全相同。通过改变该未知波形的时间原点和剩余未知反射延迟，两套五站混合观测保持一致：

- 源位置距离：**{w['changed_source_separation_m']:.6f}m**。
- 混合IQ相对差：**{w['iq_relative_difference']:.3e}**。
- 保存系数独立重放后的两解释相对差：{error:.3e}。
- 替代解释的NLOS超额路径长度：{w['remaining_nlos_excess_m']:.6f}m，仍为正；逐链路一次镜面反射定律通过。

约1e-14的差异是浮点舍入。相同噪声分布下仍无法由观测区分两套参数，不是增加迭代、调SNR或扩大Top数量能解决的。

这21m是两种合法参数解释的距离，不是算法RMSE，也不应把“1/6”叫作16.7%的总体失败概率。若成功门槛较宽，两个不同位置甚至可能同时落入门槛，因此不能用成功率替代位置可辨识性判断。

## 3 为什么局部满秩还会双解

四个直达站提供3条独立距离差，可以在某个解附近确定三个坐标。但方程是非线性的，在另一个区域可能还有第二个解。

令参考距离为d₀，三个距离差为rᵢ。平方相减后得到关于位置p和d₀的3条线性关系，可写成p=a+b·d₀。再代入“参考距离等于位置到参考站的距离”，会得到一个二次方程，因此可能留下两个合法根。只有回代原始距离方程、传播约束和有依据的工作区域后，才能判断是否排除其中一个。

这一机制是已有TDOA几何结果，本轮是对当前输入模型的验证，不作为新论文贡献。详细方程见[推导说明](推导说明.md)。

## 4 高度边界不能用来挑结果

平面范围均为±600m；三个高度范围只用于预定敏感性检查。

| 高度域 | 已知真实LOS集合仍有替代解 | 枚举未知LOS集合后有替代解 | 其中需要改变LOS集合的组合 |
|---|---|---|---|
{table}

上一轮120.53m的第二解在扩大到150m后进入范围；本轮新见证的两个高度90m与111.12m则都在原20—120m范围内。因此不能通过把上限微调到某个值来“通过门禁”。实际飞行区域若有独立应用依据，可以作为公开先验；不能由算法成绩倒推区域。

其余未见域内假解的组合，只说明本次精确路径时延、有限候选模型下没有同类歧义，不代表未知波形盲分解已被完整证明唯一。

## 5 五LOS对照说明什么

六组位置的五LOS已知支持增广线性系统均满秩，最大位置重建误差为{max(x['position_error_m'] for x in result['five_los_known_support']):.3e}m。这说明实现能够在增加独立直达约束后恢复原位置，结果不是求解代码普遍失效。

但让所有源在所有5站都有直达，会改变当前“部分LOS”问题。因此本轮**不建议直接靠要求五LOS来宣布新路线成立**，也没有把它改成正式主场景。

## 6 当前决定与下一步边界

四LOS版本暂不进入36条混合观测的性能比较；先把可用的几何条件说清楚。

如果继续保持5站、部分LOS、三维和无地图，下一项有针对性的工作应是：**检查具有实际部署依据的站点几何与飞行区域，能否排除主要四LOS双解，同时不人为提供真高度。** 这不是再调搜索参数，也不是看到假解后缩小区域。若需要额外高度信息、共享反射结构或新部署约束，必须明确作为新增条件再决定。

目标并不是要求任何构型、整个空间都必须唯一。可以研究有明确应用依据的可辨识工作域，把歧义域如实列为边界；但不能按定位成绩挑选构型或排除失败。本轮暂停的是尚未写清这些条件的性能排名，不是否定全部四LOS研究。

目前不自动增加第6站，不擅自设定高度真值/地图，不假定反射必须来自某一公共墙面。新规则和方法贡献仍未确认。

## 7 证据与检查

- [执行合同](执行合同.md) · [推导说明](推导说明.md)
- [全部候选CSV]({RUN.name}/all_candidates.csv) · [完整结果]({RUN.name}/result.json)
- [独立IQ重放]({RUN.name}/independent_replay.json) · [源码/输入身份]({RUN.name}/preflight.json)
- [核验源码](check_four_los.py) · [重放与报告源码](replay_report.py)
- 6项数值检查、独立重放、Python语法编译与Ruff F/E9通过。脚本内部计算约{result['wall_seconds']:.3f}秒，不含启动，不是定位耗时。
- 没有改旧定位器、论文、Zotero或覆盖三LOS证据，没有运行正式性能比较。

状态：`FOUR_LOS_GLOBAL_AUDIT_COMPLETED / DISCRETE_AMBIGUITY_WITNESSED / PERFORMANCE_STAGE_HOLD`。
'''
    (ROOT/'四直达核验报告.md').write_text(text, encoding='utf-8')
    print(json.dumps(check))


if __name__ == '__main__':
    main()
