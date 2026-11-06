"""Independent synthesis and documentation of saved deployment audit; no fitting."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RUN = ROOT/'run_20261025_142854_848860'


def main():
    result = json.loads((RUN/'result.json').read_text(encoding='utf-8'))
    hashes = {name: hashlib.sha256((RUN/name).read_bytes()).hexdigest()
              for name in ['result.json', 'iq_witness.npz', 'all_candidates.csv']}
    d = np.load(RUN/'iq_witness.npz')
    signals = []
    for coeff, lengths in [(d['coefficients'], d['original_lengths_m']),
                           (d['alternate_coefficients'], d['alternate_lengths_m'])]:
        transfer = np.zeros((5, len(d['frequency_hz'])), complex)
        for source in range(2):
            transfer += (d['gains'][source, :, None]*coeff[source, None, :]
                         * np.exp(-2j*np.pi*lengths[source, :, None]
                                  * d['frequency_hz'][None, :]/299792458.))
        signals.append(transfer@np.exp(2j*np.pi*d['frequency_hz'][:, None]*d['t_s'][None, :]))
    errors = [float(np.linalg.norm(signals[i]-d[key])/np.linalg.norm(d[key]))
              for i, key in enumerate(['original', 'alternate'])]
    pair = float(np.linalg.norm(signals[0]-signals[1])/np.linalg.norm(signals[0]))
    identity = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
                   for p, h in result['identity']['input_sha256'].items())
    # Independently recompute distance differences and condition-level counts.
    max_residual = 0.
    summary = {}
    for mode, stations in result['mode_stations_m'].items():
        s = np.array(stations)
        cases = [c for c in result['cases'] if c['mode'] == mode]
        summary[mode] = {'conditions': len(cases)}
        for c in cases:
            length = np.linalg.norm(np.array(c['source_m'])-s, axis=1)
            length[c['missing_station']] += c['excess_m']
            for r in c['candidates']:
                ids = r['support']
                ranges = np.linalg.norm(np.array(r['position_m'])-s[ids], axis=1)
                residual = np.max(abs((ranges-ranges[0])-(length[ids]-length[ids[0]])))
                max_residual = max(max_residual, float(residual))
        for domain in ['in_halfspace', 'in_working_box']:
            summary[mode][domain] = {}
            for label, filt in [('known_support_ambiguity', lambda r: r['true_support']),
                                ('any_support_ambiguity', lambda r: True),
                                ('changed_support_ambiguity', lambda r: not r['true_support'])]:
                summary[mode][domain][label] = sum(any(r[domain] and r['positive_excess']
                    and r['separation_m'] > 1e-3 and filt(r) for r in c['candidates']) for c in cases)
    unchanged = all(hashlib.sha256((RUN/name).read_bytes()).hexdigest() == h for name, h in hashes.items())
    audit = {'input_sha256': hashes, 'solver_inputs_unchanged': identity, 'data_unchanged': unchanged,
             'saved_signal_replay_relative_errors': errors, 'independent_pair_difference': pair,
             'max_original_distance_difference_residual_m': max_residual,
             'summary_matches': summary == result['summary'],
             'pass': bool(identity and unchanged and summary == result['summary']
                          and max(*errors, pair) < 1e-10 and max_residual < 1e-6
                          and result['all_checks_pass'])}
    (RUN/'independent_replay.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    if not audit['pass']:
        raise RuntimeError('Saved evidence replay failed')
    w = result['witness']
    rows = []
    labels = {'flat': '同高平面', 'slope': '统一斜平面', 'roof': '原屋顶高差'}
    for mode, values in summary.items():
        half, box = values['in_halfspace'], values['in_working_box']
        rows.append(f"| {labels[mode]} | {half['known_support_ambiguity']}/180 | {half['any_support_ambiguity']}/180 | {box['known_support_ambiguity']}/180 | {box['any_support_ambiguity']}/180 |")
    text = f'''# 合理部署解决了一层歧义，路径归属仍需核验

2026-10-25｜五站、每源四直达、未知波形、无地图。完成部署与路径解释核验，未启动定位性能实验。

## 先看结论

**你的判断抓住了多径的重要困难：错误路径也可能对应一个看似合理的假位置。** 本轮进一步把它与上一轮的纯几何双解分开了。

- 同高或统一斜平面部署，配合“无人机在安装面上方”，在非退化且直达集合正确时可以排除上下镜像双解。
- 但不知道哪一个站缺直达时，本批仍找到换一套路径归属的合法假位置；有一对位置相距约{w['separation_m']:.2f}m，五站双源IQ相同到数值精度。
- 这是当前自由反射延迟/增益模型中的明确反例，不是定位算法的失败率，也不是所有真实低空环境都不可定位。

## 1 本轮怎么执行

沿用五站平面位置，分别检查原屋顶高差、同高安装和统一斜平面。每种部署使用4个预先固定平面位置×3个高度×5种缺直达站×3档反射绕行长度，共180条源级路径条件；总计540条。先登记合同，再运行，没有按算法成绩挑选位置。

每条条件枚举全部五种四直达集合及其合法位置根，回代未平方距离差，检查剩余站反射长度是否大于直线距离。已给定源分组和精确到达长度，这是信息更充分的几何检查，不冒充盲分离。全部真位置恢复、根回代、输入身份和IQ等价检查通过；本批没有奇异线性系统。

## 2 部署核验结果

| 部署 | 已知直达集合：半空间内有其他解 | 未知集合：半空间内有其他解 | 已知集合：工作域内有其他解 | 未知集合：工作域内有其他解 |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

工作域在运行前固定为平面坐标±700m、参照面上方0—150m；不是围绕假解位置事后裁剪。斜平面只是解析部署对照，不代表已实测地形。各列可以重合，不应相加。

**这些数字是确定性条件计数，不是独立重复试验的概率或算法成功率。** 同高已知支持的唯一性另有满秩条件下的代数推导，而未知支持的计数只描述这批检查。

## 3 看一个具体假位置

| 项目 | 原解释 | 替代解释 |
|---|---|---|
| 源1位置（m） | (-460, -120, 34) | (-470.400, -122.731, 108.211) |
| 直达站（本表从1编号） | 2、3、4、5 | 1、2、3、5 |
| 唯一反射站 | 1 | 4 |
| 反射超额长度 | 10m | {w['alternate_nlos_excess_m']:.6f}m |
| LOS路径系数模平方之和 | {w['original_los_energy_sum']:.2f} | {w['alternate_los_energy_sum']:.2f} |
| NLOS路径系数模平方 | {w['original_nlos_energy']:.2f} | {w['alternate_nlos_energy']:.2f} |

第二个源保持不变。两种解释都在上述工作域内，直达系数能量合计均大于反射。但替代解释有一条反射比其中一条直达强，且反射绕行仅约5.7cm。**它满足当前“总体直达占主导、路径增益未知”的模型；不满足更强的“每条直达必然强于每条反射”规则。** 当前没有采用后一规则，不能不加说明地拿它排除反例。

按候选位置同步改变未知波形的时间原点及反射延迟，可保持完整观测不变。原合成相对差{w['iq_relative_difference']:.3e}；独立按频域传递因子重建后相对差{pair:.3e}。这不是只让一张空间谱相近。

替代解释中改变归属的那条反射链路，已通过一次镜面反射的长度和反射定律检查；没有验证一组共享建筑墙面能同时解释全部链路。因此，这一结果针对独立反射参数模型，不用来断言现实街区的歧义出现率。

## 4 文献与你的理解怎样对应

| 原文 | 对多径困难的描述 | 需要注意的适用条件 |
|---|---|---|
| [Scheuing & Yang，2008](https://doi.org/10.1109/TASL.2008.2004533)，§VI-D、Fig.10 | 反射路径形成的错误TDOA图也可能通过几何循环检查，称为“虚假一致性” | 一致性通过并不等于真位置；后续定位残差有时仍能消歧 |
| [Du等，2019](https://doi.org/10.1109/ACCESS.2018.2889958)，引言、§III-E | 多径相关函数多峰、路径关联困难；另有因目标退化产生的虚假极值 | 有些假峰是评分范数/归一化问题，不是另一物理位置真的等价 |
| [Garcia等，2017](https://doi.org/10.1109/TSP.2017.2666779)，§III-A | 联合利用LOS指向共同位置、NLOS作为局部分量的结构 | 使用阵列方向信息及已知波形，不能直接等同单天线盲定位 |
| [Garcia等，2015预印本](https://arxiv.org/abs/1505.03193)，Assumption 3、Theorem 1 | 明确假设：只有真位置与足够多的源路径一致 | 路径几何唯一性是保证的前提，而非算法对任意多径自动解决 |

直观例子是墙面反射像一面镜子：反射折线路径有时可以当成从“镜像源”直达的路径。但多个站可能看见不同反射面，也混有直达，不能说所有假峰都是同一个镜像源。

## 5 现在应怎样推进

这轮并非又否定了一个部署：**它把已知集合的几何问题与未知路径归属问题分开了。** “合理部署＋上方空间”值得保留，但还不能据此放行单一真位置的性能比较。

下一步应先明确对“少量反射”的物理约束究竟是什么。若继续采用当前自由路径模型，可先研究可辨识条件与多解管理；若采用可验证的共享环境或路径结构，则需先论证它为什么能提供额外信息。不能为了删除本例临时添加最小绕行、幅度排序或缩窄飞行高度。

没有启动新算法、正式性能批次或修改论文；也未据本次标准几何推导宣称新的理论贡献。

<details><summary>展开：复现材料、工程修复与验证</summary>

- [运行前合同](执行合同.md) · [推导与原文页码](推导与文献对应.md) · [枚举源码](check_deployment.py) · [独立重放源码](replay_report.py)
- [原始结果JSON]({RUN.name}/result.json) · [全部候选CSV]({RUN.name}/all_candidates.csv) · [独立重放审计]({RUN.name}/independent_replay.json) · [输入身份]({RUN.name}/preflight.json)
- 首次运行`run_20261025_142832_742287`在CSV写入时失败：共面与非共面求解器的诊断字段不同。仅改为字段并集后，同参数在`{RUN.name}`重新执行；失败尝试保留，不作为通过证据。
- 独立重放重新检查全部候选原距离差；最大残差{max_residual:.3e}m。原始数据与求解输入SHA保持不变。
- 脚本内部枚举约{result['wall_seconds']:.3f}s，不是完整定位算法的耗时。
- 原三个直达和四个直达核验报告保留，不用最新结果覆盖历史结论的适用范围。

</details>
'''
    (ROOT/'部署与多径歧义报告.md').write_text(text, encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == '__main__':
    main()
