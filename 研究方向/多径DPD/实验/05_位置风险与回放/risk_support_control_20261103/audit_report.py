"""Saved-result audit and report; performs no optimization."""
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import control as c

HERE = Path(__file__).resolve().parent


def main():
    out = Path(json.loads((HERE/'latest.json').read_text(encoding='utf-8'))['path']).resolve(strict=True)
    r = c.read(out/'result.json')
    e = c.read(out/'evaluation_only.json')
    pre = c.read(out/'preflight.json')
    post = c.read(out/'postflight.json')
    checks = {}
    for p, h in post['hashes'].items():
        checks['sha:'+p] = c.ref.sha(p) == h
    checks['all_pre_in_post'] = all(post['hashes'].get(p) == h for p, h in pre['hashes'].items())
    checks['24_local_fits_no_new_blind'] = r['local_fit_count'] == 24 and r['new_blind_localizations'] == 0
    checks['14_receiver_refits'] = r['receiver_refits'] == 14
    checks['no_adaptive_arm_claim'] = not r['strict_adaptive_arm_run']
    decisions = c.read(out/'decisions_before_truth.json')
    checks['decisions_identical'] = decisions['rows'] == r['rows']
    checks['decisions_before_evaluation'] = (out/'decisions_before_truth.json').stat().st_mtime_ns <= (out/'evaluation_only.json').stat().st_mtime_ns
    groups, smalls, donors = {}, {}, {}
    for cond in ('L2', 'L3'):
        d = c.read(out/(cond+'_donor.json'))
        donors[cond] = d
        checks[cond+'/donor_applicable'] = d['diagnostic']['applicable']
        checks[cond+'/donor_checks'] = all(d['diagnostic']['derivative_checks'].values())
        rows = [x for x in r['rows'] if x['condition'] == cond and x['kind'] == 'receiver']
        checks[cond+'/receivers_1_to_7'] = sorted(x['repeat'] for x in rows) == list(range(1, 8))
        for row in [x for x in r['rows'] if x['condition'] == cond]:
            checks[row['id']+'/raw_equal'] = c.read(out/(row['id']+'.json')) == row
            checks[row['id']+'/start_donor'] = row['initial_x'] == d['x']
            checks[row['id']+'/paths_donor'] = row['paths'] == d['paths']
            checks[row['id']+'/correctness'] = all(row['correctness'].values())
            checks[row['id']+'/stationary'] = row['status']['stationary']
            predicted = np.asarray(row['prediction_m'])
            observed = np.asarray(row['actual_change_m'])
            checks[row['id']+'/error_replay'] = bool(np.isclose(np.linalg.norm(observed-predicted), row['response_error_m']))
        recomputed = c.empirical(rows, np.asarray(e[cond]['common_label_truth']))
        checks[cond+'/mse_replay'] = bool(np.isclose(recomputed['mse_m2'], e[cond]['fixed']['mse_m2']))
        groups[cond] = dict(n=7, **{k:e[cond]['fixed'][k] for k in ('rmse_m','mean_response_error_m')},
                           old_adaptive_rmse_m=e[cond]['old_adaptive_rmse_m'],
                           relative_error_min=min(x['response_relative_error'] for x in rows),
                           relative_error_max=max(x['response_relative_error'] for x in rows),
                           delay_movement_max=max(x['branch']['max_delay_movement_Btau'] for x in rows),
                           seconds=sum(x['fit']['seconds'] for x in rows))
        smalls[cond] = {str(a):c.read(out/(cond+f'_centered_{a}.json')) for a in (.1,.01)}
    assert all(checks.values()), [k for k,v in checks.items() if not v]
    costs = dict(total=r['seconds'], receiver=sum(g['seconds'] for g in groups.values()),
                 donor_fit=sum(d['fit']['seconds'] for d in donors.values()),
                 donor_diagnostic=sum(d['diagnostic']['seconds'] for d in donors.values()),
                 small=sum(x['fit']['seconds'] for x in r['rows'] if x['kind']=='local_perturbation'))
    evidence = dict(utc_local=datetime.now().isoformat(), checks=checks, checks_passed=len(checks),
                    frozen_file_count=len(post['hashes']), groups=groups, smalls=smalls, costs=costs)
    c.save(out/'readback_audit.json', evidence)
    lines = [
        '# 固定观测支持的局部响应核验', '',
        '日期：2026-11-03。类型：固定拓扑的条件局部重拟合，不是新全盲定位或新选模算法。', '',
        '## 主要结论', '',
        '在不重新选择反射清单的条件下，足够小的观测扰动可以被现有实际Hessian局部公式准确描述；但把同一个线性公式用于完整噪声实现之间的变化，仍出现明显预测误差。因此，不能把此前风险评分不准全部归因于重新选择路径。', '',
        '本轮没有检验风险评分的跨模型排序，也没有比较新旧风险选模成绩。当前证据支持区分“导数/局部公式正确”与“在实际噪声幅度下可准确校准风险”，不支持新增评分系数或宣称解决风险选模。', '',
        '## 实际执行', '',
        f'- 结果目录：`{out.name}`。总墙钟 {r["seconds"]:.9f} 秒。',
        '- L2、L3 各以 R00 作为独立编号的观测支持 donor；R00 不计受试定位表现。随后每条件 R01–R07，共14次固定清单重拟合。',
        '- donor 完整路径清单分别为21、30个反射分量。它们由原观测算法决定，不是真实反射个数。所有受试记录从同条件 donor 的完全相同参数向量出发。',
        '- 每条件另对 R01 方向做预登记的 ±0.1、±0.01 倍观测差分，共8次小扰动拟合；加2次donor局部拟合，合计24次fit。未新增盲定位。',
        '- 两条R01短测通过数值与身份门禁后自动执行余下条目；定位误差没有参与放行。中途会话中断时进程已继续完成，恢复后只回读，没有重跑。',
        f'- 编译、Ruff、4项单元测试通过；本次回读 {len(checks)} 项检查通过，{len(post["hashes"])} 项输入/源码SHA前后及恢复后相同。',
        '- 所有14条受试和8条小扰动均达到原1e-7投影梯度条件；原300次预算、单路径幅度约束和延迟范围不变。没有放松门槛或调参。', '',
        '## 小扰动：公式在附近是否正确？', '',
        '中心响应定义为 [p(y0+αΔy)−p(y0−αΔy)]/2，与 αLΔy 比较。绝对误差为四维位置向量之差的欧氏范数，不是两源RMSE。', '',
        '| 条件 | α | 中心响应相对误差 | 绝对误差/m |',
        '|---|---:|---:|---:|']
    for cond in ('L2','L3'):
        for alpha, row in smalls[cond].items():
            lines.append(f'| {cond} | {alpha} | {100*row["relative_error"]:.6f}% | {row["error_m"]:.8g} |')
    lines += ['',
        'L3 从0.1缩到0.01时，中心响应相对误差由约7.22%降至0.0665%，说明局部近似的适用幅度需要重视。L2两种幅度均很小；0.01的相对误差略大但绝对误差更小，接近微米级数值停止误差，不能据此认定更小扰动更不可靠。', '',
        '中心差分会抵消部分偶次非线性项，不能将中心误差直接当作单侧变化误差。单侧结果也全部保存：例如L3、α=0.1正负方向相对误差分别约3.39%和14.55%。', '',
        '## 完整噪声差分：是否仍然准确？', '',
        '| 条件 | n | 平均四维响应预测误差/m | 单例相对误差范围 | 最大时延变化 BΔτ |',
        '|---|---:|---:|---:|---:|']
    for cond,g in groups.items():
        lines.append(f'| {cond} | 7 | {g["mean_response_error_m"]:.6f} | {100*g["relative_error_min"]:.2f}%–{100*g["relative_error_max"]:.2f}% | {g["delay_movement_max"]:.6f} |')
    lines += ['',
        '没有路径增删或源站归属改变，14条均为可行驻点，且未观察到同链路路径时延排序改变或约束活跃。但是排序不变不等于留在相同的局部极小盆：连续时延可明显移动，尤其L3最大约6.04/B。因此本轮不能将完整差分误差唯一归为光滑非线性曲率，也不能排除局部盆迁移。', '',
        'donor实际Hessian与残差GN矩阵相对差异均约15.9%；实际Hessian步长稳定性、混合观测导数及零残差GN极限检查全部通过。这里已经使用非零残差公式，不应再次把“补上Hessian”提出为新改进。', '',
        '## 定位精度只作诊断，不作算法排名', '',
        '| 条件 | 固定清单局部重拟合RMSE/m | 原自适应完整流程RMSE/m（相同R01–R07） |',
        '|---|---:|---:|']
    for cond,g in groups.items():
        lines.append(f'| {cond} | {g["rmse_m"]:.6f} | {g["old_adaptive_rmse_m"]:.6f} |')
    lines += ['',
        '两列起点和选择过程不同。固定组借用了另一条记录推定的完整清单和局部起点，原流程则各自盲初始化；所以这些数据不能表明“自适应一定更好”或“冻结支持一定更差”。仅用于确认本轮并未隐藏定位质量，不能替代公平的完整方法比较。', '',
        '每条件7条差分共享有噪声donor，不能当作7个独立donor实验。条件于固定donor，接收噪声可独立；边缘差分仍共享donor误差。本轮不提供简单独立样本置信区间。L2和L3使用不同固定波形/信道银行，也不能以它们的差别量化纯路径条数效应。', '',
        '## 为什么没有执行“重选支持”严格配对组？', '',
        '原冻结程序只有逐步追加路径，没有删除并重选全部已有路径的算子。从完整donor参数继续追加不是重新选择全部支持；清空路径又改变共同起点。为避免偷偷增加新算法，本轮只完成已批准控制中可直接执行的固定组。没有开发删除器或固定顺序增径器；原自适应结果仅为描述性参照。', '',
        '## 下一边界', '',
        '当前已经得到必要的否定性信息：即使冻结路径清单，单点线性影响公式也不自动准确描述完整噪声变化。因此不宜只追加一个“搜索自由度/选择惩罚”就期待风险选模恢复。是否采用有限扰动风险估计、如何处理参照误差及额外求解成本，需要新的明确理论规则；本轮不自动开发。主线优先完成信息条件相容的完整近邻比较。', '',
        '## 成本与复现', '',
        f'- donor拟合 {costs["donor_fit"]:.6f}s；实际Hessian/局部检查 {costs["donor_diagnostic"]:.6f}s；14条受试fit {costs["receiver"]:.6f}s；8条小扰动fit {costs["small"]:.6f}s。其余为IO、评价和身份检查。',
        '- 原始事实：result.json、每条JSON、donor诊断、*_influence.npz、decisions_before_truth.json、evaluation_only.json、preflight/postflight.json、readback_audit.json。',
        '- control.py 为新目录内入口；运行会新建批次，不覆盖已完成结果。本轮已完成，不需要再次执行。audit_report.py仅离线回读生成本报告。', '',
        '## Material Passport', '',
        '- Origin Skill: scientific-toolkit-skill。',
        '- Origin Mode: approved fixed-topology local refit and saved-result audit。',
        '- Verification Status: ANALYZED；不是新算法独立验证。',
        '- Scope: 2 donor、14固定组接收记录、8数学扰动；无新场景、无新阈值、无640、无严格重选支持组。',
        '- Uncertainty: 单donor条件、小样本；未隔离局部盆迁移、支持选择、参照偏差的单独因果作用。', '']
    (out/'固定清单局部响应报告.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(checks=len(checks), frozen_files=len(post['hashes']), costs=costs, groups=groups), ensure_ascii=False))


if __name__ == '__main__':
    main()
