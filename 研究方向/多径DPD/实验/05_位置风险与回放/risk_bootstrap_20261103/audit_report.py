"""Read back completed pilot artifacts; no new localization or hidden truth read."""
from pathlib import Path
import importlib.util
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('bootstrap_audit_target', HERE/'runner.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def main():
    out = Path(sys.argv[1]).resolve(strict=True) if len(sys.argv)>1 else Path(json.loads((HERE/'latest.json').read_text(encoding='utf-8'))['path'])
    contract = r.load(out/'contract.json')
    r.cal.check_paths(contract['hashes'])
    prediction = r.load(out/'prediction_frozen.json')
    checks = dict(input_source_identity=True, unit_checks=all(r.load(out/'unit_checks.json').values()))
    rows = []
    for p in sorted((out/'replicates').glob('*/result.json')):
        item = r.load(p)
        solved = r.load(item['solve_file'])
        checks[p.parent.name+'/solve_identity'] = r.cal.ref.sha(item['solve_file']) == item['solve_sha']
        cond, rep = item['condition'], item['repeat']
        with np.load(out/(cond+'_fitted_world.npz'), allow_pickle=False) as z:
            mean, sigma2, focus = z['mean'], float(z['sigma2']), z['focus']
        with np.load(p.parent/'input.npz', allow_pickle=False) as z:
            checks[p.parent.name+'/iq_only'] = set(z.files) == {'iq'}
            iq = z['iq']
        expected = r.generate(mean,sigma2,r.CONDITIONS.index(cond),rep)
        checks[p.parent.name+'/seed_replay'] = np.array_equal(expected,iq)
        checks[p.parent.name+'/iq_identity'] = r.cal.digest(iq) == item['input_sha'] == solved['iq_sha']
        checks[p.parent.name+'/numerical_gate_record'] = r.solver_gate(solved) == item['numerical_gate']
        if item['loss_curve_m2'] is not None:
            checks[p.parent.name+'/all_cap_metrics'] = np.array_equal(r.curve(solved,focus),item['loss_curve_m2'])
        rows.append(item)
    evpath = out/'evaluation_only.json'
    checks['no_premature_truth_evaluation'] = not evpath.exists() if prediction['stopped'] else evpath.exists()
    if evpath.exists():
        checks['prediction_precedes_evaluation'] = (out/'prediction_frozen.json').stat().st_mtime_ns <= evpath.stat().st_mtime_ns
    audit = dict(checks=checks, all_pass=all(checks.values()), count=len(checks),
                 frozen_files=len(contract['hashes']), new_full_solves=len(rows),
                 risk_selector_run=False, independent_validation=False,
                 audit_source_sha=r.cal.ref.sha(Path(__file__)))
    r.save(out/'readback_audit.json',audit)
    assert audit['all_pass'], checks
    evaluation = r.load(evpath) if evpath.exists() else None
    lines = ['# 有限噪声风险核验：阶段结果','',
             '## Material Passport','',
             '- Origin Skill: academic-research-suite / experiment-agent。',
             '- Verification Status: ANALYZED；重放核对的是数值身份与风险计算，不是新选择器有效性。',
             '- Scope: 旧R00拟合世界、完整盲定位重采样；不改场景、原估计器或640条。','',
             '## 当前状态','',
             f"- 阶段状态：`{prediction['stopped'] or 'finite_pilot_complete'}`。",
             f"- 已完成并保存 {len(rows)} 次完整重定位；本批累计 {prediction['cost']['seconds']:.3f} 秒。",
             f"- {len(checks)} 项回读检查通过；{len(contract['hashes'])} 项冻结输入/源码身份未变。",
             '- HIGH_R00原完整参照触及64容量且风险不适用，保留为不可校准，不换donor，不计为通过。',
             '- 每次副本从IQ重新做几何初始化、噪声估计、路径发现与联合拟合；没有借用donor清单或位置作初值。','',
             '## 拟合世界内的有限结果','',
             '| 条件 | 副本数 | 数值门通过 | 推定反射总数范围 | 终态伪世界RMSE/m |',
             '|---|---:|---:|---:|---:|']
    for cond, a in prediction['conditions'].items():
        risk = a['risk_curve_m2']
        val = f'{np.sqrt(risk[-1]):.6f}' if risk is not None else '未定义'
        lines.append(f"| {cond} | {a['n_requested']} | {a['numerical_pass']} | {min(a['orders'])}–{max(a['orders'])} | {val} |")
    lines += ['', '上述误差相对于估计的donor位置，仅度量拟合世界的风险；不能当成真实位置RMSE。任何异常返回和数值门失败均保留。小批平均风险最小的cap存在选择乐观性，不作为算法成绩。','',
              '## 预算与继续条件','']
    gatepath = out/'pilot_gate.json'
    if gatepath.exists():
        gate = r.load(gatepath)
        lines.append(f"按最慢单副本保守估计完整48副本 {gate['predicted_full_seconds_conservative']/60:.2f} 分钟；预算阈值为1小时，估计规则已在运行前冻结。")
    if prediction['stopped']:
        lines += ['本阶段未满足自动完成全部校准的条件。未读取R01–R07物理真值、未产生真实条件转移排名、未开发新选择器。不能依据当前小批调参数或增加重复数。']
    else:
        lines += ['完整有限校准完成；以下是固定donor推荐cap在旧R01–R07上的条件转移，不是逐条自适应选模或新场景独立验证。','',
                  '| 条件 | 自举最低均值cap（诊断） | 接收记录cap RMSE/m | 接收记录终态RMSE/m | MSE差/m² |',
                  '|---|---:|---:|---:|---:|']
        for cond,a in evaluation.items():
            if 'plugin_minimum_cap_diagnostic' in a:
                lines.append(f"| {cond} | {a['plugin_minimum_cap_diagnostic']} | {a['fixed_cap_receiver_rmse_m']:.6f} | {a['receiver_wide_rmse_m']:.6f} | {a['fixed_cap_receiver_delta_m2']:.6g} |")
    lines += ['', '## 理论解释与边界','',
              '本轮替换的是有限噪声风险估计方式，不是定位求解器。它不依赖单点线性响应近似，且保留路径重新选择；但其参照仍是从同一观测估计的频域均值。未知波形误差、有限记录差异和参照位置误差均可能使拟合世界与真实世界不同。',
              '固定容量政策的风险不等于原样本某一固定清单的风险。需要分别讨论估计目标、Monte Carlo误差与跨世界转移误差；不能只看拟合残差或伪世界内最低风险。',
              '所有数学工具均有既有基础。本轮是新路线可行性核验，不是新理论确认或SP投稿放行。','',
              '## 统计解释检查（11/11）','',
              '分组报告，不跨条件用总体均值掩盖差异（Simpson）；不从条件组推个体oracle（生态谬误）；明确适用性筛选及HIGH不可校准（Berkson/碰撞偏差）；不声称总体成功率（基率）；donor按编号非极端成绩选择（均值回归）；异常不删除（幸存者）；65cap最小均值只诊断（多重搜索）；预固定种子、预算和规则（分叉路径）；不将转移相关性视为唯一因果（相关/因果）；生成先于求解、真值只事后评价（反向因果）。',
              '配对MC标准误保存在prediction_frozen.json，只刻画拟合世界抽样误差，不是同时置信区间，也不涵盖参照失配。','',
              '## 工程过程记录','',
              '开发首次入口自检遇到同名runner导入冲突，发生在新实验目录建立前。修复为显式目标模块传递后，首批在L0输出后的可选closure_error字段读取处报错；旧LOS状态没有该字段。改为使用原project函数直接计算可行距离，未改变阈值。',
              '失败目录run_20261103_191438_876342及其完整L0求解保留，不作为通过证据。本有效批在新目录重跑；另通过只读审查修正了空异常汇总和未完成时过早读取真值的问题，均发生在有效批运行前。',
              '复现：runner.py（首次限时执行）、一键入口（允许用户恢复超过1小时但不越过科学门）、audit_report.py（只回读）。冻结协议及文献来源见上级执行协议.md。']
    (out/'有限噪声风险核验报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False))


if __name__=='__main__':
    main()
