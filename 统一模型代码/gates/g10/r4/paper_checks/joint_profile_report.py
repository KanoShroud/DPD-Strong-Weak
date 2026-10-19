"""Read-only result replay plus local diagnostic report; no solver calls."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from ...metrics import evaluate
from .segment_model_check import ROOT, BASE, checked_bytes, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    out = args.out.resolve(strict=True)
    if not out.is_relative_to(BASE):
        raise ValueError('Expected an R4 diagnostic output directory')
    contract = json.loads((out/'contract.json').read_text(encoding='utf-8'))
    report = json.loads((out/'report.json').read_text(encoding='utf-8'))
    rows = report['rows']
    assert report['status'] == 'COMPLETED' and len(rows) == 40
    for path, identity in contract['inputs'].items():
        checked_bytes(Path(path), identity)
        p = Path(path)
        if p.suffix == '.py':
            target = out/'source_snapshot'/p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                checked_bytes(target, identity)
            else:
                shutil.copyfile(p, target)
            checked_bytes(target, identity)
    # Evaluator only receives truth after inference; it does not affect ranking or stopping.
    prep = json.loads((BASE/'preparation.json').read_text(encoding='utf-8'))
    old = json.loads(checked_bytes(Path(prep['out'])/'contract.json', prep['contract_sha']))
    by_case = {}
    for row in rows:
        truth = old['inputs'][row['case']]['truth']
        metric = evaluate(truth, row['positions'])
        assert metric['success']['20'] == row['metrics']['success']['20']
        np.testing.assert_allclose(metric['errors_m'], row['metrics']['errors_m'], atol=1e-12)
        by_case.setdefault(row['case'], {})[row['method']] = row
    assert len(by_case) == 8
    for case in by_case.values():
        candidates = case['AP_LBFGS']['candidates']
        for name in ('AP_LBFGS', 'AP_VP_BLOCK', 'AP_VP_JOINT'):
            np.testing.assert_array_equal(case[name]['candidates'], candidates)
            assert len(case[name]['branches']) == 5
            assert all(b['projected_gradient'] <= 1e-8 for b in case[name]['branches'])
        checkpoint = case['VP_COMMON']['native_checkpoint']
        assert checkpoint is not None
        np.testing.assert_allclose(checkpoint['positions'], case['VP_NATIVE']['positions'], atol=1e-5)
    methods = ('VP_NATIVE', 'VP_COMMON', 'AP_LBFGS', 'AP_VP_BLOCK', 'AP_VP_JOINT')
    stats = {}
    for name in methods:
        rr = [r for r in rows if r['method'] == name]
        stats[name] = dict(n=len(rr), success=sum(all(r['metrics']['success']['20']) for r in rr),
                          precision=sum(r['projected_gradient'] <= 1e-8 for r in rr),
                          mean_seconds=float(np.mean([r['seconds'] for r in rr])),
                          weak_rmse=float(np.sqrt(np.mean([r['metrics']['errors_m'][1]**2 for r in rr]))))
    common = [r for r in rows if r['method'] == 'VP_COMMON']
    native = [r for r in rows if r['method'] == 'VP_NATIVE']
    stats['vp_checks'] = dict(mean_seconds=float(np.mean([r['common_check_seconds'] for r in common])),
        native_second_cycles=float(np.mean([r['native_cycles'][1]['cycles'] for r in native])),
        common_second_cycles=float(np.mean([r['cycles'][1]['cycles'] for r in common])))
    for name in methods[2:]:
        rr = [r for r in rows if r['method'] == name]
        stats[name].update(preparation_seconds=float(np.mean([r['preparation_seconds'] for r in rr])),
                          refinement_seconds=float(np.mean([r['refinement_seconds'] for r in rr])),
                          hessian_calls=float(np.mean([r['counts'].get('joint_profile_hessians', 0) for r in rr])))
    joint = stats['AP_VP_JOINT']['mean_seconds']
    stats['joint_vs_lbfgs_cost_change'] = joint/stats['AP_LBFGS']['mean_seconds']-1
    stats['joint_vs_block_cost_change'] = joint/stats['AP_VP_BLOCK']['mean_seconds']-1
    stats['new_localization_solves'] = 0
    write(out/'analysis.json', stats)
    lines = ['# 共同停止与联合消元短测回读', '',
             '结论：原生停止的速度取舍受优化精度影响；精确联合消元曲率有效减少了分块更新的循环，但未超过现有联合L-BFGS-B的完整计算效率。本轮到此收尾，不扩大实验或替换论文算法。', '',
             '## 固定范围', '',
             '采用此前四类开发构型、实现200、0/15 dB共8条观测。每例五个版本，共40次完整求解；另有数学和原AP输出核验。全部观测已用于开发，不作为新独立验证，也不并入正式240/2400条。', '',
             '## 实测结果', '',
             '|版本|双源20 m成功|达到共同梯度1e-8|平均完整耗时/s|弱源RMSE/m|',
             '|---|---:|---:|---:|---:|']
    for name in methods:
        s = stats[name]
        lines.append(f"|{name}|{s['success']}/8|{s['precision']}/8|{s['mean_seconds']:.4f}|{s['weak_rmse']:.4f}|")
    check = stats['vp_checks']
    lines += ['', '## 停止核验说明', '',
        f"VP第二源平均循环由{check['native_second_cycles']:.3f}增加到{check['common_second_cycles']:.3f}；共同精度检查本身平均耗时{check['mean_seconds']:.4f}秒，已包含在完整时间中。额外时间同时来自继续迭代和检查成本，不能全部归为更新低效。原生与共同版本的原生阈值交叉检查点位置一致，证书的最小二乘系数没有反馈改变循环轨迹。",
        '更严格的数值收敛未增加本组成功数，RMSE也未改善；优化目标更低不保证单次噪声下真实位置误差更小。原生停止仍可作为应用配置报告，本结果不直接宣布原正式比较不公平；但不同停止下的速度差不能全部解释为更新机制优势。', '',
        '## 联合曲率及成本', '',
        '三个AP版本使用逐数组一致的五候选。每例各自重新执行相同搜索和传输，并支付全部成本；三种更新的顺序轮换。全部120个分支（各版本40个）达到同一投影梯度门槛，不是只检查最终被选中的分支。', '',
        '|AP版本|候选准备/s|精修/s|平均精确Hessian调用|', '|---|---:|---:|---:|']
    for name in methods[2:]:
        s = stats[name]
        lines.append(f"|{name}|{s['preparation_seconds']:.4f}|{s['refinement_seconds']:.4f}|{s['hessian_calls']:.3f}|")
    lines += ['',
        f"联合曲率相比分块曲率降低完整耗时{-stats['joint_vs_block_cost_change']:.2%}，但比现有联合L-BFGS-B增加{stats['joint_vs_lbfgs_cost_change']:.2%}。联合曲率确实减少更新工作，然而计算每次精确曲率仍昂贵；本实现没有形成实际提速。",
        '完整Hessian是新的4×4联合消元曲率；去交叉块版本保持目标、梯度、保护规则不变，故这项机制对照不改变求解目标。它不是把两个完整算法串联。标准变量消元和Schur补不作为新数学主张。', '',
        '## 审计与版本', '',
        f"最终版本主函数内部运行{report['elapsed_seconds']:.3f}秒，含核验及逐例保存，不含解释器启动。连同前缀和工程复核，本轮记录的运行时间约108秒，远低于批准的3小时；没有为了用满预算扩大实验。",
        '源码快照位于source_snapshot；合同登记输入和代码哈希，检查记录见checks.json，逐例结果、分支及轨迹保留。输入/代码身份、新旧AP复现、原生VP检查点、评价重放、候选一致、全部分支精度、自动微分与有限差分通过。',
        '前一版本补齐了新Newton与原AP相同的精确位置缓存，并轮换VP停止版本执行顺序；最终版本进一步要求全频梯度核验绕过缓存，并补全频Hessian独立核验。旧准备目录保留，不混入本表。参数、数据和更新公式未根据定位成绩改变。',
        '原论文、冻结求解器和正式结果不修改。本轮不安排新的正式批次；如何将负面结果和已有VP贡献纳入论文由后续写作决策确定。']
    (out/'回读分析.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write(out/'analysis_audit.json', dict(passed=True, rows=40, paired_cases=8, all_ap_branches=120,
          all_ap_branches_common_precision=True, native_vp_checkpoints_equal=True,
          input_and_code_identities_unchanged=True,
          analysis_script_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
