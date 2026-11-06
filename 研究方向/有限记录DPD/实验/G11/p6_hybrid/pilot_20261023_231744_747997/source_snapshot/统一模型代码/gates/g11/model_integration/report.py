"""Read saved P5 evidence only; never starts an optimizer or selects cases."""
import csv
import io
from datetime import datetime

import numpy as np

from .runner import BASE, ROOT, read, sha, write


def main():
    destination = BASE/('report_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    destination.mkdir()
    model_path = ROOT/'统一模型代码/gates/g11/model_integration/models.py'
    solver_path = ROOT/'统一模型代码/gates/g11/model_integration/solvers.py'
    required = {str(p.relative_to(ROOT)): sha(p) for p in (model_path, solver_path)}
    paths, cases, cost, audits = {}, {}, {}, []
    for folder in sorted(BASE.iterdir()):
        if not (folder/'contract.json').exists():
            continue
        contract = read(folder/'contract.json')
        if any(contract['sources'].get(name) != value for name, value in required.items()):
            continue
        audit = folder/'audit.json'
        if audit.exists():
            audits.append(dict(folder=folder.name, **read(audit)))
        for name in ('math_checks.json', 'frozen_checks.json', 'smoke.json', 'physical.json'):
            if (folder/name).exists():
                paths[name] = str(folder/name)
        if (folder/'cost.json').exists():
            for row in read(folder/'cost.json'):
                cost[(row['n'], row['kind'])] = dict(row, source=str(folder/'cost.json'))
        for p in sorted((folder/'cases').glob('*.json')):
            row = read(p)
            from ...g10.r4.paper_checks.assignment_metrics import evaluate
            assert evaluate(row['truth'], row['result']['positions']) == row['metrics']
            cases[(row['n'], row['kind'], row['algorithm'])] = dict(row, source=str(p))
    expected = {(1024, kind, algorithm) for kind in ('periodic', 'hann', 'finite_fourier', 'slepian')
                for algorithm in ('AP5', 'VP')}
    complete = expected.issubset(cases)
    if complete:
        assert len({cases[k]['input_sha'] for k in expected}) == 1, 'Pilot inputs are not byte-identical'
    data = dict(paths=paths, costs=list(cost.values()), cases=list(cases.values()), audits=audits,
                formal_released=False, native_pilot_complete=complete,
                status='NATIVE_PILOT_COMPLETE_COST_HOLD' if complete else 'MATH_CHECKED_NATIVE_PILOT_PARTIAL_COST_HOLD',
                reason='Dense finite/Slepian whole-grid cost is not a practical formal budget; remaining pilot is explicit')
    write(destination/'summary.json', data)
    f = io.StringIO(newline='')
    writer = csv.writer(f)
    writer.writerow(['N', 'representation', 'algorithm', 'seconds', 'source_A_error_m', 'source_B_error_m', 'stop', 'source'])
    table = []
    for row in cases.values():
        r, metric = row['result'], row['metrics']
        writer.writerow([row['n'], row['kind'], row['algorithm'], r['seconds'], *metric['errors_m'], r['stop_reason'], row['source']])
        table.append(f"|{row['n']}|{row['kind']}|{row['algorithm']}|{r['seconds']:.3f}|{metric['errors_m'][0]:.4f}|{metric['errors_m'][1]:.4f}|{r['stop_reason']}|")
    (destination/'native_pilot.csv').write_text(f.getvalue(), encoding='utf-8-sig')
    text = '# G11-P5准备核验报告\n\n'
    text += '本报告是实现、有限短测和成本核验，不是正式方法排名。未批准启动大规模批次。\n\n'
    text += '## 已保存的完整原生短测\n\n'
    text += '|N|表示|算法|完整秒数|源A误差/m|源B误差/m|停止原因|\n|---:|---|---|---:|---:|---:|---|\n'+'\n'.join(table)+'\n\n'
    text += '同一开发构型、实现28000、15 dB；没有据此调参数。误差是单例距离，不能写成总体RMSE。CPU二线程，完整成本包含本例基准备。\n\n'
    text += '## 成本判断\n\n'
    text += '原生AP两次完整网格搜索分别使用单源、条件多源目标；VP两次使用残差块拟合。有限表示每次都要求跨频系数求解，不能用局部精修耗时估算全批。\n\n'
    text += '|N|表示|AP两网格外推/s|VP两网格外推/s|\n|---:|---|---:|---:|\n'
    for row in cost.values():
        text += f"|{row['n']}|{row['kind']}|{row['AP_two_grid_extrapolated_seconds']:.1f}|{row['VP_two_grid_extrapolated_seconds']:.1f}|\n"
    text += '\n这些是三个公开探针点的组件外推，未包含精修，不是完整实测、严格下界或方法固有复杂度。冻结前不据此宣称精确总预算。早期未复用Fourier公共Gram的计时保留于原目录，不与当前优化实现混算。\n\n'
    text += '## 接下来\n\n仅交付1024点单开发观测的原生核验入口；完成后自动保存各方法结果。20构型×20实现×6功率差×3长度的正式实验暂不放行。应先解决常规有限表示的全域计算可行性，再决定是否采用明确登记的粗搜索/精修组合；不能隐式退回旧公共前端。\n\n'
    text += '输入、源码快照、数学与物理核验来源在summary.json。历史论文和求解器未修改。\n'
    (destination/'准备报告.md').write_text(text, encoding='utf-8')
    print(destination)
    assert np.isfinite([r['result']['seconds'] for r in cases.values()]).all()


if __name__ == '__main__':
    main()
