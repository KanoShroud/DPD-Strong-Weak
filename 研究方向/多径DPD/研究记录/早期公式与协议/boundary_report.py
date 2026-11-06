"""Saved-result review and compact HTML; no numerical optimization."""
from pathlib import Path
import argparse
import json
import numpy as np
from structural_probe import sha, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    out = args.directory.resolve(strict=True)
    d = json.loads((out/'result.json').read_text(encoding='utf-8'))
    protocol = json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    oldroot = Path(protocol['source'])
    old = json.loads((oldroot/'result.json').read_text(encoding='utf-8'))
    prefix = json.loads((out.parent/'boundary_20261028_213951_223966/result.json').read_text(encoding='utf-8'))
    checks = {'hash/'+p: sha(p) == h for p, h in protocol['hashes'].items()}
    checks['prefix_output'] = all(np.array_equal(a['p'], b['p']) and a['cost'] == b['cost']
                                 for a, b in zip(prefix['rows'][0]['branches'], d['rows'][0]['branches']))
    checks['all_four_cases'] = d['complete'] and len(d['rows']) == 4
    audit_rows = []
    for new, previous in zip(d['rows'], old['rows']):
        checks[new['case']+'/same_candidates'] = [b['pair_id'] for b in new['branches']] == [b['pair_id'] for b in previous['branches']]
        checks[new['case']+'/selection'] = new['chosen'] == min(range(3), key=lambda k: new['branches'][k]['cost'])
        checks[new['case']+'/finite'] = all(np.all(np.isfinite(b['p'])) and np.all(np.isfinite(b['v'])) for b in new['branches'])
        b = new['branches'][new['chosen']]
        sub = [h['subproblem'] for branch in new['branches'] for h in branch['history'] if 'subproblem' in h]
        audit_rows.append(dict(case=new['case'], old_rmse=previous['evaluation']['rmse_m'],
            new_rmse=new['evaluation']['rmse_m'], errors=new['evaluation']['source_errors_m'],
            old_cost=previous['branches'][previous['chosen']]['final']['cost'], new_cost=b['cost'],
            mapping=b['mapping'], reason=b['reason'], feasible_descent=b['feasible_descent'],
            local_seconds=sum(t['seconds'] for t in new['branches']),
            inner_calls=len(sub), inner_abnormal=sum(not t['success'] for t in sub),
            fallback=sum(t['fallback'] for t in sub), inner_iterations=sum(t['iterations'] for t in sub),
            warning_count=sum(len(t['warnings']) for t in sub),
            max_closure_repair=max([t['closure_repair_norm'] for t in sub] or [0])))
    write(out/'review.json', dict(rows=audit_rows, checks=checks, passed=bool(all(checks.values())),
                                 source_sha=sha(Path(__file__))))
    assert all(checks.values())
    lines = ['| 条件 | 旧RMSE/m | 新RMSE/m | 源A/B误差/m | 新终点投影梯度映射 |',
             '|---|---:|---:|---|---:|']
    htmlrows = []
    for row in audit_rows:
        e = row['errors']
        lines.append(f'| {row["case"]} | {row["old_rmse"]:.6g} | {row["new_rmse"]:.6g} | {e[0]:.6g} / {e[1]:.6g} | {row["mapping"]:.3g} |')
        htmlrows.append(f'<tr><td>{row["case"]}</td><td>{row["old_rmse"]:.6g}</td><td>{row["new_rmse"]:.6g}</td><td>{e[0]:.6g} / {e[1]:.6g}</td></tr>')
    summary = '''# 固定盲起点的边界更新核验

2026-10-28。结果：边界卡住的问题得到缓解，但困难源仍偏约50m。仅修正边界更新不足以恢复这一批观测产生的起点；不继续追加迭代，不把剩余误差直接包装成理论瓶颈。

## 1. 保持了什么，改变了什么

原四条96点理想频率无噪观测、每条三个保留候选、位置和信道初值、原残差目标、几何及路径约束全部不变。未重新搜索、重新提取路径、添加噪声或读取真值选起点。前轮全部九对位置仍保留，本轮只复用当时按目标选定的三个完整初值。

原来先求无约束阻尼GN步，越界就拒绝；本轮求步时加入位置/时延盒约束和每链路反射比值单位球约束。线性子问题为

`min_d 0.5||r+Jd||² + λ/2||D d||², subject to x+d in closure(C)`。

采用SLSQP求解这一凸二次目标与凸约束子问题，解析目标/约束梯度，最多100次内迭代、ftol=1e-10。外层仍最多60步；每步最多12次Armijo试探。对子问题终点作闭包投影，必要时使用投影负梯度方向；所有实际接受点必须满足原严格功率约束，目标严格下降。

0.99倍内部步是从当前点向试探点移动的比例，不是规定反射功率≤0.99，更未把生成器真实DRR=6dB给算法。约束闭包仅用于求方向；不把等功率边界当作合法最终物理状态。

停止以实际坐标投影梯度映射≤1e-7核查；这比原无约束梯度适合边界，但两种停止表达并不相同。因此只报告实际终点及工作量，不宣称等精度加速。工具依据为[官方SLSQP文档](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)，本地SciPy版本1.15.2；没有把通用约束优化登记为新方法。

## 2. 配对结果

TABLE

RMSE为单条观测两源平方位置误差均值开根号，不是多随机实现的统计估计。L0/L1为两个原构型，P0/P2为无反射/两反射。输出仍由原观测残差最低的分支决定，未使用误差选择。

两个无反射结果不变；构型0多径约0.36m，微小变化不构成精度优势。构型1多径约35.21m，源B仍偏49.75m。困难分支原目标从0.020410805降到0.016003048，下降21.60%；A误差从2.87m减至2.12m，B从49.73m变为49.75m。拟合改善主要没有转化为困难源的位置恢复。

困难分支投影梯度映射由前轮0.00560456减至3.11545e-7，固定四档可行梯度探测未找到超过1e-12的下降，已明显不同于此前边界卡死状态。但仍未达到预设1e-7，且60步触顶，不能称已严格收敛或证明位置所在区域是全局/局部最优。

## 3. 执行质量及限制

12个分支中，两个无反射正确分支达到投影梯度条件，其余10个均触及60步预算。10个非平凡分支各接受60次下降更新，共600次；没有再发生旧式因越界拒绝而提前终止，但不等于全部收敛。

600次子问题中21次库返回非成功状态，全部来自L0_P0未选中错误分支2_0；其最大闭包修正约0.2963，不可称舍入误差。实际步骤仍通过可行性和原目标下降检查，此分支没有被选为最终输出。其余子问题正常返回；困难选中分支有25次切换为投影梯度方向，也如实保存，不能称600次均精确求解。120条SLSQP内部越界裁剪警告已归档，不逐行刷屏、不隐藏告警。

约束导数有限差分、可解析球/盒二次问题、投影幂等、严格内部可行步和真实下降等6项基础检查通过；完整运行54项检查通过；额外26项只读身份/候选/选择/重复核验通过。首个测试夹具误将源置于接收站，触发几何合法性检查；已修正测试点而非实验配置，后续通过。单例前缀与完整批第一条输出和目标一致；只改变告警收集，不改变算法。

本次12分支局部更新合计9.73秒，不含复用的候选搜索/路径初始化。不能与前轮完整初始化55.82秒作加速比较。内层迭代增加了工作量，相同60步不代表相同运算预算。原输入/源码哈希在运行前后复核；旧结果未覆盖。

## 4. 研究判断与下一步

本轮已完成批准的限定核验：旧边界处理确有不足，修正后仍未恢复困难源。剩余问题更值得从观测驱动初始化和进入的位置区域追查，而不是继续拉长优化预算。

建议下一项仅做保存案例的源分量归因：固定原观测得到的首源位置/拟合算子，检查扣除时是否同时损失第二源信息、剩余反射如何影响第二源评分。可以在生成端用已知分量作事后分解，但不得将这些分量提供给盲算法，也不得把此诊断当算法成绩。普通与困难两构型配对，不额外挑失败数据、不重新扫描模块或参数。

先得到具体机制，再决定一个必要改动。当前未达到规模仿真准入，也未确认论文创新；不恢复Xia调参或墙面路线。

## 5. 文件

[运行协议](protocol.json)、[完整结果](result.json)、[只读复核](review.json)、[源码快照](source/boundary_update.py)。输入及起点逐项指向上一轮冻结目录。本轮未修改论文、Zotero或原初始化算法。
'''.replace('TABLE', '\n'.join(lines))
    (out/'边界更新核验报告.md').write_text(summary, encoding='utf-8')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>边界更新核验</title><style>body{background:#f4f5f0;color:#183742;font:17px/1.8 system-ui,"Microsoft YaHei",sans-serif;margin:0}main{max-width:1000px;margin:auto;padding:40px 24px}h1{font-size:34px;line-height:1.4}h2{font-size:23px}section,.lead{border:1px solid #d8e2df;border-radius:16px;background:white;padding:24px;margin:22px 0}.lead{background:#e7f0eb}small{color:#60757b}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{text-align:left;border-bottom:1px solid #dce4df;padding:10px}.scroll{overflow:auto}a{color:#00757b}details{margin:12px 0}summary{cursor:pointer;font-weight:600}</style><main><small>2026-10-28 · 四条原观测、十二个固定起点 · 未新增全域搜索</small><h1>让更新沿约束边界走，<br>能救回那个偏了50米的源吗？</h1><div class="lead"><strong>本轮答案：还不能。</strong><p>边界不再把更新直接卡住，拟合目标明显下降，但困难源仍偏约49.75m。仅修改边界更新不足以恢复这组起点，不继续追加迭代。</p></div>
<section><h2>只改变更新方式</h2><p>原方式：先求一个步，越界就拒绝。<br>本轮：求步时纳入功率和时延约束，再检查原目标是否下降。</p><p>观测、候选、路径初值、物理先验及60步外层预算保持不变。没有用真位置挑分支，没有新增直达功率优势。</p></section>
<section><h2>结果没有形成新的定位恢复</h2><div class="scroll"><table><tr><th>案例</th><th>旧RMSE/m</th><th>新RMSE/m</th><th>新源A/B误差/m</th></tr>ROWS</table></div><small>L0/L1为两个构型，P0/P2为无反射/两反射；单条观测的两源误差统计，不是独立重复置信区间。</small></section>
<section><h2>这轮排除了哪一种简单解释？</h2><p>困难分支目标下降21.60%，投影梯度映射从0.00560减至约3.12×10⁻⁷，但困难源位置基本不动。此前边界处理不足确实存在，却不是靠修好它就能恢复位置。</p><p>终点仍略高于1×10⁻⁷停止门槛、达到60步上限。因此不声称已充分收敛，更不声称证明不可辨识。</p><details><summary>是否掩盖了未成功的数值计算？</summary><p>没有。600次子问题中21次非成功返回，全部在一个未选中的错误分支；最大闭包修正0.2963，不能称舍入误差。困难选中分支有25次投影梯度替代步。所有接受点均合法且降低原目标，异常与库警告均保存。</p></details></section>
<section><h2>下一步回到起点形成机制</h2><p>建议固定原首源拟合，检查：扣除第一个源时，是否连带削弱了第二源？剩余反射又把第二源评分推向哪里？先对保存案例作分量归因，不再扩大搜索或拉长迭代。</p><small>生成真分量只用于事后诊断，不提供给盲算法。本轮局部更新9.73秒，不含前端，不能拿来与55.82秒完整流程比较。</small></section>
<footer><a href="边界更新核验报告.md">完整报告</a> · <a href="protocol.json">协议</a> · <a href="result.json">原始结果</a> · <a href="review.json">复核记录</a></footer></main></html>'''.replace('ROWS', '\n'.join(htmlrows))
    (out/'边界更新核验报告.html').write_text(page, encoding='utf-8')
    for name in ['边界更新核验报告.md', 'protocol.json', 'result.json', 'review.json']:
        assert (out/name).is_file()
    print('audit',len(checks),'passed',all(checks.values()))
    print(out/'边界更新核验报告.html')


if __name__ == '__main__':
    main()
