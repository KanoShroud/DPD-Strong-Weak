"""Frozen one-change pilot review; no solver calls."""
from pathlib import Path
import argparse
import json
import numpy as np
from structural_probe import sha, write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
    out=parser.parse_args().directory.resolve(strict=True)
    data=json.loads((out/'result.json').read_text(encoding='utf-8'))
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    oldroot=Path(protocol['input_root'])
    old=json.loads((oldroot/'result.json').read_text(encoding='utf-8'))
    prefix=json.loads((out.parent/'first_refine_20261028_231012_716206/result.json').read_text(encoding='utf-8'))
    checks={'hash/'+p:sha(p)==h for p,h in protocol['hashes'].items()}
    original_protocol=json.loads((oldroot/'protocol.json').read_text(encoding='utf-8'))
    checks['original_truth_input_identity']=all(sha(p)==h for p,h in original_protocol['hashes'].items())
    checks['matrix']=data['passed'] and data['complete'] and len(data['rows'])==4
    checks['prefix_output']=np.array_equal(prefix['rows'][0]['output'],data['rows'][0]['output'])
    checks['prefix_costs']=[b['final']['cost'] for b in prefix['rows'][0]['branches']]==[b['final']['cost'] for b in data['rows'][0]['branches']]
    for row in data['rows']:
        checks[row['case']+'/chosen']=row['chosen']==min(range(3),key=lambda i:row['branches'][i]['final']['cost'])
        checks[row['case']+'/finite']=all(np.all(np.isfinite(b['final']['p'])) and np.all(np.isfinite(b['final']['v'])) for b in row['branches'])
    before=next(r for r in old['rows'] if r['case']=='L1_P0')['pairs'][0]
    after=next(r for r in data['rows'] if r['case']=='L1_P0')['pairs'][0]
    checks['regression_same_second_grid_point']=np.array_equal(before['coarse'][1],after['coarse'][1])
    review=dict(checks=checks,passed=bool(all(checks.values())),
        source_sha=sha(Path(__file__)),regression_old_pair=before,regression_new_pair=after,
        first_refinement_seconds=sum(t['fit']['seconds'] for r in data['rows'] for t in r['first']),
        measured_pipeline_seconds=sum(r['seconds'] for r in data['rows']),
        first_gradient_not_met=[dict(case=r['case'],first=t['index'],gradient=t['fit']['gradient_inf']) for r in data['rows'] for t in r['first'] if t['fit']['gradient_inf']>1e-7])
    write(out/'review.json',review);assert review['passed']
    lines=['| 案例 | 原双源RMSE/m | 首源预精修后RMSE/m | 最终源A/B误差/m |','|---|---:|---:|---|']
    hrows=[]
    for r in data['rows']:
        e=r['evaluation']['source_errors_m'];a=r['control_evaluation']['rmse_m'];b=r['evaluation']['rmse_m']
        lines.append(f'| {r["case"]} | {a:.7g} | {b:.7g} | {e[0]:.7g} / {e[1]:.7g} |')
        hrows.append(f'<tr><td>{r["case"]}</td><td>{a:.6g}</td><td>{b:.6g}</td><td>{e[0]:.6g} / {e[1]:.6g}</td></tr>')
    md='''# 首源先精修再扣除：单改动配对

2026-10-28。结论：本轮不支持将“首源先连续精修再扣除”替代原流程。普通多径结果不变，困难多径未恢复；困难构型的无反射控制反而退化。停止继续围绕这一模块调参，不将该现象归为多径特有瓶颈。

## 1. 公平性范围

四条原无噪理想频率观测、原三个首源粗候选、20m网格、100m去重规则、每首源三个第二候选、九对LOS联合拟合、保留三个多径分支全部固定。两边均使用上一轮已核验的约束边界后端；对照结果复用boundary_20261028_214016_411172，而非回退到早先越界拒绝后端。

唯一改变是在扣除前连续优化首源位置；每次试探位置重新以秩一LS求常复站增益和未知波形。目标仍为单源时延补偿相关矩阵最大特征值，不提供真实波形、位置或路径。每个粗候选独立L-BFGS-B，最多100迭代/1000函数调用、ftol=1e-10、gtol=1e-7；按观测目标保留所见最好状态，不按定位误差保留。精修后的第一源坐标自然成为相同100m排斥规则的中心，没有额外改去重尺度。

第一源导数采用最大单特征值的解析导数，包含几何时延梯度；已与有限差分核对。全部输出完成后才做真位置匹配。第二源候选可随残差改变，这是该单一改动的下游结果，不是冻结结果后再人工选候选。

## 2. 结果

TABLE

RMSE为每个案例两个位置误差的平方均值开根号，不是多随机实现统计。无反射控制在相同数据上原本可解；新失败不涉及反射，因此不能用它证明多径信息不足。

## 3. 第一源确实变准，但不保证全流程变好

首候选0的近真位置误差变化如下：普通无反射14.14→2.51m、普通多径14.14→3.40m、困难无反射14.14→1.66m、困难多径14.14→3.53m。相应单源拟合目标均下降。

困难多径的第一源变准后，第二源候选仅重新排序，没有进入正确位置区域；最终仍选择从B附近错误局部位置开始的分支，源B误差约49.75m。此前“首源粗位置下过早扣除”不足以独自解释该例失败。

## 4. 无反射退化具体发生在哪里

对照原0_0分支与新0_0分支：

- 原第一源初值(2090,1710)m，新值(2098.9441,1701.2753)m。
- **第二源网格初值均为(2510,2170)m，没有变。**
- 第一源增益初值和残差导出的第二源增益初值随精修改变；不是只改了位置后把其他辅助状态完全保留。
- 原LOS联合拟合13次函数调用后达到梯度条件，目标约6.62e-18，两个位置正确。
- 新LOS联合拟合54次函数调用后按ftol停止，目标0.0326003，报告optimality=6.19e-5，未达到设置的gtol=1e-7；两个位置误差约1.24/23.75m。
- 该新分支并未被删除，仍保留至多径后端；后端源B仍偏21.89m。最终另一分支具有更低的观测目标，被选择后源B偏49.77m。

因此，退化不是“候选名额把原来的第二源网格点删了”。它发生在更改第一源状态及其导出的辅助初值后，联合拟合进入不同终点；其中至少有非梯度停止，不能把库的正常返回一概称为充分收敛。仅靠位置初值更接近真值，不保证波形/增益耦合求解更可靠。

## 5. 这轮能确认和不能确认的结论

可以确认：这个常规改动没有稳定收益，不作为推荐替代版本；当前初始化—辅助量—联合拟合链条对状态敏感，无反射控制也会受影响。不能继续把所有失败笼统归为多径问题。

不能确认：错误终点是不可逃逸的局部最优、模型不可辨识，或“先精修再扣除”在其他场景无用。12个首源局部优化均库正常返回，但两项没有达到单独的1e-7梯度阈值；最终只有普通无反射所选分支达到后端投影梯度条件，其余11个分支触及60步，均如实保存。

建议不再追加候选、迭代或精修模块。下一步先对当前残差评分、辅助量初始化与联合目标做一次数学一致性审查，明确哪些是同一条件最小二乘，哪些只是启发式：例如固定首源响应时，条件拟合第二源需要同时处理残差观测和第二源候选响应，不能仅凭原单源评分变高推导双源目标一定改善。这是常规估计准则检查，不预设新理论贡献。本轮未改评分，未执行新的方法组合。

如果基础无反射盲求解仍不可靠，应暂停当前原型的多径性能扩测，而不是继续把常规适配修补包装成创新。原无预精修版本保留，本轮版本作为负面配对证据归档。

## 6. 验证、成本和证据

基础5项和完整59项检查通过；另有CHECKCOUNT项只读身份/选择/重复核验通过。首源粗点由原评分重新计算完全一致，输入和依赖源码SHA不变，单例前缀与完整批首例输出及分支目标一致。py_compile和Ruff F/E9通过。检查通过不等于收敛/性能通过。

十二次首源精修本身约0.0158秒；四条处理合计52.31秒，计时不含复用的原始首源全域图，不能直接与旧55.82秒完整流程作加速比较。旧对照不同批次计时也不用于严格速度归因。

[协议](protocol.json) · [全部结果](result.json) · [退化分支复核](review.json) · [源码快照](source/first_refinement.py)。未改旧求解器、论文、Zotero，未扩大数据或追加参数扫描。
'''.replace('TABLE','\n'.join(lines)).replace('CHECKCOUNT',str(len(checks)))
    (out/'首源预精修配对报告.md').write_text(md,encoding='utf-8')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>首源预精修配对</title><style>body{margin:0;background:#f5f5ef;color:#1b3540;font:17px/1.8 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1000px;margin:auto;padding:40px 24px}h1{font-size:34px;line-height:1.4}h2{font-size:23px}section,.lead{border:1px solid #dce3de;border-radius:16px;padding:24px;margin:22px 0;background:white}.lead{background:#f2ebe0}table{width:100%;border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #dce3de;text-align:left}.scroll{overflow:auto}small{color:#60747a}a{color:#00757d}summary{cursor:pointer;font-weight:600}</style><main><small>2026-10-28 · 固定四条观测 · 只新增首源预精修 · 相同边界后端</small><h1>先把第一个位置修准，<br>并没有让整个定位流程更可靠</h1><div class="lead"><strong>本轮不采用该改动作为替代流程。</strong><p>两个多径案例基本不变；困难构型的无反射控制从正确定位退化。首源位置更接近真值，不能保证与它一起估计的波形、增益以及后续联合求解更好。</p></div>
<section><h2>结果</h2><div class="scroll"><table><tr><th>案例</th><th>原RMSE/m</th><th>预精修后RMSE/m</th><th>新源A/B误差/m</th></tr>ROWS</table></div><small>L0/L1为原两构型，P0/P2为无反射/两反射；每案例的两源误差统计，非独立重复置信区间。</small></section>
<section><h2>退化并非因为删掉了第二源网格点</h2><p>困难无反射案例中，第一源从约14.14m误差改善到1.66m。原、新0_0分支的第二源初值都是(2510,2170)m。</p><p>但两源联合拟合：原版达到正确位置，新版进入较差终点。第一源精修同时改变了增益和残差信号，后续辅助量初值也不同。</p><details open><summary>停止条件是否充分？</summary><p>原分支13次计算后达到梯度条件；新分支54次计算后按目标变化ftol停止，optimality约6.19×10⁻⁵，未达到1×10⁻⁷。不能把这个终点宣称为充分收敛。它仍被保留到多径后端，最终也未恢复。</p></details></section>
<section><h2>对研究推进的影响</h2><p>无反射控制也会出现这种退化，说明当前首先要面对初始化、辅助量与联合目标之间的可靠衔接，不能直接包装成多径特有创新问题。</p><p><strong>停止给这个模块加参数或迭代。</strong>建议下一步先审查残差评分与完整条件最小二乘的关系，区分一致的目标处理与启发式拼接，再决定是否保留这套初始化。</p></section>
<section><h2>执行边界</h2><p>原观测、三个首源粗候选、第二源网格/名额、路径初始化及后端均固定；真值只供事后评价。原版本和全部负面结果保留，未扩大实验、修改论文或Zotero。</p><small>首源预精修累计约0.0158秒；整段52.31秒不含原首源地图，不能与旧完整耗时直接比较。基础与身份检查通过不代表定位通过。</small></section>
<footer><a href="首源预精修配对报告.md">完整报告</a> · <a href="result.json">原始结果</a> · <a href="review.json">配对复核</a> · <a href="protocol.json">冻结协议</a></footer></main></html>'''.replace('ROWS','\n'.join(hrows))
    (out/'首源预精修配对报告.html').write_text(page,encoding='utf-8')
    (out/'source'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print('review checks',len(checks),review['passed']);print(out/'首源预精修配对报告.html')


if __name__=='__main__':main()
