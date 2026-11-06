"""Offline replay and report from saved outputs; no localization rerun."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

from preflight import write, sha

ROOT = Path(__file__).resolve().parent
CONTROL = ROOT/'control_20261027_142859_457483'
PAIR = ROOT/'pair_20261027_143405_945211'
DIAG = ROOT/'diagnosis_20261027_143129_089792'
PAPER = ROOT/'paper_scale_20261027_143232_712407'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    truth = np.array([[-260., -180.], [230., 120.]])
    audits, rows = [], []
    for folder in [CONTROL, PAIR]:
        metadata = read(folder/'preflight.json')
        audits.append(all(sha(Path(p)) == h for p, h in metadata['source'].items()))
        for path in sorted(folder.glob('*_XIA.json')) + sorted(folder.glob('*_LOS_LS.json')):
            row = read(path)
            positions = row['output']['positions']
            ids = [0, 1] if row['case'].startswith('dual') else [0] if row['case'].startswith('A') else [1]
            with np.load(folder/(row['case']+'_input.npz')) as d:
                audits.append(hashlib.sha256(d['iq'].tobytes()).hexdigest() == row['observation_sha256'])
            if positions is not None:
                p = np.array(positions)
                best = min(sum(np.sum((p[i]-truth[ids[j]])**2) for i, j in enumerate(order))
                           for order in itertools.permutations(range(len(ids))))
                audits.append(abs(np.sqrt(best/len(ids))-row['evaluation']['rmse_m']) < 1e-10)
            else:
                audits.append(row['evaluation']['rmse_m'] is None)
            rows.append(row)
    assert all(audits)
    diag = read(DIAG/'result.json')
    write(ROOT/'continuation_replay_audit.json', {'all_checks_pass': all(audits), 'checks': len(audits),
                                               'observations_and_source_identity_verified': True,
                                               'evaluation_replayed_by_enumeration': True})
    pair = [r for r in rows if r['stage'] == 'pair']
    table = []
    labels = {'A_los': '孤立A / 无多径', 'A_mp': '孤立A / 有多径',
              'B_los': '孤立B / 无多径', 'B_mp': '孤立B / 有多径',
              'dual_los': '双源 / 无多径', 'dual_mp': '双源 / 有多径'}
    for case, label in labels.items():
        a = next(r for r in pair if r['case'] == case and r['method'] == 'LOS_LS')
        b = next(r for r in pair if r['case'] == case and r['method'] == 'XIA')
        e = a['evaluation']
        xe = b['evaluation']
        xtext = f"{xe['rmse_m']:.3f} m" if xe['complete'] else b['output']['status']
        table.append((label, ' / '.join(f'{v:.3f}' for v in e['errors_m']),
                      f"{e['rmse_m']:.3f}", xtext))
    mdrows = '\n'.join('| '+' | '.join(r)+' |' for r in table)
    htmlrows = ''.join('<tr>'+''.join('<td>'+v+'</td>' for v in r)+'</tr>' for r in table)
    # Save the actual empirical threshold distributions, not a hand-drawn cartoon.
    threshold_rows = []
    for r in rows:
        if r['method'] != 'XIA':
            continue
        for s in r['output']['steps']:
            v = np.array(s['scores'])
            counts, edges = np.histogram(v, 10)
            tr = 2*edges[np.argmax(counts)+1]-v.min()
            threshold_rows.append({'stage': r['stage'], 'case': r['case'], 'round': s['round'],
                                   'threshold': tr, 'minimum': v.min(), 'maximum': v.max(),
                                   'histogram': counts, 'threshold_above_max': bool(tr > v.max()),
                                   'status': s['output']['status']})
    write(ROOT/'threshold_replay.json', threshold_rows)
    md = f'''# ≤0约定后的核验与小规模配对

## 执行结论

用户批准的≤0解释已实现；死循环问题关闭。完成7项新的数学检查、无噪控制6次、单实现有噪配对12次、原文数值尺度3次与额外粒子诊断3次，共24次求解调用。没有扩大成正式多布局性能批次。

当前Xia适配版仍有阈值筛选后无粒子/无第二簇；不能当充分适配的强基线排名。直达LS在本例有噪双源无多径/有多径均得到亚米级误差，尚未在本例建立必须新增多径算法的缺口。

## 1. 原文停止解释与实现边界

原字面审计源码和结果保留。inclusive_solver.py采用已批准≤0；空第二簇/无保留粒子如实返回，不用真值或Top2兜底。原式(9)—(13)评分/直方图阈值不变。1000粒子、3轮；共同域±1000m按原±3000m等比例缩放Δh400→133.333m、重采样标准差10→3.333m，未据成绩优化。单源是显式K=1适配。该尺度迁移不保证新宽带场景的粒子覆盖已充分。

LOS_LS是已核验未知常复增益直达模型的二维限制，不使用生成器真实增益；公共站点z=0仅实现纯二维距离，不把未知高度真值提供给算法。20m网格、6个单源粗峰连续精修、残差3分支、联合波形消元LS，300次残差调用上限。2D解析Jacobian相对有限差分误差2.27e-6；7项数值检查通过。

## 2. 无噪控制说明什么

同一非周期/Hann记录，无加性噪声也无多径。Xia单源A/B误差约1.230/5.192m，LOS_LS约0.000379/0.000338m。双源LOS_LS两源误差约0.002249/0.001582m；Xia第二轮丢失第二簇。公共有限窗仍有模型残差，因此不称记录模型严格无噪精确。

首轮粒子靠近两源的最近距离均约22.6m，第二轮仍距A最近18.1m、距B最近14.2m；但筛选后A附近100m内粒子从2个变为0，B附近177个保留。A真实位置评分约8.32e-6、B约1.77e-6，均高于第二轮阈值约1.36e-7。说明不能说真实位置本身没有高分：候选覆盖与全局阈值会共同丢失一源。真实位置评分仅在输出完成后作离线归因，未用于求解。

两个额外1000粒子种子及一次10000粒子诊断仍未输出完整双源，全部保存，不选最好的一次。20m规则网格也只保留一峰；该网格A恰在点上、B在点间，因此不能把它当连续全域最优或证明评分只有一峰。

在后续有噪单源及原文数值尺度控制中，还出现原阈值高于全部粒子最高评分的情况。均匀全域粒子到峰附近高斯重采样后，评分分布已改变，式(11)—(13)的全局直方图阈值可能删除整簇或全部点。只确认本实现的具体输出机制，不推断作者实际程序如此，也不直接称理论创新。

## 3. 四类配对：本例结果，不是Monte Carlo总体RMSE

只有一个预先固定布局和随机实现，0dB基准平均直达功率；A/B孤立分别保存，所以四类问题对应6条观测。两方法逐例使用相同IQ、Hann、频带和公共信息。A/B及混合共享原噪声、波形和信道，不重标SNR。

| 观测 | LOS_LS逐源误差 m | 本例误差RMS m | Xia输出（非排名） |
|---|---|---|---|
{mdrows}

单实现的误差RMS不是跨随机重复的统计RMSE。缺失输出不填造巨大距离，也不从总体分母中偷偷删除；当前没有足够完整输出可给Xia可信总体RMSE。

LOS_LS单源均梯度停止；有噪双源分别ftol停止，梯度无穷范数4.29e-7、2.06e-6，未达到1e-7梯度条件。全部保留原生状态，不称所有求解达到共同梯度门槛。耗时见原始结果，此轮不主张公平加速优势。

## 4. 原文数值尺度核验的限度

另外运行±3000m、1MHz采样、200kbps BPSK、2000点、1000粒子、Δh400m、Γ=100m²、原式全FFT及矩形记录的3个无噪LOS控制。按式(17)COST231及5dB阴影幅度；未记载的具体站/源坐标、RRC成形和路径相位已显式选定。仍出现阈值清空/第二簇丢失。

这不是原论文图线复现：没有原始代码和种子、采用公开说明的补全选择、未跑其1000次Monte Carlo；全FFT/矩形记录也与共同Hann带限不同，不拿其误差和共同场景直接比较。不能据此宣称原论文结果错误。

## 5. 研究判断与下一步

本开发信道的额外时延0.2—2us，相当于带宽时延积BΔτ=2.5—25；LOS存在且反射相对幅度-7/-15dB。本例并不是对未分辨近延迟多径的检验，因此“普通LOS模型也能定位”不能推广成所有多径都容易。也不因为本例容易就自动缩短时延、增大反射直到制造失败。

先不再调ε、粒子数、阈值或修改场景。建议下一项只做：保持Xia评分不变，单列常规局部峰提取的适配诊断，判断评分能否稳定保留双源信息；不冒称论文原版，不称新算法。与此同时保留现有直达LS配对框架，待参照稳定后才决定多布局/正常参数范围的比较。此建议尚未实施。

## 证据

- [无噪控制]({CONTROL.name}/result.json)
- [有噪配对]({PAIR.name}/result.json)
- [候选/阈值归因]({DIAG.name}/result.json)
- [原文尺度核验]({PAPER.name}/result.json)
- [独立评价/身份重放](continuation_replay_audit.json)
- [阈值重放](threshold_replay.json)
- [已批准实施协议](继续执行协议.md)
- 原文：[Xia 2021, §IV—V, pp.2856—2857](https://doi.org/10.1109/JSYST.2020.3005918)
'''
    (ROOT/'继续核验结论.md').write_text(md, encoding='utf-8')
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>双源多径：继续核验</title>
<style>body{{max-width:1040px;margin:auto;padding:38px 20px;background:#f6f5ef;color:#203a45;font-family:"Microsoft YaHei",system-ui;line-height:1.85}}h1{{font-size:32px}}h2{{font-size:22px}}section{{background:white;border:1px solid #dce5e3;border-radius:14px;padding:22px 28px;margin:20px 0}}a{{color:#087784}}.hint{{background:#fff3db;border-left:4px solid #aa7c28;padding:14px}}table{{width:100%;border-collapse:collapse;font-size:15px}}td,th{{border-bottom:1px solid #e1e7e5;text-align:left;padding:10px}}.scroll{{overflow:auto}}.muted{{color:#65757c}}.flow{{display:flex;gap:12px;flex-wrap:wrap}}.flow div{{background:#edf5f2;border-radius:10px;padding:15px;flex:1;min-width:180px}}summary{{cursor:pointer;font-weight:bold}}</style>
<p class="muted">2026-10-27 · 已批准≤0解释 · 开发诊断，不是正式排名</p><h1>停止规则修清了，<br>但参照算法还不能直接进入排名</h1>
<section><p>本轮完成 <strong>24次求解调用</strong>：无噪控制、有噪配对、原文数值尺度及有限覆盖诊断。没有启动大规模实验。</p><div class="flow"><div><b>已解决</b><br>“累计增量&lt;0”无法结束的问题。</div><div><b>已发现</b><br>阈值可能删除一源的整簇，甚至所有候选。</div><div><b>尚未证明</b><br>新多径算法的必要性与有效贡献。</div></div></section>
<section><h2>为什么仍然没有第二个位置？</h2><p>先在全区域撒点，再在高分位置附近重新撒点。这个过程改变了评分的分布：原来大多数点得分低，后来多数点已经接近峰。</p><p>原文仍按每轮评分直方图重算一个全局阈值。本次看到两种结果：一整簇被删掉；或者阈值比最高分还高，所有点都被删掉。</p><p class="hint">这是对当前明确化实现的诊断，不是在证明原论文无效。不能把缺失输出记成一个巨大定位误差后宣布我们获胜。</p><details><summary>展开：无噪双源中实际发生了什么？</summary><p>首轮两源附近都保留了候选；第二轮A附近100m内从2点变为0点，B附近保留177点。两个真实位置的评分都高于该轮阈值，但粒子没有足够接近它们。</p><p>另外两个1000粒子种子和一次10000粒子仍未给出完整双源。真实位置只用于事后归因，没有喂给定位器，也没有选最好种子作为结果。</p></details></section>
<section><h2>共同输入下：这一个例子有多难？</h2><p>普通未知增益直达LS在本例的双源误差RMS，从无多径约<strong>0.350m</strong>变为有多径约<strong>0.437m</strong>。这说明本例有小幅精度变化，但尚未显示必须研发新方法的困难。</p><div class="scroll"><table><tr><th>条件</th><th>LS逐源误差 / m</th><th>本例RMS / m</th><th>Xia输出状态</th></tr>{htmlrows}</table></div><p class="muted">一个布局、一个随机实现；本例RMS不等于Monte Carlo RMSE。缺失输出不编造误差。这张表用于诊断，不作为充分适配后的方法排名。</p></section>
<section><h2>为什么不能现在说“多径没问题”？</h2><p>本次反射额外时延为0.2—2μs，带宽12.5MHz，BΔτ为2.5—25；直达存在、反射较弱。因此没有检验更难分辨的近延迟多径。</p><p>我们不会为了让算法失败而临时改变这些参数。需要先稳定近邻参照，再按有依据的信道范围设计比较。</p></section>
<section><h2>下一步只建议做一件事</h2><p>固定Xia评分，用常规空间局部峰提取单列一个适配诊断，确认问题在评分还是输出。保留原文版本，不把常规修正包装成创新；不继续扫阈值、粒子数或墙面参数。</p><p class="muted">这项建议尚未执行。本轮7项新数学检查、源码/输入身份与独立匹配重放通过；旧字面审计全部保留。</p></section>
<p><a href="继续核验结论.md">完整说明与数值边界</a> · <a href="{PAIR.name}/result.json">有噪配对</a> · <a href="{CONTROL.name}/result.json">无噪控制</a> · <a href="{DIAG.name}/result.json">归因证据</a> · <a href="{PAPER.name}/result.json">原文尺度控制</a></p></html>'''
    (ROOT/'继续核验报告.html').write_text(html, encoding='utf-8')
    print(json.dumps({'checks': len(audits), 'report': str(ROOT/'继续核验报告.html'),
                      'control_and_pair_rows': len(rows), 'dense_truth_scores': diag['truth_score']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
