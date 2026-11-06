"""Audit and summarize the frozen reconstruction pilot, including missingness."""
import csv
import json
from itertools import permutations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pilot import sha, write, observe, select_output

ROOT = Path(__file__).resolve().parent
RUN = ROOT/'pilot_20261027_170421_805496'


def main():
    result = json.loads((RUN/'result.json').read_text(encoding='utf-8'))
    manifest = json.loads((RUN/'preflight.json').read_text(encoding='utf-8'))
    checks = dict(result['checks'])
    for p, digest in manifest['sources'].items():
        checks['snapshot/'+Path(p).name] = sha(RUN/Path(p).name) == digest
    inputs = {i: dict(np.load(RUN/f'input_{i:02d}.npz')) for i in range(12)}
    for r in result['rows']:
        name = f'r{r["realization"]:02d}_snr{r["snr_db"]:g}_sir{r["sir_db"]:g}_G{r["paths"]}'
        rec = json.loads((RUN/(name+'.json')).read_text(encoding='utf-8'))
        z = inputs[r['realization']]
        checks[name+'/input'] = sha(RUN/f'input_{r["realization"]:02d}.npz') == r['input_sha']
        y, _, _ = observe(z, r['snr_db'], r['sir_db'], r['paths'])
        import hashlib
        checks[name+'/observation'] = hashlib.sha256(y.tobytes()).hexdigest() == r['observation_sha']
        selected, round_id, retained = select_output(rec['output'])
        checks[name+'/selection'] = selected == rec['selected_positions'] and round_id == r['selected_round'] and retained == r['retained_previous']
        if selected is not None:
            p = np.asarray(selected)
            candidates = [np.sum((p[list(order)]-z['truth'])**2, axis=1) for order in permutations(range(2))]
            best = min(candidates, key=lambda a: a.sum())
            checks[name+'/matching'] = bool(np.allclose(np.sqrt(best), r['evaluation']['error_m']))
        checks[name+'/snr'] = abs(r['target_los_snr_db']-r['snr_db']) < 1e-10
    assert all(checks.values()), [k for k,v in checks.items() if not v]
    write(ROOT/'audit.json', {'all_pass': True, 'count': len(checks), 'checks': checks,
                             'result_sha': sha(RUN/'result.json'), 'report_scope': '12-realization pilot'})
    groups = []
    rows = result['rows']
    for g, snr, sir in sorted(set((r['paths'], r['snr_db'], r['sir_db']) for r in rows)):
        sub = [r for r in rows if (r['paths'], r['snr_db'], r['sir_db']) == (g,snr,sir)]
        good = [r for r in sub if r['evaluation']['complete']]
        error = np.asarray([r['evaluation']['error_m'] for r in good])
        rms = np.sqrt(np.mean(error**2, axis=0)) if len(error) else [None,None]
        groups.append({'G': g, 'SNR_db': snr, 'SIR_db': sir, 'total': len(sub),
                       'complete': len(good), 'terminal_complete': sum(r['terminal_complete'] for r in sub),
                       'target_conditional_RMSE_m': rms[0], 'other_conditional_RMSE_m': rms[1],
                       'both20_count': sum(r['evaluation']['both20'] for r in sub),
                       'mean_seconds': float(np.mean([r['seconds'] for r in sub])),
                       'received_LOS_SIR_min_db': min(r['actual_los_received_sir_db'] for r in sub),
                       'received_LOS_SIR_max_db': max(r['actual_los_received_sir_db'] for r in sub)})
    write(ROOT/'summary.json', groups)
    with (ROOT/'summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(groups[0]))
        writer.writeheader()
        writer.writerows(groups)
    plt.rcParams.update({'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for sir, color, marker in [(-3., '#007d7b', 'o'), (3., '#bd632f', 's')]:
        sub = [r for r in groups if r['G']==3 and r['SIR_db']==sir]
        x = np.array([r['SNR_db'] for r in sub])
        axes[0].plot(x, [r['complete'] for r in sub], color=color, marker=marker, label=f'Tx SIR {sir:+g} dB')
        axes[1].scatter(x, [r['target_conditional_RMSE_m'] for r in sub], color=color, marker=marker, label=f'Tx SIR {sir:+g} dB')
        for r in sub:
            axes[1].annotate(f'n={r["complete"]}', (r['SNR_db'], r['target_conditional_RMSE_m']),
                             xytext=(4, 7), textcoords='offset points', fontsize=8)
    axes[0].set(ylabel='Complete outputs / 12 trials', ylim=(-.3,12.5), yticks=range(0,13,2),
                title='Output availability (not success rate)')
    axes[1].set(ylabel='Target RMSE on available outputs (m)', title='Conditional only; missing trials excluded')
    axes[1].set_ylim(800, max(r['target_conditional_RMSE_m'] for r in groups)*1.12)
    for ax in axes:
        ax.set_xlabel('Target mean received LOS SNR (dB)')
        ax.set_xticks([-10,5,20])
        ax.set_xlim(-13, 24)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Disclosed reconstruction pilot — G=3, 12 shared realizations; not author curves', fontsize=12)
    fig.savefig(ROOT/'pilot_summary.png', dpi=200)
    fig.savefig(ROOT/'pilot_summary.pdf')
    plt.close(fig)
    table = ''.join(f'<tr><td>{r["G"]}</td><td>{r["SNR_db"]:g}</td><td>{r["SIR_db"]:+g}</td><td>{r["complete"]}/{r["total"]}</td>'
                    f'<td>{r["target_conditional_RMSE_m"]:.1f}</td><td>{r["other_conditional_RMSE_m"]:.1f}</td></tr>' for r in groups)
    available = sum(r['evaluation']['complete'] for r in rows)
    terminal = sum(r['terminal_complete'] for r in rows)
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Xia公开信息重建</title>
<style>body{{max-width:1080px;margin:auto;padding:36px 22px;background:#f5f4ee;color:#203c45;font:16px/1.85 "Microsoft YaHei",system-ui}}section{{background:white;border:1px solid #dde5e1;border-radius:15px;padding:22px 28px;margin:20px 0}}h1{{font-size:32px;line-height:1.4}}h2{{font-size:23px}}.cards{{display:flex;flex-wrap:wrap;gap:14px}}.cards div{{flex:1;min-width:200px;background:#edf4ef;padding:15px;border-radius:10px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:10px;border-bottom:1px solid #dce4e0;text-align:left}}.scroll{{overflow:auto}}.note{{background:#fff0d7;border-left:4px solid #bd9549;padding:14px}}a{{color:#087b73}}img{{width:100%}}.muted{{color:#667a80}}summary{{cursor:pointer;font-weight:bold}}</style>
<p class="muted">2026-10-27 · 用户授权补全不确定项 · 固定约定v1 · 有噪短测</p><h1>不再等待作者代码，<br>用一套有依据、可重跑的约定核验</h1>
<section><div class="cards"><div><b>已完成</b><p>12个独立基础实现、84次求解，约{result['seconds']:.1f}秒。配置事先固定。</p></div><div><b>当前结果</b><p>原终态完整输出{terminal}/84；保留最近有效估计后{available}/84有双源输出。</p></div><div><b>结论边界</b><p>这套重建未复现论文曲线，不能把成绩当作作者方法的能力上限。</p></div></div></section>
<section><h2>不确定项怎样定下来？</h2><table><tr><th>项目</th><th>本次固定处理</th></tr><tr><td>BPSK</td><td>矩形脉冲、每符号5点；无RRC、Hann或额外带限</td></tr><tr><td>SNR</td><td>目标LOS五站平均接收功率 / 公共噪声功率；不按混合信号重标噪声</td></tr><tr><td>SIR</td><td>发射功率比；实际接收功率比保留几何与阴影影响</td></tr><tr><td>信道</td><td>原COST231、表I幅度；独立均匀常相位，不额外加Rayleigh幅度起伏</td></tr><tr><td>时延</td><td>式(4)频域相位；明确为循环频域重建，不伪装成非周期记录</td></tr><tr><td>部署</td><td>原区域内均匀抽取，参考站原点；不按成绩筛选</td></tr><tr><td>输出</td><td>后轮失效保留最近完整估计；首轮失败仍缺失，不给Top2兜底</td></tr></table><p>这些补全都有明确记录，不等于作者已确认。原评分、阈值、粒子数、聚类宽度、重采样方差均没有按结果调整。</p></section>
<section><h2>噪声标定为什么这样选？</h2><p>把两源、各站所有通道乘同一个尺度，令目标LOS的五站平均功率为1，再设置噪声方差σ²=10<sup>−SNR/10</sup>。这等价于在原绝对单位按目标LOS功率设置噪声，不改变两源或各站的相对增益。</p><p>改变SIR只改变另一源功率；增加反射也不增加噪声。它避免了“干扰越强，噪声也跟着变强”的混杂。</p><p class="muted">这是给定部署下的受控接收SNR实验，不代表固定热噪声下的全部距离损耗影响。保留原发射SIR，不能把发射SIR直接当每站接收SIR。</p></section>
<section><h2>短测实际表现</h2><p>G=3为直达加表I前两条反射；SNR取−10、5、20dB，SIR取−3、+3dB。另有G=1、5dB/+3dB直达控制。跨条件复用12个基础实现，但每个基础实现独立生成部署、符号、信道和噪声。</p>
<img src="pilot_summary.png" alt="输出可用数及仅可用样本的条件目标RMSE"><p class="note">有输出不等于定位成功。右图和下表的RMSE只针对有完整输出的子集，不是全样本RMSE，不能与作者曲线直接比较。</p>
<div class="scroll"><table><tr><th>G</th><th>SNR/dB</th><th>发射SIR/dB</th><th>有输出</th><th>目标条件RMSE/m</th><th>另一源条件RMSE/m</th></tr>{table}</table></div>
<p>所有原终态都因阈值清空或第二簇缺失而未完成；25个保护输出只是回到较早完整轮次，并没有改善求解轨迹。没有把其余缺失样本捏造成巨大误差或静默删掉后称总体RMSE。</p></section>
<section><h2>现在能判断什么？</h2><p><b>可以：</b>通行处理约定已形成可执行版本，原参数有噪小批已实际运行；该版本仍与作者报告明显不一致。</p><p><b>不可以：</b>把不一致直接认定为Xia算法本身的普遍缺陷，或据此证明新方法创新。也没有理由继续跑1000次只是把同样问题统计得更精细。</p><p>下一步若继续，应限定为核对从式(9)评分到式(13)—(16)输出的完整链路，必要的常规输出适配单独登记，不更换信道、调整SNR定义或筛选布局求成功。本轮没有自动增加这一算法修改。</p></section>
<section><h2>依据与复现入口</h2><p><a href="实现约定.md">完整实现约定</a> · <a href="pilot.py">短测入口</a> · <a href="{RUN.name}/result.json">逐例结果</a> · <a href="audit.json">{len(checks)}项审计</a> · <a href="summary.csv">统计CSV</a> · <a href="pilot_summary.pdf">图表PDF</a></p><p><a href="https://www.mathworks.com/help/comm/ref/rectpulse.html">矩形脉冲</a> · <a href="https://www.mathworks.com/help/comm/ref/awgn.html">AWGN功率标定</a> · <a href="https://www.mathworks.com/help/comm/ug/fading-channels.html">多径模型约定</a> · <a href="https://www.dsprelated.com/freebooks/mdft/Shift_Theorem.html">DFT移位定理</a> · <a href="https://doi.org/10.1109/JSYST.2020.3005918">Xia原论文</a></p></section>
<p class="muted">全部新数据、代码和协议有快照与哈希；实现检查不等于性能复现成功。未修改论文或Zotero，未接回50MHz场景，未启动1000次正式重复。</p></html>'''
    trace = ROOT/'trace_20261027_180535_576489/逐轮链路核查.html'
    if trace.exists():
        banner = '<section id="latest-trace"><h2>新增：逐轮评分—筛选—分组核查</h2><p><a href="trace_20261027_180535_576489/逐轮链路核查.html">查看已保存84次结果的失效环节、第一目标评价和示意图 →</a></p></section>'
        html = html.replace('<section>', banner+'<section>', 1)
    (ROOT/'公开信息重建报告.html').write_text(html, encoding='utf-8')
    md = f'''# 公开信息重建短测结论

依据用户2026-10-27授权，冻结[唯一实现约定](实现约定.md)，不再等作者代码。以通行矩形BPSK、受控目标五站平均LOS接收SNR、原发射SIR、原表I幅度及独立静态相位、原尺度随机几何、式(4)精确频域模型完成有噪短测。不是作者代码或其曲线的独立复现。

结果目录：{RUN.name}。12个基础实现、84次求解，{result['seconds']:.3f}秒；前置2次工程短测不计入84次统计。15项预检和{len(checks)}项离线审计通过。

原终态完整输出{terminal}/84；最近完整输出保护后{available}/84有双源输出，均未达到双源20m。原终态77例EMPTY_SECOND_CLUSTER、7例NO_RETAINED_PARTICLE。保留较早输出并不解决轨迹失效，不依据真值选择轮次。

条件RMSE及各组分母见[CSV](summary.csv)。例如G3/SNR5dB/SIR+3dB有5/12输出，目标/另一源条件RMSE约1274.57/2910.84m；未输出7例，因此这些不是该条件全样本RMSE。不能凭其远差于论文直接推断作者错误或新颖贡献。

尚未达到原图线复现水平，不启动1000次正式批次、不自动变更场景/阈值。下一步如继续，只限原评分至输出链路的有依据核对/单列常规适配，不做按成绩选参数。原作者代码缺失不再是暂停理由；暂停扩大规模的原因是本实现的小批表现仍明显异常。

图表[PNG](pilot_summary.png)、[PDF](pilot_summary.pdf)、[HTML](公开信息重建报告.html)。
'''
    (ROOT/'短测结论.md').write_text(md, encoding='utf-8')
    print(json.dumps({'checks': len(checks), 'available': available, 'terminal': terminal,
                      'out': str(ROOT)}, ensure_ascii=False))


if __name__ == '__main__': main()
