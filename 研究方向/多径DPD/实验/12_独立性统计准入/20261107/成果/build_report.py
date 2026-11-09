"""Render saved diagnostics; all oracle quantities stay evaluation-only."""
import csv
import hashlib
import html
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[3]
RAW = HERE.parent/'运行数据/saved30'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def main():
    rows = read(RAW/'rows.json')
    done = read(RAW/'completion.json')
    assert len(rows) == done['count'] == 30
    for p, digest in {**done['source_hashes'], **done['input_hashes']}.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest, p
    # Exact population enumeration: independent signs at each deterministic phase.
    signs = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
    a = np.sqrt(np.array([1.5,.5]))[:,None,None]*signs[None,:,:]
    pwr = np.mean(a*a,axis=1)
    fourth = np.mean(a[:,:,0]**2*a[:,:,1]**2,axis=1)
    naive = float(fourth.mean()-pwr[:,0].mean()*pwr[:,1].mean())
    conditional = fourth-pwr[:,0]*pwr[:,1]
    assert abs(naive-.25)<1e-14 and np.max(abs(conditional))<1e-14
    rng = np.random.default_rng(2026110714)
    h = rng.normal(size=(64,4,2))+1j*rng.normal(size=(64,4,2))
    s = rng.normal(size=(64,2))+1j*rng.normal(size=(64,2))
    d = np.array([2.,.4])[None,:]*np.exp(1j*np.arange(64)[:,None]*np.array([.13,-.17]))
    y = np.einsum('fmq,fq->fm',h,s)
    yp = np.einsum('fmq,fq->fm',h/d[:,None,:],s*d)
    relative = float(np.linalg.norm(y-yp)/np.linalg.norm(y))
    assert relative < 1e-14
    table=[]
    for r in rows:
        c = r['waveform']['matched_correlation']
        table.append(dict(case=r['tag'],arm=r['arm'],position_rmse_m=r['position_rmse'],
            matched_corr_1=c[0],matched_corr_2=c[1],energy_peak_hz=r['energy_cycle_peak_hz'],
            naive_mixed_cumulant=r['stats']['naive_mixed_cumulant'],
            truth_naive_mixed_cumulant=r['truth_stats']['naive_mixed_cumulant'],normal=r['normal']))
    with (HERE/'逐状态诊断.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    # Declared 1 FFT-bin proximity is evaluation, never a detection threshold.
    groups=[]
    for cond in ['L0','L1_P075','L2_P075']:
        rr=[r for r in rows if r['arm']=='blind' and r['condition']==cond]
        groups.append(dict(condition=cond,n=len(rr),within_one_bin_of_generator_rate=sum(
            abs(r['energy_cycle_peak_hz']-10e6)<=50e6/8192 for r in rr)))
    summary=dict(groups=groups,states=table,completion=done,
        algebra=dict(naive_independent_cycle_cumulant=naive,conditional_cumulant=conditional.tolist(),
                     diagonal_channel_waveform_invariance_relative_error=relative),
        scope='saved_state_and_mathematical_diagnostics_not_new_localization',
        gate='HOLD_GENERIC_ICA_AND_NAIVE_CUMULANT_SELECTION_NO_FORMAL_BATCH')
    (HERE/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 独立性统计准入：结果','',
        '同一批18条旧观测，重算18个盲终态与12个近真候选控制终态，共30个状态。没有重新定位，没有修改公共算法，没有使用独立重复。近真控制含真值选候选，不是可部署方法。','',
        '| 条件 | 观测数 | 最大能量谱峰落在真实符号率附近（±1频率格） |',
        '|---|---:|---:|']
    for g in groups: lines.append(f"| {g['condition']} | {g['n']} | {g['within_one_bin_of_generator_rate']}/6 |")
    lines += ['', '这里的10MHz来自生成器，只用于探测结果的事后评价；算法探测时不知道它。只试了预登记的全频带最大峰，未调阈值、窗口、峰数或频段。它不是完整符号率估计器，更不是非共轭频偏检测器。','',
        '## 所有状态（不删除失败案例）','',
        '| 案例 | 状态 | 位置RMSE/m | 波形相关1 | 波形相关2 | 普通混合四阶量 |',
        '|---|---|---:|---:|---:|---:|']
    for r in table:
        arm='盲终态' if r['arm']=='blind' else '近真候选控制'
        lines.append(f"| {r['case']} | {arm} | {r['position_rmse_m']:.4f} | {r['matched_corr_1']:.4f} | {r['matched_corr_2']:.4f} | {r['naive_mixed_cumulant']:.5f} |")
    lines += ['', '波形真值代理是参考站无噪直达分量，采用相同频带；幅相、源置换和循环整数时移消歧。相关性并非去噪后SINR、独立性检验或任意滤波歧义下的恢复保证。完整字段、状态正常性与相关矩阵保留在CSV和原始JSON；控制与盲解可以交换源序。','',
        '## 核心发现','',
        '- IN102双反射：1931.160m错误解的匹配相关为0.6071/0.1250，近真控制为0.9272/0.9466；存在明显分源损失，但本轮没有判定其由混合、信道误差和噪声各占多少。',
        '- IN105单反射：47.052m错误解相关为0.9122/0.9382，0.326m控制为0.9138/0.9397；波形接近不等于位置接近。不能只用波形恢复质量替代位置目标。',
        '- IN100双反射：28.945m错误解的普通混合四阶量绝对值小于0.316m控制；“越接近零位置越好”并未成立。这是描述性反例，不是筛选器性能实验。','',
        '## 正确性与成本','',
        f"精确符号枚举：独立但功率共同周期变化时，普通混合量为{naive:.2f}，按相位计算为0。随机代数回代：逐源可逆幅相/时移变换的观测相对差{relative:.3g}。这些是数学单元检查，不是新定位结果。",
        f"保存状态重算阶段墙钟{done['seconds']:.3f}秒，包含本轮输入复制等操作；不能当作定位算法耗时。输入、原求解器源码和冻结诊断脚本身份前后核对通过。",
        '本轮没有证明统计盲分离无效，也没有形成理论创新或新定位器。停止直接ICA/普通四阶选分支的迁移；下一步必须明确几何—通道歧义如何处理。']
    text='\n'.join(lines)+'\n'
    (HERE/'核验结果.md').write_text(text,encoding='utf-8')
    path=BASE/'实验/10_文献机制综合/20261107/成果/summarize.py'
    spec=importlib.util.spec_from_file_location('renderer',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    theory=(HERE/'理论边界.md').read_text(encoding='utf-8')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>独立性统计准入报告</title><style>
body{background:#f3f5f1;color:#20353d;font:16px/1.85 "Segoe UI","Microsoft YaHei",sans-serif;margin:0}main{max-width:1100px;margin:auto;padding:32px}header{background:#173f49;color:white;padding:28px;border-radius:16px}h1{font-size:30px}h2{margin-top:36px}section,details{background:white;padding:24px;border-radius:12px;margin:20px 0}summary{cursor:pointer;font-weight:600}a{color:#086b78}nav{display:flex;gap:20px;flex-wrap:wrap;margin:22px 0}.note{background:#fff0d5;border-left:4px solid #bf902f;padding:18px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #d7e1dc;padding:9px;text-align:left}th{background:#e4efeb}.scroll{overflow-x:auto}pre{background:#edf2f1;padding:15px;white-space:pre-wrap}.flow{display:flex;gap:12px;flex-wrap:wrap}.flow div{background:#e6efec;flex:1;padding:16px;min-width:190px}footer{font-size:13px;color:#637679}@media(max-width:600px){main{padding:12px}section{padding:14px}}</style><main>
<header>多径DPD · 第12轮 · 2026-11-07<h1>分开两个源，还不等于找对两个位置</h1><p>30个保存状态核验＋统计定义检查。没有改场景，没有重跑定位。</p></header>
<nav><a href="#result">结果与全部案例</a><a href="#theory">理论边界与文献</a><a href="逐状态诊断.csv">下载逐状态表</a><a href="summary.json">复现指标与身份</a></nav>
<div class="flow"><div>第一关：观测是否能提供所需统计特征？<br>最简单的能量峰规则尚不可靠。</div><div>第二关：分源改善是否一定带来定位改善？<br>存在波形相近、位置明显不同的解。</div><div>第三关：是否启动完整新算法？<br>暂不。先补足几何—通道歧义处理。</div></div>
<p class="note">没有否定统计独立性方向，也没有宣称找到新方法。不能把近真控制或已知真实波形的相关性作为盲定位成绩。</p>'''
    page+='<section id="result">'+mod.render(text)+'</section><section id="theory">'+mod.render(theory)+'</section>'
    page+='<footer><a href="执行合同.md">预登记合同</a> · <a href="probe.py">冻结重算脚本</a> · <a href="理论边界.md">推导源文档</a> · <a href="https://arxiv.org/html/1102.2706v1">Florian等作者全文</a><p>证据：'+html.escape(str(RAW))+'</p></footer></main></html>'
    (HERE/'独立性统计准入报告.html').write_text(page,encoding='utf-8')
    print(json.dumps(dict(groups=groups,algebra=summary['algebra']),ensure_ascii=False))


if __name__=='__main__':main()
