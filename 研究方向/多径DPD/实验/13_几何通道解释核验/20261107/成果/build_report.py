"""Reproducible alias diagnostics and finite-record operator unit check."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[3]
RAW=HERE.parent/'运行数据/alias30'


def main():
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    done=read(RAW/'completion.json');rows=read(RAW/'rows.json')
    assert len(rows)==done['count']==30
    for p,h in {**done['source_hashes'],**done['input_hashes']}.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    probe_spec=importlib.util.spec_from_file_location('alias_probe',HERE/'probe.py')
    probe=importlib.util.module_from_spec(probe_spec);probe_spec.loader.exec_module(probe)
    objective_errors=[]
    for row in rows:
        folder=RAW/'inputs'/row['tag']
        solved=read(folder/(row['arm']+'.json'))
        with np.load(folder/'iq.npz',allow_pickle=False) as z:iq=z['iq']
        engine=probe.gate.api.create_engine();freq,y,nrec,_=engine.fc.cal.spectral(iq)
        x=np.asarray(solved['final']['x']);xx=x.copy()
        period=nrec/engine.ref.FS*engine.ref.B
        xx[engine.D+2::3]-=(xx[engine.D+2::3]>period/2)*period
        residual=engine.core(x,freq,y,solved['paths'])[3]
        alias_residual=engine.core(xx,freq,y,solved['paths'])[3]
        objective_errors.append(float(abs(np.linalg.norm(residual)**2-np.linalg.norm(alias_residual)**2)/np.linalg.norm(y)**2))
    assert max(objective_errors)<1e-10
    n=64;f=np.fft.fftfreq(n)
    phase_error=float(np.max(abs(np.exp(-2j*np.pi*f*(n-1))-np.exp(2j*np.pi*f))))
    rng=np.random.default_rng(2026110715)
    s=rng.normal(size=3*n)+1j*rng.normal(size=3*n)
    delayed=s[n+np.arange(n)-(n-1)];advanced=s[n+np.arange(n)+1]
    finite_difference=float(np.linalg.norm(delayed-advanced)/np.linalg.norm(delayed))
    assert phase_error<1e-12 and finite_difference>0
    simple=[{k:v for k,v in r.items() if k not in ('paths','matched_correlation')} for r in rows]
    with (HERE/'逐状态结果.csv').open('w',encoding='utf-8-sig',newline='') as fcsv:
        w=csv.DictWriter(fcsv,fieldnames=list(simple[0]));w.writeheader();w.writerows(simple)
    summary=dict(completion=done,rows=rows,alias_max_normalized_rss_difference=max(objective_errors),finite_window_unit=dict(n=n,seed=2026110715,
        phase_error=phase_error,arbitrary_finite_sequence_relative_difference=finite_difference),
        gate='HOLD_NO_NEW_DELAY_BOUND_OR_ZERO_PADDING_WITHOUT_PROTOCOL_DECISION')
    (HERE/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 第13轮保存状态核验','',
        '18条旧观测，18个盲终态和12个近真候选控制。后者含真值选分支，不能作为盲方法成绩。全部状态均列出；没有新定位求解。','',
        '| 案例 | 状态 | RMSE/m | 正常终止 | 负主值路径数 | 最大相对幅度 | 原RSS | 删除后RSS |',
        '|---|---|---:|---|---:|---:|---:|---:|']
    for r in rows:
        arm='盲' if r['arm']=='blind' else '近真控制'
        lines.append(f"| {r['tag']} | {arm} | {r['position_rmse']:.3f} | {r['normal']} | {r['negative_alias_count']} | {r['max_negative_alias_amplitude']:.3f} | {r['normalized_rss']:.5f} | {r['deletion_normalized_rss']:.5f} |")
    lines+=['','RSS统一除以观测能量。删除控制固定位置和剩余通道，仅重新消元波形；不是新定位器、受限全局最优值或因果效应估计。负主值标记不构成真实时延上界。','',
        f"将标记延迟减去T后，30个状态H最大相对差{max(r['channel_alias_relative_error'] for r in rows):.3g}，验证实际实现的周期等价性。记录长度163.84微秒。",
        '',f"独立回代核对消元目标：归一化RSS最大绝对差{max(objective_errors):.3g}。",
        '',f"独立算子单元：N=64，循环相位差{phase_error:.3g}，任意非周期测试序列的两种截取相对差{finite_difference:.3f}。它不是本项目定位场景数据。",
        '',f"阶段耗时{done['seconds']:.3f}秒，仅为诊断，不能用作完整定位成本。源码和输入身份核对通过。没有删除负面记录、调整参数或启动独立批。"]
    text='\n'.join(lines)+'\n';(HERE/'核验结果.md').write_text(text,encoding='utf-8')
    spec=importlib.util.spec_from_file_location('render',BASE/'实验/10_文献机制综合/20261107/成果/summarize.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    theory=(HERE/'分析与下一步.md').read_text(encoding='utf-8')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>几何与通道解释核验</title><style>body{margin:0;background:#f3f5f1;color:#20353d;font:16px/1.85 "Segoe UI","Microsoft YaHei",sans-serif}main{max-width:1150px;margin:auto;padding:30px}header{background:#173e47;color:white;border-radius:16px;padding:30px}h1{font-size:30px}section,details{background:white;padding:24px;border-radius:12px;margin-top:22px}summary{cursor:pointer;font-weight:600}a{color:#087080}nav{display:flex;gap:20px;flex-wrap:wrap;margin:20px 0}.notice{padding:18px;background:#fff0d6;border-left:4px solid #bc913c}.diagram{display:flex;gap:14px;flex-wrap:wrap}.diagram div{background:#e3efeb;padding:18px;flex:1;min-width:220px}table{width:100%;border-collapse:collapse;font-size:14px}td,th{border:1px solid #d6dfda;padding:9px;text-align:left}th{background:#e6eeea}.scroll{overflow-x:auto}pre{white-space:pre-wrap;background:#edf1ef;padding:18px}footer{color:#63777a;font-size:13px;margin-top:24px}@media(max-width:650px){main{padding:14px}section{padding:14px}}</style><main><header>多径DPD · 第13轮 · 2026-11-07<h1>参数写着“晚到”，模型却可能把它当成“提前”</h1><p>先核验传播解释，再决定如何使用统计独立性。原场景与公共求解器保持不变。</p></header><nav><a href="#meaning">原因与研究决定</a><a href="#data">全部30个状态</a><a href="逐状态结果.csv">CSV</a><a href="summary.json">JSON与身份</a></nav><div class="diagram"><div>真实有限记录<br>晚到163.56μs意味着读取更早的发射历史。</div><div>当前DFT模型<br>记录163.84μs，晚到163.56μs与提前0.28μs相位等价。</div><div>对定位的影响<br>错误几何可能由这样的通道分量补偿；独立性不会自动解除它。</div></div><p class="notice">不能据此直接删路径或把最大时延设成T/2。需要决定是否允许公开的短信道记忆条件，或改用显式有限记录模型。没有声称已经证明全部失败根因。</p>'''
    page+='<section id="meaning">'+mod.render(theory)+'</section><section id="data">'+mod.render(text)+'</section>'
    page+='<footer><a href="执行合同.md">执行合同</a> · <a href="probe.py">诊断入口</a> · <a href="分析与下一步.md">推导源文件</a><p>本报告区分模型性质、保存结果诊断和待审批的设计选择；不是正式新算法成绩。</p></footer></main></html>'
    (HERE/'几何与通道解释核验.html').write_text(page,encoding='utf-8')
    print('30 states; identity checks PASS; HTML written; finite-window unit',phase_error,finite_difference)


if __name__=='__main__':main()
