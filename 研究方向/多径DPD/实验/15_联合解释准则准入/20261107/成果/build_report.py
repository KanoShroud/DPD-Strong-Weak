"""Report all closure diagnostics and a path-component counterexample."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

from probe import HERE, BASE, gate, closure


def main():
    raw=HERE.parent/'运行数据/closure18'
    rows=gate.read(raw/'rows.json');done=gate.read(raw/'completion.json')
    assert len(rows)==18
    assert all(gate.sha(Path(p))==h for p,h in {**done['source_hashes'],**done['input_hashes']}.items())
    e=gate.api.create_engine();p=np.array([-900.,800.]);false=np.array([-600.,600.])
    d=np.linalg.norm(e.stations-p,axis=1)/e.ref.C
    df=np.linalg.norm(e.stations-false,axis=1)/e.ref.C
    shift=max(d-df)+3/e.ref.B
    extra=shift+df-d
    assert extra.min()>0 and extra.max()<20e-6
    mismatch=float(np.max(abs((d+extra-(d+extra)[0])-(df-df[0]))))
    assert mismatch<1e-18
    gains=np.sqrt(.75)*np.exp(1j*np.array([.2,1.1,-.8,.7]))
    gram=gains.conj()[:,None]*gains[None,:]
    pure_phase=float(closure(gram[None])[0][0]);assert abs(pure_phase-1)<1e-13
    algebra=dict(true_position=p.tolist(),false_position=false.tolist(),reflection_power_ratio=.75,
        extra_delays_us=(extra*1e6).tolist(),TDOA_max_error_s=mismatch,phase_closure=pure_phase,
        scope='resolved_component_identity_not_full_IQ_or_estimator_performance')
    table=[]
    for r in rows:
        for q in range(2):
            table.append(dict(case=r['tag'],evaluated_source=q+1,pool_size=r['pool_size'],
                nearest_candidate_error_m=r['near_errors_m'][q],phase_rank=r['phase_rank'][q],
                coherent_product_rank=r['product_rank'][q],near_phase=r['near_phase'][q],near_product=r['near_product'][q]))
    with (HERE/'候选诊断.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    summary=dict(completion=done,rows=rows,component_counterexample=algebra,
                 decision='DO_NOT_PROMOTE_CLOSURE_SCORE_TO_LOCATOR',new_localization_runs=0)
    (HERE/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 第15轮全量诊断','',
        '全部18条已有观测。每行的两个排名按真源位置匹配，仅用于诊断，不是盲输出；候选有重复邻近点，未按成绩去重。','',
        '| 案例 | 候选数 | 近真点误差/m | 相位排名 | 相干乘积排名 |',
        '|---|---:|---|---|---|']
    for r in rows:
        lines.append(f"| {r['tag']} | {r['pool_size']} | {r['near_errors_m'][0]:.3f}, {r['near_errors_m'][1]:.3f} | {r['phase_rank']} | {r['product_rank']} |")
    lines+=['','## 保存分支的逐源对照','',
        '| 案例 | 评价源 | 盲误差/m | 控制误差/m | 盲相位分数 | 控制相位分数 | 盲乘积分数 | 控制乘积分数 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if 'oracle' not in r['states']:continue
        a,b=r['states']['blind'],r['states']['oracle']
        for q in range(2):lines.append(f"| {r['tag']} | {q+1} | {a['errors_m'][q]:.3f} | {b['errors_m'][q]:.3f} | {a['phase'][q]:.4f} | {b['phase'][q]:.4f} | {a['product'][q]:.5f} | {b['product'][q]:.5f} |")
    lines+=['',f"重算阶段耗时{done['seconds']:.3f}秒，不是完整定位耗时。没有新位置选择或MP运行。输入/源码身份通过；理想闭合、站相位不变与单频平凡闭合单元通过。",
        '',f"代数反例：每条反射功率比0.75，额外延迟微秒为{algebra['extra_delays_us']}；虚假位置时差等价误差{mismatch:.3g}秒，反射组相位分数{pure_phase:.6f}。不用于算法成绩。"]
    text='\n'.join(lines)+'\n';(HERE/'核验结果.md').write_text(text,encoding='utf-8')
    spec=importlib.util.spec_from_file_location('render',BASE/'实验/10_文献机制综合/20261107/成果/summarize.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    theory=(HERE/'推导与研究判断.md').read_text(encoding='utf-8')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>联合解释准则准入</title><style>body{background:#f3f5f1;color:#20353d;margin:0;font:16px/1.85 "Segoe UI","Microsoft YaHei",sans-serif}main{max-width:1150px;margin:auto;padding:30px}header{background:#173e47;color:white;padding:30px;border-radius:16px}h1{font-size:30px}section{background:white;border-radius:12px;padding:24px;margin:20px 0}nav{display:flex;gap:20px;flex-wrap:wrap;margin:20px 0}a{color:#086f7b}.notice{background:#fff0d5;border-left:4px solid #b58b31;padding:18px}table{width:100%;border-collapse:collapse;font-size:14px}td,th{border:1px solid #d9e2dc;padding:9px;text-align:left}th{background:#e6efeb}.scroll{overflow-x:auto}footer{font-size:13px;color:#66797e}@media(max-width:650px){main{padding:12px}section{padding:14px}}</style><main><header>多径DPD · 第15轮 · 2026-11-07<h1>一致的路径组，不一定是直达路径组</h1><p>旧MP转为精修与参照；新联合准则先通过理论和小规模准入。</p></header><nav><a href="#decision">推导与研究判断</a><a href="#data">全部结果</a><a href="候选诊断.csv">CSV</a><a href="summary.json">JSON</a></nav><p class="notice">本轮没有形成新定位器。复增益闭合未通过准入，不加权修补、不扩大Top-K、不启动数千分支试算。</p>'''
    page+='<section id="decision">'+mod.render(theory)+'</section><section id="data">'+mod.render(text)+'</section>'
    page+='<footer><a href="执行合同.md">冻结合同</a> · <a href="probe.py">诊断脚本</a> · <a href="推导与研究判断.md">推导源文档</a> · <a href="https://doi.org/10.1029/2018RS006537">闭合量文献</a> · <a href="https://radionavlab.ae.utexas.edu/images/stories/files/papers/tdoa_based_gnss_ils.pdf">GNSS关联原文</a><p>保存状态与数学反例分开报告，不把oracle排名当盲算法表现。</p></footer></main></html>'
    (HERE/'联合解释准则准入报告.html').write_text(page,encoding='utf-8')
    print(json.dumps(algebra,ensure_ascii=False))


if __name__=='__main__':main()
