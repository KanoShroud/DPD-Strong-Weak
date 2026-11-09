"""Aggregate all paired outputs and independently recompute final objectives."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from run import HERE, BASE, gate, engine_for


def main():
    raw=HERE.parent/'运行数据/paired18';cold=HERE.parent/'运行数据/oracle12'
    done=gate.read(raw/'completion.json');cdone=gate.read(cold/'completion.json')
    rows=gate.read(raw/'rows.json');controls=gate.read(cold/'rows.json')
    assert len(rows)==54 and len(controls)==12
    for record in [done,cdone]:
        assert all(gate.sha(Path(p))==h for p,h in {**record['source_hashes'],**record['input_hashes']}.items())
    errors=[]
    for row in rows:
        solved=gate.read(raw/f"{row['tag']}_{row['arm']}_decision.json")
        with np.load(raw/'inputs'/row['tag']/'iq.npz',allow_pickle=False) as z:iq=z['iq']
        e=engine_for(row['arm']);f,y,n,_=e.fc.cal.spectral(iq)
        x=np.asarray(solved['final']['x']);cost,g=e.objective(x,f,y,solved['paths'])
        pg=float(np.max(abs(x-e.ref.project(x-g,n))))
        ce=abs(cost-row['cost']);pe=abs(pg-row['projected_gradient'])
        assert ce<1e-10 and pe<1e-8,(row['tag'],row['arm'],ce,pe)
        errors.append((ce,pe))
    groups=[]
    for cond in ['L0','L1_P075','L2_P075']:
        for arm in ['FULL','CAP_CONSERVATIVE','CAP_COUNTED']:
            rr=[r for r in rows if r['condition']==cond and r['arm']==arm]
            groups.append(dict(condition=cond,arm=arm,n=len(rr),rmse=float(np.sqrt(np.mean([r['rmse']**2 for r in rr]))),
                mean_seconds=float(np.mean([r['seconds'] for r in rr])),
                max_seconds=max(r['seconds'] for r in rr),normal=sum(r['normal'] for r in rr),
                mean_K=float(np.mean([r['K'] for r in rr]))))
    evidence=dict(paired=done,oracle=cdone,groups=groups,rows=rows,controls=controls,
        verification=dict(cost_max_error=max(c for c,_ in errors),PG_max_error=max(p for _,p in errors)),
        scope='same_development_data_no_independent_generalization',decision='NO_FREEZE_NO_BOUND_TUNING')
    (HERE/'summary.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    with (HERE/'配对结果.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=[k for k in rows[0] if k!='errors_m'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        w.writeheader();w.writerows(rows)
    lines=['# 第14轮：公开20微秒上界的配对结果','',
        '已完成54次盲定位（18条同观测×三臂）和12次既有近真候选控制；控制不是盲定位成绩。六构型各一个实现，不能作为独立泛化或高置信性能结论。','',
        '## 全部条件汇总','',
        '| 条件 | 版本 | RMSE/m | 平均秒数 | 最大秒数 | 正常终止/6 | 平均估计反射数 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for g in groups:
        lines.append(f"| {g['condition']} | {g['arm']} | {g['rmse']:.4f} | {g['mean_seconds']:.3f} | {g['max_seconds']:.3f} | {g['normal']}/6 | {g['mean_K']:.1f} |")
    lines+=['','FULL为原流程；CAP_CONSERVATIVE只缩小支持范围、保留原检验次数；CAP_COUNTED同时按实际候选数重算阈值。正常终止不是定位成功率，非正常但有限的位置全部计入RMSE。总反射数是两源四站合计，不向算法提供真值条数。','',
        '## 逐例位置误差与近真控制','',
        '| 案例 | FULL/m | CAP_CONSERVATIVE/m | CAP_COUNTED/m | CAP_COUNTED近真控制/m |',
        '|---|---:|---:|---:|---:|']
    tags=list(dict.fromkeys(r['tag'] for r in rows));lookup={(r['tag'],r['arm']):r for r in rows}
    cl={r['tag']:r for r in controls}
    for tag in tags:
        cells=[f"{lookup[tag,a]['rmse']:.4f}" for a in ['FULL','CAP_CONSERVATIVE','CAP_COUNTED']]
        last=f"{cl[tag]['rmse']:.4f}" if tag in cl else '未安排'
        lines.append('| '+tag+' | '+' | '.join(cells)+' | '+last+' |')
    lines+=['','## 执行与验证','',
        f"三臂批次墙钟{done['seconds']:.3f}秒；近真控制批次{cdone['seconds']:.3f}秒。单方法时间包含其前端，批次时间另含输入复制与核验等；三臂顺序按案例轮换，两个批次未并发。",
        f"54个终态独立回代：目标最大差{evidence['verification']['cost_max_error']:.3g}，投影梯度映射最大差{evidence['verification']['PG_max_error']:.3g}。原FULL位置与09轮在1e-5m内复现，三臂LOS前端在1e-8m内相同。",
        '单元检查：支持域内目标/梯度相同；凸积集投影检查通过；候选池8000项与原池对应子集数值一致；FFT核验通过。首轮名义阈值从29.5582变为25.3522，显著性预算未变，但未新增有限样本统计校准保证。',
        '运行中SLSQP出现越界试步后裁剪的常规警告；没有据此忽略失败。最终接受状态与全部历史接受状态的上界已检查，未把非正常结束当收敛。',
        '20微秒没有使频域近似变成严格非周期有限记录模型；小延迟边界误差仍需在未来正式方法中核对。当前没有新算法创新声明，没有扫描其他上界，没有新增真实路径先验或独立样本。']
    text='\n'.join(lines)+'\n';(HERE/'核验结果.md').write_text(text,encoding='utf-8')
    spec=importlib.util.spec_from_file_location('renderer',BASE/'实验/10_文献机制综合/20261107/成果/summarize.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    discussion=(HERE/'结论与推进判断.md').read_text(encoding='utf-8')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>公开信道记忆核验</title><style>body{margin:0;background:#f3f5f1;color:#243a43;font:16px/1.85 "Segoe UI","Microsoft YaHei",sans-serif}main{max-width:1150px;margin:auto;padding:30px}header{background:#163f48;color:#fff;padding:30px;border-radius:16px}section{background:white;padding:24px;border-radius:12px;margin-top:22px}h1{font-size:30px}nav{display:flex;gap:18px;flex-wrap:wrap;margin:20px 0}a{color:#087080}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;border:1px solid #d6e0da;text-align:left}th{background:#e6efeb}.scroll{overflow-x:auto}.note{background:#fff0d7;padding:18px;border-left:4px solid #b88b2e}footer{font-size:13px;color:#65777b;margin-top:24px}@media(max-width:600px){main{padding:12px}section{padding:14px}}</style><main><header>多径DPD · 第14轮 · 2026-11-07<h1>公开时延上界修正了搜索范围，但没有解决错误分支</h1><p>54次盲定位＋12次近真控制；没有调上界，没有改变观测。</p></header><nav><a href="#decision">结论与下一步</a><a href="#results">全量结果</a><a href="配对结果.csv">CSV</a><a href="summary.json">JSON</a></nav><p class="note">近真控制仍然使用既有真值辅助候选索引，不能写成算法成绩。不能把缩小已知信道范围的收益包装为新方法创新。</p>'''
    page+='<section id="decision">'+mod.render(discussion)+'</section><section id="results">'+mod.render(text)+'</section>'
    page+='<footer><a href="执行合同.md">三臂合同</a> · <a href="重建控制合同.md">近真控制合同</a> · <a href="run.py">隔离执行入口</a> · <a href="https://www.itu.int/rec/R-REC-M.1225">ITU原始依据</a><p>成果按冻结数据生成；程序检查不等于方法创新验证。</p></footer></main></html>'
    (HERE/'公开信道记忆核验.html').write_text(page,encoding='utf-8')
    print(json.dumps(dict(groups=groups,oracle_rmse=float(np.sqrt(np.mean([r['rmse']**2 for r in controls]))),
                          oracle_normal=sum(r['normal'] for r in controls),verification=evidence['verification']),ensure_ascii=False))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
