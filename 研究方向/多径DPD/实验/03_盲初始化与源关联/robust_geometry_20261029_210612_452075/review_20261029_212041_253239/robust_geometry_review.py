"""Independent replay of saved geometry outputs; no estimator rerun."""
from pathlib import Path
from datetime import datetime
from itertools import permutations
import argparse
import json

import numpy as np

from geometry_association_gate import sha, write, C, STATIONS, PAIRS

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'finite_iq_clues_20261029_204430_528859'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run')
    parser.add_argument('--prefix')
    args=parser.parse_args()
    run=Path(args.run).resolve(strict=True)
    saved=json.loads((run/'result.json').read_text(encoding='utf-8'))
    contract=json.loads((run/'protocol.json').read_text(encoding='utf-8'))
    checks={f'hash:{f}':sha(f)==h for f,h in contract['hashes'].items()}
    original=json.loads((SOURCE/'result.json').read_text(encoding='utf-8'))
    source={r['tag']:r for r in original['rows']}
    summary={}
    failures=[]
    detailed=[]
    edm_output_diagnostic=[]
    for row in saved['rows']:
        raw=source[row['tag']]
        truth=np.load(SOURCE/f'L{row["layout"]}_R{row["realization"]}_components.npz')['truth'][raw['active_sources']]
        checks[row['tag']+'/iq']=row['iq_sha']==raw['iq_sha']
        arrays=np.load(run/(row['tag']+'_'+row['front']+'.npz'))
        lags=np.array([sorted(np.array(raw['extraction'][row['front']][f'{m}_{n}']['lag_s'][:6])*C) for m,n in PAIRS])
        for name,rec in row['methods'].items():
            key=f'{row["tag"]}/{row["front"]}/{name}'
            p=np.array(rec['p']).reshape(-1,2)
            pool=arrays[name+'_p']
            checks[key+'/selection']=np.array_equal(p,pool[rec['selected']])
            if len(p)==len(truth):
                variants=[np.linalg.norm(p[list(order)]-truth,axis=1) for order in permutations(range(len(p)))]
                errors=min(variants,key=lambda v:np.sum(v*v))
                checks[key+'/errors']=bool(np.allclose(errors,rec['evaluation']['errors_m'],atol=1e-8))
                checks[key+'/success']=bool(max(errors)<=20)==rec['evaluation']['success20']
            else:
                errors=None
                checks[key+'/missing']=rec['evaluation']['errors_m'] is None
            if len(pool):
                near=np.linalg.norm(pool[:,None]-truth[None],axis=2).min(axis=0)
                checks[key+'/coverage']=bool(np.allclose(near,rec['coverage_m'],atol=1e-8))
            if name.startswith('GEO'):
                ranges=np.linalg.norm(pool[:,None]-STATIONS[None],axis=2)
                pred=np.column_stack([ranges[:,n]-ranges[:,m] for m,n in PAIRS])
                residual=np.min(abs(pred[:,:,None]-lags[None]),axis=2)
                keep=10 if name=='GEO_ALL' else 8
                cost=np.mean(np.sort(residual**2,axis=1)[:,:keep],axis=1)
                checks[key+'/objective']=bool(np.allclose(cost,arrays[name+'_scores'],atol=1e-7,rtol=1e-9))
                initial=arrays['initial']
                ir=np.linalg.norm(initial[:,None]-STATIONS[None],axis=2)
                ip=np.column_stack([ir[:,n]-ir[:,m] for m,n in PAIRS])
                initial_residual=np.min(abs(ip[:,:,None]-lags[None]),axis=2)
                initial_cost=np.mean(np.sort(initial_residual**2,axis=1)[:,:keep],axis=1)
                checks[key+'/descent']=bool(np.all(cost<=initial_cost+1e-7))
            else:
                u=arrays['EDM_RF_u']
                alpha=arrays['EDM_RF_alpha']
                dd=(alpha[:,None]+np.column_stack([np.zeros(len(alpha)),u]))**2
                station_dd=np.sum((STATIONS[:,None]-STATIONS[None])**2,axis=2)
                direct=np.zeros((len(alpha),6,6))
                centered=STATIONS-STATIONS.mean(axis=0)
                direct[:,:5,:5]=centered@centered.T
                cross=-.5*(dd-dd.mean(axis=1)[:,None]-station_dd.mean(axis=1)[None]+station_dd.mean())
                direct[:,:5,5]=cross
                direct[:,5,:5]=cross
                direct[:,5,5]=dd.mean(axis=1)-.5*station_dd.mean()
                eigen=np.linalg.eigvalsh(direct/(3000.**2))
                checks[key+'/edm_direct_formula']=bool(np.allclose(np.sum(abs(eigen[:,:-2]),axis=1),arrays['EDM_RF_scores'],atol=1e-11))
                if len(truth)==2 and len(pool):
                    order=np.argsort(arrays['EDM_RF_scores'],kind='stable')
                    first=int(order[0])
                    alternatives=[int(i) for i in order if np.sum(u[i]!=u[first])>=4]
                    diag_errors=None
                    if alternatives:
                        outp=pool[[first,alternatives[0]]]
                        variants=[np.linalg.norm(outp[list(o)]-truth,axis=1) for o in permutations(range(2))]
                        diag_errors=min(variants,key=lambda e:e@e)
                    edm_output_diagnostic.append(dict(tag=row['tag'],front=row['front'],paths=row['paths'],
                         errors_m=diag_errors,success20=bool(diag_errors is not None and max(diag_errors)<=20),
                         selected=[first]+alternatives[:1]))
            group=f'{row["signal"]}/{row["paths"]}/{row["front"]}/{name}'
            detail=dict(tag=row['tag'],front=row['front'],method=name,layout=row['layout'],
                        errors_m=errors,success20=rec['evaluation']['success20'],seconds=rec['seconds'],
                        coverage_m=rec['coverage_m'],selected_stop=rec.get('chosen_stop'))
            selected_ranges=np.linalg.norm(p[:,None]-STATIONS[None],axis=2)
            selected_lags=np.column_stack([selected_ranges[:,n]-selected_ranges[:,m] for m,n in PAIRS])
            selected_residual=np.min(abs(selected_lags[:,:,None]-lags[None]),axis=2)
            detail['output_order_star_rms_m']=np.sqrt(np.mean(selected_residual[:,:4]**2,axis=1))
            detail['output_order_other_edges_rms_m']=np.sqrt(np.mean(selected_residual[:,4:]**2,axis=1))
            summary.setdefault(group,[]).append(detail)
            detailed.append(detail)
            if not detail['success20']:
                failures.append(detail)
    if args.prefix:
        prefix=json.loads((Path(args.prefix)/'result.json').read_text(encoding='utf-8'))
        byid={(r['tag'],r['front']):r for r in saved['rows']}
        for row in prefix['rows']:
            later=byid[(row['tag'],row['front'])]
            for name,rec in row['methods'].items():
                checks[f'prefix/{row["tag"]}/{row["front"]}/{name}']=bool(np.array_equal(rec['p'],later['methods'][name]['p']))
    stats={}
    for key,rows in summary.items():
        complete=[r for r in rows if r['errors_m'] is not None]
        ee=np.array([r['errors_m'] for r in complete])
        stats[key]=dict(n=len(rows),complete=len(complete),success20=sum(r['success20'] for r in rows),
            rmse_m=float(np.sqrt(np.mean(ee**2))) if len(ee) else None,
            source_rmse_m=np.sqrt(np.mean(ee**2,axis=0)) if len(ee) else None,
            p95_m=float(np.quantile(ee,.95)) if len(ee) else None,
            mean_seconds=float(np.mean([r['seconds'] for r in rows])),
            coverage20=sum(r['coverage_m'] is not None and max(r['coverage_m'])<=20 for r in rows))
    # Both the improvement and damage of trimming are retained, by case.
    paired=[]
    for row in saved['rows']:
        if row['signal']!='AB':
            continue
        a=row['methods']['GEO_ALL']['evaluation']
        b=row['methods']['GEO_TRIM8']['evaluation']
        paired.append(dict(tag=row['tag'],front=row['front'],paths=row['paths'],
             all_m=a['rmse_m'],trim_m=b['rmse_m'],improved20=(not a['success20'] and b['success20']),
             damaged20=(a['success20'] and not b['success20'])))
    sensitivity=[]
    for row in saved['rows']:
        if not (row['layout']==1 and row['realization']==5 and row['signal']=='AB'):
            continue
        raw=source[row['tag']]
        truth=np.load(SOURCE/'L1_R5_components.npz')['truth']
        delta=truth[:,None]-STATIONS[None]
        ranges=np.linalg.norm(delta,axis=2)
        unit=delta/ranges[:,:,None]
        jac=np.stack([unit[:,n]-unit[:,m] for m,n in PAIRS],axis=1)
        pred=np.column_stack([ranges[:,n]-ranges[:,m] for m,n in PAIRS])
        lags=np.array([sorted(np.array(raw['extraction'][row['front']][f'{m}_{n}']['lag_s'][:6])*C) for m,n in PAIRS])
        residual=pred[:,:,None]-lags[None]
        nearest=np.argmin(abs(residual),axis=2)
        error=np.take_along_axis(residual,nearest[:,:,None],axis=2)[:,:,0]
        dp=np.array([-np.linalg.lstsq(jac[q],error[q],rcond=None)[0] for q in range(2)])
        sensitivity.append(dict(tag=row['tag'],front=row['front'],posthoc_oracle_diagnostic=True,
             edge03_residual_m=error[:,2],edge03_sensitivity=np.linalg.norm(jac[:,2],axis=1),
             all_edge_sensitivity=np.linalg.norm(jac,axis=2),linear_position_error_m=np.linalg.norm(dp,axis=1),
             measured_position_error_m=row['methods']['GEO_ALL']['evaluation']['errors_m']))
    assert all(checks.values()),[k for k,v in checks.items() if not v]
    out=run/('review_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    write(out/'review.json',dict(checks=checks,stats=stats,failures=failures,paired=paired,
          edm_four_different_posthoc=edm_output_diagnostic,
          position_sensitivity_diagnostic=sensitivity,
          inputs={str(run/'result.json'):sha(run/'result.json'),str(Path(__file__)):sha(__file__)},details=detailed))
    lines=['# 有限IQ的容错几何关联核验','',
       '本轮是从已保存RF相关峰恢复位置的开发短测，不是原始IQ-DPD最终性能；不重新生成数据、不加窗、不调提取器。',
       '',f'结果目录：`{run.name}`。{len(saved["rows"])}组前端输入，{len(checks)}项回读检查通过；总执行墙钟{saved["seconds"]:.2f}s。',
       '', '## 定位结果','',
       '位置RMSE对正确一一匹配的所有完整输出计算，不截断大误差；缺失输出另列。单源每组只有2个控制样本。',
       '', '| 观测/前端/方法 | 完整输出 | 20m成功 | 位置RMSE/m | P95/m | 候选池覆盖20m | 关联均耗时/s |',
       '|---|---:|---:|---:|---:|---:|---:|']
    for key,v in stats.items():
        lines.append(f'|{key}|{v["complete"]}/{v["n"]}|{v["success20"]}/{v["n"]}|{v["rmse_m"]:.3f}|{v["p95_m"]:.3f}|{v["coverage20"]}/{v["n"]}|{v["mean_seconds"]:.3f}|')
    lines += ['', '## 公平性与解释边界','',
       '- GEO_ALL和GEO_TRIM8使用相同解析候选池、全部10条边、相同精修器与停止条件；只有保留的残差数10/8不同。',
       '- EDM_RF使用4条参考边，实际优化参考距离，再做特征分解与Procrustes。与GEO差异还包含观测边数和输出恢复方式，不能全部归因于容错。',
       '- 三版统一按c/B约24m空间去重。不采用原6站声学论文的四时差不同规则，也不称原论文完整复现。',
       '- EDM代码以完整网格+逐组合标量优化核对数学目标，尚未做批量优化；本轮秒数不是该方法最优实现的复杂度结论。',
       '- GEO是常规枚举多边定位与修剪最小二乘参照，不是Flood最优传输算法，也没有新颖性主张。',
       '- 计时仅包括已提取峰之后的关联，不能称完整IQ端到端定位耗时；两种前端共享基础信道，不是独立样本。',
       '- 基础数据为20个开发实现、两个固定布局。本轮不宣称任意几何、SNR或多径范围的泛化。',
       '- 约24m是冻结的输出去重尺度，不是位置分辨率的理论下界。新有限IQ合同与旧理想频点合同不同，不能跨批次直接归因性能提升。',
       '', '## 未成功样本与机制配对','',
       '|样本|前端|方法|逐源误差/m|候选池最近距离/m|', '|---|---|---|---|---|']
    for r in failures:
        lines.append(f'|{r["tag"]}|{r["front"]}|{r["method"]}|{np.round(r["errors_m"],3).tolist() if r["errors_m"] is not None else "缺失"}|{np.round(r["coverage_m"],3).tolist() if r["coverage_m"] is not None else "无"}|')
    lines += ['',f'GEO_TRIM8相对GEO_ALL：改对{sum(r["improved20"] for r in paired)}项，改错{sum(r["damaged20"] for r in paired)}项。包含两种相关前端，不按独立样本解读。逐样本RMSE变化保存在review.json。',
       '', '## 文献及实现', '',
       '- Brümann–Doclo, IEEE TASLP 2026：[正式出处](https://doi.org/10.1109/TASLPRO.2026.3717238)，式(22)、(24)–(29)；具体适配见冻结协议。',
       '- Flood–Elvander, EUSIPCO2024：[正式出处](https://doi.org/10.23919/EUSIPCO63174.2024.10715317)，候选生成和误检/漏检关联思想参照，未复现最优传输。',
       '- 代码、输入哈希、冻结协议见运行目录source/与protocol.json；完整逐例候选见NPZ，后续不依据失败例调本轮参数。',
       '', '本报告仅汇总关联证据；是否接入共同多径DPD须根据起点质量另行判断，不能直接把TDOA拟合当作DPD。']
    lines += ['', '## 原文四时差不同规则的事后重放', '',
       '仅从相同EDM优化候选中按原文实验规则重选第二个输出，不重新优化，不替换预定主结果。用于核查输出规则的影响，不能据结果逐样本挑选最有利规则。', '',
       '|条件|前端|双源成功|位置RMSE/m|', '|---|---|---:|---:|']
    for paths in ['LOS','MP']:
        for front in ['CC_ABS','PHAT_ABS']:
            group=[r for r in edm_output_diagnostic if r['paths']==paths and r['front']==front]
            if group:
                valid=[r['errors_m'] for r in group if r['errors_m'] is not None]
                value=float(np.sqrt(np.mean(np.array(valid)**2))) if valid else None
                lines.append(f'|{paths}|{front}|{sum(r["success20"] for r in group)}/{len(group)}|{value}|')
    if sensitivity:
        lines += ['', '## 为什么一条时差偏了，位置仍然准确？', '',
          '对距离差d_n(p)-d_m(p)，位置导数是两条源站单位向量之差。局部线性最小二乘给出δp≈−J⁺e，其中e为当前位置处的时差残差（乘光速换成米）。这是常规误差传播，不是新定理。', '',
          '预先已知的L1/R5困难站对0–3，对两个源的导数模分别约0.064、0.042；其余站对约0.84–1.99。该条边对位置变化很不敏感，其他边仍提供有效约束。时差错误的大小不能直接当成位置错误。下表使用真值作事后局部归因，不进入估计器。', '',
          '|输入|前端|0–3距离差误差/m|线性预测位置误差/m|实际位置误差/m|', '|---|---|---|---|---|']
        for v in sensitivity:
            lines.append(f'|{v["tag"]}|{v["front"]}|{np.round(v["edge03_residual_m"],3).tolist()}|{np.round(v["linear_position_error_m"],3).tolist()}|{np.round(v["measured_position_error_m"],3).tolist()}|')
        lines += ['', '这支持本批无需把每一条直达时差全部修准才形成位置起点；不证明任何漏峰都无害，也不把位置导数分析本身称为创新。']
    (out/'容错几何关联核验.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('REVIEW',out,flush=True)
    print(json.dumps({'checks':len(checks),'stats':stats,'failures':len(failures)},ensure_ascii=False,default=lambda x:x.tolist()),flush=True)


if __name__=='__main__':
    main()
