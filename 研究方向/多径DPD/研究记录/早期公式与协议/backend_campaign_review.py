"""Read-only numerical replay and reproducible aggregate tables for the campaign."""
from datetime import datetime
from collections import Counter
import csv
import json

import numpy as np

from geometry_association_gate import ROOT, B, C, PAIRS, sha, write, signature
from robust_geometry import evaluate
from bounded_consensus import quantities, joint_select, selftest

RUNS={
 'model_contract':'backend_contract_20261029_221413_270882',
 'pilot':'blind_backend_20261029_221536_252934',
 'independent_failure':'independent_backend_20261029_222111_026721',
 'trim_attribution':'independent_attribution_20261029_222816_600594',
 'bounded_diagnostic':'bounded_consensus_20261029_223120_983207',
 'development':'consensus_development_20261029_223235_334946',
 'validation':'consensus_validation_20261029_223520_495993',
}


def main():
    paths=[ROOT/run/'result.json' for run in RUNS.values()]
    paths+=[ROOT/'backend_campaign_review.py']
    hashes={str(p.resolve(strict=True)):sha(p) for p in paths}
    out=ROOT/('backend_review_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    write(out/'protocol.json',dict(hashes=hashes,runs=RUNS))
    datasets={k:json.loads((ROOT/v/'result.json').read_text(encoding='utf-8')) for k,v in RUNS.items()}
    checks=selftest()
    for name,data in datasets.items():
        checks[name+'/original_checks']=all(data['checks'].values())
        proto=json.loads((ROOT/RUNS[name]/'protocol.json').read_text(encoding='utf-8'))
        for path,h in proto['hashes'].items():
            # Source snapshots preserve the exact version even if later test-only changes occurred.
            from pathlib import Path
            p=Path(path)
            snapshot=ROOT/RUNS[name]/'source'/p.name
            if p.suffix in ['.py','.md'] and snapshot.exists():
                checks[name+'/snapshot/'+p.name]=sha(snapshot)==h
            elif p.suffix not in ['.py','.md']:
                checks[name+'/input/'+p.name]=sha(p)==h
    # Frozen solver identity is identical between development and truly new validation.
    dev=json.loads((ROOT/RUNS['development']/'protocol.json').read_text(encoding='utf-8'))['hashes']
    val=json.loads((ROOT/RUNS['validation']/'protocol.json').read_text(encoding='utf-8'))['hashes']
    for p,h in dev.items():
        if p in val:
            checks['freeze/'+p]=h==val[p]
    aggregate=[]
    detail=[]
    for phase in ['pilot','independent_failure','development','validation']:
        data=datasets[phase]
        for condition in ['LOS','MP']:
            subset=[r for r in data['rows'] if r['tag'].rsplit('_',1)[1]==condition]
            for method in ['geometry','los','mp']:
                e=np.array([r[method+'_errors_m'] for r in subset])
                aggregate.append(dict(phase=phase,path=condition,method=method,n=len(e),
                  rmse_m=float(np.sqrt(np.mean(e*e))),source_A_rmse_m=float(np.sqrt(np.mean(e[:,0]**2))),
                  source_B_rmse_m=float(np.sqrt(np.mean(e[:,1]**2))),max_error_m=float(e.max()),
                  p95_m=float(np.percentile(e,95)),both20=int(sum(e.max(axis=1)<=20))))
        for row in data['rows']:
            history=row['mp']['history']
            checks[phase+'/'+row['tag']+'/monotone']=all(b['cost']<=a['cost']+1e-10 for a,b in zip(history,history[1:]))
            last=np.array(row['mp']['p']).reshape(2,2)
            idx=min(40,len(history)-1)
            earlier=np.array(history[idx]['positions']).reshape(2,2)
            detail.append(dict(phase=phase,tag=row['tag'],path=row['tag'].rsplit('_',1)[1],
               geometry_mse=float(np.mean(np.array(row['geometry_errors_m'])**2)),
               los_mse=float(np.mean(np.array(row['los_errors_m'])**2)),
               mp_mse=float(np.mean(np.array(row['mp_errors_m'])**2)),
               stop=row['mp']['reason'],mapping=row['mp']['mapping'],
               history_length=len(history),position_change_from_saved40_m=float(np.max(np.linalg.norm(last-earlier,axis=1))),
               solver_seconds=row['seconds'],frontend_seconds=row.get('frontend_seconds'),
               reported_whole_seconds=row.get('whole_seconds')))
    for phase in ['development','validation']:
        source=ROOT/RUNS['independent_failure'] if phase=='development' else ROOT/RUNS[phase]
        for row in datasets[phase]['rows']:
            tag=row['tag']
            data=dict(np.load(source/(tag.rsplit('_',1)[0]+'_input.npz')))
            iq=(data['los']+(data['reflected'] if row['path']=='MP' else 0)).sum(axis=0)+data['noise']
            checks[phase+'/'+tag+'/IQ']=signature(iq)==row['iq_sha']
            z=np.load(ROOT/RUNS[phase]/(tag+'_geometry.npz'))
            lags=np.array([sorted(np.array(row['peaks'][f'{m}_{n}']['lag_s'][:6])*C) for m,n in PAIRS])
            rr,_,near,_,cost=quantities(z['p'],lags)
            checks[phase+'/'+tag+'/score']=np.allclose(cost,z['cost'])
            rebuilt=np.zeros((len(z['p']),10,6))
            np.put_along_axis(rebuilt,near[:,:,None],np.maximum(0,1-rr*rr/(C/B)**2)[:,:,None],axis=2)
            checks[phase+'/'+tag+'/reward']=np.allclose(rebuilt.reshape(len(z['p']),-1),z['reward'])
            ids,_=joint_select(z['reward'])
            checks[phase+'/'+tag+'/decode']=ids==z['chosen'].tolist()
            for method,pos in [('geometry',row['initial']),('los',row['los_p']),('mp',row['mp']['p'])]:
                ev=evaluate(np.array(pos),data['p'])
                checks[phase+'/'+tag+'/'+method+'/metric']=np.allclose(ev['errors_m'],row[method+'_errors_m'])
            checks[phase+'/'+tag+'/band_snr']=abs(data['nominal_average_snr_db'][1]-6)<1e-10
    for name,rows in [('metrics',aggregate),('paired_details',detail)]:
        with (out/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    checks.update({p:sha(p)==h for p,h in hashes.items()})
    assert all(checks.values()),[k for k,v in checks.items() if not v]
    summaries={}
    for phase in ['development','validation']:
        selected=[r for r in detail if r['phase']==phase]
        summaries[phase]=dict(
          stop_counts=dict(Counter(r['stop'] for r in selected)),
          max_position_change_from_saved40_m=max(r['position_change_from_saved40_m'] for r in selected),
          mean_solver_seconds=float(np.mean([r['solver_seconds'] for r in selected])),
          mean_frontend_seconds=float(np.mean([r['frontend_seconds'] for r in selected])),
          mp_better_than_los_mse=sum(r['mp_mse']<r['los_mse'] for r in selected if r['path']=='MP'),
          mp_better_than_geometry_mse=sum(r['mp_mse']<r['geometry_mse'] for r in selected if r['path']=='MP'))
    write(out/'audit.json',dict(checks=checks,metrics=aggregate,summary=summaries,runs=RUNS,
        original_run_checks={k:len(d['checks']) for k,d in datasets.items()},
        run_seconds={k:d['seconds'] for k,d in datasets.items()},
        timing_note='reported_whole_seconds excludes the backend IQ-to-FFT conversion; not a strict end-to-end timing comparison'))
    lines=['# 盲起点与多径DPD：多轮核验汇总','',
       '本表由原始保存输出重算。RMSE为两个目标等权、全部案例未截断平方误差的平方根；源标签仅用于离线最小平方距离匹配。',
       '', '|批次|信道|方法|观测数|整体RMSE/m|最大单源误差/m|双源20m成功|',
       '|---|---|---|---:|---:|---:|---:|']
    for r in aggregate:
        lines.append(f"|{r['phase']}|{r['path']}|{r['method']}|{r['n']}|{r['rmse_m']:.6f}|{r['max_error_m']:.6f}|{r['both20']}/{r['n']}|")
    lines+=['','## 审计与读取边界','',f'保存结果重放与身份检查：{len(checks)}项全部通过。',
       '开发与独立复核共享完整源码身份，独立数据不参加参数选择。每阶段包含12个基础随机实现的直达/多径配对，不把24观测算成24个独立信道实现。',
       '后端触顶照实保留，不把位置变化小等同全部参数满足梯度停止。',
       '耗时字段whole_seconds未包含后端FFT转换，因此仅为已记录组件成本；本轮不据此主张端到端加速。',
       '独立复核仍只含四个新布局、每布局三次实现，未证明任意几何、低SNR、近时延反射或无LOS条件。',
       '', '## 阶段摘要','', '```json',json.dumps(summaries,ensure_ascii=False,indent=2),'```','',
       '## 证据目录','']
    for label,run in RUNS.items():
        lines.append(f'- {label}: [{run}](../{run}/result.json)')
    (out/'核验汇总.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('COMPLETE',out,'CHECKS',len(checks),json.dumps(summaries,ensure_ascii=False))


if __name__=='__main__':
    main()
