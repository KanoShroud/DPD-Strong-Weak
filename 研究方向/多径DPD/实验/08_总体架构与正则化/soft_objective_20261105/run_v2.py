"""Preregistered soft-objective math and position-response checks."""
import argparse
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from soft_v2 import HERE,BASE,PREV,INPUT,SAVED,cfg,ad,stable,PositionProblem,calibrate


def sources():
    ds=cfg.read(PREV/'discrepancy_v1/blind_decisions.json')
    interior={r['tag'] for r in cfg.read(PREV/'conditional_v1/rows.json') if r['conditional']['status']=='interior_discrepancy'}
    result=[]
    for d in ds:
        if d['tag'] in interior:result.append((d,SAVED/(d['tag']+'_ORTH_SAFE.json'),'BLIND'))
        if d['condition'] in ['L2_P075','L3_P150']:
            result.append((d,BASE/'source_path_gate_20261105/run_v1'/(d['tag']+'_DIAGNOSTIC_NEAREST.json'),'DIAGNOSTIC'))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['math','response'],required=True);args=parser.parse_args()
    out=HERE/(args.mode+'_v2');out.mkdir(exist_ok=False)
    hashes=cfg.read(PREV/'discrepancy_v1/identity.json');stable.inside.rr.check_hashes(hashes)
    for p in [Path(__file__).resolve(),HERE/'soft_v2.py',HERE/'protocol.md',PREV/'conditional_precision.py']:
        hashes[str(p)]=cfg.sha(p)
    for d,p,arm in sources():hashes[str(p)]=cfg.sha(p)
    cfg.write(out/'identity.json',hashes);started=time.perf_counter()
    if args.mode=='math':
        records=[]
        for r in cfg.read(PREV/'zero_plateau_v1/decisions.json'):
            if r['tag'] not in {x['tag'] for x in records}:
                old=cfg.read(SAVED/(r['tag']+'_ORTH_SAFE.json'))['result']['los']
                with np.load(INPUT/'cases'/r['tag']/'iq.npz',allow_pickle=False) as z:iq=z['iq']
                f,y,_,_=ad.b.fc.cal.spectral(iq);problem=PositionProblem(np.asarray(old['x']),[],f,y)
                base_value=problem.value(np.asarray(old['x'])[:4],1.)[0]
            p=np.asarray(r['p']).reshape(-1)/problem.e.ref.UNIT
            value,_,info=problem.value(p,1.)
            records.append(dict(tag=r['tag'],shift_m=r['shift_m'],old_zero_feasible=r['feasible'],value=value,
                base_value=base_value,ls_equivalent=abs(value-.5*info['rss'])<1e-14,
                different=bool(abs(value-base_value)>1e-12)))
        cfg.write(out/'plateau.json',records)
        d,p,_=sources()[0];old=cfg.read(p)['result'];x=np.asarray(old['final']['x'])
        with np.load(INPUT/'cases'/d['tag']/'iq.npz',allow_pickle=False) as z:iq=z['iq']
        f,y,_,_=ad.b.fc.cal.spectral(iq);problem=PositionProblem(x,old['paths'],f,y)
        lam=cfg.read(PREV/'conditional_v1'/f"{d['tag']}.json")['conditional']['multiplier']/problem.scale
        checks=[]
        for k in range(4):
            _,g,_=problem.value(x[:4],lam);step=1e-4;direction=np.eye(4)[k]*step
            fd=(problem.value(x[:4]+direction,lam)[0]-problem.value(x[:4]-direction,lam)[0])/(2*step)
            checks.append(dict(index=k,analytic=g[k],fd=fd,passed=bool(abs(g[k]-fd)<=1e-6+.01*max(abs(g[k]),abs(fd)))))
        cfg.write(out/'derivatives.json',checks)
        summary=dict(passed=all(r['ls_equivalent'] for r in records) and all(c['passed'] for c in checks),
            plateau_count=len(records),scores_different_from_center=sum(r['different'] for r in records),gradient_checks=len(checks),
            identity_unchanged=True,seconds=time.perf_counter()-started)
    else:
        assert cfg.read(HERE/'math_v2/summary.json')['passed']
        rows=[]
        for d,path,arm in sources():
            tag=d['tag'];print('START',tag,arm,flush=True);t=time.perf_counter()
            saved=cfg.read(path);r=saved['result'];x=np.asarray(r['final']['x'])
            with np.load(INPUT/'cases'/tag/'iq.npz',allow_pickle=False) as z:iq=z['iq']
            assert saved['iq_sha']==cfg.digest(iq)
            f,y,_,_=ad.b.fc.cal.spectral(iq);problem=PositionProblem(x,r['paths'],f,y)
            initial_h=problem.e.core(x,f,y,r['paths'])[0]
            _,initial_cal=calibrate(initial_h,y,problem.c,d['radius2'])
            row=dict(tag=tag,arm=arm,initial_p=r['final']['p'],initial_calibration=initial_cal)
            if initial_cal['status']!='interior_discrepancy':
                row['status']='initial_not_interior';rows.append(row);cfg.write(out/(tag+'_'+arm+'.json'),row);continue
            reference=problem.fit(x[:4],0.);p0=np.asarray(reference['x_position']);row['reference']=reference
            updated=x.copy();updated[:4]=p0;h=problem.e.core(updated,f,y,r['paths'])[0]
            _,cal=calibrate(h,y,problem.c,d['radius2']);row['calibration']=cal
            if cal['status']!='interior_discrepancy':row['status']='reference_not_interior'
            else:
                lam=cal['multiplier']/problem.scale;response=problem.sensitivity(p0)
                row.update(lambda_normalized=lam,sensitivity=response,trials=[],status='completed')
                for factor in [.005,.01,1.]:
                    fit=problem.fit(p0,lam*factor);actual=(np.asarray(fit['x_position'])-p0)*problem.e.ref.UNIT
                    if response['valid'] and reference['stationary']:
                        predicted=np.asarray(response['response_m'])*lam*factor
                        error=np.linalg.norm(actual-predicted)
                        fit.update(predicted_shift_m=predicted,actual_shift_m=actual,prediction_error_m=error,
                            small_response_pass=bool(error<=.1*np.linalg.norm(predicted)+.001))
                    row['trials'].append(dict(factor=factor,fit=fit))
            row['seconds']=time.perf_counter()-t;rows.append(row);cfg.write(out/(tag+'_'+arm+'.json'),row);cfg.write(out/'decisions.json',rows)
            print('END',tag,arm,row['status'],row['seconds'],flush=True)
        cfg.write(out/'decisions.json',rows)
        evaluations=[]
        for row in rows:
            with np.load(INPUT/'banks'/(row['tag'].split('_')[1]+'_00.npz'),allow_pickle=False) as z:truth=z['p']
            record=dict(tag=row['tag'],arm=row['arm'],status=row['status'],initial_errors=stable.inside.rr.evaluation(np.asarray(row['initial_p']),truth))
            if 'reference' in row:record['reference_errors']=stable.inside.rr.evaluation(np.asarray(row['reference']['p']),truth)
            record['trials']=[dict(factor=t['factor'],errors=stable.inside.rr.evaluation(np.asarray(t['fit']['p']),truth),stationary=t['fit']['stationary']) for t in row.get('trials',[])]
            evaluations.append(record)
        cfg.write(out/'evaluation.json',evaluations)
        summary=dict(completed=len(rows),eligible=sum(r['status']=='completed' for r in rows),seconds=time.perf_counter()-started,identity_unchanged=True,
            full_blind_runs=0,scope='position_only_fixed_channel_mechanism')
    stable.inside.rr.check_hashes(hashes);cfg.write(out/'summary.json',summary);print(summary,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
