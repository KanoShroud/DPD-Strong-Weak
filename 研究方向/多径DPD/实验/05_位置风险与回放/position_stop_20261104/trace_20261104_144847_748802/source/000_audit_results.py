"""Readback only after timing runs; re-evaluate outputs on original IQ."""
import json
from pathlib import Path
import shutil

import numpy as np
from threadpoolctl import threadpool_limits
import fast_core as fc
import run_gate as rg

HERE=Path(__file__).resolve().parent
RUNS={
 'exact':'paired_20261104_123748_731072',
 'grouped':'grouped_20261104_124051_478393',
 'pilot':'budget_pilot_20261104_124330_620289',
 'followup':'budget_followup_20261104_124445_393210',
}


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    contract=fc.load(rg.FORMAL/'contract.json')['contract'];checks={};conditions={};rows=[]
    for label,name in RUNS.items():
        root=HERE/name
        pre=read(root/'preflight.json')
        checks[label+'/hashes']=all(fc.ref.sha(p)==h for p,h in pre['hashes'].items())
        checks[label+'/postflight']=read(root/'postflight.json')['unchanged']
        data=read(root/('results.json' if label=='exact' else 'result.json'))
        checks[label+'/complete']=data['complete']
        for old in data['rows']:
            tag=old['case'];variant=old.get('variant','grouped');cond=tag.split('_')[0]
            file=root/(tag+('_'+variant if label!='grouped' else '')+'.json')
            result=read(file);state=result['final'];paths=[tuple(p) for p in result['paths']]
            iq=rg.inputs(tag);f,y,n,_=fc.cal.spectral(iq)
            r,j=fc.ref.evaluate(np.asarray(state['x']),f,y,paths)
            J=float(.5*r@r);gradient=j.T@r;x=np.asarray(state['x'])
            pg=float(np.max(abs(x-fc.ref.project(x-gradient,n))))
            with np.load(contract['conditions'][cond]['bank'],allow_pickle=False) as z:truth=z['p']
            err=np.asarray(fc.ref.error(state['p'],truth));rmse=float(np.sqrt(np.mean(err**2)))
            prefix=label+'/'+tag+'/'+variant
            checks[prefix+'/objective']=bool(np.isclose(J,state['cost'],rtol=1e-9,atol=1e-12))
            checks[prefix+'/rmse']=bool(np.isclose(rmse,old['rmse_m'],rtol=1e-10,atol=1e-9))
            checks[prefix+'/PG']=bool(np.isclose(pg,state['projected_gradient'],rtol=1e-4,atol=1e-9))
            checks[prefix+'/feasible']=bool(np.linalg.norm(x-fc.ref.project(x,n))<1e-7)
            baseline=fc.load(rg.FORMAL/'cases'/tag/'solve.json')
            be=np.asarray(fc.ref.error(baseline['final']['p'],truth));le=np.asarray(fc.ref.error(baseline['los']['p'],truth))
            rows.append(dict(stage=label,case=tag,condition=cond,variant=variant,rmse_m=rmse,baseline_rmse_m=float(np.sqrt(np.mean(be**2))),LOS_rmse_m=float(np.sqrt(np.mean(le**2))),seconds=old['seconds'],J=J,PG=pg,stationary=pg<=1e-7,K=len(paths),stop=result['stop'],time_limit=result.get('budget'),over_5=old['seconds']>5,over_3=old['seconds']>3,selected_region=result.get('selected_region')))
        snapshot=root/'source_snapshot';snapshot.mkdir(exist_ok=True)
        for p,h in pre['hashes'].items():
            source=Path(p)
            if source.parent==HERE:
                assert fc.ref.sha(source)==h
                shutil.copyfile(source,snapshot/source.name)
    for label in ('pilot','followup'):
        conditions[label]={}
        for cond in ('L0','L2','L3','HIGH'):
            conditions[label][cond]={}
            for variant in ('GEO_3','REGION_3','REGION_5'):
                rr=[r for r in rows if r['stage']==label and r['condition']==cond and r['variant']==variant]
                a=np.array([r['seconds'] for r in rr]);e=np.array([r['rmse_m'] for r in rr])
                conditions[label][cond][variant]=dict(n=len(rr),rmse_m=float(np.sqrt(np.mean(e**2))),mean_seconds=float(a.mean()),max_seconds=float(a.max()),min_seconds=float(a.min()),over5=int((a>5).sum()),over3=int((a>3).sum()),stationary=sum(r['stationary'] for r in rr),normal_order_stop=sum(r['stop']=='no_detected_increment' for r in rr),K0=sum(r['K']==0 for r in rr),baseline_rmse_m=float(np.sqrt(np.mean([r['baseline_rmse_m']**2 for r in rr]))),LOS_rmse_m=float(np.sqrt(np.mean([r['LOS_rmse_m']**2 for r in rr]))),large_errors=sum(r['rmse_m']>100 for r in rr))
    out=dict(checks=checks,all_pass=all(checks.values()),conditions=conditions,rows=rows,description='saved-result audit, old fixed-geometry observations, no independent new geometry')
    fc.write(HERE/'audit.json',out)
    print(json.dumps(dict(all_pass=out['all_pass'],checks=len(checks),conditions=conditions),ensure_ascii=False))
    assert all(checks.values()),[k for k,v in checks.items() if not v]


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
