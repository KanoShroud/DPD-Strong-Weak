"""Report all terminal outputs, including negative outcomes and mixed timing bases."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from reference import HERE, sha, write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['repair','unseen']);args=parser.parse_args()
    run=Path(json.loads((HERE/f'feasible_{args.stage}_latest.json').read_text(encoding='utf-8'))['path'])
    data=json.loads((run/'result.json').read_text(encoding='utf-8'))
    pre=json.loads((run/'preflight.json').read_text(encoding='utf-8'))
    checks=dict(identity=all(sha(p)==h for p,h in pre['hashes'].items()),
                recorded_checks=data['all_checks_pass'],
                complete_count=len(data['rows'])==(12 if args.stage=='repair' else 2),
                statistical_stop=all(r['stop']=='no_detected_increment' for r in data['rows']))
    summaries=[];compact=[];trajectory=[]
    old=json.loads((Path(data['historical_source'])/'result.json').read_text(encoding='utf-8'))
    bytag={r['tag']:r for r in old['rows']}
    for row in data['rows']:
        tag=row['tag'];fits=[e['fit'] for e in row['history'] if 'fit' in e]
        checks[tag+'/finite']=bool(np.isfinite(row['errors_m']).all())
        checks[tag+'/budget']=all(s['iterations']<=300 for s in fits)
        checks[tag+'/closure']=all(s['closure_error']<1e-7 for s in fits)
        checks[tag+'/stationarity']=all(s['stationary'] and s['projected_gradient']<=1e-7 for s in fits)
        checks[tag+'/descent']=all(e.get('decrease',True) for e in row['history'])
        if not row['reused'] and args.stage=='repair':
            prior=bytag[tag]
            checks[tag+'/blind_start']=np.allclose(row['initial'],prior['initial'],atol=1e-10,rtol=0)
            checks[tag+'/iq_identity']=row['iq_sha']==prior['iq_sha']
            audit=Path(json.loads((HERE/'feasible_audit_latest.json').read_text(encoding='utf-8'))['path'])
            ar=next(r for r in json.loads((audit/'result.json').read_text(encoding='utf-8'))['rows'] if r['tag']==tag)
            first=next(e['step'] for e in ar['events'] if not e['same'])
            checks[tag+'/unchanged_prefix']=all(np.array_equal(e['fit']['x'],prior['history'][i]['fit']['x']) for i,e in enumerate(row['history'][:first-1]))
        compact.append(dict(tag=tag,reused=row['reused'],timing_basis=row['timing_basis'],stop=row['stop'],
            errors_m=row['errors_m'],los_errors_m=row['los_errors_m'],orders=len(row['paths']),
            seconds=row['seconds'],frontend_seconds=row['frontend_seconds'],
            refit_seconds=sum(s['seconds'] for s in fits),alias=row['delay_alias'],
            acceptance_seconds=sum(s['seconds']-s['feasible_acceptance']['before']['seconds'] for s in fits if 'feasible_acceptance' in s),
            projection_changed=sum(s.get('feasible_acceptance',{}).get('projection_change',0)>0 for s in fits),
            pg_attempts=sum(s.get('feasible_acceptance',{}).get('pg_attempted',False) for s in fits)))
        for budget in (0,5,10,20,40):
            available=[e for e in row['history'] if 'fit' in e and e['step']<=budget]
            ev=available[-1] if available else None
            state=ev['fit'] if ev else row['los']
            trajectory.append(dict(tag=tag,condition=row['condition'],budget=budget,
                actual_order=ev['step'] if ev else 0,held_terminal=budget>len(row['paths']),
                errors_m=ev['errors_m'] if ev else row['los_errors_m'],cost=state['cost'],
                interpretation='offline saved-state diagnostic, not a validated stopping rule'))
    for condition in sorted({r['condition'] for r in data['rows']}):
        rows=[r for r in data['rows'] if r['condition']==condition]
        a=np.array([r['los_errors_m'] for r in rows]);b=np.array([r['errors_m'] for r in rows])
        summaries.append(dict(condition=condition,count=len(rows),los_rmse=float(np.sqrt(np.mean(a*a))),
            final_rmse=float(np.sqrt(np.mean(b*b))),improved=int(np.sum(np.sum(a*a-b*b,axis=1)>1e-8)),
            harmed=int(np.sum(np.sum(a*a-b*b,axis=1)<-1e-8)),
            orders=[len(r['paths']) for r in rows],seconds=[r['seconds'] for r in rows],
            reused=[r['reused'] for r in rows]))
    out=HERE/(f'feasible_{args.stage}_review_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir(exist_ok=False)
    (out/'feasible_review.py').write_bytes(Path(__file__).read_bytes())
    write(out/'result.json',dict(source=str(run),source_sha=sha(run/'result.json'),checks=checks,
        all_checks_pass=all(checks.values()),summaries=summaries,rows=compact,trajectory=trajectory,
        execution_seconds=data['seconds']))
    write(HERE/f'feasible_{args.stage}_review_latest.json',dict(path=str(out)))
    print(json.dumps(summaries,ensure_ascii=False,indent=2))
    print('检查',len(checks),all(checks.values()),[k for k,v in checks.items() if not v],out)


if __name__=='__main__':main()
