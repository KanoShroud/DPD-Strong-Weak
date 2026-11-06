"""Audit fresh-IQ transfer without selecting intermediate truth-best states."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
from reference import HERE, sha, write, FS, B


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--unseen',action='store_true')
    args=parser.parse_args()
    pointer='unseen_latest.json' if args.unseen else 'cross_latest.json'
    run=Path(json.loads((HERE/pointer).read_text(encoding='utf-8'))['path'])
    data=json.loads((run/'result.json').read_text(encoding='utf-8'))
    pre=json.loads((run/'preflight.json').read_text(encoding='utf-8'))
    checks=dict(source_identity=all(sha(p)==h for p,h in pre['hashes'].items()),
                all_recorded_checks=data['all_checks_pass'],count=len(data['rows'])==(2 if args.unseen else 12))
    start=time.perf_counter()
    summaries=[]
    conditions=sorted({r['condition'] for r in data['rows']})
    for condition in conditions:
        rows=[r for r in data['rows'] if r['condition']==condition]
        complete=all('errors_m' in r for r in rows)
        entry=dict(condition=condition,count=len(rows),complete=complete,
                   stopped=sum(r['stop']=='no_detected_increment' for r in rows))
        if complete:
            a=np.array([r['los_errors_m'] for r in rows]);b=np.array([r['errors_m'] for r in rows])
            entry.update(geometry_rmse=float(np.sqrt(np.mean(np.array([r['geometry_errors_m'] for r in rows])**2))),
                         los_rmse=float(np.sqrt(np.mean(a*a))),final_rmse=float(np.sqrt(np.mean(b*b))),
                         improved=int(np.sum(np.sum(a*a-b*b,axis=1)>1e-8)),
                         harmed=int(np.sum(np.sum(a*a-b*b,axis=1)<-1e-8)),
                         max_error=float(b.max()),orders=[len(r['paths']) for r in rows],
                         mean_seconds=float(np.mean([r['seconds'] for r in rows])))
        summaries.append(entry)
    compact=[]
    for row in data['rows']:
        tag=row['tag']
        with np.load(run/(tag+'_input.npz'),allow_pickle=False) as inp:
            coefficients=inp['relative_coefficients'];counts=inp['counts'];iq=inp['iq']
            import hashlib
            checks[tag+'/saved_iq_identity']=hashlib.sha256(np.ascontiguousarray(iq).tobytes()).hexdigest()==row['iq_sha']
            checks[tag+'/individual_LOS_dominant']=bool(np.max(abs(coefficients))<1)
            ratios=np.sum(abs(coefficients)**2,axis=2)
            checks[tag+'/generated_order']=bool(np.array_equal(np.sum(abs(coefficients)>0,axis=2),counts))
            checks[tag+'/total_energy']=bool(np.allclose(ratios[counts>0],1.2 if row['condition']=='HIGH' else 10**(-.6)))
        entry=dict(tag=tag,stop=row['stop'],physical_counts=counts,total_power_ratios=ratios)
        if 'final' in row:
            fits=[e['fit'] for e in row['history'] if 'fit' in e]
            checks[tag+'/iteration_budget']=all(s['iterations']<=300 for s in fits)
            checks[tag+'/monotone']=all(e.get('decrease',True) for e in row['history'])
            orders=np.zeros((5,2),int)
            for m,q in row['paths']:
                orders[m,q]+=1
            entry.update(geometry_errors=row['geometry_errors_m'],los_errors=row['los_errors_m'],errors=row['errors_m'],
                         orders=orders,order=len(row['paths']),mapping=row['final']['projected_gradient'],
                         pg_attempted=sum('pg_finish' in s for s in fits),
                         pg_accepted=sum(s.get('pg_finish',{}).get('accepted',False) for s in fits),
                         alias=row['delay_alias'],power_boundary=row['power_boundary'],seconds=row['seconds'],
                         frontend_seconds=row['frontend_seconds'],
                         refit_seconds=sum(s['seconds'] for s in fits),
                         search_seconds=sum(e['search']['seconds'] for e in row['history']))
            if row['paths']:
                vals=np.array(row['final']['x'][22::3])
                entry['max_extra_delay_us']=float(vals.max()/B*1e6)
                entry['record_length_us']=8192/FS*1e6
        compact.append(entry)
    out=HERE/(('unseen_review_' if args.unseen else 'cross_review_')+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    (out/'cross_review.py').write_bytes(Path(__file__).read_bytes())
    write(out/'result.json',dict(source=str(run),source_sha=sha(run/'result.json'),summaries=summaries,rows=compact,
                               checks=checks,all_checks_pass=all(checks.values()),batch_seconds=data['seconds'],analysis_seconds=time.perf_counter()-start))
    write(HERE/('unseen_review_latest.json' if args.unseen else 'cross_review_latest.json'),dict(path=str(out)))
    print(json.dumps(summaries,ensure_ascii=False,indent=2))
    print('审核',all(checks.values()),'目录',out)


if __name__=='__main__':
    main()
