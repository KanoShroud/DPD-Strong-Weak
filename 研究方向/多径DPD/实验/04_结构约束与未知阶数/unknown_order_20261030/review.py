"""Saved-result audit only; never selects a state using truth."""
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np

from reference import HERE, sha, write


def main():
    start = time.perf_counter()
    run = Path(json.loads((HERE/'full_latest.json').read_text(encoding='utf-8'))['path'])
    data = json.loads((run/'result.json').read_text(encoding='utf-8'))
    pre = json.loads((run/'preflight.json').read_text(encoding='utf-8'))
    checks = dict(input_and_source_identity=all(sha(p)==h for p,h in pre['hashes'].items()),
                  expected_48=len(data['rows'])==48, raw_checks=data['all_checks_pass'])
    previous=HERE.parent/'risk_step_20261030'
    oldrun=Path(json.loads((previous/'full_latest.json').read_text(encoding='utf-8'))['path'])
    oldrows={r['tag']:r for r in json.loads((oldrun/'result.json').read_text(encoding='utf-8'))['rows']}
    checks['same_los_endpoints_as_prior']=all(np.allclose(r['los']['p'],oldrows[r['tag']]['los']['p'],rtol=0,atol=1e-7) for r in data['rows'])
    summaries = []
    for mode in ('LOS','MP'):
        rows = [r for r in data['rows'] if r['mode']==mode]
        for label,group in [('all',rows)]+[(f'L{li}',[r for r in rows if r['tag'].startswith(f'L{li}_')]) for li in [20,21,22,23,40,41,42,43]]:
            if not group:
                continue
            before=np.array([r['los_errors_m'] for r in group])
            after=np.array([r['errors_m'] for r in group])
            delta=np.sum(before**2-after**2,axis=1)
            summaries.append(dict(mode=mode,group=label,count=len(group),los_rmse=float(np.sqrt(np.mean(before**2))),
                final_rmse=float(np.sqrt(np.mean(after**2))),source_rmse=np.sqrt(np.mean(after**2,axis=0)),
                los_p95=float(np.percentile(before,95)),final_p95=float(np.percentile(after,95)),
                improved=int(np.sum(delta>1e-8)),harmed=int(np.sum(delta < -1e-8)),
                stopped=sum(r['stop']=='no_detected_increment' for r in group),
                stationary=sum(r['final']['stationary'] for r in group),
                mean_paths=float(np.mean([len(r['paths']) for r in group])),
                min_paths=min(len(r['paths']) for r in group),max_paths=max(len(r['paths']) for r in group),
                added=sum(bool(r['paths']) for r in group),alias_cases=sum(r['delay_alias']>0 for r in group),
                power_boundary_cases=sum(r['power_boundary']>0 for r in group),
                near_zero_delay_cases=sum(any(abs(v)<1e-8 for v in r['final']['x'][22::3]) for r in group),
                mean_los_seconds=float(np.mean([r['los']['seconds'] for r in group])),
                mean_total_stage_seconds=float(np.mean([r['los']['seconds']+r['incremental_seconds'] for r in group]))))
    compact=[]
    for row in data['rows']:
        perlink=np.zeros((5,2),int)
        for m,q in row['paths']:
            perlink[m,q]+=1
        fits=[ev['fit'] for ev in row['history'] if 'fit' in ev]
        checks[row['tag']+'/bounded_iterations']=all(s['iterations']<=300 for s in fits)
        checks[row['tag']+'/monotone_fit']=all(ev.get('decrease',True) for ev in row['history'])
        compact.append(dict(tag=row['tag'],stop=row['stop'],order=len(row['paths']),link_orders=perlink,
            los_errors_m=row['los_errors_m'],errors_m=row['errors_m'],projected_gradient=row['final']['projected_gradient'],
            recentered=sum('attempts' in s for s in fits),calls=sum(s['calls'] for s in fits),
            last_score=row['history'][-1].get('pick'),last_threshold=row['history'][-1]['search']['threshold'],
            alias=row['delay_alias'],power_boundary=row['power_boundary']))
    out=HERE/('review_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    (out/'review.py').write_bytes(Path(__file__).read_bytes())
    write(out/'result.json',dict(source=str(run),source_sha=sha(run/'result.json'),summaries=summaries,
          rows=compact,checks=checks,all_checks_pass=all(checks.values()),batch_seconds=data['seconds'],
          all_order_stopped=data['all_order_stopped'],prior_los_result_sha=sha(oldrun/'result.json'),
          analysis_seconds=time.perf_counter()-start))
    write(HERE/'review_latest.json',dict(path=str(out)))
    print(json.dumps([s for s in summaries if s['group']=='all'],ensure_ascii=False,indent=2,
                     default=lambda v:v.tolist() if isinstance(v,np.ndarray) else bool(v)))
    print('AUDIT',all(checks.values()),len(checks),'BATCH_SECONDS',data['seconds'],'OUT',out)


if __name__=='__main__':
    main()
