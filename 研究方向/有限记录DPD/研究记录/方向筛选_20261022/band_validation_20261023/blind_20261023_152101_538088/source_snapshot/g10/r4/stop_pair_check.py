"""Check extended NOMP_JOINT keeps the extended NOMP prefix; final cost replay."""
import copy
from pathlib import Path
import sys
import time
import numpy as np
from ..metrics import evaluate
from .runtime import read,write,sha,seal,load,check_contract,require_exclusive,resource_check
from .solvers import run
from .config import BASE,ROOT


def main():
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):
            stream.reconfigure(encoding='utf-8',errors='replace')
    require_exclusive();resource_check()
    pointer=read(BASE/'stop_review.json')
    source=Path(pointer['out']).resolve(strict=True)
    review=read(source/'contract.json')
    if sha(source/'contract.json')!=pointer['contract_sha']:
        raise RuntimeError('Stop review identity changed')
    original=check_contract(Path(review['reference']))
    source_audit=read(source/'final_audit_report.json')
    out=source/'joint_prefix_and_final_report'
    if out.exists():
        raise FileExistsError('Fresh paired supplement required')
    out.mkdir()
    write(out/'contract.json',dict(reference=str(source),source_audit_sha=sha(source/'final_audit_report.json'),
        revision='CAP_ONLY_PAIRED_PREFIX',cycles=review['caps']['NOMP'],formal_started=False,
        source=str(Path(__file__).relative_to(ROOT)),source_sha=sha(__file__)))
    (out/'stop_pair_check.py').write_bytes(Path(__file__).read_bytes())
    cfg=copy.deepcopy(original['configurations']['NOMP_JOINT'])
    cfg['cycles']=review['caps']['NOMP']
    jobs=[j for j in review['jobs'] if j['method']=='NOMP']
    _,iq,_,_=load(original,jobs[0]['task'])
    start=time.perf_counter();run(iq,'NOMP_JOINT',cfg,resource_check)
    write(out/'warmup.json',dict(seconds=time.perf_counter()-start))
    rows=[]
    for index,job in enumerate(jobs,1):
        _,iq,truth,identity=load(original,job['task'])
        key='_'.join(map(str,job['task'][:3]))
        path=source/'rows'/(key+'_NOMP.json')
        assert sha(path)==source_audit['rows'][path.name]
        native=read(path)
        require_exclusive()
        result=run(iq,'NOMP_JOINT',cfg,resource_check)
        np.testing.assert_array_equal(result['mechanism']['prefix_positions'],native['result']['positions'])
        assert result['mechanism']['prefix_counts']==native['result']['counts']
        assert result['mechanism']['prefix_objective']==native['result']['native_objective']
        row=dict(task=job['task'],method='NOMP_JOINT',input_sha=identity,truth=truth.tolist(),
            result=result,metrics=evaluate(truth,result['positions']),native_row_sha=sha(path))
        row['record_sha']=seal(row);write(out/'rows'/(key+'.json'),row);rows.append(row)
        print(f'延长前缀配对 {index}/8：一致，完整耗时{result["wall_solver_seconds"]:.3f}s',flush=True)
    native_report=read(source/'report_v2/comparison_report.json')
    prior=read(Path(review['reference'])/'budget_estimate.json')
    native_rows=[read(p) for p in (source/'rows').glob('*.json')]
    measurements={m:[r['result']['wall_solver_seconds']+r['result']['certificate_seconds']
        for r in native_rows if r['job']['method']==m] for m in ('NOMP','BNOMP_REF')}
    measurements['NOMP_JOINT']=[r['result']['wall_solver_seconds']+r['result']['certificate_seconds'] for r in rows]
    mean_seconds=sum(v['mean_based_seconds'] for m,v in prior['by_method'].items() if m not in measurements)
    p95_seconds=sum(v['p95_based_seconds'] for m,v in prior['by_method'].items() if m not in measurements)
    mean_seconds+=2400*sum(float(np.mean(v)) for v in measurements.values())
    p95_seconds+=2400*sum(float(np.quantile(v,.95)) for v in measurements.values())
    other=prior['generation_io_seconds']+prior['recording_save_audit_seconds']
    forecast=dict(conditional=True,formal_cases=2400,formal_solves=16800,
        mean_hours=(1.25*(mean_seconds+other)+600)/3600,
        p95_scenario_hours=(1.25*(p95_seconds+other)+600)/3600,
        basis='BNOMP/NOMP/NOMP_JOINT measured on 8 paired observations with extended native caps; '
              'four AP methods retain previous 16-observation cost. No new geometry timed.',
        per_method_native_mean_seconds={m:float(np.mean(v)) for m,v in measurements.items()},
        native_cap_configuration=dict(BNOMP_REF=10000,NOMP=200,NOMP_JOINT=200),
        formal_configuration_adopted=False)
    report=dict(native_report,provisional_cost=forecast,
        nomp_joint=dict(cases=8,both20=sum(all(r['metrics']['success']['20']) for r in rows),
            mean_seconds=float(np.mean([r['result']['wall_solver_seconds'] for r in rows])),
            prefix_equivalent_count=8,stop_reasons=[r['result']['stop_reason'] for r in rows]),
        status='STOP_REVIEW_PASS_CONFIG_BUDGET_PENDING')
    write(out/'comparison_report.json',report)
    write(out/'budget_estimate.json',forecast)
    check_contract(Path(review['reference']))
    audit=dict(passed=True,count=8,extended_prefix_equivalent=True,
        rows={p.name:sha(p) for p in (out/'rows').glob('*.json')},formal_started=False,
        original_solver_rows_unchanged=all(sha(source/'rows'/n)==h for n,h in source_audit['rows'].items()))
    assert audit['original_solver_rows_unchanged']
    write(out/'final_audit_report.json',audit)
    write(BASE/'stop_review.json',dict(out=str(source),report=str(out),
        contract_sha=sha(source/'contract.json'),status=report['status']))
    print('最终回读目录：',out)
    print(forecast)


if __name__=='__main__':
    main()
