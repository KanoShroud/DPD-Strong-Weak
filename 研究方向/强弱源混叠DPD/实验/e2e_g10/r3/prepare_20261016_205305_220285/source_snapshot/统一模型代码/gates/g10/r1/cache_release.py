"""Release five-repeat contract from finite cache checks and preserved pilot evidence."""
import argparse
from pathlib import Path
from .config import BASE, METHODS, GAPS, TIMES, SEED, FS, N, FORMAL_REPEATS, method_configs
from .runtime import fresh, preflight, read, write, sha, finish_audit
from .scenes import task_list, geometry_audit
from .test_shared import suite, check_shared


def verify_evidence(folder):
    audit = read(folder/'final_audit_report.json')
    assert audit['identity_passed']
    contract = read(folder/'contract.json')
    for relative, expected in {**contract['sources'], **contract['r1_sources']}.items():
        assert sha(folder/'source_snapshot'/relative) == expected
    for name, expected in audit['rows'].items():
        assert sha(folder/'rows'/name) == expected
    for relative, expected in audit.get('states',{}).items():
        assert sha(folder/relative) == expected
    for relative, expected in audit.get('output_replays',{}).items():
        assert sha(folder/relative) == expected
    for name in audit['inputs']:
        info = read(folder/'inputs'/name)
        assert sha(folder/info['path']) == info['sha256'] == audit['inputs'][name]
    return audit


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--probe',type=Path,required=True)
    args=parser.parse_args()
    probe=args.probe.resolve(strict=True)
    report=read(probe/'report.json')
    assert report['status']=='PASSED' and len(report['comparisons'])==4
    prior=Path(report['prior']).resolve(strict=True)
    assert not (BASE/'active_run.json').exists(), 'Do not replace an active formal contract'
    verify_evidence(prior); verify_evidence(probe)
    out=fresh('cache_delivery')
    preflight(out,dict(stage='cache_repeat5_delivery',prior=str(prior),probe=str(probe),
                       comparison_evidence_dir=str(prior),
                       prior_audit_sha=sha(prior/'final_audit_report.json'),
                       probe_audit_sha=sha(probe/'final_audit_report.json')))
    print('OUT='+str(out),flush=True)
    tests=suite(out/'checks')
    assert tests['passed']
    tests['cached_shared']=check_shared(out/'checks_cached',dictionary_cache=True)
    tasks=list(task_list(FORMAL_REPEATS,GAPS,100))
    assert len(tasks)==260 and len(set(tasks))==260
    assert {t[1] for t in tasks}==set(range(100,105))
    tests['formal_tasks']=dict(observations=len(tasks),solves=len(tasks)*len(METHODS),
                               repeats=FORMAL_REPEATS,realizations=list(range(100,105)))
    write(out/'numerical_tests.json',tests)
    cfg=dict(revision='G10_R1_SHARED_R5_CACHE_V1',methods=method_configs(),gaps=list(GAPS),
             repeats=FORMAL_REPEATS,realization_start=100,sample_rate_hz=FS,N=N,
             geometry=geometry_audit(),times=list(TIMES),sbl_checkpoints=[25,50,100,200,300],
             group_diameter_m=.05,expected_observations=len(tasks),expected_rows=len(tasks)*len(METHODS),
             protocol='EVAL-20261011-v2',seed=SEED)
    write(out/'frozen_config.json',cfg)
    write(out/'formal_task_manifest.json',dict(tasks=tasks,method_count=len(METHODS)))
    oldrows=[read(p) for p in (prior/'rows').glob('*.json')]
    newrows=[read(p) for p in (probe/'rows').glob('*.json')]
    factors={kind:max(c['ratio'] for c in report['comparisons'] if c['kind']==kind)
             for kind in ('weak','strong','mixed')}
    costs,aux={},{}
    for method in METHODS:
        costs[method]=aux[method]=0.
        for kind,count in (('weak',20),('strong',120),('mixed',120)):
            rows=[r for r in oldrows if r['method']==method and r['kind']==kind]
            costs[method]+=count*max(r['result']['seconds'] for r in rows)*(factors[kind] if method=='SBL' else 1.)
            observed=rows+[r for r in newrows if r['method']==method and r['kind']==kind]
            aux[method]+=count*max(r['result'].get('recording_seconds',0.)+
                                  r['result'].get('output_replay_wall_seconds',0.) for r in observed)
    hours=(1.25*(sum(costs.values())+sum(aux.values()))+1800)/3600
    oldbudget=read(prior/'budget_estimate.json')
    write(out/'budget_estimate.json',dict(hours=hours,per_method_solver_seconds=costs,
          per_method_recording_and_replay_seconds=aux,factor=1.25,overhead_seconds=1800,
          estimated_sbl_state_gib=oldbudget['estimated_sbl_state_gib']*.5,
          input_role_counts=dict(weak=20,strong=120,mixed=120),cache_time_ratio_by_kind=factors,
          evidence='Prior 38-solve maximum by kind, scaled by slowest observed cache ratio within kind; four full cache pairs, not a runtime guarantee',
          prior=str(prior),probe=str(probe),hard_per_solver_time_cap=False))
    # Replay/schema tests from the unchanged reporter are reused; no new localization figures are claimed.
    write(out/'reused_checks.json',dict(prior_numerical_sha=sha(prior/'numerical_tests.json'),
          reused='Offline plotting and completed-row skip; reporter, observer and decoder unchanged',
          new_checks='Current numerical suite, cached checkpoint continuation, formal task enumeration'))
    finish_audit(out)
    contract=read(out/'contract.json')
    ready=dict(ready=hours<=12,engineering_ready=True,budget_ready=hours<=12,
               reason='READY' if hours<=12 else 'BUDGET_CONFIRMATION_REQUIRED',
               estimated_hours=hours,approved_hours=12.,config_sha=sha(out/'frozen_config.json'),
               sources={**contract['sources'],**contract['r1_sources']},
               numerical_passed=True,resume_ok=True,completed_solver_rows=4,
               reused_prior_solver_rows=38,comparison_evidence_dir=str(prior))
    write(out/'readiness.json',ready)
    write(out/'status.json',dict(status='PREPARATION_COMPLETED',formal_ready=ready['ready'],
                               formal_started=False,expected_observations=260,expected_rows=2080))
    write(BASE/'release.json',dict(preparation=str(out.resolve()),readiness_sha=sha(out/'readiness.json')))
    write(BASE/'preparation_pointer.json',dict(out=str(out.resolve())))
    print(f'工程核验通过；5重复正式估算 {hours:.2f} 小时，预算上限仍为12小时。',flush=True)


if __name__=='__main__':
    main()
