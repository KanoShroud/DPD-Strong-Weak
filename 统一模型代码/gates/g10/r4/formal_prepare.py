"""Finite release checks on previously used IQ only; formal observations stay ungenerated."""
from pathlib import Path
import copy
import os
import subprocess
import sys
import time
import traceback
from unittest.mock import patch
import numpy as np
from ..metrics import evaluate
from ..runtime import verify
from ..r2.band.solvers import run as old_band_run
from .runtime import load,check_contract,paired_checks
from . import formal_adapter as adapter
from .formal_protocol import (ROOT,BASE,ENTRY,REVISION,CORE,METHODS,layouts,jobs,
    job_key,solver_seed,frozen_configurations,LIMIT_SECONDS)
from .formal_runtime import (read,write,sha,seal,fresh,lock,require_exclusive,resource_check,
    sources,snapshot,environment,Ledger,compact_result)
from .formal_report import generate

PROBES=tuple((g,200,0 if g%2==0 else 15,'mixed') for g in range(4))


def command_check(args,cwd):
    env=os.environ.copy(); env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8:replace')
    r=subprocess.run([sys.executable,str(ENTRY),*args],cwd=str(cwd),env=env,
        capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=45)
    if r.returncode:
        raise RuntimeError(r.stdout+'\n'+r.stderr)
    if '\ufffd' in r.stdout+r.stderr:
        raise RuntimeError('Non-ASCII output capture was not lossless')
    return dict(command=args,cwd=str(cwd),returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)


def forecast(rows,ga_rows):
    oldprep=Path(read(BASE/'preparation.json')['out']).resolve(strict=True)
    old=read(oldprep/'budget_estimate.json')
    ref=read(oldprep/'contract.json')['reference']['out']; ref=Path(ref)
    ref=(ref if ref.is_absolute() else ROOT/ref).resolve(strict=True)
    r2=read(ref/'comparison_report.json')['summaries']
    means={}; details={}
    for method in METHODS:
        values=[r['result']['complete_seconds'] for r in rows if r['job']['method']==method]
        prior=next((r['seconds'] for r in r2 if r['method']==method and r['kind']=='mixed'),None)
        measured=float(np.mean(values))
        # Long-run legacy cost guards against an optimistic four-case pilot.
        cost=max(measured,prior or 0.)
        count=2400 if method in CORE else 240
        means[method]=cost
        details[method]=dict(pilot_n=len(values),pilot_mean=measured,prior_mean=prior,
            planning_seconds_per_case=cost,planned_count=count,planned_seconds=cost*count,
            pilot_p95=float(np.quantile(values,.95)))
    total=sum(v['planned_seconds'] for v in details.values())
    job_count=sum(len(jobs(s)) for s in ('core','supplement','sbl'))
    overhead=job_count*old['observed_job_overhead_seconds']+old['generation_io_seconds']
    # Additional report sets and hierarchical bootstrap receive 20 min reserve.
    estimate=(1.25*(total+overhead)+1200)/3600
    return dict(estimate_hours=estimate,approved_hours=12,within_budget=estimate<=12,
        by_method=details,cost_means=means,solves=job_count,formal_cases=2400,panorama_cases=240,
        safety_factor=1.25,report_reserve_seconds=1200,
        basis='Four existing geometries, one realization, alternating 0/15 dB; for frozen legacy methods use max(pilot mean, prior 120-case mean).',
        ga_diagnostic_solves=len(ga_rows))


def recovery_check(out,c,row,cache):
    from . import formal_runner as runner
    toy=out/'recovery_fixture'; job=row['job']; fixture=copy.deepcopy(c)
    iq,truth,identity=cache[tuple(job['task'])]
    clock=Ledger(toy)
    with patch.object(runner,'ensure_input',return_value=(iq,truth,identity)):
        with patch.object(runner.adapter,'run',side_effect=KeyboardInterrupt('intentional')):
            try:
                runner.record(toy,job,fixture,clock)
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError('Interruption was swallowed')
        assert not (toy/'rows'/(job_key(job)+'.json')).exists()
        with patch.object(runner.adapter,'run',return_value=row['result']) as mock:
            _,new=runner.record(toy,job,fixture,clock)
            assert new and mock.call_count==1
        with patch.object(runner.adapter,'run',side_effect=AssertionError('Completed solve reran')):
            assert not runner.record(toy,job,fixture,clock)[1]
    before=clock.total(); clock.flush()
    restored=Ledger(toy); assert restored.total()>=before
    return dict(passed=True,interrupted_row_not_committed=True,completed_row_reused=True,
                cumulative_ledger_not_reset=True)


def main():
    require_exclusive(); resource_check()
    cfgs,ref=frozen_configurations()
    original=check_contract(Path(ref['preparation']))
    cached={}; raw_scenes={}
    for task in PROBES:
        scene,iq,truth,digest=load(original,task)
        cached[task]=(iq,truth,dict(iq_sha=digest,reference_task=list(task)))
        raw_scenes[task]=scene
    out=fresh('formal_prepare')
    if out.exists():
        raise FileExistsError('Fresh preparation required')
    out.mkdir()
    identity=sources()
    c=dict(revision=REVISION,stage='preparation',reference=ref,configurations=cfgs,
        sources=identity,environment=environment(),layouts=layouts(),budget_seconds=LIMIT_SECONDS,
        tasks=PROBES,formal_observations_generated=False,
        ga_rule='200/600 generations; select 600 only if a capped 200 case improves native joint objective by >1e-8',
        expected={s:len(jobs(s)) for s in ('core','supplement','sbl')})
    assert c['expected']==dict(core=14400,supplement=1920,sbl=240)
    assert len(c['layouts'])==20
    write(out/'contract.json',c); snapshot(out,identity)
    print('正式接入检查目录：'+str(out),flush=True)
    with lock(out):
        clock=Ledger(out)
        try:
            write(out/'console_check.json',command_check(['--help'],ENTRY.parent))
            warm=copy.deepcopy(cfgs['NOMP']); warm['cycles']=1
            start=time.perf_counter(); adapter.run(cached[PROBES[0]][0],'NOMP',warm)
            write(out/'warmup.json',dict(seconds=time.perf_counter()-start))
            print('阶段1/3：GA重复初值与200/600代有限停止核验。',flush=True)
            ga_rows=[]; needs_extension=False
            for task in PROBES:
                iq,truth,identity_iq=cached[task]; pair=[]
                for cap in (200,600):
                    cfg=copy.deepcopy(cfgs['GA']); cfg['generations']=cap
                    r=adapter.run(iq,'GA',cfg,solver_seed(task),clock.check)
                    row=dict(task=list(task),cap=cap,result=r,input_identity=identity_iq,metrics=evaluate(truth,r['positions']))
                    write(out/'ga_stop'/f'{task[0]}_{cap}.json',row); pair.append(r); ga_rows.append(row)
                a,b=pair
                np.testing.assert_array_equal(a['mechanism']['initial_population'],b['mechanism']['initial_population'])
                for x,y in zip(a['history'],b['history']):
                    np.testing.assert_array_equal(x['positions'],y['positions'])
                    assert x['native_objective']==y['native_objective']
                gain=a['joint_objective']-b['joint_objective']
                if a['stop_reason']=='ITERATION_LIMIT' and gain>1e-8:
                    needs_extension=True
                clock.flush()
                print(f"GA布局{task[0]}：200代{a['stop_reason']}，600代{b['stop_reason']}；目标改善{gain:.3g}",flush=True)
            cfgs['GA']['generations']=600 if needs_extension else 200
            c['configurations']=cfgs
            write(out/'ga_stop_review.json',dict(passed=True,selected_generations=cfgs['GA']['generations'],
                rows=8,decision_uses_truth=False,remaining_caps=sum(r['result']['stop_reason']=='ITERATION_LIMIT' for r in ga_rows if r['cap']==600),
                interpretation='Finite GA budget; iteration cap is not a proof of convergence'))
            write(out/'contract.json',c)
            print('阶段2/3：四条已有观测、15版本接入与成本检查；正式新布局未求解。',flush=True)
            rows=[]; compatibility={}
            for i,task in enumerate(PROBES):
                iq,truth,identity_iq=cached[task]; group={}
                order=METHODS[i:]+METHODS[:i]
                for method in order:
                    if clock.total()>90*60:
                        raise RuntimeError('90 minute preparation schedule exceeded; no next solve started')
                    require_exclusive()
                    stage='core' if method in CORE else 'sbl' if method=='SBL' else 'supplement'
                    job=dict(task=list(task),method=method,stage=stage)
                    result=adapter.run(iq,method,cfgs[method],solver_seed(task),clock.check,True)
                    row=dict(job=job,input_identity=identity_iq,truth=truth.tolist(),
                        result=compact_result(result,True),metrics=evaluate(truth,result['positions']))
                    row['record_sha']=seal(row)
                    write(out/'rows'/(job_key(job)+'.json'),row); rows.append(row); group[method]=row
                    if i==0 and method in ('LS','MUSIC','MVDR','AP_FULL','SBL'):
                        old=old_band_run(iq,method,2,cfgs[method],solver_seed(task),clock.check)
                        np.testing.assert_allclose(old['positions'],result['positions'],rtol=0,atol=1e-5)
                        assert old['stop_reason']==result['stop_reason']
                        compatibility[method]=dict(positions_match=True,stop_matches=True,
                            max_difference_m=float(abs(np.array(old['positions'])-result['positions']).max()))
                    clock.flush()
                paired_checks(group)
                print(f'接入检查完成{i+1}/4布局（{len(rows)}/60次）；累计{clock.total()/60:.1f}分钟。',flush=True)
            write(out/'adapter_equivalence.json',compatibility)
            print('阶段3/3：保存恢复、工作目录、统计重绘与预算核对。',flush=True)
            recovery=recovery_check(out,c,rows[0],cached)
            write(out/'recovery_checks.json',recovery)
            plot_contract=copy.deepcopy(c)
            plot_contract['layouts']=[dict(id=t[0],category=t[0],role='analysis',positions=raw_scenes[t]['positions'].tolist()) for t in PROBES]
            with patch.object(adapter,'run',side_effect=AssertionError('Redraw called solver')):
                generate(out,plot_contract,['core','supplement','sbl'],preparation=True)
            budget=forecast(rows,ga_rows); write(out/'budget_estimate.json',budget)
            c['cost_means']=budget['cost_means']; write(out/'contract.json',c)
            verify(c['sources']); check_contract(Path(ref['preparation']))
            for task in PROBES:
                load(original,task)
            ids={p.name:sha(p) for p in (out/'rows').glob('*.json')}
            for p in (out/'rows').glob('*.json'):
                row=read(p)
                assert row['record_sha']==seal(row)
                assert row['metrics']==evaluate(row['truth'],row['result']['positions'])
            write(out/'final_audit_report.json',dict(passed=True,solves=60,ga_diagnostic_solves=8,rows=ids,
                inputs_and_frozen_sources_unchanged=True,core_prefix_checks=True,
                legacy_adapter_equivalence=compatibility,recovery=recovery,redraw_did_not_solve=True))
            ready=dict(ready=budget['within_budget'],status='READY_FOR_USER' if budget['within_budget'] else 'BUDGET_CONFIRMATION_REQUIRED',
                forecast_hours=budget['estimate_hours'],approved_hours=12,formal_started=False)
            write(out/'readiness.json',ready)
            write(BASE/'formal_release.json',dict(out=str(out),contract_sha=sha(out/'contract.json'),readiness_sha=sha(out/'readiness.json')))
            if ready['ready']:
                # Real invocation from the exact PyCharm script directory, with a
                # live frozen release; it must never create a formal run.
                write(out/'working_directory_check.json',command_check(['--check-only'],ENTRY.parent))
            clock.flush()
            write(out/'status.json',dict(status=ready['status'],seconds=clock.total(),formal_started=False))
            print(f"接入完成：{ready['status']}，预计{budget['estimate_hours']:.2f}小时，正式尚未启动。",flush=True)
        except BaseException:
            clock.flush(keep=True)
            if (out/'readiness.json').exists():
                write(out/'readiness.json',dict(ready=False,status='FAILED_PREPARATION_RETAINED'))
            write(out/'status.json',dict(status='FAILED_PREPARATION_RETAINED',error=traceback.format_exc(),formal_started=False))
            raise


if __name__=='__main__':
    main()
