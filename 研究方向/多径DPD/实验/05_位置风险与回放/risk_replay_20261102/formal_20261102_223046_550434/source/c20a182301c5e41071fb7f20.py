"""Engineering fixtures plus one fixed intermediate-state derivative replay."""
from datetime import datetime
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parent))
import runner as r  # noqa: E402


def main():
    start=time.perf_counter()
    out=r.HERE/('qa_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir()
    contract=r.make_contract()
    r.save(out/'contract.json',contract)
    checks={}
    def check(name,okay):
        checks[name]=bool(okay)
        if not okay:
            raise AssertionError(name)
    r.verify(contract)
    check('32_records',len(contract['inventory'])==32)
    check('four_conditions',len({x['case'].split('_')[0] for x in contract['inventory']})==4)
    check('no_new_formal_pointer',not (r.HERE/'latest_formal.json').exists())
    check('all_original_trajectories_reconstructed',all(x['count']==len(x['orders']) for x in contract['inventory']))
    check('two_capacity_cases_preserved',sum(x['stop']=='capacity' for x in contract['inventory'])==2)
    p=out/'中文封装.json'
    r.save(p,dict(text='非ASCII输出',x=[1,2]))
    check('unicode_artifact',r.load(p)['text']=='非ASCII输出')
    bad=r.read(p);bad['payload']['x']=[3];r.write(out/'tamper.json',bad)
    try:r.load(out/'tamper.json')
    except RuntimeError:check('artifact_tamper_rejected',True)
    else:check('artifact_tamper_rejected',False)
    with r.lock(out):
        try:
            with r.lock(out):pass
        except RuntimeError:check('duplicate_process_lock',True)
        else:check('duplicate_process_lock',False)
    calls=[]
    def function():calls.append(1);return dict(v=1)
    r.stage(out,'cached',out/'cached.json',function)
    r.stage(out,'cached',out/'cached.json',function)
    check('completed_stage_not_repeated',len(calls)==1)
    def fail():raise ValueError('fixture failure')
    try:r.stage(out,'failed',out/'failed.json',fail)
    except ValueError:pass
    check('failed_stage_not_cached',not (out/'failed.json').exists())
    r.write(out/'active.json',dict(id='interrupted_fixture',stage='fixture',seconds=2.5))
    before=r.cost(out)['seconds'];r.recover(out);after=r.cost(out)['seconds'];r.recover(out)
    check('interruption_cost_once',abs(after-before-2.5)<1e-8 and r.cost(out)['seconds']==after)
    solved={'final':{'p':[[1,2],[3,4]]}}
    tie=[dict(order=2,applicable=True,risk=-1.,position=[[1,2],[3,4]]),dict(order=1,applicable=True,risk=-1.,position=[[1,2],[3,4]])]
    choice=r.choose(tie,solved)
    check('negative_risk_preserved',choice['selected_risk']==-1.)
    check('exact_tie_lower_order',choice['selected_order']==1)
    empty=r.choose([dict(order=0,applicable=False,position=solved['final']['p'])],solved)
    check('no_score_fallback',empty['fallback'] and empty['selected_position']==solved['final']['p'])
    part=r.choose(tie+[dict(order=0,applicable=False)],solved)
    check('partial_coverage_explicit',not part['full_trajectory_coverage'] and not part['fallback'])
    check('risk_formula_scalar',r.score_value(np.ones(4),np.zeros(4),np.ones((4,3)),np.ones((4,3)),2.)==14.)
    ctx=r.get_context('L0_R00')
    st,pa=ctx['traj'][0]
    d=r.score_state(ctx,st,pa)
    check('LOS_endpoint_reuse',d['provenance']=='original_endpoint_influence' and d['applicable'])
    d0=r.choose([d],ctx['solved'])
    r.save(out/'L0_decision_before_evaluation.json',d0)
    evaluated=r.evaluate(d0,ctx,contract)
    check('decision_before_truth_evaluation',(out/'L0_decision_before_evaluation.json').exists())
    check('no_choice_no_gain',abs(evaluated['losses']['SELECTED_OR_FALLBACK']-evaluated['losses']['WIDE'])<1e-20)
    ctx2=r.get_context('L2_R00');state,paths=ctx2['traj'][1]
    cached=r.score_state(ctx2,state,paths)
    actual=r.score_state(ctx2,state,paths,reuse=False)
    r.save(out/'fixed_K1_recompute.json',dict(cached=cached,recomputed=actual))
    check('fixed_intermediate_diagnostic_pass',actual['applicable'])
    check('saved_score_replay_agrees',abs(cached['risk']-actual['risk'])<1e-10)
    check('state_sha_stable',cached['state_sha']==actual['state_sha'])
    altered=dict(state);altered['x']=list(state['x']);altered['x'][0]+=1e-8
    check('state_sha_detects_change',r.state_id(altered,paths)!=actual['state_sha'])
    for ident in ('HIGH_R00','HIGH_R05'):
        context=r.get_context(ident)
        s,p=context['traj'][0]
        val=r.score_state(context,s,p)
        check(ident+'_abstains',not val['applicable'] and val['reason']=='no_reliable_wide_reference')
    nonstationary=[]
    for rep in range(8):
        ctxh=r.get_context(f'HIGH_R{rep:02d}')
        if ctxh['solved']['stop']=='no_detected_increment':
            val=r.score_state(ctxh,*ctxh['traj'][0])
            if not val['applicable']:nonstationary.append(rep)
    check('five_original_LOS_not_silently_repaired',len(nonstationary)==5)
    try:r.check_diagnostic(dict(derivative_checks={'example':False}),'fixture',0)
    except AssertionError:check('derivative_failure_halts',True)
    else:check('derivative_failure_halts',False)
    original_score=r.score_state
    def bad_score(*args,**kwargs):
        raise r.CorrectnessFailure('fixture correctness failure',dict(derivative_checks={'example':False}))
    r.score_state=bad_score
    try:
        r.protected_score(out,ctx,st,pa)
    except r.CorrectnessFailure:
        check('correctness_failure_artifact_preserved',len(list((out/'failures').glob('*.json')))==1)
    finally:
        r.score_state=original_score
    report=r.load_report()
    check('singleton_interval_not_invented',report.interval([1.])['ci95'] is None)
    check('empty_interval',report.interval([])['n']==0)
    check('paired_constant_interval',report.interval([1.,1.])['ci95']==[1.,1.])
    synthetic=dict(evaluated)
    synthetic['case']='L0_R00'
    dest=out/'cases/L0_R00';dest.mkdir(parents=True)
    r.save(dest/'record.json',synthetic)
    summary=report.summarize(out,contract)
    check('QA_report_incomplete',summary['completed']==1 and not summary['complete'])
    data=(out/'records.csv').read_bytes()
    check('csv_BOM_and_newlines',data.startswith(b'\xef\xbb\xbf') and b'\r\r\n' not in data)
    check('HTML_local_links',all((out/name).exists() for name in ('summary.json','records.csv','完整风险回放报告.md','contract.json')))
    r.verify(contract)
    result=dict(all_pass=all(checks.values()),checks=checks,count=len(checks),
                directory=str(out),contract=contract,seconds=time.perf_counter()-start,
                actual_new_state_diagnostics=1,new_localization_runs=0)
    r.save(out/'test_result.json',result)
    r.save(r.HERE/'test_result.json',result)
    r.write(r.HERE/'budget.json',r.budget(contract))
    print('工程/科学短测通过：',len(checks),'项；新增诊断1状态，不是正式回放。',flush=True)
    print('证据目录：',out,flush=True)
    print(r.encode(r.budget(contract)),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
