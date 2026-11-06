"""Engineering/numerical contract checks; synthetic fixtures are not experiments."""
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import runner
from report import stats,interval
from storage import save,load,read,write,lock,recover,stage,cost,append


def main():
    contract=runner.source_contract();runner.verify(contract)
    out=Path(__file__).resolve().parent/('qa_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    checks={}
    manifest=runner.cases();smoke=runner.cases(True)
    checks['32_cases']=len(manifest)==32
    checks['four_by_eight']=all(sum(c['condition']==name for c in manifest)==8 for name in runner.CONDITIONS)
    checks['unique_noise_seeds']=len({tuple(c['seed']) for c in manifest})==32
    checks['development_seed_not_formal']=not {tuple(c['seed']) for c in smoke}&{tuple(c['seed']) for c in manifest}
    checks['frozen_backend']=runner.engine.fit is runner.frozen_fit
    for condition in runner.CONDITIONS:
        case=next(c for c in manifest if c['condition']==condition)
        iq=runner.generate(contract,case);again=runner.generate(contract,case)
        other=runner.generate(contract,dict(case,seed=[2026110203,runner.CONDITIONS.index(condition),7]))
        checks[condition+'/deterministic']=np.array_equal(iq,again)
        checks[condition+'/different_noise']=not np.array_equal(iq,other)
        checks[condition+'/shape_finite']=iq.shape==(5,8192) and np.isfinite(iq).all()
        with np.load(contract['conditions'][condition]['bank'],allow_pickle=False) as z:bank={k:z[k] for k in z.files}
        old,_=runner.engine.observation(bank,condition);clean=old-bank['noise']
        relative=float(np.mean(abs(iq-clean)**2)/float(bank['variance']))
        checks[condition+'/noise_scaling']=abs(relative-1)<.04
    folder,iq,info=runner.input_case(out,contract,smoke[0])
    _,again,again_info=runner.input_case(out,contract,smoke[0])
    checks['input_resume_identity']=np.array_equal(iq,again) and info==again_info
    with np.load(folder/'input.npz',allow_pickle=False) as z:checks['input_has_only_iq']=z.files==['iq']
    target=out/'测试完整性.json';save(target,dict(说明='中文路径与日志',number=3))
    checks['unicode_roundtrip']=load(target)['说明']=='中文路径与日志'
    corrupted=read(target);corrupted['payload']['number']=4;write(target,corrupted)
    try:load(target);checks['tamper_detected']=False
    except RuntimeError:checks['tamper_detected']=True
    with lock(out):
        try:
            with lock(out):checks['second_writer_blocked']=False
        except RuntimeError:checks['second_writer_blocked']=True
    calls=[]
    def compute():calls.append(1);return dict(result=42)
    p=out/'cached.json'
    a,new=stage(out,'test_stage',p,compute)
    b,new_again=stage(out,'test_stage',p,compute)
    checks['completed_stage_not_recomputed']=len(calls)==1 and new and not new_again and a==b
    append(out,dict(kind='start',id='simulated_kill',stage='synthetic_interruption'))
    write(out/'active.json',dict(id='simulated_kill',stage='synthetic_interruption',seconds=2.5))
    before=cost(out)['seconds'];recover(out);once=cost(out)['seconds'];recover(out);twice=cost(out)['seconds']
    checks['interrupted_cost_preserved_once']=abs(once-before-2.5)<1e-9 and twice==once and cost(out)['has_lower_bound']
    try:
        stage(out,'synthetic_failure',out/'not_completed.json',lambda:(_ for _ in ()).throw(ValueError('测试异常')))
    except ValueError:pass
    checks['failure_not_cached']=not (out/'not_completed.json').exists()
    checks['active_cleared_after_error']=read(out/'active.json') is None
    fixtures=[]
    for i in range(8):
        error=np.array([[i*.1,.2],[-.1,.3]])
        loss=float(np.sum(error**2)/2)
        fixtures.append(dict(solver_status='returned',stop='no_detected_increment',
            evaluation={name:dict(position_error=error,loss_m2=loss,stationary=True) for name in ('LOS','WIDE')},
            risk=dict(applicable=True,LOS=loss,WIDE=loss,difference=0.)))
    s=stats(fixtures)
    checks['mse_bias_variation_identity']=s['methods']['LOS']['decomposition_error']<1e-12
    checks['paired_difference_zero']=s['risk_difference_minus_loss_difference']['mean']==0
    fixtures.append(dict(solver_status='exception',stop=None,evaluation={},risk=dict(applicable=False,reason='failed')))
    ss=stats(fixtures,9)
    checks['failure_denominator_kept']=ss['completed']==9 and ss['methods']['LOS']['missing']==1 and ss['joint_subset_n']==8
    checks['one_repeat_no_interval']=interval([1.])['interval95'] is None
    fake=dict(los=dict(p=[[0.,0.],[100.,0.]]),final=dict(p=[[100.,0.],[0.,0.]]))
    info=dict(applicable=True,L=np.eye(4),sigma2_est=1.)
    checks['label_swap_not_silently_matched']=not runner.risk_compare(fake,info,info)['applicable']
    altered=dict(contract);altered['hashes']=dict(contract['hashes'])
    key=next(iter(altered['hashes']));altered['hashes'][key]='0'*64
    try:runner.verify(altered);checks['source_drift_blocked']=False
    except RuntimeError:checks['source_drift_blocked']=True
    result=dict(checks=checks,all_pass=all(checks.values()),hashes=contract['hashes'],
                output=str(out),kind='engineering_and_synthetic_fixture_checks_not_localization',time=time.time())
    save(out/'result.json',result);save(runner.HERE/'test_result.json',result)
    print('检查完成：',len(checks),all(checks.values()),out,flush=True)
    if not all(checks.values()):raise RuntimeError(str({k:v for k,v in checks.items() if not v}))


if __name__=='__main__':main()
