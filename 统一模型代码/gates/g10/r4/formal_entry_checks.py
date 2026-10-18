"""Mocked stage-transition checks; consumes saved prep outputs, never solves new IQ."""
import copy
import io
import shutil
from contextlib import redirect_stdout, ExitStack
from unittest.mock import patch
from pathlib import Path
import numpy as np
from . import formal_runner as runner
from . import formal_runtime as runtime
from .formal_protocol import CORE, EXTRA, job_key, BASE
from .runtime import read,write,sha


def main(preparation=None):
    prep=Path(preparation or read(BASE/'formal_release.json')['out']).resolve(strict=True)
    if not read(prep/'final_audit_report.json')['passed']:
        raise RuntimeError('A passed numerical preparation is required')
    c=read(prep/'contract.json')
    selected={}
    for path in (prep/'rows').glob('0_200_0_*.json'):
        row=read(path); selected[row['job']['method']]=row
    methods=CORE+EXTRA+('SBL',)
    assert set(selected)==set(methods)
    task=[0,200,0,'mixed']
    identity=selected['NOMP']['input_identity']; truth=np.array(selected['NOMP']['truth'])
    def test_jobs(stage):
        mm=CORE if stage=='core' else EXTRA if stage=='supplement' else ('SBL',)
        return [dict(task=task,method=m,stage=stage) for m in mm]
    def input_fixture(*args):
        return np.zeros((4,8192),complex),truth,identity
    def result_fixture(iq,method,*args):
        return copy.deepcopy(selected[method]['result'])
    def manifest_fixture(out,*args):
        write(out/'input_manifest.json',{'fixture':'no scientific observation generated'})
        return sha(out/'input_manifest.json')
    def report_fixture(out,contract,stages):
        write(out/'reports'/stages[-1]/'comparison_report.json',{'engineering_mock':True})
    out=prep/'staged_entry_fixture'
    if out.exists():
        raise FileExistsError('Fresh test fixture required')
    write(out/'status.json',dict(status='CREATED',completed_stages=[]))
    buffer=io.StringIO()
    with ExitStack() as stack:
        stack.enter_context(patch.object(runner,'check',return_value=c))
        stack.enter_context(patch.object(runner,'jobs',side_effect=test_jobs))
        stack.enter_context(patch.object(runtime,'jobs',side_effect=test_jobs))
        stack.enter_context(patch.object(runner,'ensure_input',side_effect=input_fixture))
        stack.enter_context(patch.object(runtime,'ensure_input',side_effect=input_fixture))
        stack.enter_context(patch.object(runtime,'make',side_effect=AssertionError('Unexpected data generation')))
        solve=stack.enter_context(patch.object(runner.adapter,'run',side_effect=result_fixture))
        stack.enter_context(patch.object(runner,'manifest_at_end',side_effect=manifest_fixture))
        stack.enter_context(patch.object(runner,'generate',side_effect=report_fixture))
        with redirect_stdout(buffer):
            runner.execute(out)
            assert read(out/'status.json')['status']=='COMPLETED'
            assert solve.call_count==15
            runner.execute(out)
            assert solve.call_count==15
        assert len(list((out/'rows').glob('*.json')))==15
        assert read(out/'final_audit_report.json')['passed']
        resume=prep/'core_completed_resume_fixture'
        write(resume/'status.json',dict(status='READY_TO_SUPPLEMENT',completed_stages=['core']))
        (resume/'rows').mkdir()
        for job in test_jobs('core'):
            name=job_key(job)+'.json'
            shutil.copyfile(out/'rows'/name,resume/'rows'/name)
        for name in ('input_manifest.json','input_manifest_identity.json'):
            shutil.copyfile(out/name,resume/name)
        with redirect_stdout(buffer):
            runner.execute(resume)
        assert solve.call_count==24  # only the nine remaining methods
        assert read(resume/'status.json')['status']=='COMPLETED'
        budget=prep/'budget_boundary_fixture'
        write(budget/'status.json',dict(status='CREATED',completed_stages=[]))
        base=runner.Ledger
        class AtLimit(base):
            def total(self):
                return 12*3600
        with patch.object(runner,'Ledger',AtLimit),redirect_stdout(buffer):
            runner.execute(budget)
        assert read(budget/'status.json')['status']=='BUDGET_PAUSED'
        assert solve.call_count==24
    (prep/'staged_entry_test_log.txt').write_text(buffer.getvalue(),encoding='utf-8')
    # Exercise the real create path without touching the user's formal pointer.
    create_root=prep/'creation_fixture'
    create_root.mkdir()
    with patch.object(runtime,'BASE',create_root):
        created=runtime.create(prep,c)
    created_contract=read(created/'contract.json')
    assert len(created_contract['tasks'])==2400
    assert created_contract['expected']==dict(core=14400,supplement=1920,sbl=240)
    assert created_contract['seeds']==c['seeds']
    assert not (created/'inputs').exists()
    write(prep/'staged_entry_checks.json',dict(passed=True,mocked_execution_only=True,
        one_launch_all_stages=True,core_completed_resume_without_recompute=True,
        completed_run_no_recompute=True,budget_checked_before_next_solve=True,
        formal_manifest_2400=True,formal_manifest_seeds_preserved=True,
        paired_audit_passed=True,formal_input_generated=False,real_solver_invoked=False,
        test_source_sha=sha(__file__),runner_sha=sha(runner.__file__),
        fixture_jobs=[job_key(j) for s in ('core','supplement','sbl') for j in test_jobs(s)]))
    print('连续入口模拟通过：一次15次调用；重复点击不重算；核心完成后恢复只补9次；预算边界不启动下一例。')


if __name__=='__main__':
    main()
