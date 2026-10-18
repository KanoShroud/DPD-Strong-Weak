"""Verify a Windows IO-only fix and preserve interrupted evidence in a fresh run."""
import ast
import copy
import ctypes
from pathlib import Path
import shutil
import threading
from unittest.mock import patch
from .. import runtime as io
from .formal_protocol import ROOT,BASE,ENTRY,jobs,job_key,key
from .formal_runtime import (read,write,sha,fresh,sources,snapshot,environment,
    ensure_input,validate,paired_checks,require_exclusive,resource_check)
from .formal_prepare import command_check


def io_tests(out):
    target=out/'中文账本.json'
    write(target,{'seconds':12.5,'备注':'原账本'})
    previous=target.read_bytes()
    real_replace=io.os.replace
    transient=PermissionError(13,'simulated sharing conflict'); transient.winerror=5
    calls=[]
    def briefly_busy(src,dst):
        calls.append(1)
        if len(calls)<3:
            raise transient
        return real_replace(src,dst)
    with patch.object(io.os,'replace',side_effect=briefly_busy),patch.object(io.time,'sleep'):
        write(target,{'seconds':13.5,'备注':'有限重试成功'})
    assert len(calls)==3 and read(target)['seconds']==13.5
    retained=target.read_bytes()
    with patch.object(io.os,'replace',side_effect=transient) as replace,patch.object(io.time,'sleep'):
        try:
            write(target,{'seconds':14.5})
        except PermissionError:
            pass
        else:
            raise AssertionError('Permanent denial was swallowed')
        assert replace.call_count==8
    assert target.read_bytes()==retained
    assert read(target.with_name(target.name+'.pending'))['seconds']==14.5
    with patch.object(io.os,'replace',side_effect=OSError(28,'disk full')) as replace:
        try:
            write(target,{'seconds':15.5})
        except OSError:
            pass
        else:
            raise AssertionError('Non-sharing failure was swallowed')
        assert replace.call_count==1
    # Real Windows handle permits reading/writing but denies rename/delete.
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateFileW.argtypes=[ctypes.c_wchar_p,ctypes.c_uint32,ctypes.c_uint32,
        ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p]
    kernel.CreateFileW.restype=ctypes.c_void_p
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]; kernel.CloseHandle.restype=ctypes.c_int
    handle=kernel.CreateFileW(str(target),0x80000000,3,None,3,0,None)
    if handle==ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    pending=target.with_name(target.name+'.pending')
    observed=None; timer=None
    try:
        try:
            real_replace(pending,target)
        except OSError as error:
            observed=error.winerror
        assert observed in (5,32,33)
        timer=threading.Timer(.2,kernel.CloseHandle,args=(handle,)); timer.start()
        write(target,{'seconds':16.5,'备注':'真实共享锁释放后成功'})
        assert read(target)['seconds']==16.5
    finally:
        if timer:
            timer.join()
        else:
            kernel.CloseHandle(handle)
    report=dict(passed=True,transient_attempts=3,persistent_max_attempts=8,
        old_file_preserved_on_denial=True,pending_retained=True,other_errors_not_retried=True,
        windows_lock_error=observed,real_shared_handle_recovered=True,
        original_fixture_bytes=len(previous),formal_solver_invoked=False)
    write(out/'io_tests.json',report)
    return report


def verify_io_only(oldrun,old_sources,new_sources):
    changed=[p for p,h in old_sources.items() if new_sources.get(p)!=h]
    assert {p.replace('\\','/') for p in changed}=={'统一模型代码/gates/g10/runtime.py'}
    rel=changed[0]
    old=ast.parse((oldrun/'source_snapshot'/rel).read_text(encoding='utf-8'))
    new=ast.parse((ROOT/rel).read_text(encoding='utf-8'))
    new.body=[node for node in new.body if not isinstance(node,ast.FunctionDef) or node.name!='atomic_replace']
    class UndoReplace(ast.NodeTransformer):
        def visit_Call(self,node):
            self.generic_visit(node)
            if isinstance(node.func,ast.Name) and node.func.id=='atomic_replace':
                node.func=ast.Attribute(value=ast.Name(id='os',ctx=ast.Load()),attr='replace',ctx=ast.Load())
            return node
    assert ast.dump(old,include_attributes=False)==ast.dump(UndoReplace().visit(new),include_attributes=False)
    return {rel:dict(before=old_sources[rel],after=new_sources[rel])}


def main():
    require_exclusive(); resource_check()
    oldpointer=read(BASE/'formal_run.json'); oldrun=Path(oldpointer['out']).resolve(strict=True)
    oldrelease=read(BASE/'formal_release.json'); oldprep=Path(oldrelease['out']).resolve(strict=True)
    assert oldrun.is_relative_to(BASE.resolve()) and oldprep.is_relative_to(BASE.resolve())
    c=read(oldrun/'contract.json'); pc=read(oldprep/'contract.json')
    assert sha(oldrun/'contract.json')==oldpointer['contract_sha']
    assert sha(oldprep/'contract.json')==oldrelease['contract_sha']==c['preparation_sha']
    assert sha(oldprep/'readiness.json')==oldrelease['readiness_sha']
    assert read(oldrun/'status.json')['status']=='STOPPED'
    assert 'PermissionError' in read(oldrun/'status.json')['error']
    identity=sources(); changes=verify_io_only(oldrun,c['sources'],identity)
    assert c['sources']==pc['sources'] and environment()==c['environment']
    repair=fresh('io_repair'); repair.mkdir()
    write(repair/'previous_pointers.json',dict(run=oldpointer,release=oldrelease))
    tests=io_tests(repair)
    plan=jobs('core'); paths=sorted((oldrun/'rows').glob('*.json'))
    assert {p.stem for p in paths}=={job_key(j) for j in plan[:len(paths)]}
    grouped={}; inputs={}; row_hashes={}
    for path in paths:
        row=read(path); validate(row,row['job'],c)
        name=key(row['job']['task'])
        if name not in inputs:
            assert (oldrun/'inputs'/(name+'.json')).is_file()
            _,truth,ident=ensure_input(oldrun,row['job']['task'],c)
            inputs[name]=(truth.tolist(),ident)
        assert (row['truth'],row['input_identity'])==inputs[name]
        grouped.setdefault(name,{})[row['job']['method']]=row
        row_hashes[path.name]=sha(path)
    for group in grouped.values():
        if len(group)==6:
            paired_checks(group)
    # Commit nothing to the old run. Copy byte-identical inputs/results to a fresh directory.
    prep=fresh('formal_ready_io'); prep.mkdir()
    newpc=copy.deepcopy(pc); newpc['sources']=identity
    provenance=dict(previous_run=str(oldrun),previous_contract_sha=sha(oldrun/'contract.json'),
        previous_release=str(oldprep),previous_release_contract_sha=sha(oldprep/'contract.json'),
        changes=changes,inherited_rows=row_hashes,repair_report=str(repair/'repair_report.json'),
        scientific_configuration_unchanged=True)
    newpc['io_recovery']=provenance
    write(prep/'contract.json',newpc); snapshot(prep,identity)
    ready=read(oldprep/'readiness.json'); ready.update(io_fix_verified=True,engineering_recovery=str(repair))
    write(prep/'readiness.json',ready)
    shutil.copyfile(oldprep/'budget_estimate.json',prep/'budget_estimate.json')
    out=fresh('formal_recovered'); out.mkdir()
    nc=copy.deepcopy(c); nc.update(sources=identity,preparation=str(prep),
        preparation_sha=sha(prep/'contract.json'),io_recovery=provenance)
    write(out/'contract.json',nc); snapshot(out,identity)
    copied={}
    for folder in ('rows','inputs'):
        for path in (oldrun/folder).iterdir():
            assert path.is_file() and not path.is_symlink() and not path.name.endswith('.pending')
            target=out/folder/path.name; target.parent.mkdir(exist_ok=True)
            shutil.copyfile(path,target)
            before=sha(path); assert sha(target)==before
            copied[str(target.relative_to(out))]=before
    # The successful exception-handler flush already retained the elapsed time.
    ledger=read(oldrun/'ledger.json'); assert ledger['seconds']>0 and ledger['pending'] is None
    shutil.copyfile(oldrun/'ledger.json',out/'ledger.json')
    assert sha(out/'ledger.json')==sha(oldrun/'ledger.json')
    write(out/'status.json',dict(status='RECOVERY_READY',completed_stages=[],
        previous_failure=str(oldrun),seconds=ledger['seconds']))
    # Check every inherited row without calling the solver or generating any input.
    from . import formal_runner as runner
    from ..r1.runtime import Ledger
    fixture=repair/'resume_fixture'; fixture.mkdir()
    with patch.object(runner.adapter,'run',side_effect=AssertionError('Recovery tried to solve')):
        clock=Ledger(fixture)
        for job in plan[:len(paths)]:
            _,new=runner.record(out,job,nc,clock)
            assert not new
        clock.flush()
    report=dict(passed=True,old_run=str(oldrun),recovered_run=str(out),release=str(prep),
        completed_rows=len(paths),consumed_observations=len(inputs),
        next_job=plan[len(paths)],ledger_seconds=ledger['seconds'],copied_files=copied,
        changes=changes,io_tests=tests,scientific_configuration_unchanged=True,
        completed_results_reused_without_solving=True,old_evidence_unchanged=True,
        formal_solver_invoked=False)
    write(repair/'repair_report.json',report)
    write(BASE/'formal_release.json',dict(out=str(prep),contract_sha=sha(prep/'contract.json'),readiness_sha=sha(prep/'readiness.json')))
    write(BASE/'formal_run.json',dict(out=str(out),contract_sha=sha(out/'contract.json')))
    write(repair/'working_directory_check.json',command_check(['--check-only'],ENTRY.parent))
    summary=f'# G10-R4恢复已准备\n\n已保留{len(paths)}条结果；累计{ledger["seconds"]/60:.2f}分钟不清零。\n\n下一任务：{job_key(plan[len(paths)])}。再次点击原入口继续，未启动新求解。\n\n原中断现场：{oldrun}\n\n恢复目录：{out}\n'
    (out/'运行摘要.md').write_text(summary,encoding='utf-8')
    (repair/'修复说明.md').write_text(summary,encoding='utf-8')
    print(summary); print('修复检查：'+str(repair))


if __name__=='__main__':
    main()
