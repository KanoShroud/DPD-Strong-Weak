# ruff: noqa: F401
# Preserve imported globals for legacy function cloning and reexports.
"""User-run 32-record conditional risk calibration; frozen estimators."""
import argparse
from datetime import datetime
import hashlib
import importlib.util
from pathlib import Path
import shutil
import sys
import time
import traceback
import uuid

import numpy as np
import scipy
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
PROJECT=next(p for p in HERE.parents if (p/'AGENTS.md').is_file())
OLD=BASE/'unknown_order_20261030'
PILOT=BASE/'risk_reference_pilot_20261102'
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(OLD))
import cross_order as engine  # noqa: E402
import reference as ref  # noqa: E402
from reference_feasible import fit as frozen_fit  # noqa: E402
from storage import read,write,save,load,lock,recover,stage,cost,append  # noqa: E402

spec=importlib.util.spec_from_file_location('frozen_risk_pilot',PILOT/'pilot.py')
risk=importlib.util.module_from_spec(spec)
spec.loader.exec_module(risk)
engine.fit=frozen_fit
CONDITIONS=('L0','L2','L3','HIGH')
ENTRY=PROJECT/'运行入口/E2E/G11/G11位置风险32条条件校准一键运行.py'


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def check_paths(hashes):
    for p,h in hashes.items():
        pp=Path(p).resolve(strict=True)
        if not pp.is_relative_to(PROJECT):raise RuntimeError('输入越出当前项目：'+str(pp))
        if ref.sha(pp)!=h:raise RuntimeError('冻结文件已变更：'+str(pp))


def source_contract():
    pointer_names=('cross','feasible_unseen')
    roots={k:Path(read(OLD/(k+'_latest.json'))['path']).resolve(strict=True) for k in pointer_names}
    files=[HERE/'执行协议.md',ENTRY,HERE/'runner.py',HERE/'storage.py',HERE/'report.py',HERE/'selftest.py']
    for name,root in roots.items():
        pre=read(root/'preflight.json')['hashes'];check_paths(pre)
        files += [OLD/(name+'_latest.json'),root/'preflight.json',root/'L20_generation.npz']
    definitions={}
    for condition in CONDITIONS:
        root=roots['feasible_unseen' if condition=='L3' else 'cross']
        tag='L20_'+condition
        bankpath=root/'L20_generation.npz';oldinp=root/(tag+'_input.npz')
        with np.load(bankpath,allow_pickle=False) as z:bank={k:z[k] for k in z.files}
        iq,_=engine.observation(bank,condition)
        with np.load(oldinp,allow_pickle=False) as z:previous=z['iq']
        if not np.array_equal(iq,previous):raise RuntimeError('原生成银行不能精确重放：'+condition)
        clean=iq-bank['noise']
        definitions[condition]=dict(bank=str(bankpath),old_input=str(oldinp),clean_sha=digest(clean),
                                   original_iq_sha=digest(previous))
        files.append(oldinp)
    for mod in tuple(sys.modules.values()):
        filename=getattr(mod,'__file__',None)
        if filename and Path(filename).suffix=='.py' and Path(filename).resolve().is_relative_to(BASE):
            files.append(Path(filename).resolve())
    files.append(PILOT/'pilot.py')
    hashes={str(p.resolve(strict=True)):ref.sha(p) for p in files}
    return dict(hashes=hashes,conditions=definitions,numpy=np.__version__,scipy=scipy.__version__,
                python_major_minor=list(sys.version_info[:2]),threads=2,
                seed=2026110203,development_seed=2026110299,repeats=8,expected_total=32)


def verify(contract):
    check_paths(contract['hashes'])
    if contract['numpy']!=np.__version__ or contract['scipy']!=scipy.__version__:
        raise RuntimeError('数值库版本与冻结构建不一致。')
    if contract['python_major_minor']!=list(sys.version_info[:2]):
        raise RuntimeError('Python版本与冻结构建不一致。')


def cases(smoke=False):
    return [dict(id=f'{condition}_R{rep:02d}',condition=condition,repeat=rep,
                 seed=[2026110299 if smoke else 2026110203,i,rep])
            for rep in range(1 if smoke else 8)
            for i,condition in enumerate(CONDITIONS if not smoke else ('L0','L2'))]


def generate(contract,case):
    source=contract['conditions'][case['condition']]
    with np.load(source['bank'],allow_pickle=False) as z:bank={k:z[k] for k in z.files}
    old,_=engine.observation(bank,case['condition'])
    clean=old-bank['noise']
    if digest(clean)!=source['clean_sha']:raise RuntimeError('固定信号身份漂移。')
    rng=np.random.default_rng(np.random.SeedSequence(case['seed']))
    noise=np.sqrt(float(bank['variance'])/2)*(rng.normal(size=clean.shape)+1j*rng.normal(size=clean.shape))
    return clean+noise


def input_case(out,contract,case):
    folder=out/'cases'/case['id'];folder.mkdir(parents=True,exist_ok=True)
    path=folder/'input.npz';meta=folder/'input.json'
    if path.exists() and meta.exists():
        info=load(meta)
        if info['case']!=case:raise RuntimeError('样本编号/种子改变。')
        with np.load(path,allow_pickle=False) as z:
            if set(z.files)!={'iq'}:raise RuntimeError('求解输入意外包含真值字段。')
            iq=z['iq']
        if digest(iq)!=info['iq_sha']:raise RuntimeError('IQ保存身份不匹配。')
        return folder,iq,info
    # If interrupted during input creation, deterministic regeneration is safe.
    iq=generate(contract,case)
    tmp=folder/'input.pending.npz'
    np.savez_compressed(tmp,iq=iq)
    tmp.replace(path)
    info=dict(case=case,iq_sha=digest(iq),source_clean_sha=contract['conditions'][case['condition']]['clean_sha'])
    save(meta,info)
    return folder,iq,info


def solve(iq,folder,ident):
    attempt=folder/('solve_attempt_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    attempt.mkdir(exist_ok=False)
    original=digest(iq)
    try:
        result=engine.solve_iq(iq,attempt,ident)
        if digest(iq)!=original:raise AssertionError('求解器修改输入IQ')
        result.update(status='returned',iq_sha=original,attempt=str(attempt))
        return result
    except (ValueError,IndexError,np.linalg.LinAlgError) as exc:
        return dict(status='numerical_or_frontend_exception',exception=repr(exc),
                    traceback=traceback.format_exc(),iq_sha=original,attempt=str(attempt))


def spectral(iq):
    f=np.fft.fftfreq(iq.shape[1],1/ref.FS);use=abs(f)<=ref.B/2
    z=np.fft.fft(iq,axis=1,norm='ortho')
    return f[use],z[:,use].T,iq.shape[1],float(np.mean(abs(z[:,~use])**2))


def diagnostic(solved,iq,model,seed):
    if solved['status']!='returned':return dict(applicable=False,reason='no_finite_solver_output')
    if model=='WIDE' and solved['stop']!='no_detected_increment':
        return dict(applicable=False,reason='wide_model_incomplete',stop=solved['stop'])
    state=solved['los' if model=='LOS' else 'final']
    paths=[] if model=='LOS' else solved['paths']
    f,y,n,sigma2=spectral(iq)
    info,l=risk.diagnose(np.asarray(state['x']),f,y,paths,n,sigma2,np.random.default_rng(seed))
    return dict(applicable=info['applicable'],diagnostic=info,L=l,sigma2_est=sigma2)


def risk_compare(solved,a,b):
    if not a['applicable'] or not b['applicable']:
        return dict(applicable=False,reason='one_or_both_endpoints_inapplicable')
    p=np.asarray(solved['los']['p']);q=np.asarray(solved['final']['p'])
    same=float(np.sum((p-q)**2));swapped=float(np.sum((p-q[::-1])**2))
    if swapped < same:
        return dict(applicable=False,reason='estimated_source_label_change',same_distance=same,swapped_distance=swapped)
    lm,lw=np.asarray(a['L']),np.asarray(b['L']);s=a['sigma2_est']
    if not np.isclose(s,b['sigma2_est'],rtol=1e-12,atol=0):raise AssertionError('同IQ噪声尺度不一致')
    wide=float(s*np.sum(lw*lw)/2)
    los=float(same/2+s*np.sum(lm*lw)-wide)
    return dict(applicable=True,LOS=los,WIDE=wide,difference=los-wide,
                interpretation='uncalibrated_endpoint_plugin_not_selector')


def evaluate(solved,contract,case):
    # Only this post-decision evaluation reads physical positions.
    with np.load(contract['conditions'][case['condition']]['bank'],allow_pickle=False) as z:truth=z['p']
    result={}
    if solved['status']!='returned':return result
    for name,key in (('LOS','los'),('WIDE','final')):
        p=np.asarray(solved[key]['p'])
        if p.shape!=(2,2) or not np.isfinite(p).all():continue
        if np.sum((p[::-1]-truth)**2)<np.sum((p-truth)**2):p=p[::-1]
        error=p-truth
        result[name]=dict(position_error=error,loss_m2=float(np.sum(error**2)/2),
                          stationary=bool(solved[key]['stationary']))
    return result


def initialize(contract,smoke=False,resume=None):
    pointer=HERE/('latest_smoke.json' if smoke else 'latest_formal.json')
    if resume:
        out=Path(resume).resolve(strict=True)
    elif not smoke and pointer.exists():
        out=Path(read(pointer)['path']).resolve(strict=True)
    else:
        if shutil.disk_usage(HERE).free < 8*1024**3:
            raise RuntimeError('可用磁盘不足8GB，请先确认空间，不减少保存内容。')
        out=HERE/(('smoke_' if smoke else 'formal_')+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        out.mkdir(exist_ok=False)
        save(out/'contract.json',dict(contract=contract,smoke=smoke,cases=cases(smoke)))
        snapshots={}
        for p in contract['hashes']:
            source=Path(p)
            if source.suffix in ('.py','.md'):
                dst=out/'source'/(contract['hashes'][p][:24]+source.suffix)
                dst.parent.mkdir(parents=True,exist_ok=True)
                if dst.exists() and ref.sha(dst)!=contract['hashes'][p]:raise RuntimeError('快照短名冲突。')
                shutil.copyfile(source,dst)
                snapshots[p]=str(dst.relative_to(out))
        write(out/'source_map.json',snapshots)
        write(pointer,dict(path=str(out)))
    if not out.is_relative_to(HERE):raise RuntimeError('恢复目录越界。')
    saved=load(out/'contract.json')
    if saved!=dict(contract=contract,smoke=smoke,cases=cases(smoke)):
        raise RuntimeError('恢复合同与本次冻结配置不同，禁止混批。')
    return out


def execute(out,contract,smoke=False):
    from report import summarize
    with lock(out):
        recover(out);verify(contract)
        todo=cases(smoke);print('结果目录：',out,flush=True)
        print('开发联调，不计正式32条。' if smoke else '正式32条：4条件 × 8新噪声；一次运行全部完成，重复点击自动恢复。',flush=True)
        for index,case in enumerate(todo):
            start=time.perf_counter();before=cost(out)['seconds']
            folder,iq,info=input_case(out,contract,case)
            final=folder/'record.json'
            if final.exists():
                cached=load(final)
                if cached['iq_sha']!=info['iq_sha']:raise RuntimeError('完成记录IQ身份改变。')
                print(f'[{index+1}/{len(todo)}] {case["id"]} 已完成，跳过。',flush=True)
                continue
            print(f'[{index+1}/{len(todo)}] {case["id"]} 盲定位/恢复阶段；当前累计 {cost(out)["seconds"]/3600:.2f} h',flush=True)
            solved,_=stage(out,case['id']+'/solve',folder/'solve.json',lambda:solve(iq,folder,case['id']))
            if solved['iq_sha']!=info['iq_sha']:raise RuntimeError('求解输出IQ身份不一致。')
            if solved['status']=='returned' and (solved['stop']=='correctness_failure' or not all(solved['checks'].values())):
                raise RuntimeError('冻结求解器正确性断言未通过，已保存，不继续批次。')
            diag=[]
            for model in ('LOS','WIDE'):
                print('  风险诊断',model,'（不修改定位输出）',flush=True)
                seed=[2026110210,*case['seed'],0 if model=='LOS' else 1]
                a,_=stage(out,case['id']+'/'+model,folder/(model+'_risk.json'),
                          lambda model=model,seed=seed:diagnostic(solved,iq,model,seed))
                if 'diagnostic' in a and not all(a['diagnostic']['derivative_checks'].values()):
                    raise RuntimeError('导数正确性检查失败，已保存；不得把公式错误当不适用继续。')
                diag.append(a)
            comparison=risk_compare(solved,*diag)
            save(folder/'comparison_before_evaluation.json',comparison)
            evaluation=evaluate(solved,contract,case)
            record=dict(case=case,iq_sha=info['iq_sha'],solver_status=solved['status'],
                        stop=solved.get('stop'),path_count=len(solved.get('paths',[])),
                        risk=comparison,evaluation=evaluation,
                        applicability={name:{k:v for k,v in a.items() if k!='L'} for name,a in zip(('LOS','WIDE'),diag)})
            save(final,record)
            verify(contract)
            summarize(out)
            overhead=max(0.,time.perf_counter()-start-(cost(out)['seconds']-before))
            append(out,dict(kind='end',id=uuid.uuid4().hex,stage=case['id']+'/input_audit_report_overhead',seconds=overhead))
            print(f'  已保存，风险适用={comparison["applicable"]}，本次阶段耗时 {time.perf_counter()-start:.1f}s',flush=True)
        verify(contract)
        summary=summarize(out)
        print('全部完成。' if summary['complete'] else '阶段报告已保存，尚未完整。',flush=True)
        print('回读报告：',out/'条件校准报告.html',flush=True)
        return summary


def release():
    contract=source_contract()
    qa=load(HERE/'test_result.json')
    if not qa['all_pass'] or qa['hashes']!=contract['hashes']:raise RuntimeError('测试未通过或源码已改变。')
    smoke=Path(read(HERE/'latest_smoke.json')['path'])
    if load(smoke/'contract.json')['contract']!=contract:raise RuntimeError('联调合同不一致。')
    summary=read(smoke/'summary.json')
    if not summary['complete'] or summary['completed']!=2:raise RuntimeError('真实盲求解联调未完成。')
    for case in cases(True):
        row=load(smoke/'cases'/case['id']/'record.json')
        if row['solver_status']!='returned' or row['stop']!='no_detected_increment' or not row['risk']['applicable']:
            raise RuntimeError('联调出现数值异常，需要先审查。')
    save(HERE/'release.json',dict(contract=contract,smoke=str(smoke),tests=str(HERE/'test_result.json')))
    print('正式入口已放行；32条未运行。',flush=True)


# Historical main retained only in the evidence archive.


# Historical batch entry disabled; use the public mp_dpd API.
