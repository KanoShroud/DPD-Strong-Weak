"""Prepare/qualify R1, then user-driven full six-gap comparison and offline replot."""
import argparse
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from .config import BASE, DEFAULT, METHODS, GAPS, TIMES, SEED, FS, N
from .scenes import make, observation, digest, geometry_audit, task_list
from .runtime import (fresh, preflight, read, write, sha, verify, sources,
                      save_scene, load_scene, Ledger, finish_audit)
from .methods import run
from ..metrics import evaluate
from ..runtime import Progress


def scene_for(out, group, realization, gap):
    key = f'{group}_{realization}_{gap:g}'
    if not (out/'inputs'/f'{key}.json').exists():
        save_scene(out, key, make(group, realization, gap))
    scene = load_scene(out, key)
    if scene['meta'].get('sample_rate_hz') != FS or scene['meta']['N'] != N:
        raise RuntimeError('Scene belongs to a different sampling contract')
    return scene


def record(out, clock, task, method, cfg, clean=False, suffix=''):
    group, realization, gap, kind = task
    tag = f'{group}_{realization}_{gap:g}_{kind}_{method}{suffix}'
    path = out/'rows'/f'{tag}.json'
    scene = scene_for(out, group, realization, gap)
    iq, truth, k = observation(scene, kind)
    if clean:
        iq = scene['signal'][1 if kind == 'weak' else 0].copy() if kind != 'mixed' else scene['signal'].sum(0)
    identity = digest(iq)
    if path.exists():
        row = read(path)
        if row['input_sha'] != identity or row['result']['config'] != cfg:
            raise RuntimeError('Resume input/config differs: '+tag)
        return row
    clock.flush(dict(tag=tag, started_utc_seconds=time.time()))
    result = run(iq, method, k, cfg, seed=SEED+group*100+realization, guard=clock.check)
    native = result.pop('native_map')
    # Prespecified displays: first realization of each geometry at 0/15 dB.
    if native is not None and realization in (0, 100) and gap in (0, 15) and kind == 'mixed':
        write(out/'maps'/f'{tag}.json', native)
    row = dict(tag=tag, scene=scene['meta'], kind=kind, method=method,
               input_sha=identity, truth=truth.tolist(), clean=clean, result=result,
               metrics=evaluate(truth, result['positions']))
    write(path, row)
    clock.flush()
    return row


def warmup():
    from .physics import Model, Spectrum, small_fit
    rng = np.random.default_rng(38)
    start = time.perf_counter()
    m = Model(rng.normal(size=(4, 128))+1j*rng.normal(size=(4, 128)))
    pp = np.array([[173., 117.], [-137., -213.]])
    m.joint(pp, True)
    m.joint_batch(np.stack([pp, pp+1]))
    small_fit(m.response(pp), m.x)
    for method in ('MUSIC', 'MVDR'):
        Spectrum(m, method, 2, 8, 1e-3).evaluate(pp)
    torch.cuda.synchronize()
    return time.perf_counter()-start


def prepare(out, resume=False, qualification_only=False):
    cfgs = {m:dict(DEFAULT, sbl_update='fp_half') for m in METHODS}
    if not resume:
        preflight(out, dict(stage='prepare', methods=cfgs, geometry=geometry_audit(),
                            gaps=[0,15], repeats=2, scope='Approved short development only'))
    else:
        contract = read(out/'contract.json')
        verify(contract['sources']); verify(contract['r1_sources'])
    clock = Ledger(out)
    write(out/'warmup.json',dict(seconds=warmup(),included_in_solver_time=False))
    from .test_r1 import suite
    tests = suite()
    write(out/'numerical_tests.json', tests)
    if not tests['passed']:
        write(out/'readiness.json', dict(ready=False, reason='NUMERICAL_TESTS_FAILED'))
        return
    print('阶段1/4：谱方法有限配对适配（统一网格，J与汇聚方式）', flush=True)
    tuning = []
    # Do not score six-gap comparison data to pick per-gap settings.
    for method in ('MUSIC','MVDR'):
        options = []
        for segments in (4,8,16):
            for weight in (('uniform','excess_power') if method == 'MUSIC' else ('uniform',)):
                cfg = dict(cfgs[method], segments=segments, music_weight=weight)
                selected = []
                for group in range(4):
                    row = record(out, clock, (group,0,0,'weak'), method, cfg,
                                 suffix=f'_tune_{segments}_{weight}')
                    selected.append(row)
                success = sum(all(r['metrics']['success']['20']) for r in selected)
                error = float(np.mean([r['metrics']['errors_m'][0] for r in selected]))
                seconds = sum(r['result']['seconds'] for r in selected)
                score = (-success, error, seconds)
                options.append((score,cfg))
                tuning.append(dict(method=method, segments=segments, weight=weight,
                                   successes=success, mean_error_m=error, seconds=seconds))
        cfgs[method] = min(options,key=lambda x:x[0])[1]
    write(out/'tuning.json', tuning)
    print('阶段2/4：无噪声孤立源与原生停止核验', flush=True)
    clean_rows=[]
    for group in range(4):
        for method in METHODS:
            print(f'无噪声：布局{group} / {method}', flush=True)
            r=record(out,clock,(group,0,0,'weak'),method,cfgs[method],clean=True,suffix='_clean')
            clean_rows.append(r)
    clean_ok=all(len(r['result']['positions']) == 1 and
                 np.isfinite(r['result']['positions']).all() for r in clean_rows)
    write(out/'clean_qualification.json',dict(passed=clean_ok,
           success20_count=sum(all(r['metrics']['success']['20']) for r in clean_rows),
           criterion='finite complete output; localization success is reported, not a release condition',
           rows=[dict(tag=r['tag'],errors=r['metrics']['errors_m'],stop=r['result']['stop_reason']) for r in clean_rows]))
    if not clean_ok:
        write(out/'readiness.json',dict(ready=False,reason='CLEAN_SINGLE_ADAPTATION_REQUIRES_REVIEW'))
        finish_audit(out)
        print('去噪压力检查输出缺失或非有限：保留证据，暂停正式放行。',flush=True)
        return
    if qualification_only:
        write(out/'readiness.json',dict(ready=False,reason='QUALIFICATION_ONLY_PAIRED_SHORT_PENDING'))
        finish_audit(out); clock.flush()
        print('无噪声核验结束；配对短测尚未执行，正式入口未放行。',flush=True)
        return
    print('阶段3/4：4布局×2次实现×0/15dB配对短测',flush=True)
    tasks=list(task_list(2,(0,15)))
    rows=[]
    progress=Progress(len(tasks)*len(METHODS),'G10-R1短测')
    for i,task in enumerate(tasks):
        for method in METHODS[i%8:]+METHODS[:i%8]:
            r=record(out,clock,task,method,cfgs[method]); rows.append(r)
            progress.update(len(rows),f'{method} {task}')
    print('阶段4/4：充分性对照、恢复、图表与成本估算',flush=True)
    sufficient=[]
    for method in ('AP','BNOMP','SBL','MULTISTART','GA'):
        cfg=dict(cfgs[method])
        cfg.update(cycles=cfg['cycles']*2,sbl_iterations=cfg['sbl_iterations']*2,
                   starts=min(64,cfg['starts']*2),generations=cfg['generations']*2)
        base=next(r for r in rows if r['scene']['group']==0 and r['scene']['realization']==0
                  and r['scene']['gap_db']==15 and r['kind']=='mixed' and r['method']==method)
        extended=record(out,clock,(0,0,15,'mixed'),method,cfg,suffix='_sufficiency')
        p=np.asarray(base['result']['positions']); q=np.asarray(extended['result']['positions'])
        shift=min(np.linalg.norm(p-q,axis=1).max(),np.linalg.norm(p-q[::-1],axis=1).max())
        gain=base['result']['events'][-1]['native_objective']-extended['result']['events'][-1]['native_objective']
        sufficient.append(dict(method=method,solution_shift_m=float(shift),native_gain=float(gain),
                               base_stop=base['result']['stop_reason'],extended_stop=extended['result']['stop_reason']))
    write(out/'sufficiency.json',sufficient)
    # Inference-only extension comparison; truth not used to select a stopping point.
    enough=all(r['solution_shift_m'] <= 1 and r['extended_stop'] != 'ITERATION_LIMIT' for r in sufficient)
    proof=rows[0]; before=sha(out/'rows'/f"{proof['tag']}.json")
    task=(proof['scene']['group'],proof['scene']['realization'],proof['scene']['gap_db'],proof['kind'])
    record(out,clock,task,proof['method'],cfgs[proof['method']])
    resume_ok=sha(out/'rows'/f"{proof['tag']}.json")==before
    estimates={}
    for method in METHODS:
        estimate=0.
        for kind,count in [('weak',40),('strong',240),('mixed',240)]:
            subset=[r['result']['seconds'] for r in rows if r['method']==method and r['kind']==kind]
            estimate+=count*max(subset)
        estimates[method]=estimate
    predicted=1.25*sum(estimates.values())+1800
    write(out/'budget_estimate.json',dict(hours=predicted/3600,per_method_seconds=estimates,
                                         factor=1.25,overhead_seconds=1800,hard_time_cap=False))
    frozen=dict(methods=cfgs,gaps=list(GAPS),repeats=10,realization_start=100,
                sample_rate_hz=FS,N=N,geometry=geometry_audit(),times=list(TIMES),
                expected_observations=520,expected_rows=4160,protocol='EVAL-20261011-v2',seed=SEED)
    write(out/'frozen_config.json',frozen)
    from .report import generate
    # Explicit row selection separates tuning/clean/sufficiency from noisy paired results.
    write(out/'plot_selection.json',dict(tags=[r['tag'] for r in rows]))
    generate(out)
    audit=finish_audit(out)
    ready=clean_ok and enough and resume_ok and audit['identity_passed']
    write(out/'readiness.json',dict(ready=ready,reason='READY' if ready else 'HOLD_SUFFICIENCY',
          numerical_passed=True,clean_passed=clean_ok,sufficiency_passed=enough,resume_ok=resume_ok,
          config_sha=sha(out/'frozen_config.json'),sources={**read(out/'contract.json')['sources'],**sources()},
          estimated_hours=predicted/3600))
    if ready:
        write(BASE/'release.json',dict(preparation=str(out),readiness_sha=sha(out/'readiness.json')))
    print(f'准备完成 ready={ready}，六档保守估算{predicted/3600:.1f}小时；尚未运行完整比较。',flush=True)


def formal(out, preparation, resume=False):
    ready=read(preparation/'readiness.json')
    if not ready['ready'] or sha(preparation/'frozen_config.json') != ready['config_sha']:
        raise RuntimeError('R1 preparation not ready or changed')
    verify(ready['sources'])
    cfg=read(preparation/'frozen_config.json')
    if not resume:
        preflight(out,dict(stage='formal',config=cfg,preparation=str(preparation)))
    else:
        old=read(out/'contract.json')
        if old['config'] != cfg: raise RuntimeError('Resume configuration changed')
        verify(old['sources']); verify(old['r1_sources'])
    clock=Ledger(out)
    if cfg.get('sample_rate_hz') != FS or cfg.get('N') != N:
        raise RuntimeError('Frozen configuration belongs to another sampling contract')
    tasks=list(task_list(cfg['repeats'],cfg['gaps'],cfg['realization_start']))
    progress=Progress(len(tasks)*len(METHODS),'G10-R1六档比较')
    done=0
    try:
        print(f'公共预热{warmup():.2f}s（单列）',flush=True)
        for i,task in enumerate(tasks):
            for method in METHODS[i%8:]+METHODS[:i%8]:
                record(out,clock,task,method,cfg['methods'][method]); done+=1
                progress.update(done,f'{method} 布局{task[0]} 实现{task[1]} {task[2]}dB {task[3]}')
        print('阶段：离线统计、图表和审计',flush=True)
        from .report import generate
        generate(out)
        finish_audit(out); clock.flush()
        write(out/'status.json',dict(status='COMPLETED',rows=done,seconds=clock.state['seconds']))
    except BaseException as exc:
        pending=clock.state['pending']; clock.flush()
        write(out/'status.json',dict(status='STOPPED',error=str(exc),traceback=traceback.format_exc(),pending=pending))
        raise


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['prepare','formal','replot'])
    p.add_argument('--out',type=Path)
    p.add_argument('--preparation',type=Path)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--qualification-only',action='store_true')
    a=p.parse_args(); out=a.out or fresh(a.stage)
    print('OUT='+str(out),flush=True)
    if a.stage=='prepare': prepare(out,a.resume,a.qualification_only)
    elif a.stage=='formal': formal(out,a.preparation,a.resume)
    else:
        from .report import generate
        generate(out)


if __name__=='__main__': main()
