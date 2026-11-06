"""Prepare/qualify R1, then user-driven full six-gap comparison and offline replot."""
import argparse
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from .config import BASE, METHODS, GAPS, TIMES, SEED, FS, N, FORMAL_REPEATS, method_configs
from .scenes import make, observation, digest, geometry_audit, task_list
from .runtime import (fresh, preflight, read, write, sha, verify,
                      save_scene, load_scene, Ledger, finish_audit)
from .methods import run
from .sbl_outputs import StateStore, replay
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
    clock.flush(dict(tag=tag, started_utc_seconds=time.time(), ledger_start_seconds=clock.total()))
    seed = SEED+group*100+realization
    store = StateStore(out/'states'/tag, dict(input_sha=identity, config=cfg, count=k, seed=seed)) if method == 'SBL' else None
    result = run(iq, method, k, cfg, seed=seed, guard=clock.check, state_store=store)
    if store is not None:
        start = time.perf_counter()
        decoded = replay(store.folder, iq, k)
        result['output_replay_wall_seconds'] = time.perf_counter()-start
        chosen = decoded['states'][decoded['roles']['selected']]
        np.testing.assert_allclose(result['positions'], chosen['outputs']['SBL'], rtol=0, atol=0)
        decoded.update(tag=tag, input_sha=identity, count=k)
        write(out/'sbl_outputs'/f'{tag}.json', decoded)
        result['shared_state'] = dict(manifest=str(store.path.relative_to(out)),
                                    manifest_sha=sha(store.path),
                                    replay=str((out/'sbl_outputs'/f'{tag}.json').relative_to(out)),
                                    replay_sha=sha(out/'sbl_outputs'/f'{tag}.json'),
                                    resumed=store.was_resumed)
    result['interrupted_attempts'] = [p for p in clock.state['interruptions'] if p['tag'] == tag]
    native = result.pop('native_map')
    # Prespecified displays: first realization of each geometry at 0/15 dB.
    if native is not None and realization in (0, 30, 100) and gap in (0, 15) and kind == 'mixed':
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


def short_tasks():
    base = [(0,30,0,'weak'), (0,30,15,'strong'), (0,30,0,'mixed'), (0,30,15,'mixed')]
    jobs = [(task, method) for i,task in enumerate(base)
            for method in METHODS[i%8:]+METHODS[:i%8]]
    jobs += [((group,30,gap,kind),'SBL') for group in (1,2,3)
             for gap,kind in ((0,'weak'),(15,'mixed'))]
    return jobs


def approved_hours(config_sha):
    path = BASE/'budget_confirmation.json'
    if not path.exists():
        return 12.
    info = read(path)
    if info['config_sha'] != config_sha or info['hours'] < 12:
        raise RuntimeError('Budget confirmation does not match frozen configuration')
    return float(info['hours'])


def prepare(out, resume=False, qualification_only=False, limit_minutes=90.):
    if qualification_only:
        raise ValueError('Old 320-case qualification is replaced by shared-state preparation')
    cfgs = method_configs()
    jobs = short_tasks()
    if not resume:
        preflight(out, dict(stage='prepare_shared', methods=cfgs, geometry=geometry_audit(),
                            short_jobs=jobs, revision='G10_R1_SHARED_R5_CACHE_V1', maximum_sbl_iterations=300))
        write(BASE/'preparation_pointer.json', dict(out=str(out.resolve())))
    else:
        contract = read(out/'contract.json')
        verify(contract['sources']); verify(contract['r1_sources'])
        if contract.get('revision') != 'G10_R1_SHARED_R5_CACHE_V1':
            raise RuntimeError('Use a fresh directory for the shared-state protocol')
    clock = Ledger(out)
    tests_path = out/'numerical_tests.json'
    if not tests_path.exists():
        print('阶段1/3：数值、共同状态、恢复和中文日志检查', flush=True)
        from .test_shared import suite, pipeline_smoke
        tests = suite(out/'engineering_checks')
        if tests['passed']:
            smoke = out/'engineering_pipeline'
            preflight(smoke,dict(stage='engineering_pipeline'))
            tests['pipeline'] = pipeline_smoke(smoke)
            finish_audit(smoke)
        write(tests_path, tests)
    tests = read(tests_path)
    if not tests['passed']:
        write(out/'readiness.json', dict(ready=False, reason='NUMERICAL_TESTS_FAILED'))
        return
    if not (out/'warmup.json').exists():
        write(out/'warmup.json', dict(seconds=warmup(),included_in_solver_time=False))
    print('阶段2/3：38次有限求解；不因定位失败追加迭代', flush=True)
    rows = []
    progress = Progress(len(jobs), 'G10-R1共同状态短测')
    for task, method in jobs:
        tag = f'{task[0]}_{task[1]}_{task[2]:g}_{task[3]}_{method}'
        existing = (out/'rows'/f'{tag}.json').exists()
        costs = [r['result']['seconds']+r['result'].get('recording_seconds',0.)
                 +r['result'].get('output_replay_wall_seconds',0.)
                 for r in rows if r['method']==method]
        estimate = max(costs) if costs else 0.
        if not existing and limit_minutes is not None and (
                clock.total() >= limit_minutes*60 or clock.total()+estimate > limit_minutes*60):
            clock.flush()
            write(out/'status.json', dict(status='PREPARATION_PAUSED',completed=len(rows),planned=38,
                                          seconds=clock.state['seconds'],next_task=task,next_method=method))
            write(out/'readiness.json', dict(ready=False,reason='PREPARATION_HANDOFF',
                                             numerical_passed=True,completed=len(rows)))
            finish_audit(out)
            print('准备排期边界：当前单例已完成，剩余准备可由一键入口继续。',flush=True)
            return
        r = record(out,clock,task,method,cfgs[method])
        rows.append(r)
        progress.update(len(rows),f'{method} 布局{task[0]} {task[3]} {task[2]}dB')
    print('阶段3/3：配对统计、候选重放、图表、完整成本估算',flush=True)
    resumed = record(out,clock,jobs[0][0],jobs[0][1],cfgs[jobs[0][1]])
    resume_ok = resumed['input_sha']==rows[0]['input_sha']
    observation_counts = dict(weak=4*FORMAL_REPEATS,strong=4*FORMAL_REPEATS*len(GAPS),
                              mixed=4*FORMAL_REPEATS*len(GAPS))
    observations = sum(observation_counts.values())
    frozen = dict(revision='G10_R1_SHARED_R5_CACHE_V1', methods=cfgs, gaps=list(GAPS),repeats=FORMAL_REPEATS,
                  realization_start=100,sample_rate_hz=FS,N=N,geometry=geometry_audit(),
                  times=list(TIMES),sbl_checkpoints=[25,50,100,200,300],group_diameter_m=.05,
                  expected_observations=observations,expected_rows=observations*len(METHODS),protocol='EVAL-20261011-v2',seed=SEED)
    write(out/'frozen_config.json',frozen)
    estimates, auxiliary, storage = {}, {}, 0.
    for method in METHODS:
        estimates[method] = auxiliary[method] = 0.
        for kind,count in observation_counts.items():
            subset = [r for r in rows if r['method']==method and r['kind']==kind]
            estimates[method] += count*max(r['result']['seconds'] for r in subset)
            auxiliary[method] += count*max(r['result'].get('recording_seconds',0.)
                                         +r['result'].get('output_replay_wall_seconds',0.) for r in subset)
    for r in rows:
        if r['method']=='SBL':
            size = sum(p.stat().st_size for p in (out/'states'/r['tag']).glob('*') if p.is_file())
            storage = max(storage,size)
    predicted = 1.25*(sum(estimates.values())+sum(auxiliary.values()))+1800
    write(out/'budget_estimate.json',dict(hours=predicted/3600,per_method_solver_seconds=estimates,
          per_method_recording_and_replay_seconds=auxiliary,factor=1.25,overhead_seconds=1800,
          estimated_sbl_state_gib=storage*observations/2**30,
          evidence='Maximum observed cost by method/input kind; limited pilot, not a runtime guarantee',
          input_role_counts=observation_counts,hard_per_solver_time_cap=False))
    write(out/'plot_selection.json',dict(tags=[r['tag'] for r in rows]))
    from .report import generate
    generate(out)
    audit = finish_audit(out)
    cfg_sha = sha(out/'frozen_config.json')
    hours = approved_hours(cfg_sha)
    engine_ok = bool(resume_ok and audit['identity_passed'])
    ready = engine_ok and predicted <= hours*3600
    write(out/'readiness.json',dict(ready=ready,engineering_ready=engine_ok,budget_ready=ready,
          reason='READY' if ready else 'BUDGET_CONFIRMATION_REQUIRED',
          numerical_passed=True,resume_ok=resume_ok,config_sha=cfg_sha,
          sources={**read(out/'contract.json')['sources'],**read(out/'contract.json')['r1_sources']},
          estimated_hours=predicted/3600,approved_hours=hours,completed_solver_rows=len(rows)))
    # The release pointer identifies verified engineering even while budget is pending.
    write(BASE/'release.json',dict(preparation=str(out.resolve()),readiness_sha=sha(out/'readiness.json')))
    clock.flush()
    write(out/'status.json',dict(status='PREPARATION_COMPLETED',formal_ready=ready,rows=len(rows),
                                 seconds=clock.state['seconds']))
    print(f'准备完成：工程通过={engine_ok}，正式估算{predicted/3600:.1f}小时，已确认预算{hours:g}小时。',flush=True)
    print(f'短测不是正式排名；未启动{observations*len(METHODS)}次正式求解。',flush=True)


def formal(out, preparation, resume=False):
    ready=read(preparation/'readiness.json')
    if not ready.get('engineering_ready') or sha(preparation/'frozen_config.json') != ready['config_sha']:
        raise RuntimeError('R1 preparation not ready or changed')
    hours = approved_hours(ready['config_sha'])
    if ready['estimated_hours'] > hours:
        raise RuntimeError(f"完整规模估算{ready['estimated_hours']:.1f}小时，超过已确认{hours:g}小时；等待预算确认，不缩样本。")
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
    prior_costs = read(preparation/'budget_estimate.json')['per_method_solver_seconds']
    observed_wall = []
    try:
        print(f'公共预热{warmup():.2f}s（单列）',flush=True)
        for i,task in enumerate(tasks):
            for method in METHODS[i%8:]+METHODS[:i%8]:
                tag = f'{task[0]}_{task[1]}_{task[2]:g}_{task[3]}_{method}'
                if not (out/'rows'/f'{tag}.json').exists():
                    next_cost = max([c for m,c in observed_wall if m==method], default=prior_costs[method]/cfg['expected_observations'])
                    if clock.total()+next_cost > hours*3600:
                        clock.flush()
                        write(out/'status.json',dict(status='BUDGET_PAUSED',rows=done,seconds=clock.state['seconds'],
                                                    next_task=task,next_method=method,approved_hours=hours))
                        finish_audit(out)
                        print('已到预算单例边界：保存进度，等待追加预算；未截断当前算法。',flush=True)
                        return
                r = record(out,clock,task,method,cfg['methods'][method]); done+=1
                observed_wall.append((method,r['result']['seconds']+r['result'].get('recording_seconds',0.)
                                      +r['result'].get('output_replay_wall_seconds',0.)))
                progress.update(done,f'{method} 布局{task[0]} 实现{task[1]} {task[2]}dB {task[3]}')
        print('阶段：离线统计、图表和审计',flush=True)
        from .report import generate
        generate(out)
        finish_audit(out); clock.flush()
        write(out/'status.json',dict(status='COMPLETED',rows=done,seconds=clock.state['seconds']))
    except BaseException as exc:
        pending=clock.state['pending']; clock.flush(keep=True)
        write(out/'status.json',dict(status='STOPPED',error=str(exc),traceback=traceback.format_exc(),pending=pending))
        raise


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['prepare','formal','replot'])
    p.add_argument('--out',type=Path)
    p.add_argument('--preparation',type=Path)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--qualification-only',action='store_true')
    p.add_argument('--continue-preparation',action='store_true',help='User-run preparation without Codex 90-minute scheduling limit')
    a=p.parse_args(); out=a.out or fresh(a.stage)
    print('OUT='+str(out),flush=True)
    if a.stage=='prepare': prepare(out,a.resume,a.qualification_only,None if a.continue_preparation else 90.)
    elif a.stage=='formal': formal(out,a.preparation,a.resume)
    else:
        from .report import generate
        generate(out)


if __name__=='__main__': main()
