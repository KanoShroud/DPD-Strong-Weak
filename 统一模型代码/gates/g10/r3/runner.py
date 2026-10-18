"""One click prepares then exits; a later click runs only a released contract."""
import argparse
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch
import numpy as np
from ..metrics import evaluate
from ..r1.runtime import Ledger
from .config import BASE, METHODS, PREP_METHODS
from .runtime import (read, write, sha, seal, lock, reference, require_exclusive,
    active_project_processes, fresh, create, check_contract, job_key, load_input,
    validate, audit, resolve_root)


class Progress:
    def __init__(self, total):
        self.total, self.start, self.last = total, time.perf_counter(), 0.

    def show(self, done, computed, detail):
        now = time.perf_counter()
        if now-self.last < 15 and done != self.total:
            return
        elapsed = now-self.start
        eta = elapsed/max(1, computed)*(self.total-done)
        count = int(24*done/max(1, self.total))
        text = (f"G10-R3 [{'#'*count}{'-'*(24-count)}] {done}/{self.total} "
                f"| {elapsed/60:.1f} min | ETA {eta/60:.1f} min | {detail}")
        # PyCharm supports carriage-return progress; never emit per-iteration lines.
        print('\r'+text.ljust(150), end='\n' if done == self.total else '', flush=True)
        self.last = now


def record(out, job, clock, contract):
    from . import solvers
    scene, iq, truth, identity = load_input(out, job['task'], contract)
    cfg = contract['configurations'][job['method']]
    path = out/'rows'/(job_key(job)+'.json')
    if path.exists():
        row = read(path); validate(row, job, identity, cfg, truth)
        return row, False
    require_exclusive()  # outside solver timing; check before every new solve
    clock.flush(dict(tag=path.stem, started_utc_seconds=time.time(), ledger_start_seconds=clock.total()))
    last = time.perf_counter()

    def guard():
        nonlocal last
        clock.check()
        if time.perf_counter()-last > 5:
            require_exclusive(); last = time.perf_counter()

    seed = int(np.random.SeedSequence([2026101619, *job['task'][:3]]).generate_state(1)[0])
    result = solvers.run(iq, job['method'], 2, cfg, seed, guard)
    row = dict(job=job, task=job['task'], method=job['method'], input_sha=identity,
        truth=truth.tolist(), scene=scene['meta'], result=result,
        metrics=evaluate(truth, result['positions']),
        prior_attempts=[x for x in clock.state['interruptions'] if x['tag'] == path.stem])
    row['record_sha'] = seal(row)
    write(path, row)
    clock.flush()
    return row, True


def warmup(out, contract):
    import torch
    from .solvers import run
    from .runtime import load_r2_input
    _, iq, _, _, _ = load_r2_input(resolve_root(contract['reference']['out']), (0, 200, 0, 'mixed'))
    start = time.perf_counter()
    for method in (PREP_METHODS if contract['stage'] == 'prepare' else METHODS):
        run(iq, method, 2, contract['configurations'][method], 123)
    seconds = time.perf_counter()-start
    history = read(out/'warmups.json') if (out/'warmups.json').exists() else []
    environment = dict(python=sys.version, numpy=np.__version__, torch=torch.__version__,
                       gpu=torch.cuda.get_device_name(0), torch_cpu_threads=torch.get_num_threads())
    if history and history[-1]['environment'] != environment:
        raise RuntimeError('恢复进程环境与已有计时不同，先核对版本/CPU线程，不混合统计。')
    if contract['stage'] == 'formal':
        prep = resolve_root(contract['preparation']['preparation'])
        expected = read(prep/'warmups.json')[-1]['environment']
        if expected != environment:
            raise RuntimeError('正式运行环境与短测不同，先核对版本/CPU线程。')
    history.append(dict(seconds=seconds, time=time.time(), separate_from_solver=True, environment=environment))
    write(out/'warmups.json', history)
    print(f'公共预热 {seconds:.2f}s（单列，每次进程启动重新预热）', flush=True)


def execute(out):
    from .checks import cpu_suite, gpu_suite
    from . import solvers
    from .report import generate
    require_exclusive()
    c = check_contract(out)
    with lock(out):
        clock = Ledger(out)
        try:
            if c['stage'] == 'prepare':
                if not (out/'cpu_checks.json').exists():
                    cpu_suite(out)
                if not (out/'gpu_checks.json').exists():
                    print('开始GPU核验：原搜索前缀、统一停止、缓存与记录开关。', flush=True)
                    gpu_suite(out)
            warmup(out, c)
            progress = Progress(len(c['jobs'])); computed = 0
            for i, job in enumerate(c['jobs']):
                _, new = record(out, job, clock, c)
                computed += int(new)
                progress.show(i+1, computed, job_key(job))
            write(out/'phase_timing.json', dict(batch_wall_seconds=time.perf_counter()-progress.start,
                  newly_computed=computed, total_jobs=len(c['jobs'])))
            final = audit(out)
            # Same record path must resume without invoking any solver.
            with patch.object(solvers, 'run', side_effect=AssertionError('Resume/replot invoked solver')):
                record(out, c['jobs'][0], clock, c)
                report = generate(out)
            clock.flush()
            write(out/'status.json', dict(status='COMPLETED_PREPARATION' if c['stage'] == 'prepare' else 'COMPLETED',
                stage=c['stage'], completed=final['completed'], seconds=clock.total(),
                resume_verified=True, offline_replot_verified=True))
            return report
        except BaseException as error:
            clock.flush(keep=True)
            write(out/'status.json', dict(status='STOPPED', stage=c['stage'], error=str(error),
                traceback=traceback.format_exc(), seconds=clock.total()))
            raise


def readiness(out, report):
    """Do not silently launch independent validation if the premise disappeared."""
    from .report import flatten
    rows = [read(p) for p in (out/'rows').glob('*.json')]
    flat = flatten(rows)
    paired = report['paired']
    subsets = {m: [x for x in flat if x['method'] == m] for m in METHODS}
    a, b = subsets['AP_1_ALT'], subsets['AP_1_JOINT']
    # No truth-dependent stop or retuning: these are post-run progression checks.
    checks = dict(engineering=read(out/'cpu_checks.json')['passed'] and read(out/'gpu_checks.json')['passed'],
        complete=report['completed'] == 240,
        same_precision=paired['both_precision_count'] == 24,
        same_objective=paired['maximum_objective_difference'] <= 1e-8,
        no_success_loss=sum(x['both20'] for x in b) >= sum(x['both20'] for x in a),
        lower_full_time=paired['mean_seconds_saved'] > 0,
        common_precision_observed=paired['first_precision_pairs'] == 24,
        lower_time_to_common_precision=paired['mean_first_precision_seconds_saved'] is not None and paired['mean_first_precision_seconds_saved'] > 0,
        lower_core_time_to_common_precision=paired['mean_first_precision_core_seconds_saved'] is not None and paired['mean_first_precision_core_seconds_saved'] > 0,
        fewer_joint_evaluations=paired['mean_joint_calls_saved'] > 0)
    contract = read(out/'contract.json')
    prior = contract.get('previous_preparation')
    if not prior:
        raise RuntimeError('Original R3 preparation needed for unchanged NOMP/ML cost reference')
    original = read(resolve_root(prior['out'])/'comparison_report.json')
    old_summary = {r['method']: r for r in original['summaries']}
    estimates = {}
    for method in METHODS:
        if subsets[method]:
            per_solve = float(np.mean([x['wall_solver_seconds']+x['certificate_seconds']
                                      for x in subsets[method]]))
        else:
            per_solve = old_summary[method]['wall_solver_seconds']+old_summary[method]['certificate_seconds']
        estimates[method] = per_solve*120
    timing = read(out/'phase_timing.json')
    measured_overhead = None
    if timing['newly_computed'] == len(rows):
        algorithm = sum(r['result']['wall_solver_seconds']+r['result']['certificate_seconds'] for r in rows)
        measured_overhead = max(0., timing['batch_wall_seconds']-algorithm)/len(rows)
    forecast = None if measured_overhead is None else 1.25*(sum(estimates.values())+480*measured_overhead)+120
    result = dict(status='READY_FOR_INDEPENDENT' if all(checks.values()) else 'HOLD_FOR_REVIEW',
        checks=checks, formal_solves=480, solver_seconds_by_method=estimates,
        estimated_total_seconds_with_25pct_and_120s=forecast,
        measured_job_overhead_seconds=measured_overhead,
        unchanged_method_cost_reference=prior,
        note='24 existing IQs, five timing repeats for the two common-stop APs. NOMP/ML costs reused from prior preparation. '
             'Forecast includes measured job overhead when the batch was uninterrupted. Estimate, not a promise. '
             'Failing progression checks does not mean an implementation error; do not retune automatically.')
    write(out/'readiness.json', result)
    if result['status'] == 'READY_FOR_INDEPENDENT':
        write(BASE/'release.json', dict(preparation=str(out), readiness_sha=sha(out/'readiness.json'),
            audit_sha=sha(out/'final_audit_report.json'), contract_sha=sha(out/'contract.json')))
    return result


def inspect():
    ref = reference()
    return dict(reference=ref['out'], r2_frozen_sources=len(ref['protected']),
                active_processes=active_project_processes(),
                preparation_pointer=read(BASE/'preparation.json') if (BASE/'preparation.json').exists() else None,
                release_exists=(BASE/'release.json').exists())


def main(argv=None):
    parser = argparse.ArgumentParser(description='R3：短测完成即退出；再次点击才执行已放行独立比较。')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--new-prepare', action='store_true')
    parser.add_argument('--replot', type=Path)
    args = parser.parse_args(argv)
    if args.check_only:
        print(inspect()); return
    if args.replot is not None:
        # No GPU or solver execution. Avoid concurrent load while SBL runs.
        require_exclusive()
        from .report import generate
        generate(args.replot.resolve(strict=True)); return
    active = active_project_processes()
    if active:
        print('R3未启动：检测到R2/SBL或其他项目任务仍在运行。请待其结束后再次点击。')
        print('运行中PID：'+', '.join(str(p['pid']) for p in active))
        return
    BASE.mkdir(parents=True, exist_ok=True)
    with lock(BASE):
        ref = reference()
        pointer = BASE/'preparation.json'
        if args.new_prepare or not pointer.exists():
            out = create(fresh('prepare'), 'prepare', ref)
            write(pointer, dict(out=str(out)))
        else:
            out = resolve_root(read(pointer)['out'])
        c = check_contract(out)
        if c['stage'] != 'prepare':
            raise RuntimeError('Wrong preparation pointer')
        state = read(out/'status.json')['status']
        if state != 'COMPLETED_PREPARATION':
            report = execute(out)
            result = readiness(out, report)
            print('\n短测完成并已退出：', out)
            print('放行状态：', result['status'])
            print('请回读运行摘要、readiness.json、comparison_report.json；通过时再次点击运行正式120例。')
            return
        if args.prepare:
            print('准备已完成，不重复计算：', out); return
        release_file = BASE/'release.json'
        if not release_file.exists():
            print('短测尚未放行正式比较，请先回读：', out/'readiness.json'); return
        release = read(release_file)
        if resolve_root(release['preparation']) != out:
            raise RuntimeError('Release belongs to a different preparation')
        for file, key in (('readiness.json', 'readiness_sha'), ('final_audit_report.json', 'audit_sha'), ('contract.json', 'contract_sha')):
            if sha(out/file) != release[key]:
                raise RuntimeError('Release evidence changed: '+file)
        ready = read(out/'readiness.json')
        if ready['status'] != 'READY_FOR_INDEPENDENT':
            raise RuntimeError('Preparation did not pass')
        estimate = ready['estimated_total_seconds_with_25pct_and_120s']
        print('正式480次求解，'+(f'短测估计约 {estimate/60:.1f} 分钟。' if estimate is not None else '恢复短测无完整吞吐估计，耗时待确认。'))
        formal_pointer = BASE/'active_run.json'
        if formal_pointer.exists():
            formal = resolve_root(read(formal_pointer)['out'])
            fc = check_contract(formal)
            if fc['preparation'] != release:
                raise RuntimeError('Existing formal run uses a different release')
        else:
            formal = create(fresh('independent'), 'formal', ref, preparation=release)
            write(formal_pointer, dict(out=str(formal)))
        if read(formal/'status.json')['status'] == 'COMPLETED':
            print('已完成，不重复计算。回读：', formal); return
        execute(formal)
        print('\nG10-R3独立比较完成。回读目录：', formal)


if __name__ == '__main__':
    main(sys.argv[1:])
