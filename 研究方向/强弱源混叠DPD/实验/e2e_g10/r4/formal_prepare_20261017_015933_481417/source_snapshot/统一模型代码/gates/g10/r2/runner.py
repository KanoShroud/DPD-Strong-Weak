"""Finite preparation and user-run development. No automatic independent stage."""
import argparse
from pathlib import Path
import time
import traceback
import numpy as np
from ..runtime import Progress, save_scene
from ..metrics import evaluate
from ..r1.scenes import make
from ..r1.runner import warmup
from .config import BASE, CONFIG, METHODS, tasks
from .runtime import (read, write, sha, verify, Ledger, fresh, preflight, reference_manifest,
                      load_input, seal, validate_row, audit, lock)
from . import solvers


def record(out, task, method, clock):
    key = f'{task[0]}_{task[1]}_{task[2]:g}'
    path = out/'rows'/f'{key}_{method}.json'
    scene, iq, truth, identity, refs = load_input(out, task)
    if path.exists():
        row = read(path); validate_row(row, identity)
        return row
    clock.flush(dict(tag=path.stem, started_utc_seconds=time.time(), ledger_start_seconds=clock.total()))
    result = solvers.run(iq, method, guard=clock.check)
    if not np.isfinite(result['positions']).all():
        raise ArithmeticError('Invalid estimated positions')
    row = dict(tag=path.stem, task=list(task), method=method, kind='mixed', scene=scene['meta'],
               input_sha=identity, truth=truth.tolist(), result=result,
               metrics=evaluate(truth, result['positions']), references=refs,
               prior_attempts=[x for x in clock.state['interruptions'] if x['tag'] == path.stem])
    row['record_sha'] = seal(row)
    write(path, row)
    clock.flush()
    return row


def run_rows(out, stage, clock, estimated=None):
    plan = tasks(stage)
    progress = Progress(len(plan)*len(METHODS), 'G10-R2准备' if stage == 'prepare' else 'G10-R2双候选比较')
    rows = []
    for index, task in enumerate(plan):
        # Rotate order to avoid always assigning one method the same warm/cache slot.
        order = METHODS[index % len(METHODS):]+METHODS[:index % len(METHODS)]
        for method in order:
            tag = f'{task[0]}_{task[1]}_{task[2]:g}_{method}'
            cost = (estimated or {}).get(method, 0.)
            if not (out/'rows'/f'{tag}.json').exists() and clock.total()+cost > CONFIG['cumulative_hours']*3600:
                clock.flush()
                write(out/'status.json', dict(status='BUDGET_PAUSED', seconds=clock.total(), next_task=tag))
                audit(out)
                return None
            rows.append(record(out, task, method, clock))
            progress.update(len(rows), f'{method} 布局{task[0]} 实现{task[1]} {task[2]}dB')
    return rows


def prepare(out, resume=False):
    if not resume:
        preflight(out, dict(stage='prepare', config=CONFIG, jobs=tasks('prepare')))
        write(out/'input_manifest.json', reference_manifest())
        write(BASE/'preparation_pointer.json', dict(out=str(out.resolve())))
    contract = read(out/'contract.json')
    if contract['stage'] != 'prepare' or contract['config'] != CONFIG:
        raise RuntimeError('Preparation contract mismatch')
    verify(contract['sources']); verify(contract['r2_sources'])
    with lock(out):
        clock = Ledger(out)
        try:
            print('阶段1/3：新算法、理论关系与工程不变量短测', flush=True)
            from .tests import suite, encoding_check
            if not (out/'numerical_tests.json').exists():
                write(out/'numerical_tests.json', suite())
                write(out/'encoding_test.json', encoding_check())
            checks = read(out/'numerical_tests.json')
            if not checks['passed'] or not read(out/'encoding_test.json')['passed']:
                raise RuntimeError('Preparation correctness failed; evidence preserved')
            for task in tasks('prepare'):
                key = f'{task[0]}_{task[1]}_{task[2]:g}'
                if not (out/'inputs'/f'{key}.json').exists():
                    save_scene(out, key, make(*task))
            write(out/'warmup.json', dict(seconds=warmup(), included_in_algorithm_time=False))
            print('阶段2/3：8条开发观测×7版本；不以成功率决定放行', flush=True)
            rows = run_rows(out, 'prepare', clock)
            if rows is None:
                return
            # The actual pipeline must resume completed rows without a solver call.
            from unittest.mock import patch
            with patch.object(solvers, 'run', side_effect=AssertionError('Unexpected solve on resume')):
                replay = record(out, tasks('prepare')[0], METHODS[0], clock)
            if replay['record_sha'] != read(out/'rows'/f"0_31_0_{METHODS[0]}.json")['record_sha']:
                raise RuntimeError('Resume changed a completed row')
            print('阶段3/3：一致性审计、开发图示、正式成本估算', flush=True)
            final = audit(out, len(tasks('prepare'))*len(METHODS))
            costs = {method: [r['result']['seconds']+r['result']['diagnostics_seconds']
                              for r in rows if r['method'] == method] for method in METHODS}
            typical = sum(float(np.mean(v))*120 for v in costs.values())
            conservative = 1.25*sum(max(v)*120 for v in costs.values())+600
            budget = dict(typical_solver_seconds=typical, estimated_hours=conservative/3600,
                          conservative_seconds=conservative, margin=1.25, other_overhead_seconds=600,
                          per_method_max_seconds={k: max(v) for k, v in costs.items()},
                          per_method_mean_seconds={k: float(np.mean(v)) for k, v in costs.items()},
                          measured_rows=len(rows), planned_observations=120, planned_rows=840,
                          note='8 fixed development observations; max by method across 0/15dB, all four geometries. Estimate, not guarantee.')
            write(out/'budget_estimate.json', budget)
            write(out/'frozen_config.json', CONFIG)
            from .report import generate
            with patch.object(solvers, 'run', side_effect=AssertionError('Replot called solver')):
                generate(out)
            ready = conservative <= CONFIG['cumulative_hours']*3600
            write(out/'readiness.json', dict(ready=ready, engineering_ready=True,
                  reason='READY_FOR_USER_DEVELOPMENT' if ready else 'BUDGET_CONFIRMATION_REQUIRED',
                  config_sha=sha(out/'frozen_config.json'), input_manifest_sha=sha(out/'input_manifest.json'),
                  sources=final['sources'], resume_verified=True, offline_replot_verified=True,
                  estimated_hours=conservative/3600, approved_ceiling_hours=CONFIG['cumulative_hours'],
                  independent_validation='LOCKED_PENDING_REVIEW'))
            clock.flush()
            write(out/'status.json', dict(status='PREPARATION_COMPLETED', seconds=clock.total(),
                                          ready=ready, rows=len(rows), formal_started=False))
            write(BASE/'release.json', dict(preparation=str(out.resolve()), readiness_sha=sha(out/'readiness.json')))
            print(f'准备完成：840次开发求解保守估算{conservative/3600:.2f}小时；未启动完整比较。', flush=True)
        except BaseException as exc:
            clock.flush(keep=True)
            write(out/'status.json', dict(status='STOPPED', error=str(exc), traceback=traceback.format_exc(), seconds=clock.total()))
            raise


def release_check():
    entry = read(BASE/'release.json')
    prep = Path(entry['preparation']).resolve(strict=True)
    if not prep.is_relative_to(BASE.resolve()) or sha(prep/'readiness.json') != entry['readiness_sha']:
        raise RuntimeError('Release identity differs')
    ready = read(prep/'readiness.json')
    if not ready['ready'] or sha(prep/'frozen_config.json') != ready['config_sha']:
        raise RuntimeError('Preparation/configuration not ready')
    if sha(prep/'input_manifest.json') != ready['input_manifest_sha']:
        raise RuntimeError('Input manifest changed')
    verify(ready['sources'])
    # Read-only validation: does not create a run or recalculate any localization.
    if reference_manifest() != read(prep/'input_manifest.json'):
        raise RuntimeError('Frozen R1 reference changed')
    return prep, ready


def develop(out, preparation, resume=False):
    _, ready = release_check()
    if not resume:
        preflight(out, dict(stage='develop', config=CONFIG, jobs=tasks('develop'),
                            preparation=str(preparation.resolve()), readiness_sha=sha(preparation/'readiness.json')))
        write(out/'input_manifest.json', read(preparation/'input_manifest.json'))
    contract = read(out/'contract.json')
    if contract['config'] != CONFIG or contract['stage'] != 'develop':
        raise RuntimeError('Resume configuration differs')
    verify(contract['sources']); verify(contract['r2_sources'])
    with lock(out):
        clock = Ledger(out)
        try:
            if (out/'status.json').exists() and read(out/'status.json')['status'] == 'COMPLETED':
                audit(out, 840)
                print('已经完成，身份核对通过；不重复求解。', flush=True)
                return
            print(f"开发比较：120条观测/840次求解；保守估算{ready['estimated_hours']:.2f}小时。", flush=True)
            print(f'公共预热{warmup():.2f}s（单列）；独立复核保持锁定。', flush=True)
            costs = read(preparation/'budget_estimate.json')['per_method_max_seconds']
            rows = run_rows(out, 'develop', clock, costs)
            if rows is None:
                return
            print('阶段：配对统计、机制图表和最终审计', flush=True)
            audit(out, 840)
            from .report import generate
            generate(out)
            clock.flush()
            write(out/'status.json', dict(status='COMPLETED', seconds=clock.total(), rows=len(rows),
                                          next='REVIEW_AND_FREEZE_CANDIDATES', independent_started=False))
        except BaseException as exc:
            clock.flush(keep=True)
            write(out/'status.json', dict(status='STOPPED', error=str(exc), traceback=traceback.format_exc(), seconds=clock.total()))
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'develop', 'replot', 'check'])
    parser.add_argument('--out', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.stage == 'prepare':
        out = args.out or fresh('prepare')
        print('OUT='+str(out), flush=True)
        prepare(out, args.resume)
    elif args.stage == 'check':
        prep, ready = release_check()
        print('检查通过；完整开发尚未启动：'+str(prep), flush=True)
        print(f"保守估算{ready['estimated_hours']:.2f}小时", flush=True)
    elif args.stage == 'develop':
        prep, _ = release_check()
        develop(args.out or fresh('develop'), prep, args.resume)
    else:
        from .report import generate
        if args.out is None:
            raise ValueError('Replot requires existing --out')
        generate(args.out.resolve(strict=True))


if __name__ == '__main__':
    main()
