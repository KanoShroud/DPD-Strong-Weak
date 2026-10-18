"""Finite preparation; independent complete comparison is user-launched only."""
import argparse
import copy
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch
import numpy as np
from ...metrics import evaluate
from ...runtime import Progress
from ...r1.runner import warmup
from .config import BASE, METHODS
from . import solvers
from .runtime import (read, write, sha, verify, seal, lock, Ledger, fresh, create,
                      check_contract, load_input, validate, audit, tag, job_plan)


def record(out, job, clock):
    task, method, variant = job['task'], job['method'], job['variant']
    path = out/'rows'/f'{tag(task, method, variant)}.json'
    scene, iq, truth, count, identity = load_input(out, task)
    if path.exists():
        row = read(path); validate(row, job, identity)
        return row
    clock.flush(dict(tag=path.stem, started_utc_seconds=time.time(), ledger_start_seconds=clock.total()))
    result = solvers.run(iq, method, count, job['config'], job['seed'], clock.check)
    # Keep native maps only for preregistered first-repeat mixed examples.
    first_repeat = 200 if scene['meta']['role'] == 'independent' else 40
    if not (task[1] == first_repeat and task[3] == 'mixed' and task[2] in (0, 15)):
        result.pop('native_map', None)
    row = dict(tag=path.stem, task=task, method=method, variant=variant, kind=task[3], job=job,
        scene=scene['meta'], input_sha=identity, truth=truth.tolist(), result=result,
        metrics=evaluate(truth, result['positions']),
        prior_attempts=[x for x in clock.state['interruptions'] if x['tag'] == path.stem])
    row['record_sha'] = seal(row)
    write(path, row)
    clock.flush()
    return row


def execute(out, batch='all'):
    contract = check_contract(out)
    formal = contract['stage'] == 'independent'
    plan = [j for j in contract['jobs'] if batch == 'all' or (j['method'] == 'SBL') == (batch == 'slow')]
    with lock(out):
        clock = Ledger(out)
        progress = Progress(len(plan), 'G10-R2带限正式' if formal else 'G10-R2有限校准')
        try:
            if not (out/'warmup.json').exists():
                write(out/'warmup.json', dict(seconds=warmup(), separate=True))
            if not formal and not (out/'numerical_tests.json').exists():
                from .tests import suite
                result = suite(); write(out/'numerical_tests.json', result)
                if not result['passed'] or not result['encoding']['passed']:
                    raise RuntimeError('Correctness/encoding checks failed')
            seen_slow = False
            for i, job in enumerate(plan):
                if job['method'] == 'SBL' and not seen_slow:
                    print('\n主批已保存，进入SBL慢方法；正常单例不会按秒截断。', flush=True)
                    from .report import generate
                    generate(out, selected_only=False)
                    seen_slow = True
                path = out/'rows'/f"{tag(job['task'], job['method'], job['variant'])}.json"
                if not path.exists() and clock.total() >= contract['budget_hours']*3600:
                    clock.flush()
                    write(out/'status.json', dict(status='BUDGET_PAUSED', next_job=path.stem,
                        seconds=clock.total(), formal_started=formal))
                    audit(out)
                    return False
                record(out, job, clock)
                progress.update(i+1, f"{job['method']} 布局{job['task'][0]} 实现{job['task'][1]} {job['task'][2]}dB {job['task'][3]}")
            with patch.object(solvers, 'run', side_effect=AssertionError('Resume must not solve')):
                record(out, plan[0], clock)
            complete = len(list((out/'rows').glob('*.json'))) == len(contract['jobs'])
            audit(out, complete)
            from .report import generate
            with patch.object(solvers, 'run', side_effect=AssertionError('Replot must not solve')):
                generate(out, selected_only=False)
            clock.flush()
            write(out/'status.json', dict(status=('COMPLETED' if formal else 'CALIBRATION_COMPLETED') if complete else 'BATCH_COMPLETED',
                seconds=clock.total(), formal_started=formal, completed=len(list((out/'rows').glob('*.json'))),
                planned=len(contract['jobs']), resume_verified=True, offline_replot_verified=True))
            return complete
        except BaseException as error:
            clock.flush(keep=True)
            write(out/'status.json', dict(status='STOPPED', error=str(error), traceback=traceback.format_exc(),
                seconds=clock.total(), formal_started=formal))
            raise


def select_configuration(out):
    """Predeclared mixed success, GOSPA within 0.1m, then complete mean cost."""
    rows = [read(p) for p in (out/'rows').glob('*.json')]
    selected, table = {}, []
    for method in METHODS:
        available = [r for r in rows if r['method'] == method and r['kind'] == 'mixed']
        alternatives = []
        for variant in sorted({r['variant'] for r in available}):
            subset = [r for r in available if r['variant'] == variant]
            alternatives.append(dict(method=method, variant=variant, n=len(subset),
                both=sum(all(r['metrics']['success']['20']) for r in subset),
                gospa=float(np.mean([r['metrics']['gospa_m'] for r in subset])),
                seconds=float(np.mean([r['result']['seconds'] for r in subset])), config=subset[0]['job']['config']))
        top = max(a['both'] for a in alternatives)
        quality = min(a['gospa'] for a in alternatives if a['both'] == top)
        best = min((a for a in alternatives if a['both'] == top and a['gospa'] <= quality+.1), key=lambda a:(a['seconds'], a['variant']))
        selected[method] = copy.deepcopy(best['config'])
        table.extend(dict(a, selected=a is best) for a in alternatives)
    # Shared prefixes/tolerances are never independently tuned across ablations.
    for method in ('NOMP_EXTRA12', 'NOMP_JOINT'):
        selected[method] = copy.deepcopy(selected['NOMP'])
    write(out/'configuration_selection.json', table)
    return selected


def stopping_review(out):
    items = []
    for p in (out/'rows').glob('*_SBL_v0.json'):
        row = read(p)
        point = row['result']['mechanism'].get('candidate_stop')
        if point is None:
            continue
        early = evaluate(row['truth'], point['positions'])
        late = row['metrics']
        items.append(dict(tag=row['tag'], task=row['task'], early=early, late=late,
            early_seconds=point['seconds'], late_seconds=row['result']['seconds'], iteration=point['iteration'],
            final_iteration=row['result']['counts']['iterations']))
    write(out/'stopping_review.json', dict(rows=items, note='Same trajectory; manual evidence-based freeze, no automatic performance gate.'))
    return items


def freeze(out, sbl_rule):
    audit(out, True)
    if read(out/'status.json')['status'] != 'CALIBRATION_COMPLETED':
        raise RuntimeError('Calibration incomplete')
    checks = read(out/'numerical_tests.json')
    if not checks['passed'] or not checks['encoding']['passed']:
        raise RuntimeError('Correctness failed')
    configs = select_configuration(out)
    configs['SBL']['sbl_stop_rule'] = sbl_rule
    stopping_review(out)
    rows = [read(p) for p in (out/'rows').glob('*.json')]
    costs = {}
    for method in METHODS:
        for kind in ('weak', 'strong', 'mixed'):
            base = method if kind == 'mixed' else ('AP_FULL' if method.startswith('AP_') else method)
            values = [r['result']['seconds'] for r in rows if r['method'] == base and r['kind'] == kind]
            if values:
                costs[f'{method}:{kind}'] = dict(n=len(values), mean=float(np.mean(values)), maximum=max(values))
    plan = job_plan(True, configs)
    typical = sum(costs[f"{j['method']}:{j['task'][3]}"]['mean'] for j in plan)
    conservative = 1.25*sum(costs[f"{j['method']}:{j['task'][3]}"]['maximum'] for j in plan)+600
    write(out/'frozen_config.json', configs)
    write(out/'budget_estimate.json', dict(planned_rows=len(plan), typical_hours=typical/3600,
        conservative_hours=conservative/3600, margin=1.25, overhead_seconds=600, measurements=costs,
        note='All sampled variants enter conservative estimate; SBL uses full observed trajectory cost.'))
    contract = check_contract(out)
    write(out/'readiness.json', dict(ready=conservative <= 20*3600, approved_ceiling_hours=20.,
        sbl_rule=sbl_rule, frozen_config_sha=sha(out/'frozen_config.json'),
        budget_sha=sha(out/'budget_estimate.json'), audit_sha=sha(out/'final_audit_report.json'),
        sources=contract['sources'] | contract['band_sources']))
    write(BASE/'release.json', dict(preparation=str(out), readiness_sha=sha(out/'readiness.json')))


def release():
    pointer = read(BASE/'release.json')
    prep = Path(pointer['preparation']).resolve(strict=True)
    if not prep.is_relative_to(BASE.resolve()) or sha(prep/'readiness.json') != pointer['readiness_sha']:
        raise RuntimeError('Preparation reference changed')
    ready = read(prep/'readiness.json')
    if not ready['ready']:
        raise RuntimeError('Budget confirmation required; no automatic reduction')
    for name, field in (('frozen_config.json', 'frozen_config_sha'), ('budget_estimate.json', 'budget_sha'),
                        ('final_audit_report.json', 'audit_sha')):
        if sha(prep/name) != ready[field]:
            raise RuntimeError('Frozen preparation artifact changed: '+name)
    verify(ready['sources'])
    return prep, ready, read(prep/'frozen_config.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'freeze', 'run', 'check', 'replot'])
    parser.add_argument('--out', type=Path)
    parser.add_argument('--batch', choices=['all', 'main', 'slow'], default='all')
    parser.add_argument('--sbl-rule', choices=['legacy', 'candidate'], default='candidate')
    args = parser.parse_args()
    if args.mode == 'prepare':
        out = args.out or fresh('prepare')
        if not (out/'contract.json').exists():
            create(out)
        write(BASE/'preparation_pointer.json', dict(out=str(out.resolve())))
        print('仅执行开发校准；不会启动正式独立数据。\n'+str(out), flush=True)
        if execute(out, args.batch):
            select_configuration(out); stopping_review(out)
        return
    if args.mode == 'freeze':
        freeze(args.out or Path(read(BASE/'preparation_pointer.json')['out']), args.sbl_rule)
        return
    if args.mode == 'replot':
        from .report import generate
        out = args.out or Path(read(BASE/'active_run.json')['out'])
        generate(out, selected_only=False)
        return
    prep, ready, configs = release()
    print('准备已通过，预算估算：'+str(read(prep/'budget_estimate.json')['conservative_hours'])+'小时', flush=True)
    if args.mode == 'check':
        print('只读预检完成，未生成正式数据。', flush=True)
        return
    active = BASE/'active_run.json'
    if active.exists():
        pointer = read(active)
        if pointer['preparation'] != str(prep):
            raise RuntimeError('Existing run belongs to another preparation')
        out = Path(pointer['out']).resolve(strict=True)
    else:
        out = fresh('independent')
        create(out, True, configs, ready['approved_ceiling_hours'])
        write(active, dict(out=str(out), preparation=str(prep)))
    print('结果目录：'+str(out), flush=True)
    if (out/'status.json').exists() and read(out/'status.json')['status'] == 'COMPLETED':
        audit(out, True)
        print('已完成，核验结果后退出，不重复求解。', flush=True)
        return
    execute(out, args.batch)


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    main()
