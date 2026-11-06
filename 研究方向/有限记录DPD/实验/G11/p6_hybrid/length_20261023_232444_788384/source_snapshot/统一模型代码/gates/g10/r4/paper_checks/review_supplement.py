"""Bounded review supplement; frozen solvers, separate pilot/formal observations.

FC_VP reuses frozen refine bytecode, substituting only the curvature callable.
The VP trial objective, coefficient schedule, safeguards and stops are identical.
No global solver state is patched.
"""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import time
from types import FunctionType

import numpy as np
import torch

from migration_paths import input_path
from .assignment_metrics import evaluate
from .. import adaptation_review
from ..bnomp import fixed_derivatives
from ..formal_adapter import run as frozen_run
from ..config import ROOT, BASE
from ..runtime import require_exclusive, resource_check, lock
from ...runtime import read, write, sha, save_scene, load_scene
from ...r1.scenes import make, observation, digest
from ...r2.band.scenes import geometry_valid

GAPS = (0, 3, 6, 9, 12, 15)
GEOMETRY_SEED = 20261020071
WAVEFORM_SEED = 20261020072
ACTIVE = BASE/'review_supplement_pilot.json'


def fc_vp_runner():
    namespace = dict(adaptation_review.refine.__globals__)
    namespace['profile_derivatives'] = fixed_derivatives
    refine = FunctionType(adaptation_review.refine.__code__, namespace,
                          'fc_direction_vp_acceptance', adaptation_review.refine.__defaults__)
    namespace = dict(adaptation_review.run.__globals__)
    namespace['refine'] = refine
    return FunctionType(adaptation_review.run.__code__, namespace,
                        'fc_vp_run', adaptation_review.run.__defaults__)


FC_VP = fc_vp_runner()


def geometries(seed, per_category):
    rng, rows = np.random.default_rng(seed), []
    for category in range(4):
        for _ in range(per_category):
            for attempt in range(100000):
                p = rng.uniform(-950., 950., (2, 2))
                if geometry_valid(p, category):
                    rows.append(dict(category=category, positions=p.tolist()))
                    break
            else:
                raise RuntimeError('Geometry rejection sampler exhausted')
    return rows


def create(pilot, budget_hours):
    pointer = read(BASE/'formal_run.json')
    source = input_path(pointer['out']).resolve(strict=True)
    assert source.is_relative_to(BASE.resolve())
    assert sha(source/'contract.json') == pointer['contract_sha']
    audit = read(source/'final_audit_report.json')
    assert audit['passed']
    configs = {}
    for method in ('AP_1_JOINT', 'AP_5_JOINT', 'PROFILE_DIAG'):
        p = source/'rows'/('0_400_0_'+method+'.json')
        assert sha(p) == audit['rows'][p.name]
        configs[method] = read(p)['result']['config']
    common = [[g, r, gap] for g in range(20) for r in (400, 410) for gap in GAPS]
    if pilot:
        common = [[g, 400, gap] for g in (0, 5, 10, 15) for gap in (0, 15)]
    layouts = geometries(GEOMETRY_SEED + int(pilot), 1 if pilot else 4)
    new_tasks = [[g, r, gap] for g in range(len(layouts))
                 for r in ((900,) if pilot else range(1000, 1005))
                 for gap in ((0, 15) if pilot else GAPS)]
    jobs = [('mechanism', t, ('FC_VP', 'VP_VP')) for t in common]
    jobs += [('geometry', t, ('AP_1_JOINT', 'AP_5_JOINT', 'PROFILE_DIAG')) for t in new_tasks]
    code = list((ROOT/'统一模型代码/gates/g10').rglob('*.py'))
    code += list((ROOT/'统一模型代码/gates/g8').glob('*.py'))
    code += [ROOT/'migration_paths.py']
    contract = dict(pilot=pilot, source=str(source), source_contract_sha=pointer['contract_sha'],
                    source_audit_sha=sha(source/'final_audit_report.json'), configs=configs,
                    layouts=layouts, geometry_seed=GEOMETRY_SEED+int(pilot),
                    waveform_seed=WAVEFORM_SEED+int(pilot), jobs=jobs,
                    budget_seconds=budget_hours*3600,
                    source_hashes={str(p.relative_to(ROOT)): sha(p) for p in code},
                    protocol='REVIEW_SUPPLEMENT_V1', input_policy='No outcome-based selection',
                    expected_solves=sum(len(j[2]) for j in jobs))
    out = BASE/('review_supplement_'+('pilot_' if pilot else 'formal_')+
                datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    write(out/'contract.json', contract)
    write(out/'contract_identity.json', dict(sha256=sha(out/'contract.json')))
    write(out/'ledger.json', dict(seconds=0.))
    if pilot:
        write(ACTIVE, dict(out=str(out), contract_sha=sha(out/'contract.json')))
    return out


def verify_contract(out, c):
    assert sha(out/'contract.json') == read(out/'contract_identity.json')['sha256']
    for name, value in c['source_hashes'].items():
        assert sha(ROOT/name) == value, name
    source = Path(c['source'])
    assert sha(source/'contract.json') == c['source_contract_sha']
    assert sha(source/'final_audit_report.json') == c['source_audit_sha']


def consume(out, c, stage, task):
    key = '_'.join(map(str, task))
    if stage == 'mechanism':
        source = Path(c['source'])
        audit = read(source/'final_audit_report.json')
        meta = source/'inputs'/(key+'.json')
        assert sha(meta) == audit['inputs'][key]['identity']['metadata_sha']
        scene = load_scene(source, key)
        iq, truth, _ = observation(scene, 'mixed')
        assert digest(iq) == audit['inputs'][key]['identity']['iq_sha']
    else:
        g, realization, gap = task
        key = 'new_'+key
        if not (out/'inputs'/(key+'.json')).exists():
            scene = make(g, realization, gap, n=8192, fs=50e6, snr=6.,
                         positions=c['layouts'][g]['positions'], seed=c['waveform_seed'])
            scene['meta']['role'] = 'review_pilot' if c['pilot'] else 'review_independent'
            save_scene(out, key, scene)
        scene = load_scene(out, key)
        assert scene['meta']['seed'] == c['waveform_seed']
        assert scene['meta']['positions'] == c['layouts'][g]['positions']
        assert scene['meta']['realization'] == realization and scene['meta']['gap_db'] == gap
        iq, truth, _ = observation(scene, 'mixed')
    assert iq.shape == (4, 8192)
    return iq, truth, digest(iq)


def solve(iq, c, method):
    torch.cuda.synchronize()
    started = time.perf_counter()
    if method in ('FC_VP', 'VP_VP'):
        run = FC_VP if method == 'FC_VP' else adaptation_review.run
        result = run(iq, copy.deepcopy(c['configs']['PROFILE_DIAG']), True)
    else:
        result = frozen_run(iq, method, copy.deepcopy(c['configs'][method]))
    torch.cuda.synchronize()
    complete = time.perf_counter()-started
    assert np.isfinite(result['positions']).all()
    # Identical wall-clock boundary including diagnostic recording and final certificate.
    result['review_complete_seconds'] = complete
    return result


def summary(out, c):
    groups = {}
    paths = sorted((out/'pairs').glob('*.json'))
    for p in paths:
        pair = read(p)
        for row in pair['rows']:
            groups.setdefault(pair['stage']+'/'+row['method'], []).append(row)
    stats = {}
    for label, rows in groups.items():
        seconds = np.array([r['result']['review_complete_seconds'] for r in rows])
        errors = np.array([r['metrics']['errors_m'] for r in rows])
        steps = [u['step_m'] for r in rows for u in r['result'].get('updates', [])]
        stats[label] = dict(n=len(rows), both20=sum(r['metrics']['both20'] for r in rows),
                           weak_rmse_m=float(np.sqrt(np.mean(errors[:, 1]**2))),
                           mean_seconds=float(seconds.mean()), p95_seconds=float(np.quantile(seconds, .95)),
                           mean_updates=float(np.mean([len(r['result'].get('updates', [])) for r in rows])),
                           step_quantiles_m=np.quantile(steps, [0, .5, .95, 1]).tolist() if steps else None)
    total = sum(r['n'] for r in stats.values())
    forecast = None
    if len(stats) == 5:
        forecast = 1.25*(240*sum(stats['mechanism/'+m]['mean_seconds'] for m in ('FC_VP', 'VP_VP'))+
                         480*sum(stats['geometry/'+m]['mean_seconds'] for m in
                                 ('AP_1_JOINT', 'AP_5_JOINT', 'PROFILE_DIAG')))+600
    result = dict(pilot=c['pilot'], completed_solves=total, expected_solves=c['expected_solves'],
                  summaries=stats, forecast_formal_seconds=forecast,
                  forecast_basis='Pilot means by stage/method; 25% margin plus 600s IO/audit allowance',
                  status='COMPLETED' if total == c['expected_solves'] else 'PARTIAL')
    write(out/'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--formal', action='store_true')
    parser.add_argument('--budget-hours', type=float)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--limit', type=int, help='Maximum complete input groups in this invocation')
    args = parser.parse_args()
    if args.formal and not args.resume and args.budget_hours is None:
        raise SystemExit('正式运行须先确认短测预算，再提供 --budget-hours。')
    budget = args.budget_hours if args.budget_hours is not None else 1/3
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError('Positive finite budget required')
    require_exclusive()
    resource_check()
    out = args.resume.resolve(strict=True) if args.resume else create(not args.formal, budget)
    assert out.is_relative_to(BASE.resolve()) and out != BASE.resolve()
    c = read(out/'contract.json')
    verify_contract(out, c)
    print('结果目录：'+str(out), flush=True)
    with lock(out):
        ledger = read(out/'ledger.json')
        started = time.perf_counter()
        def flush():
            write(out/'ledger.json', dict(seconds=ledger['seconds']+time.perf_counter()-started))
        try:
            # Explicit warmup is separate; results are not used as evidence.
            stage, task, methods = c['jobs'][0]
            iq, _, _ = consume(out, c, stage, task)
            warm_started = time.perf_counter()
            for method in ('FC_VP', 'VP_VP', 'AP_1_JOINT', 'AP_5_JOINT', 'PROFILE_DIAG'):
                solve(iq, c, method)
            write(out/('warmup_'+str(time.time_ns())+'.json'),
                  dict(seconds=time.perf_counter()-warm_started, separate=True))
            completed = 0
            for index, (stage, task, methods) in enumerate(c['jobs']):
                p = out/'pairs'/(stage+'_'+'_'.join(map(str, task))+'.json')
                iq, truth, identity = consume(out, c, stage, task)
                if p.exists():
                    old = read(p)
                    assert old['iq_sha'] == identity and old['truth'] == truth.tolist()
                    assert {r['method'] for r in old['rows']} == set(methods)
                    for r in old['rows']:
                        assert r['metrics'] == evaluate(truth, r['result']['positions'])
                    continue
                if ledger['seconds']+time.perf_counter()-started >= c['budget_seconds']:
                    print('预算边界：保存进度，不启动下一组。', flush=True)
                    break
                order = methods[index % len(methods):]+methods[:index % len(methods)]
                rows = []
                for method in order:
                    result = solve(iq, c, method)
                    rows.append(dict(method=method, result=result,
                                     metrics=evaluate(truth, result['positions'])))
                write(p, dict(stage=stage, task=task, iq_sha=identity, truth=truth.tolist(), rows=rows))
                flush()
                completed += 1
                print(f'{index+1}/{len(c["jobs"])} {stage} {task} 已保存完整配对', flush=True)
                if args.limit and completed >= args.limit:
                    break
            verify_contract(out, c)
            result = summary(out, c)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        finally:
            flush()


if __name__ == '__main__':
    main()
