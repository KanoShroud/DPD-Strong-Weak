"""Bounded, truth-blind VP route diagnosis on frozen IQ; no baseline edits.

The crossed refiners start from a common joint coefficient fit, so VP_START_VP
is a controlled diagnostic, not a relabelled reproduction of native VP.
"""
import argparse
import copy
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .assignment_metrics import evaluate
from ...runtime import read, write, sha, load_scene, resource_check
from ...physics import grid
from ...methods import peaks
from ...r1.scenes import observation, digest
from ...r1.physics import small_fit
from ...r2.solvers import local
from ...r3.solvers import Model, Trace, projected_gradient
from ...r3.common_stop import refine_pair
from ..config import ROOT, BASE
from ..runtime import require_exclusive, lock
from ..adaptation_review import refine as vp_step
from ..bnomp import raw_response
from ..formal_adapter import run as frozen_run

SOURCE = BASE/'review_supplement_formal_20261020_122624_540128'
AUDIT = ROOT/'paper/data/review_supplement_20261020/audit.json'
TASKS = ([(g, 1000, d) for g in (7, 8, 13, 14) for d in (0, 15)]
         + [(4, r, d) for r in (1000, 1001) for d in (0, 15)]
         + [(g, 1000, 0) for g in (0, 5, 9, 12)])
METHODS = ('AP_NATIVE', 'VP_NATIVE', 'AP_START_AP', 'AP_START_VP',
           'VP_START_AP', 'VP_START_VP')


def now():
    torch.cuda.synchronize()
    return time.perf_counter()


def fit_all(m, positions):
    a = raw_response(m, positions)
    scale = torch.linalg.vector_norm(a, dim=1)
    q, residual = small_fit(a/scale[:, None, :], m.x)
    m.counts['coefficient_joint_refits'] = m.counts.get('coefficient_joint_refits', 0)+1
    return a, q/scale, residual


def initialize(m, cfg, kind):
    points = grid(cfg['grid'])
    first = points[int(np.argmax(m.conditional(points)))]
    if kind == 'AP':
        first, _, _ = local(m, first[None], cfg)
        first = first[0]
    else:
        first, a, q, _ = vp_step(m, first, m.x, cfg, True)
        residual = m.x-a*q[:, None]
        for _ in range(cfg['bnomp_cycles']):
            before = float(residual.abs().square().sum().cpu()/m.energy.cpu())
            first, a, q, _ = vp_step(m, first, m.x, cfg, True)
            residual = m.x-a*q[:, None]
            after = float(residual.abs().square().sum().cpu()/m.energy.cpu())
            if abs(before-after) <= cfg['bnomp_energy_tolerance']:
                break
    _, _, residual = fit_all(m, first[None])
    scores = (m.conditional(points, first[None]) if kind == 'AP'
              else m.conditional(points, residual=residual))
    second = points[peaks(points, scores, cfg['top'])[0]]
    return np.vstack([first, second])


def block_refine(m, start, cfg):
    """Native conditional VP updates from a standardized joint-LS start."""
    p = np.asarray(start, float).copy()
    matrix, fitted, residual = fit_all(m, p)
    atoms = [matrix[:, :, k] for k in range(2)]
    coeffs = [fitted[:, k] for k in range(2)]
    history, attempts = [], 0
    reason = 'ITERATION_LIMIT'
    for cycle in range(cfg['bnomp_cycles']):
        m.check()
        before = float(residual.abs().square().sum().cpu()/m.energy.cpu())
        for slot in range(2):
            data = residual+atoms[slot]*coeffs[slot][:, None]
            p[slot], atoms[slot], coeffs[slot], updates = vp_step(m, p[slot], data, cfg, True)
            attempts += len(updates)
            residual = data-atoms[slot]*coeffs[slot][:, None]
        after = float(residual.abs().square().sum().cpu()/m.energy.cpu())
        if after > before+1e-12:
            raise ArithmeticError('Block refinement increased residual')
        if cycle < 3 or (cycle+1) % 25 == 0:
            history.append(dict(cycle=cycle+1, objective=after, positions=p.tolist()))
        if abs(before-after) <= cfg['bnomp_energy_tolerance']:
            reason = 'NATIVE_ENERGY_STABLE'
            break
    fit_all(m, p)
    return p, dict(cycles=cycle+1, attempts=attempts, history=history, stop_reason=reason)


def controlled(iq, method, ap_cfg, vp_cfg):
    started = now()
    t = Trace(resource_check)
    m = Model(iq, t.check, sample_rate_hz=ap_cfg['sample_rate_hz'],
              half_band_hz=ap_cfg['half_band_hz'])
    t.model = m
    init, _, refiner = method.split('_')
    start = initialize(m, ap_cfg if init == 'AP' else vp_cfg, init)
    initialized = now()
    if refiner == 'AP':
        p, _, detail = refine_pair(m, t, start, 'AP_1_JOINT', ap_cfg)
    else:
        p, detail = block_refine(m, start, vp_cfg)
    value, gradient = m.joint(p, True)
    done = now()
    t.model = None
    return {**detail, 'positions': p.tolist(), 'initial_positions': start.tolist(), 'objective': value,
            'projected_gradient': projected_gradient(p, gradient),
            'complete_seconds': done-started, 'initialization_seconds': initialized-started,
            'refinement_seconds': done-initialized, 'counts': copy.deepcopy(m.counts)}


def score_probe(iq, ap_cfg, vp_cfg, truth):
    """Truth enters only the post-score diagnostic ranks, never initialization."""
    t = Trace(resource_check)
    m = Model(iq, t.check, sample_rate_hz=ap_cfg['sample_rate_hz'],
              half_band_hz=ap_cfg['half_band_hz'])
    t.model = m
    points = grid(ap_cfg['grid'])
    rows = []
    for kind in ('AP', 'VP'):
        first = initialize(m, ap_cfg if kind == 'AP' else vp_cfg, kind)[0]
        _, _, residual = fit_all(m, first[None])
        for rule in ('residual', 'projected'):
            scores = (m.conditional(points, residual=residual) if rule == 'residual'
                      else m.conditional(points, first[None]))
            ids = peaks(points, scores, ap_cfg['top'])
            nearest = np.argmin(np.linalg.norm(points[:, None]-truth[None], axis=2), axis=0)
            rows.append(dict(first_method=kind, first=first.tolist(), rule=rule,
                             top_positions=points[ids].tolist(), top_scores=scores[ids].tolist(),
                             truth_grid_ranks=[int(1+np.sum(scores > scores[i])) for i in nearest]))
    t.model = None
    return rows


def sources():
    result = {str(p.relative_to(ROOT)): sha(p) for p in
              (ROOT/'统一模型代码/gates/g10').rglob('*.py')}
    result.update({str(p.relative_to(ROOT)): sha(p) for p in
                   (ROOT/'统一模型代码/gates/g8').glob('*.py')})
    result['migration_paths.py'] = sha(ROOT/'migration_paths.py')
    return result


def prepare_inputs(tasks):
    source = SOURCE.resolve(strict=True)
    assert source.is_relative_to((ROOT/'outputs_e2e').resolve())
    audit = read(AUDIT)
    cfgs = read(source/'contract.json')['configs']
    cache, identities = {}, {str(AUDIT): sha(AUDIT)}
    for task in tasks:
        key = '_'.join(map(str, task))
        path = source/'pairs'/('geometry_'+key+'.json')
        assert sha(path) == audit['source_files'][str(path)]
        old = read(path)
        scene = load_scene(source, 'new_'+key)
        iq, truth, count = observation(scene, 'mixed')
        assert count == 2 and iq.shape == (4, 8192)
        assert digest(iq) == old['iq_sha']
        np.testing.assert_array_equal(truth, old['truth'])
        payload = source/'inputs'/('new_'+key+'.npz')
        metadata = payload.with_suffix('.json')
        identities.update({str(path): sha(path), str(metadata): sha(metadata), str(payload): sha(payload)})
        cache[task] = (iq, truth, old)
    return cache, cfgs, identities


def summarize(out):
    records = [read(p) for p in sorted((out/'pairs').glob('*.json'))]
    all_rows, summary = [], []
    for pair in records:
        for r in pair['rows']:
            s, w = r['metrics']['errors_m']
            all_rows.append(dict(case=pair['case'], method=r['method'], both20=r['metrics']['both20'],
                                 strong_error_m=s, weak_error_m=w,
                                 seconds=r['result']['complete_seconds'],
                                 cycles=r['result'].get('cycles', sum(x['cycles'] for x in r['result'].get('native_cycles', []))),
                                 stop_reason=r['result'].get('stop_reason')))
    for method in dict.fromkeys(r['method'] for r in all_rows):
        rr = [r for r in all_rows if r['method'] == method]
        summary.append(dict(method=method, n=len(rr), success=sum(r['both20'] for r in rr),
                            mean_seconds=float(np.mean([r['seconds'] for r in rr])),
                            median_seconds=float(np.median([r['seconds'] for r in rr])),
                            weak_rmse=float(np.sqrt(np.mean([r['weak_error_m']**2 for r in rr])))))
    for name, rr in [('per_case', all_rows), ('summary', summary)]:
        if rr:
            with (out/(name+'.csv')).open('w', encoding='utf-8-sig', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rr[0])); w.writeheader(); w.writerows(rr)
    write(out/'report.json', dict(summary=summary, cases=len(records),
                                 scope='Outcome-informed mechanism diagnosis, not population performance'))
    return summary


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser()
    parser.add_argument('--deadline', required=True, help='UTC ISO timestamp, shared across this approved turn')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--stage', choices=['diagnosis', 'candidate', 'confirmation'], default='diagnosis')
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline).timestamp()
    if time.time() >= deadline-300:
        raise RuntimeError('Less than five minutes remain; do not start another pair')
    torch.set_num_threads(20)
    require_exclusive(); resource_check()
    tasks = TASKS if args.stage != 'confirmation' else [(g, r, d) for g in range(16)
                                                       for r in (1002, 1003) for d in (0, 15)]
    methods = (METHODS if args.stage == 'diagnosis' else
               ('AP_NATIVE', 'VP_NATIVE', 'SEARCH_ONLY', 'UPDATE_ONLY', 'FULL_CONDITIONAL')
               if args.stage == 'candidate' else ('AP_NATIVE', 'VP_NATIVE', 'FULL_CONDITIONAL'))
    cache, cfgs, identities = prepare_inputs(tasks)
    source_hashes = sources()
    if args.resume:
        out = args.resume.resolve(strict=True); c = read(out/'contract.json')
        assert c['sources'] == source_hashes and c['inputs'] == identities
        assert c['tasks'] == [list(t) for t in tasks] and c['deadline'] == args.deadline
        assert c['stage'] == args.stage and c['methods'] == list(methods)
    else:
        out = BASE/('vp_route_'+args.stage+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        assert not out.exists() and out.resolve().is_relative_to(BASE.resolve())
        out.mkdir(); (out/'pairs').mkdir()
        c = dict(tasks=tasks, methods=methods, stage=args.stage, sources=source_hashes, inputs=identities,
                 configs=cfgs, deadline=args.deadline, budget_seconds=7200,
                 selection=('8 from failure-concentrated layouts, 4 slow-success, 4 ordinary controls; actual native failures=6'
                            if args.stage != 'confirmation' else
                            'All 16 frozen new layouts, realizations 1002/1003, gaps 0/15; existing data, not blind validation'),
                 timing='GPU-synchronized input-to-final-certificate; no disk IO or metric truth inside solver')
        write(out/'contract.json', c)
        (out/'source.py').write_bytes(Path(__file__).read_bytes())
    print('诊断目录：'+str(out), flush=True)
    started = time.perf_counter()
    def execute(iq, method):
        if method.endswith('NATIVE'):
            m = 'AP_1_JOINT' if method == 'AP_NATIVE' else 'PROFILE_DIAG'
            return frozen_run(iq, m, copy.deepcopy(cfgs[m]))
        if args.stage == 'diagnosis':
            return controlled(iq, method, copy.deepcopy(cfgs['AP_1_JOINT']), copy.deepcopy(cfgs['PROFILE_DIAG']))
        from .vp_conditional import solve
        return solve(iq, method, copy.deepcopy(cfgs['PROFILE_DIAG']), copy.deepcopy(cfgs['AP_1_JOINT']))
    with lock(out):
        try:
            if args.stage != 'diagnosis':
                from .vp_conditional import checks
                write(out/'math_checks.json', checks(cache[tasks[-1]][0], cfgs['PROFILE_DIAG']))
            # Warm up all paths equally; warmup is not performance evidence.
            iq, _, _ = cache[tasks[-1]]
            for method in methods:
                execute(iq, method)
            warmup = time.perf_counter()-started
            done = 0
            for index, task in enumerate(tasks):
                key = '_'.join(map(str, task)); path = out/'pairs'/(key+'.json')
                if path.exists():
                    continue
                if time.time() >= deadline-300:
                    break
                require_exclusive(); resource_check()
                iq, truth, old = cache[task]
                order = methods[index % len(methods):]+methods[:index % len(methods)]
                rr = []
                for method in order:
                    if method.endswith('NATIVE'):
                        m = 'AP_1_JOINT' if method == 'AP_NATIVE' else 'PROFILE_DIAG'
                        result = execute(iq, method)
                        previous = next(x['result'] for x in old['rows'] if x['method'] == m)
                        np.testing.assert_allclose(result['positions'], previous['positions'], rtol=0, atol=1e-6)
                    else:
                        result = execute(iq, method)
                    rr.append(dict(method=method, result=result, metrics=evaluate(truth, result['positions'])))
                audit = score_probe(iq, cfgs['AP_1_JOINT'], cfgs['PROFILE_DIAG'], truth) if args.stage == 'diagnosis' else []
                write(path, dict(case=key, task=task, iq_sha=digest(iq), truth=truth.tolist(), rows=rr, scores=audit))
                done += 1
                print(f'已完成 {index+1}/{len(tasks)}：{key}，累计{(time.perf_counter()-started)/60:.1f}分钟', flush=True)
                if args.limit and done >= args.limit:
                    break
            assert sources() == source_hashes
            for path, expected in identities.items():
                assert sha(path) == expected, path
            summary = summarize(out)
            write(out/'status.json', dict(status='COMPLETED' if len(list((out/'pairs').glob('*.json'))) == len(tasks) else 'PARTIAL',
                                         invocation_seconds=time.perf_counter()-started, warmup_seconds=warmup,
                                         checked_sources_inputs_unchanged=True))
            print(json.dumps(summary, ensure_ascii=False), flush=True)
        except BaseException as exc:
            write(out/'status.json', dict(status='FAILED_RETAINED', error=repr(exc), invocation_seconds=time.perf_counter()-started))
            raise


if __name__ == '__main__':
    main()
