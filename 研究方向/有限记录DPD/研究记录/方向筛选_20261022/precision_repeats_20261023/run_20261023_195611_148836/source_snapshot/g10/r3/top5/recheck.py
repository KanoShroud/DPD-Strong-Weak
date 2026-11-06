"""Approved additive Top5 replay on frozen R3 observations; no truth in solver."""
import copy
import csv
import io
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ...physics import grid
from ...methods import peaks
from ...metrics import evaluate, aggregate
from ...r2.solvers import local as native_local
from ..common_stop import refine_pair
from ..solvers import Model, Trace, projected_gradient
from ..runtime import (read, write, sha, seal, check_contract, load_input, validate,
                       job_key, require_exclusive)
from ..config import BASE, ROOT


def run(iq, cfg):
    if cfg['top'] != 5 or not cfg['common_stop']:
        raise ValueError('Expected frozen Top5/common-stop configuration')
    trace = Trace(lambda: None)
    with trace.stage('input_fft_transfer_band_selection'):
        model = Model(iq, trace.check, sample_rate_hz=cfg['sample_rate_hz'],
                      half_band_hz=cfg['half_band_hz'])
        trace.model = model
    points = grid(cfg['grid'])
    with trace.stage('first_source'):
        scores = model.conditional(points)
        first = points[peaks(points, scores, cfg['top'])[0]]
        first, _, _ = native_local(model, first[None], cfg)
    with trace.stage('conditional_candidates'):
        scores = model.conditional(points, first)
        ids = peaks(points, scores, cfg['top'])
        starts = [np.vstack([first, points[i]]) for i in ids]
    if len(starts) != 5:
        raise RuntimeError('Expected five distinct candidate branches')
    branches = []
    with trace.stage('branch_refinement'):
        for rank, start in enumerate(starts, 1):
            before = model.joint(start)
            trace.save(start, before)
            p, value, detail = refine_pair(model, trace, start, 'AP_1_JOINT', cfg)
            branches.append(dict(rank=rank, initial_positions=start.tolist(),
                positions=p.tolist(), initial_objective=before, objective=value, **detail))
    selected = min(branches, key=lambda b: (b['objective'], b['rank']))
    torch.cuda.synchronize()
    seconds, counts = trace.elapsed(), copy.deepcopy(model.counts)
    value, gradient = model.joint(selected['positions'], True)
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise ArithmeticError('Nonfinite selected result')
    return dict(positions=selected['positions'], selected_rank=selected['rank'],
        native_objective=value, projected_gradient=projected_gradient(selected['positions'], gradient),
        seconds=seconds, wall_solver_seconds=seconds+trace.recording_seconds,
        recording_seconds=trace.recording_seconds, counts=counts, stages=trace.stages,
        branches=branches, config=cfg)


def main():
    require_exclusive()
    ref = Path(read(BASE/'active_run.json')['out']).resolve(strict=True)
    contract = check_contract(ref)
    if contract['stage'] != 'formal' or read(ref/'status.json')['status'] != 'COMPLETED':
        raise RuntimeError('Completed formal reference required')
    audit = read(ref/'final_audit_report.json')
    protected = {name: sha(ref/name) for name in
                 ('contract.json', 'input_manifest.json', 'final_audit_report.json')}
    jobs = [j for j in contract['jobs'] if j['method'] == 'AP_1_JOINT']
    cfg = contract['configurations']['AP_1_JOINT']
    # Validate every source observation and baseline before starting new computation.
    cache = []
    for job in jobs:
        _, iq, truth, identity = load_input(ref, job['task'], contract)
        references = {}
        for method in ('AP_1_JOINT', 'NOMP'):
            other = dict(job, method=method)
            name = job_key(other)+'.json'
            if sha(ref/'rows'/name) != audit['rows'][name]:
                raise RuntimeError('Reference row hash changed')
            row = read(ref/'rows'/name)
            validate(row, other, identity, contract['configurations'][method], truth)
            references[method] = row
        cache.append((job, iq, truth, identity, references))
    out = BASE/('top5_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    if out.exists() or not out.resolve().is_relative_to(BASE.resolve()) or out == ref:
        raise RuntimeError('Fresh isolated output required')
    out.mkdir()
    write(out/'contract.json', dict(reference=str(ref), protected=protected, config=cfg,
        source=str(Path(__file__).relative_to(ROOT)), source_sha=sha(Path(__file__)),
        jobs=jobs, selection='minimum final joint objective; ties use initial rank',
        role='approved diagnostic replay on previously inspected R3 data'))
    (out/'recheck.py').write_bytes(Path(__file__).read_bytes())
    print(f'结果目录：{out}', flush=True)
    warm_start = time.perf_counter()
    run(cache[0][1], cfg)
    write(out/'warmup.json', dict(seconds=time.perf_counter()-warm_start,
        torch=torch.__version__, numpy=np.__version__, gpu=torch.cuda.get_device_name(0),
        threads=torch.get_num_threads()))
    # Two predeclared probes cover a prior success and failure; correctness only.
    probes = [cache[0], next(x for x in cache if x[0]['task'] == [1, 301, 6, 'mixed'])]
    for job, iq, truth, identity, refs in probes:
        result = run(iq, cfg)
        check_prefix(result, refs['AP_1_JOINT']['result'])
        write(out/'short'/f'{job_key(job)}.json', result)
    write(out/'short_checks.json', dict(passed=True, first_branch_equivalent=True,
        all_five_finite=True, selected_minimum_objective=True))
    print('短测通过：首分支与原R3一致，五分支有限且按最终目标择优。开始120例。', flush=True)
    rows, start = [], time.perf_counter()
    for index, (job, iq, truth, identity, refs) in enumerate(cache, 1):
        require_exclusive()
        result = run(iq, cfg)
        check_prefix(result, refs['AP_1_JOINT']['result'])
        metrics = evaluate(truth, result['positions'])
        branch_metrics = [evaluate(truth, b['positions']) for b in result['branches']]
        row = dict(job=job, input_sha=identity, truth=truth.tolist(), result=result,
                   metrics=metrics, branch_metrics=branch_metrics,
                   baseline_metrics=refs['AP_1_JOINT']['metrics'],
                   nomp_metrics=refs['NOMP']['metrics'])
        row['record_sha'] = seal(row)
        write(out/'rows'/f'{job_key(job)}.json', row)
        saved = read(out/'rows'/f'{job_key(job)}.json')
        if saved['record_sha'] != seal(saved) or saved['metrics'] != evaluate(truth, saved['result']['positions']):
            raise RuntimeError('Saved row replay mismatch')
        rows.append(row)
        if index % 20 == 0 or index == len(cache):
            print(f'五候选复核 {index}/{len(cache)}，累计 {time.perf_counter()-start:.1f}s', flush=True)
    check_contract(ref)
    if any(sha(ref/name) != value for name, value in protected.items()):
        raise RuntimeError('Protected evidence changed')
    summary = aggregate([r['metrics'] for r in rows])
    summary.update(strong_rmse_m=float(np.sqrt(np.mean([r['metrics']['errors_m'][0]**2 for r in rows]))),
        weak_rmse_m=float(np.sqrt(np.mean([r['metrics']['errors_m'][1]**2 for r in rows]))),
        mean_wall_seconds=float(np.mean([r['result']['wall_solver_seconds'] for r in rows])),
        p95_wall_seconds=float(np.quantile([r['result']['wall_solver_seconds'] for r in rows], .95)),
        mean_objective_calls=float(np.mean([r['result']['counts']['joint_evaluations'] for r in rows])),
        mean_gradient_calls=float(np.mean([r['result']['counts']['gradient_calls'] for r in rows])),
        selected_precision_count=sum(r['result']['projected_gradient'] <= cfg['gtol'] for r in rows),
        branch_precision_count=sum(b['projected_gradient'] <= cfg['gtol'] for r in rows for b in r['result']['branches']),
        recovered=sum(not all(r['baseline_metrics']['success']['20']) and all(r['metrics']['success']['20']) for r in rows),
        harmed=sum(all(r['baseline_metrics']['success']['20']) and not all(r['metrics']['success']['20']) for r in rows),
        both_count=sum(all(r['metrics']['success']['20']) for r in rows),
        selected_ranks={str(i): sum(r['result']['selected_rank'] == i for r in rows) for i in range(1, 6)},
        total_batch_seconds=time.perf_counter()-start)
    write(out/'comparison_report.json', summary)
    buf = io.StringIO(newline='')
    writer = csv.writer(buf)
    writer.writerow(['layout', 'realization', 'gap_db', 'baseline_success', 'top5_success',
                     'rank', 'weak_error_m', 'wall_seconds', 'successful_branch_ranks'])
    for r in rows:
        writer.writerow(r['job']['task'][:3]+[all(r['baseline_metrics']['success']['20']),
            all(r['metrics']['success']['20']), r['result']['selected_rank'], r['metrics']['errors_m'][1],
            r['result']['wall_solver_seconds'], [i+1 for i, m in enumerate(r['branch_metrics']) if all(m['success']['20'])]])
    (out/'per_case.csv').write_text(buf.getvalue(), encoding='utf-8-sig')
    write(out/'final_audit_report.json', dict(passed=True, count=len(rows),
        prefix_passed=True, input_identity_passed=True, metric_replay_passed=True,
        protected=protected, rows={p.name: sha(p) for p in (out/'rows').glob('*.json')}))
    write(out/'status.json', dict(status='COMPLETED', count=len(rows)))
    print(summary, flush=True)


def check_prefix(result, baseline):
    first = result['branches'][0]
    old = baseline['mechanism']['branches'][0]
    np.testing.assert_allclose(first['initial_positions'], old['initial_positions'], rtol=0, atol=1e-10)
    np.testing.assert_allclose(first['positions'], baseline['positions'], rtol=0, atol=1e-7)
    np.testing.assert_allclose(first['objective'], baseline['native_objective'], rtol=0, atol=1e-12)
    assert result['native_objective'] <= first['objective']+1e-14
    assert all(np.isfinite(b['objective']) and np.isfinite(b['positions']).all() for b in result['branches'])
    assert result['selected_rank'] == min(result['branches'], key=lambda b: (b['objective'], b['rank']))['rank']


if __name__ == '__main__':
    main()
