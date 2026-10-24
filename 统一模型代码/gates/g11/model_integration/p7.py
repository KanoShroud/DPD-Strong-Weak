"""Frozen P7 stage-placement experiment; no truth in solve().

Every method runs its own complete prefix and charges all model preparation.
Residual scoring is an initialization heuristic, not conditional projection.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits

from .runner import ROOT, read, write, sha, layouts
from .models import build, FS, BAND
from .solvers import CONFIG, local
from .hybrid import array_hash
from .hybrid_runner import get_input, key, read_row, save_row
from .early_stage import taper_retained
from ...g10.physics import grid
from ...g10.methods import peaks
from ...g10.r1.scenes import make
from ...g10.r4.paper_checks.assignment_metrics import evaluate, self_test

BASE = ROOT/'outputs_e2e/unified/e2e_g11/p7_staged'
METHODS = ('A_HANN', 'B_TERMINAL', 'C_RESIDUAL', 'D_STAGED')
REFERENCE = 'E_FINITE_GRID'
REVISION = 'G11_P7_STAGED_V1'
SEED = 20261024053
FORMAL_REPS = tuple(range(29100, 29105))


def solve(iq, method):
    if method not in METHODS+(REFERENCE,):
        raise ValueError(method)
    tick = time.perf_counter()
    hann = build(iq, 'hann')
    models = {'hann': hann}
    points = grid(20.)
    first_scores = hann.scan(points)
    initial = points[peaks(points, first_scores, 1)[0]][None]
    p, first_detail = local(hann, initial, CONFIG)
    original = p.copy()
    prefix_seconds = time.perf_counter()-tick
    stage = time.perf_counter()
    end_model = hann
    if method != 'A_HANN':
        end_model = build(iq, 'finite_fourier')
        models['finite'] = end_model
    prepare_seconds = time.perf_counter()-stage
    stage = time.perf_counter()
    first_update = None
    if method in ('D_STAGED', REFERENCE):
        p, first_update = local(end_model, p.copy(), CONFIG)
    update_seconds = time.perf_counter()-stage
    stage = time.perf_counter()
    if method in ('A_HANN', 'B_TERMINAL'):
        scores = hann.scan(points, others=p)
    elif method == REFERENCE:
        scores = end_model.scan(points, others=p)
    else:
        periodic = build(iq, 'periodic')
        models['periodic'] = periodic
        residual = periodic.fit(p)['residual']
        scores = hann.scan(points, data=taper_retained(residual, periodic))
    ids = peaks(points, scores, CONFIG['top'])
    if len(ids) != 5 or not np.isfinite(scores).all():
        raise ArithmeticError('Invalid candidate spectrum')
    starts = np.stack([np.vstack([p, points[i]]) for i in ids])
    discovery_seconds = time.perf_counter()-stage
    stage = time.perf_counter()
    branches = []
    for rank, start in enumerate(starts, 1):
        positions, detail = local(end_model, start.copy(), CONFIG)
        branches.append(dict(rank=rank, start=start.tolist(), positions=positions.tolist(), **detail))
    selected = min(branches, key=lambda b: (b['objective'], b['rank']))
    refinement_seconds = time.perf_counter()-stage
    seconds = time.perf_counter()-tick
    counts = Counter()
    for model in models.values():
        counts.update(model.counts)
    stages = dict(prefix=prefix_seconds, preparation=prepare_seconds, first_update=update_seconds,
                  discovery=discovery_seconds, refinement=refinement_seconds)
    stages['other'] = max(0., seconds-sum(stages.values()))
    result = dict(method=method, input_sha=array_hash(iq), positions=selected['positions'],
        objective=selected['objective'], stop_reason=selected['stop_reason'],
        projected_gradient=selected['projected_gradient'], selected_rank=selected['rank'],
        branches=branches, starts=starts.tolist(), starts_sha=array_hash(starts),
        first_initial=initial.tolist(), first_original=original.tolist(), first=p.tolist(),
        first_detail=first_detail, first_update=first_update, grid_indices=ids.tolist(),
        scores_at_candidates=scores[ids].tolist(), counts=dict(counts), seconds=seconds,
        stage_seconds=stages, model=end_model.metadata, data_sha=array_hash(end_model.z),
        max_normal_relative=getattr(end_model, 'max_normal_relative', 0.))
    return result, scores


def task_list(mode):
    reps, gaps = ((29000,), (15,)) if mode == 'pilot' else (FORMAL_REPS, (0, 3, 6, 9, 12, 15))
    return [dict(group=g, realization=r, gap=gap, n=n)
            for g in range(4) for r in reps for gap in gaps for n in (1024, 4096)]


def jobs(contract):
    result = []
    for i, task in enumerate(contract['tasks']):
        offset = i % 4
        for method in METHODS[offset:]+METHODS[:offset]:
            result.append((task, method))
    # Expensive references never delay the main-batch saved report.
    for task in contract['tasks']:
        eligible = task['n'] == 1024 and (
            (contract['mode'] == 'pilot' and task['group'] == 0) or
            (contract['mode'] == 'formal' and task['realization'] == FORMAL_REPS[0]
             and task['gap'] in (0, 15)))
        if eligible:
            result.append((task, REFERENCE))
    return result


def source_map():
    # Reports may evolve independently; numerical dependencies may not.
    files = [p for p in (ROOT/'统一模型代码/gates').rglob('*.py')
             if p.name != 'p7_report.py']
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def historical_ids():
    """Inspect prior contracts, excluding byte-copied source snapshots."""
    observed, files = set(), []
    for path in (ROOT/'outputs_e2e/unified/e2e_g11').rglob('contract.json'):
        if path.is_relative_to(BASE) or 'source_snapshot' in path.parts:
            continue
        value = read(path)
        found = set()

        def visit(x):
            if isinstance(x, dict):
                for k, v in x.items():
                    if k in ('realization', 'realization_id') and isinstance(v, int):
                        found.add(v)
                    elif k in ('realizations', 'reps') and isinstance(v, list):
                        found.update(i for i in v if isinstance(i, int))
                    visit(v)
            elif isinstance(x, list):
                for v in x:
                    visit(v)
        visit(value)
        observed.update(found)
        files.append(dict(path=str(path.relative_to(ROOT)), sha=sha(path)))
    return observed, files


def create(mode, hours=None, preparation=None):
    if mode not in ('pilot', 'formal'):
        raise ValueError(mode)
    observed, history = historical_ids()
    if observed.intersection({29000, *FORMAL_REPS}):
        raise RuntimeError('Reserved realization was already used in prior G11 contract')
    out = BASE/(mode+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True, exist_ok=False)
    for name in ('inputs', 'rows', 'spectra'):
        (out/name).mkdir()
    sources = source_map()
    for name in sources:
        dst = out/'source_snapshot'/name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((ROOT/name).read_bytes())
    tasks = task_list(mode)
    coords, inputs = layouts(), {}
    for g, rep in sorted({(t['group'], t['realization']) for t in tasks}):
        scene = make(g, rep, gap=0., n=8192, fs=FS, snr=6., positions=coords[g], seed=SEED)
        path = out/'inputs'/f'{g}_{rep}.npz'
        np.savez_compressed(path, signal=scene['signal'], noise=scene['noise'], positions=scene['positions'])
        inputs[path.name] = sha(path)
        write(out/'inputs'/f'{g}_{rep}_meta.json', scene['meta'])
    contract = dict(revision=REVISION, mode=mode, tasks=tasks, sources=sources, inputs=inputs,
        layouts=coords, seed=SEED, generator_n=8192, fs=FS, band=BAND, snr_inband_db=6.,
        config=CONFIG, methods=METHODS, threads=2, budget_hours=hours, preparation=preparation,
        python=sys.version, numpy=np.__version__, history_scan=history,
        scope='new realizations at fixed existing geometries; not new-geometry generalization',
        timing='independent complete runs; cold model preparation; offline truth/IO/report excluded',
        candidates='five; own final objective; no oracle',
        stats='stratified realization-paired bootstrap retaining all gaps and lengths',
        reference='Hann first search then finite local update + full finite second conditional grid')
    write(out/'contract.json', contract)
    write(out/'ledger.json', dict(attempts=[]))
    return out, contract


def verify(out, contract):
    if not out.resolve(strict=True).is_relative_to(BASE.resolve()):
        raise ValueError('Outside P7 root')
    if contract['revision'] != REVISION or contract['config'] != CONFIG:
        raise RuntimeError('Protocol/configuration mismatch')
    for name, identity in contract['sources'].items():
        if sha(ROOT/name) != identity:
            raise RuntimeError('Frozen source changed: '+name)
    for name, identity in contract['inputs'].items():
        if sha(out/'inputs'/name) != identity:
            raise RuntimeError('Input changed: '+name)


def validate_row(row, iq, truth, task, method):
    if row['task'] != task or row['method'] != method or row['input_sha'] != array_hash(iq):
        raise RuntimeError('Saved row/input identity mismatch')
    result = row['result']
    if row['metrics'] != evaluate(truth, result['positions']):
        raise RuntimeError('Metric replay failed')
    if len(result['branches']) != 5:
        raise ArithmeticError('Missing branch')
    best = min(result['branches'], key=lambda b: (b['objective'], b['rank']))
    assert result['selected_rank'] == best['rank'] and result['positions'] == best['positions']
    for branch in result['branches']:
        assert np.isfinite(branch['positions']).all() and np.isfinite(branch['objective'])
        assert branch['objective'] <= branch['initial_objective']+1e-12
    np.testing.assert_allclose(sum(result['stage_seconds'].values()), result['seconds'], atol=1e-9)


def execute(out, contract, limit=None):
    verify(out, contract)
    ledger = read(out/'ledger.json')
    for previous in ledger['attempts']:
        if previous['status'] == 'RUNNING':
            previous['status'] = 'INTERRUPTED_UNKNOWN_TAIL'
    attempt = dict(start_utc=datetime.now(timezone.utc).isoformat(), observed_seconds=0., status='RUNNING')
    ledger['attempts'].append(attempt)
    start, completed_new, last_print = time.perf_counter(), 0, -1e9
    todo = jobs(contract)

    def checkpoint():
        attempt['observed_seconds'] = time.perf_counter()-start
        write(out/'ledger.json', ledger)

    def progress(index, task, method):
        nonlocal last_print
        now = time.perf_counter()
        if now-last_print < 5 and index < len(todo):
            return
        last_print = now
        total = sum(a['observed_seconds'] for a in ledger['attempts'][:-1])+now-start
        bar = '#' * int(24*index/len(todo))
        print(f'\rP7 [{bar:<24}] {index}/{len(todo)} | 累计 {total/60:.1f} min | {key(task)} {method:<14}   ', end='', flush=True)

    checkpoint()
    try:
        with threadpool_limits(limits=2):
            warm = time.perf_counter()
            dummy = np.random.default_rng(17).normal(size=(4, 128)).astype(complex)
            for kind in ('hann', 'finite_fourier'):
                build(dummy, kind).fit([[170., 120.], [-130., -210.]], gradient=True)
            write(out/f'warmup_{len(ledger["attempts"])}.json', dict(seconds=time.perf_counter()-warm))
            for index, (task, method) in enumerate(todo):
                iq, truth = get_input(out, task)
                name = key(task)+'_'+method
                path = out/'rows'/(name+'.json')
                if path.exists():
                    row = read_row(path)
                    validate_row(row, iq, truth, task, method)
                    if sha(out/'spectra'/row['spectrum']) != row['spectrum_sha']:
                        raise RuntimeError('Spectrum changed')
                    continue
                checkpoint()
                total = sum(a['observed_seconds'] for a in ledger['attempts'])
                if contract['budget_hours'] is not None and total >= contract['budget_hours']*3600:
                    attempt['status'] = 'BUDGET_PAUSED'
                    break
                if limit is not None and completed_new >= limit:
                    attempt['status'] = 'LIMIT_PAUSED'
                    break
                progress(index, task, method)
                result, scores = solve(iq, method)
                spectrum = out/'spectra'/(name+'.npz')
                np.savez_compressed(spectrum, scores=scores)
                row = dict(task=task, method=method, input_sha=array_hash(iq), result=result,
                    truth=truth.tolist(), metrics=evaluate(truth, result['positions']),
                    spectrum=spectrum.name, spectrum_sha=sha(spectrum))
                validate_row(row, iq, truth, task, method)
                save_row(path, row)
                completed_new += 1
                checkpoint()
            else:
                attempt['status'] = 'COMPLETED'
        verify(out, contract)
        checkpoint()
        count = len(list((out/'rows').glob('*.json')))
        write(out/'audit.json', dict(completed=count == len(todo), source_identity=True,
            input_identity=True, rows=count, expected=len(todo), state=attempt['status']))
        print(f'\n已保存 {count}/{len(todo)}；{attempt["status"]}：{out}', flush=True)
    except BaseException as exc:
        attempt['status'] = 'FAILED_OR_INTERRUPTED'
        checkpoint()
        write(out/f'failure_{len(ledger["attempts"])}.json', dict(error=repr(exc), traceback=traceback.format_exc()))
        raise


def tests():
    """Legacy development replay only; no new formal realization is solved."""
    from .early_stage import first_source, candidate_starts, finish
    from .early_ablation import solve as old_staged
    from .solvers import run
    from .early_stage import checks as old_checks
    from .hybrid import ap_prefix
    out = BASE/('checks_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True)
    scene = make(0, 28001, gap=0., n=8192, fs=FS, snr=6., positions=layouts()[0], seed=SEED)
    iq = (scene['signal'][0]*10**.75+scene['signal'][1]+scene['noise'])[:, :1024]
    findings = dict(metrics=self_test())
    with threadpool_limits(limits=2):
        prefix, _ = ap_prefix(iq)
        findings['input_and_conditional'] = old_checks(iq, prefix)
        first = first_source(iq)
        for method in METHODS:
            new, _ = solve(iq, method)
            if method == 'A_HANN':
                old = run(build(iq, 'hann'), 'AP5', grid(20.))
            elif method == 'D_STAGED':
                old, _ = old_staged(iq, first)
            else:
                variant = 'terminal' if method == 'B_TERMINAL' else 'periodic_residual'
                model, starts, _, info = candidate_starts(iq, variant, first)
                old = finish(model, starts, info)
            np.testing.assert_allclose(new['positions'], old['positions'], atol=1e-7, rtol=0)
            np.testing.assert_allclose([b['start'] for b in new['branches']],
                                       [b['start'] for b in old['branches']], atol=1e-8, rtol=0)
            assert new['selected_rank'] == old['selected_rank']
            findings[method] = dict(old_equivalence=True, positions=new['positions'])
    sample = dict(text='中文状态保存与断点恢复', input_sha=array_hash(iq))
    save_row(out/'中文核验.json', sample)
    assert read_row(out/'中文核验.json') == sample
    observed, history = historical_ids()
    assert not observed.intersection({29000, *FORMAL_REPS})
    findings.update(passed=True, reserved_ids_unused_in_G11=True, history_contracts=len(history),
                    sources=source_map(), unicode_roundtrip=True)
    write(out/'checks.json', findings)
    print(out, flush=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('tests', 'pilot'))
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.mode == 'tests':
        tests()
        return
    if args.resume:
        out = args.resume.resolve(strict=True)
        contract = read(out/'contract.json')
        assert contract['mode'] == 'pilot'
    else:
        out, contract = create('pilot')
    print('结果目录：', out, flush=True)
    execute(out, contract, args.limit)


if __name__ == '__main__':
    main()
