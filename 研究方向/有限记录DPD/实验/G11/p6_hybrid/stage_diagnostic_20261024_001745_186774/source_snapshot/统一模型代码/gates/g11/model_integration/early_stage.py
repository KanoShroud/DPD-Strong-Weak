"""P6 bounded early-stage diagnosis. Search routines accept no truth.

Finite residual followed by Hann single-source scanning is an initialization
heuristic, NOT the exact projected conditional objective. Every final branch
is fitted to the original unwindowed observation. No new model or tuning.
"""
import argparse
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .models import build
from .solvers import CONFIG, local
from .hybrid import array_hash
from .hybrid_runner import BASE, ROOT, read, write, sha, key, get_input, read_row, save_row
from ...g10.physics import grid
from ...g10.methods import peaks
from ...g10.r4.paper_checks.assignment_metrics import evaluate
from ...research_discovery.slepian_reference import reconstruct

VARIANTS = ('terminal', 'periodic_residual', 'finite_fit', 'finite_position', 'finite_both')
FULL_CASES = {(0, 15), (1, 0), (1, 15), (3, 15)}
REVISION = 'G11_P6_EARLY_STAGE_DIAGNOSIS_V1'


def first_source(iq):
    tick = time.perf_counter()
    model = build(iq, 'hann')
    points = grid(20.)
    scores = model.scan(points)
    initial = points[peaks(points, scores, 1)[0]][None]
    position, detail = local(model, initial, CONFIG)
    return dict(position=position.tolist(), initial=initial.tolist(), detail=detail,
                seconds=time.perf_counter()-tick, input_sha=array_hash(iq))


def taper_retained(residual, model):
    """The same band->Hann->band operator as Periodic(hann=True)."""
    signal = reconstruct(residual, model.n, model.keep)
    return np.fft.fft(signal*np.hanning(model.n), axis=-1)[:, model.keep]/np.sqrt(model.n)


def candidate_starts(iq, variant, first, progress=None):
    if first['input_sha'] != array_hash(iq):
        raise RuntimeError('Prefix/input identity mismatch')
    tick = time.perf_counter()
    finite, hann = build(iq, 'finite_fourier'), build(iq, 'hann')
    p = np.asarray(first['position'], float)
    original = p.copy()
    first_detail = None
    if variant in ('finite_position', 'finite_both', 'finite_conditional'):
        p, first_detail = local(finite, p, CONFIG)
    points = grid(20.)
    if variant in ('terminal', 'finite_position'):
        scores = hann.scan(points, others=p)
    elif variant == 'finite_conditional':
        scores = finite.scan(points, others=p, progress=progress)
    elif variant in ('periodic_residual', 'finite_fit', 'finite_both'):
        fitting = build(iq, 'periodic') if variant == 'periodic_residual' else finite
        residual = fitting.fit(p)['residual']
        scores = hann.scan(points, data=taper_retained(residual, fitting))
    else:
        raise ValueError(variant)
    indices = peaks(points, scores, CONFIG['top'])
    if len(indices) != CONFIG['top'] or not np.isfinite(scores).all():
        raise ArithmeticError('Invalid candidate spectrum')
    starts = np.stack([np.vstack([p, points[i]]) for i in indices])
    details = dict(variant=variant, first_original=original.tolist(), first=p.tolist(),
                   first_detail=first_detail, starts=starts.tolist(), starts_sha=array_hash(starts),
                   grid_indices=indices.tolist(), scores_at_candidates=scores[indices].tolist(),
                   prefix_seconds=first['seconds'], discovery_seconds=time.perf_counter()-tick,
                   representation_preparation_seconds=finite.preparation_seconds,
                   data_sha=array_hash(finite.z))
    return finite, starts, scores, details


def finish(model, starts, details):
    tick = time.perf_counter()
    branches = []
    for rank, start in enumerate(starts, 1):
        p, detail = local(model, start, CONFIG)
        branches.append(dict(rank=rank, start=start.tolist(), positions=p.tolist(), **detail))
    best = min(branches, key=lambda b: (b['objective'], b['rank']))
    return dict(**details, positions=best['positions'], objective=best['objective'],
                selected_rank=best['rank'], stop_reason=best['stop_reason'],
                projected_gradient=best['projected_gradient'], branches=branches,
                refinement_seconds=time.perf_counter()-tick,
                max_normal_relative=model.max_normal_relative, counts=dict(model.counts))


def checks(iq, old_prefix):
    first = first_source(iq)
    np.testing.assert_allclose(first['position'], old_prefix['first'], atol=1e-8, rtol=0)
    periodic, hann, finite = [build(iq, k) for k in ('periodic', 'hann', 'finite_fourier')]
    np.testing.assert_allclose(taper_retained(periodic.z, periodic), hann.z, atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(periodic.z, finite.z)
    _, starts, _, _ = candidate_starts(iq, 'terminal', first)
    np.testing.assert_allclose(starts, old_prefix['starts'], atol=1e-8, rtol=0)
    p = np.asarray(first['position'])
    probes = np.array([[-340., 180.], [80., -220.], [640., 420.]])
    scores = finite.scan(probes, others=p)
    old = finite.fit(p)['value']
    direct = np.array([old-finite.fit(np.vstack([p, q]))['value'] for q in probes])
    np.testing.assert_allclose(scores, direct, atol=1e-10, rtol=1e-8)
    residual = finite.fit(p)['residual']
    np.testing.assert_allclose(taper_retained(residual, finite),
                               build(reconstruct(residual, finite.n, finite.keep), 'hann').z,
                               atol=1e-12, rtol=1e-12)
    bad = dict(first, input_sha='wrong')
    try:
        candidate_starts(iq, 'finite_fit', bad)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Missing input guard')
    return dict(passed=True, unwindowed_finite_data=True, hann_residual_transform=True,
                old_prefix_reproduced=True, exact_conditional_vs_joint=True, input_guard=True)


def fresh(source):
    source = source.resolve(strict=True)
    if not source.is_relative_to(BASE):
        raise ValueError('Source must be local P6 evidence')
    old = read(source/'contract.json')
    if old['revision'] != 'G11_P6_FIXED_HANN_DISCOVERY_FINITE_REFINEMENT_V1':
        raise ValueError('Unexpected source contract')
    for name, identity in old['sources'].items():
        if sha(ROOT/name) != identity:
            raise RuntimeError('Frozen dependency changed: '+name)
    out = BASE/('stage_diagnostic_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    for name in ('inputs', 'rows', 'spectra', 'oracle', 'first'):
        (out/name).mkdir()
    tasks = [t for t in old['tasks'] if t['role'] == 'cohort']
    assert len(tasks) == 8 and all(t['n'] == 1024 for t in tasks)
    inputs, saved = {}, {}
    for t in tasks:
        name = f"{t['group']}_{t['realization']}.npz"
        p = source/'inputs'/name
        if sha(p) != old['inputs'][name]:
            raise RuntimeError('Raw input hash mismatch')
        if not (out/'inputs'/name).exists():
            (out/'inputs'/name).write_bytes(p.read_bytes())
        inputs[name] = sha(out/'inputs'/name)
        for path in (source/'prefixes'/(key(t)+'_ap.json'),
                     source/'rows'/(key(t)+'_AP_finite_fourier.json')):
            read_row(path)
            saved[str(path)] = sha(path)
    sources = {str(p.relative_to(ROOT)): sha(p) for p in (ROOT/'统一模型代码/gates').rglob('*.py')}
    for name in sources:
        target = out/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    contract = dict(revision=REVISION, source=str(source), source_contract_sha=sha(source/'contract.json'),
        tasks=tasks, inputs=inputs, sources=sources, saved_sources=saved,
        variants=VARIANTS, full_cases=sorted(FULL_CASES), config=CONFIG,
        scope='8 existing development observations; full conditional on predeclared failure/control subset',
        truth='only post-selection metrics and separately labelled oracle diagnostics',
        timing='cold finite preparation + common first-source search charged + own discovery + all five refinements',
        statistics='descriptive only; selected failure diagnosis, not new generalization evidence', threads=2)
    write(out/'contract.json', contract)
    return out, contract


def verify(out, contract):
    for name, identity in contract['sources'].items():
        if sha(ROOT/name) != identity:
            raise RuntimeError('Run source changed: '+name)
    for name, identity in contract['inputs'].items():
        if sha(out/'inputs'/name) != identity:
            raise RuntimeError('Run input changed: '+name)
    for name, identity in contract['saved_sources'].items():
        if sha(name) != identity:
            raise RuntimeError('Historical evidence changed')


def oracle_diagnostic(iq, truth, first):
    """Truth-assisted localization ceiling diagnostic; NEVER a blind output."""
    model = build(iq, 'finite_fourier')
    p, detail = local(model, truth.copy(), CONFIG)
    blind = np.asarray(first['position'])
    refined, refdetail = local(model, blind, CONFIG)
    points = grid(20.)
    conditional = []
    for label, fixed in (('Hann_first', blind), ('finite_first', refined)):
        # Select the non-first truth only inside this diagnostic.
        nearest = int(np.argmin(np.linalg.norm(truth-fixed[0], axis=1)))
        other = truth[1-nearest]
        grid_point = points[np.argmin(np.linalg.norm(points-other, axis=1))]
        base = model.fit(fixed)['value']
        conditional.append(dict(label=label, fixed=fixed.tolist(), truth_other=other.tolist(),
            nearest_grid=grid_point.tolist(),
            truth_gain=base-model.fit(np.vstack([fixed, other]))['value'],
            nearest_grid_gain=base-model.fit(np.vstack([fixed, grid_point]))['value']))
    return dict(truth_assisted=True, positions=p.tolist(), **detail, metrics=evaluate(truth, p),
                finite_first_detail=refdetail, conditional=conditional)


def execute(out, contract, limit=None):
    verify(out, contract)
    source = Path(contract['source'])
    ledger_path = out/'ledger.json'
    ledger = read(ledger_path) if ledger_path.exists() else dict(attempts=[])
    attempt = dict(start_utc=datetime.now(timezone.utc).isoformat(), status='RUNNING')
    ledger['attempts'].append(attempt)
    tick, done = time.perf_counter(), 0
    def checkpoint():
        attempt['seconds_observed'] = time.perf_counter()-tick
        write(ledger_path, ledger)
    try:
        with threadpool_limits(limits=2):
            task = contract['tasks'][0]
            iq, _ = get_input(out, task)
            old_prefix = read_row(source/'prefixes'/(key(task)+'_ap.json'))
            if not (out/'checks.json').exists():
                result = checks(iq, old_prefix)
                save_row(out/'中文恢复核验.json', result)
                assert read_row(out/'中文恢复核验.json') == result
                write(out/'checks.json', result)
            jobs = [(t, v) for t in contract['tasks'] for v in VARIANTS]
            jobs += [(t, 'finite_conditional') for t in contract['tasks']
                     if (t['group'], t['gap']) in FULL_CASES]
            for task, variant in jobs:
                ident = key(task)
                path = out/'rows'/(ident+'_'+variant+'.json')
                iq, truth = get_input(out, task)
                if path.exists():
                    row = read_row(path)
                    assert row['input_sha'] == array_hash(iq)
                    assert row['metrics'] == evaluate(truth, row['result']['positions'])
                    continue
                first_path = out/'first'/(ident+'.json')
                if not first_path.exists():
                    first = first_source(iq)
                    old_prefix = read_row(source/'prefixes'/(ident+'_ap.json'))
                    np.testing.assert_allclose(first['position'], old_prefix['first'], atol=1e-8, rtol=0)
                    save_row(first_path, first)
                first = read_row(first_path)
                last = [time.perf_counter()]
                def progress(n, total):
                    now = time.perf_counter()
                    if now-last[0] >= 30:
                        print(f'\r有限条件扫描 {ident} {n}/{total} | 本次{(now-tick)/60:.1f} min', end='', flush=True)
                        checkpoint()
                        last[0] = now
                print(f'\n阶段诊断 {len(list((out/"rows").glob("*.json")))}/{len(jobs)} {ident} {variant}', flush=True)
                model, starts, scores, details = candidate_starts(iq, variant, first, progress)
                np.savez_compressed(out/'spectra'/(ident+'_'+variant+'.npz'), scores=scores)
                result = finish(model, starts, details)
                result['seconds'] = result['prefix_seconds']+result['discovery_seconds']+result['refinement_seconds']
                if variant == 'terminal':
                    historical = read_row(source/'rows'/(ident+'_AP_finite_fourier.json'))['result']
                    np.testing.assert_allclose(result['positions'], historical['positions'], atol=1e-7, rtol=0)
                    assert result['selected_rank'] == historical['selected_rank']
                save_row(path, dict(task=task, variant=variant, result=result, truth=truth.tolist(),
                                   input_sha=array_hash(iq), metrics=evaluate(truth, result['positions'])))
                checkpoint()
                done += 1
                if limit and done >= limit:
                    attempt['status'] = 'LIMITED_PREFIX_COMPLETE'
                    checkpoint()
                    return
            # Run truth-assisted diagnostics only after every blind job is sealed.
            for task in contract['tasks']:
                path = out/'oracle'/(key(task)+'.json')
                if path.exists():
                    read_row(path)
                    continue
                iq, truth = get_input(out, task)
                first = read_row(out/'first'/(key(task)+'.json'))
                save_row(path, oracle_diagnostic(iq, truth, first))
            verify(out, contract)
            attempt['status'] = 'COMPLETED'
            write(out/'audit.json', dict(completed=True, source_identity=True, input_identity=True,
                historical_unchanged=True, rows=len(jobs), oracle_rows=8, formal_release=False))
            checkpoint()
            print('\n有限诊断完成：', out, flush=True)
    except BaseException as exc:
        attempt['status'] = 'FAILED_OR_INTERRUPTED'
        checkpoint()
        write(out/f'failure_{len(ledger["attempts"])}.json', dict(error=repr(exc), traceback=traceback.format_exc()))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.resume:
        out = args.resume.resolve(strict=True)
        contract = read(out/'contract.json')
    elif args.source:
        out, contract = fresh(args.source)
    else:
        parser.error('Provide --source or --resume')
    print('输出目录：', out, flush=True)
    execute(out, contract, args.limit)


if __name__ == '__main__':
    main()
