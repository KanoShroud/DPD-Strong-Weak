"""Short preparation and user-only formal benchmark. No formal run at import."""
import argparse
from datetime import datetime
import itertools
import sys
import time
import numpy as np
from .config import BASE, METHODS, DEFAULT, GAPS, CHECKPOINTS, SEED, candidates
from .runtime import preflight, read, write, sha, verify, manifest, Clock, Progress, save_scene, load_scene
from .scenes import make, observation, digest
from .methods import run, at_time
from .metrics import evaluate, aggregate


def fresh(prefix):
    return BASE/(prefix+'_'+datetime.now().strftime('%Y%m%d_%H%M%S'))


def record(out, tag, scene, kind, method, budget, cfg, clock, extra=None):
    path = out/'rows'/f'{tag}.json'
    if path.exists():
        row = read(path)
        if row['input_sha'] != digest(observation(scene, kind)[0]):
            raise RuntimeError('Resume input differs')
        return row
    iq, truth, count = observation(scene, kind)
    clock.check()
    clock.checkpoint(dict(task=tag, reserve=budget+2))
    result = run(iq, method, count, budget, cfg, SEED+scene['meta']['group'], clock.check)
    checkpoints = [t for t in CHECKPOINTS if t <= budget]
    if budget not in checkpoints:
        checkpoints.append(budget)
    row = dict(tag=tag, scene=scene['meta'], kind=kind, method=method, input_sha=digest(iq),
               truth=truth.tolist(), result=result,
               evaluations={str(t): evaluate(truth, at_time(result, t)) for t in checkpoints},
               extra=extra or {})
    write(path, row)
    clock.checkpoint()
    return row


def scene_for(out, role, group, realization=0, snr=3., gap=12., center='zero'):
    key = f'{role}_{group}_{realization}_{snr:g}_{gap:g}_{center}'
    if not (out/'inputs'/f'{key}.json').exists():
        save_scene(out, key, make(role, group, realization, snr, gap, center))
    return load_scene(out, key)


def audit(out, passed, **details):
    contract = read(out/'contract.json')
    verify(contract['sources'])
    for info_path in (out/'inputs').glob('*.json') if (out/'inputs').exists() else []:
        info = read(info_path)
        if sha(out/info['path']) != info['sha256']:
            raise RuntimeError('End-of-stage input identity mismatch: '+str(info_path))
    inputs = {p.name: sha(p) for p in (out/'inputs').glob('*')} if (out/'inputs').exists() else {}
    rows = {p.name: sha(p) for p in (out/'rows').glob('*.json')} if (out/'rows').exists() else {}
    write(out/'final_audit_report.json', dict(passed=bool(passed), sources_unchanged=True,
                                             inputs=inputs, rows=rows, **details))


def smoke(out):
    from .test_g10 import suite
    preflight(out, dict(stage='engineering_smoke', evidence='not_performance', budget=600))
    clock = Clock(out, 600)
    tests = suite()
    write(out/'algebra_tests.json', tests)
    if not tests['passed']:
        audit(out, False, reason='algebra')
        return False
    scene = make('smoke', 2, n=512)
    save_scene(out, 'smoke', scene)
    scene = load_scene(out, 'smoke')
    cfg = dict(DEFAULT, grid=200., sbl_grid=200., sbl_iterations=3, cycles=3, starts=4)
    rows = [record(out, method, scene, 'mixed', method, 8., cfg, clock) for method in METHODS]
    # Resume must return byte-identical existing evidence without re-running.
    before = sha(out/'rows/LS.json')
    record(out, 'LS', scene, 'mixed', 'LS', 8., cfg, clock)
    resume_ok = sha(out/'rows/LS.json') == before
    passed = tests['passed'] and resume_ok and all(r['result']['status'] != 'NUMERICAL_FAILURE'
                                                and r['result']['events'] for r in rows)
    from .report import generate
    generate(out, label='ENGINEERING SMOKE / NOT PERFORMANCE')
    clock.checkpoint()
    audit(out, passed, resume_ok=resume_ok, seconds=clock.consumed())
    print('短链路：'+str(passed), flush=True)
    return bool(passed)


def center_probe(out):
    from ..g9 import solver_v2
    from ..g9.probes.coverage_search import BatchedScore, dense
    preflight(out, dict(stage='paired_center_development', samples=8, budget=1800,
                       solvers='frozen G9 B and B+dense D; full time included'))
    clock = Clock(out, 1800)
    rows = []
    progress = Progress(8*2*2*2, '中心对照')
    for group, center, kind in itertools.product(range(8), ('random', 'zero'), ('weak', 'mixed')):
        scene = scene_for(out, 'center', group, center=center)
        iq, truth, count = observation(scene, kind)
        for method in ('B', 'D'):
            clock.check()
            start = time.perf_counter()
            # Same frozen B initialization; D adds full-domain replacement.
            initial = solver_v2.run(iq, 'B', 30., count, clock.check)
            pp = np.asarray(initial['events'][-1]['positions']) if initial['events'] else np.empty((0, 2))
            if method == 'D' and len(pp) == count:
                calc = BatchedScore(iq, clock.check, chunk=64)
                for k in range(count):
                    if count == 2:
                        pp[k] = dense(calc, pp[1-k], pp[k], 5., clock.check)['position']
                    # count=1: D matches B; dense() requires another source.
            row = dict(group=group, center=center, center_hz=scene['meta']['center_hz'],
                       kind=kind, method=method, predictions=pp.tolist(), metrics=evaluate(truth, pp),
                       seconds=time.perf_counter()-start, input_sha=digest(iq))
            rows.append(row)
            write(out/'rows'/f'{group}_{center}_{kind}_{method}.json', row)
            progress.update(len(rows))
            clock.checkpoint()
    groups = {f'{center}_{kind}_{method}': aggregate([r['metrics'] for r in rows
              if r['center'] == center and r['kind'] == kind and r['method'] == method])
              for center, kind, method in itertools.product(('random', 'zero'), ('weak', 'mixed'), ('B', 'D'))}
    write(out/'center_report.json', dict(groups=groups, note='Development paired data, not final performance',
        single_source_D='Same frozen B solution; D global replacement applies to two-source case'))
    from .report import center_figure
    center_figure(out, rows)
    audit(out, True, seconds=clock.consumed())


def prepare(out, smoke_dir, center_dir):
    for path in (smoke_dir, center_dir):
        if not read(path/'final_audit_report.json')['passed']:
            raise RuntimeError('Required short stage did not pass')
        sources = read(path/'contract.json')['sources']
        if path == center_dir:
            # The completed centre experiment calls frozen G9 solvers, G10 scenes,
            # and v2 metrics, not the new G10 solvers/report/launcher. Reuse only
            # after those actual numerical dependencies match their original hashes.
            sources = {k: v for k, v in sources.items() if '/g9/' in k.replace('\\', '/')
                       or '/g8/' in k.replace('\\', '/')
                       or k.replace('\\', '/').endswith(('g10/scenes.py', 'g10/metrics.py'))}
            for info_path in (path/'inputs').glob('*.json'):
                info = read(info_path)
                if sha(path/info['path']) != info['sha256']:
                    raise RuntimeError('Center input changed')
        verify(sources)
    preflight(out, dict(stage='development_calibration_selection_pilot', budget=2700,
                       smoke=str(smoke_dir), center=str(center_dir), seed=SEED,
                       snr_candidates=[3, 6, 9], qualification='at least 30/32 at 20m'))
    clock = Clock(out, 2700)
    selected_snr, qualifications = None, []
    print('阶段：独立单源校准（最多32例×3档）', flush=True)
    for snr in (3., 6., 9.):
        rows = []
        progress = Progress(32, f'单源 {snr:g}dB')
        for group, realization in itertools.product(range(16), range(2)):
            scene = scene_for(out, 'calibration', group, realization, snr=snr)
            row = record(out, f'cal_{snr}_{group}_{realization}', scene, 'weak', 'AP', 30., DEFAULT, clock)
            rows.append(row)
            progress.update(len(rows))
        count = sum(r['evaluations']['30.0']['success']['20'][0] for r in rows)
        qualifications.append(dict(snr=snr, successes=count, total=32))
        write(out/'qualification.json', qualifications)
        if count >= 30:
            selected_snr = snr
            break
    if selected_snr is None:
        write(out/'readiness.json', dict(ready=False, reason='ISOLATED_SOURCE_QUALIFICATION_FAILED'))
        audit(out, True, scientific_status='HOLD_QUALIFICATION')
        return
    print('阶段：有限开发选参，不读取正式布局', flush=True)
    chosen, tuning = {}, []
    for method in METHODS:
        options = []
        for ci, cfg in enumerate(candidates(method)):
            rows = []
            for group, gap in enumerate((0., 9., 15.)):
                scene = scene_for(out, 'tuning', group, snr=selected_snr, gap=gap)
                rows.append(record(out, f'tune_{method}_{ci}_{group}', scene, 'mixed', method, 15., cfg, clock))
            if any(r['result']['status'] == 'NUMERICAL_FAILURE' for r in rows):
                raise RuntimeError(f'{method} numerical failure: stop method qualification')
            ms = [r['evaluations']['15.0'] for r in rows]
            score = (-sum(all(m['success']['20']) for m in ms),
                     np.mean([m['errors_m'][-1] if m['errors_m'][-1] is not None else 1e9 for m in ms]),
                     sum(r['result']['seconds'] for r in rows), ci)
            options.append((score, cfg))
            tuning.append(dict(method=method, candidate=ci, score=list(score), config=cfg))
        chosen[method] = min(options, key=lambda x: x[0])[1]
        print('选参完成：'+method, flush=True)
    write(out/'tuning.json', tuning)
    print('阶段：最终配置先导计时（两条混合＋一条单源/方法）', flush=True)
    pilot = []
    for method in METHODS:
        for group, gap, kind in ((8, 0., 'mixed'), (9, 15., 'mixed'), (10, 0., 'weak')):
            scene = scene_for(out, 'tuning', group, snr=selected_snr, gap=gap)
            row = record(out, f'pilot_{method}_{group}', scene, kind, method,
                         60. if kind == 'mixed' else 30., chosen[method], clock)
            pilot.append(row)
            if row['result']['status'] == 'NUMERICAL_FAILURE' or not row['result']['events']:
                raise RuntimeError(f'{method} pilot did not produce a qualified result')
    # Max observed time, 25% reserve plus 30 minutes for generation/figures/audit.
    estimates = {}
    for method in METHODS:
        mixed = [r['result']['seconds'] for r in pilot if r['method'] == method and r['kind'] == 'mixed']
        single = [r['result']['seconds'] for r in pilot if r['method'] == method and r['kind'] == 'weak']
        estimates[method] = 72*max(mixed)+84*max(single)
    predicted = 1.25*sum(estimates.values())+1800
    config = dict(methods=chosen, snr=selected_snr, layouts=list(range(12)), seed=SEED,
                  gaps=list(GAPS), N=16384, center='zero', mixed_budget=60., single_budget=30.,
                  total_budget=43200., protocol='EVAL-20261011-v2', times=list(CHECKPOINTS))
    write(out/'frozen_config.json', config)
    write(out/'budget_estimate.json', dict(seconds=predicted, hours=predicted/3600,
                                           per_method_seconds=estimates, factor=1.25, overhead_seconds=1800))
    from .report import generate
    generate(out, label='DEVELOPMENT / NOT FORMAL', rows=pilot)
    clock.checkpoint()
    audit(out, True, seconds=clock.consumed())
    ready = predicted <= 43200
    write(out/'readiness.json', dict(ready=ready, reason='READY' if ready else 'HOLD_BUDGET',
        config_sha=sha(out/'frozen_config.json'), sources=manifest(), estimated_seconds=predicted,
        smoke=str(smoke_dir), center=str(center_dir)))
    if ready:
        write(BASE/'release.json', dict(preparation=str(out), readiness_sha=sha(out/'readiness.json')))
    print(f'准备完成：ready={ready}，保守估算{predicted/3600:.2f}小时', flush=True)


def warmup(out, clock):
    """Per-process common kernel warmup; separate from per-case solve time."""
    import torch
    from .physics import Model, Spectrum, sbl_posterior
    clock.checkpoint(dict(task='common_warmup', reserve=10.))
    start = time.perf_counter()
    rng = np.random.default_rng(2401)
    m = Model(rng.normal(size=(4, 128))+1j*rng.normal(size=(4, 128)), clock.check)
    pp = np.array([[-311., 227.], [531., -179.]])
    m.conditional(pp)
    m.joint(pp, True)
    m.joint_batch(np.array([pp, pp+1]))
    for name in ('MUSIC', 'MVDR'):
        Spectrum(m, name, 2).evaluate(pp)
    sbl_posterior(m.response(pp), m.x,
                  torch.ones(2, dtype=torch.float64, device='cuda'), 1/128)
    torch.cuda.synchronize()
    seconds = time.perf_counter()-start
    path = out/'warmups.json'
    records = read(path) if path.exists() else []
    records.append(dict(seconds=seconds, synthetic=True, excluded_from_per_case=True))
    write(path, records)
    clock.checkpoint()
    return seconds


def formal(out, preparation, resume=False):
    readiness = read(preparation/'readiness.json')
    if not readiness['ready'] or sha(preparation/'frozen_config.json') != readiness['config_sha']:
        raise RuntimeError('Preparation not ready or frozen config changed')
    verify(readiness['sources'])
    cfg = read(preparation/'frozen_config.json')
    if not resume:
        preflight(out, dict(stage='formal', config=cfg, preparation=str(preparation)))
    else:
        contract = read(out/'contract.json')
        if contract['config'] != cfg:
            raise RuntimeError('Resume contract differs')
        verify(contract['sources'])
    clock = Clock(out, cfg['total_budget'])
    # Weak-only once per layout. Paired strong-only at each gap.
    tasks = []
    for group in cfg['layouts']:
        tasks.append((group, 0, 'weak'))
        for gap in cfg['gaps']:
            tasks.extend([(group, gap, 'mixed'), (group, gap, 'strong')])
    progress = Progress(len(tasks)*len(METHODS), 'G10 正式比较')
    done = 0
    try:
        print(f'公共预热：{warmup(out, clock):.2f}秒（单列，计入总任务预算）', flush=True)
        for index, (group, gap, kind) in enumerate(tasks):
            scene = scene_for(out, 'formal', group, snr=cfg['snr'], gap=gap)
            # Balanced rotation controls order/thermal/cache effects.
            order = METHODS[index % len(METHODS):]+METHODS[:index % len(METHODS)]
            for method in order:
                budget = cfg['mixed_budget'] if kind == 'mixed' else cfg['single_budget']
                row = record(out, f'{group}_{gap}_{kind}_{method}', scene, kind, method,
                             budget, cfg['methods'][method], clock)
                done += 1
                progress.update(done, f'{method} {kind} gap={gap}dB')
                if row['result']['status'] == 'NUMERICAL_FAILURE':
                    raise RuntimeError('Formal numerical failure retained; stop for diagnosis')
        print('阶段：统计、图表与审计', flush=True)
        clock.checkpoint(dict(task='maps_statistics_audit', reserve=1800.))
        from .report import generate, spatial_maps
        spatial_maps(out, cfg, clock)
        generate(out, label='FORMAL', guard=clock.check)
        clock.checkpoint()
        audit(out, True, completed=len(tasks)*len(METHODS), seconds=clock.consumed())
        write(out/'status.json', dict(status='COMPLETED', seconds=clock.consumed()))
    except BaseException as exc:
        pending = clock.state['pending']
        clock.checkpoint()
        write(out/'status.json', dict(status='STOPPED', error=str(exc), pending=pending,
                                     seconds=clock.consumed()))
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('smoke', 'center', 'prepare', 'formal', 'replot'))
    p.add_argument('--out', type=str)
    p.add_argument('--smoke', type=str)
    p.add_argument('--center', type=str)
    p.add_argument('--preparation', type=str)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    from pathlib import Path
    out = Path(args.out) if args.out else fresh(args.stage)
    print('输出：'+str(out), flush=True)
    if args.stage == 'smoke':
        if not smoke(out):
            sys.exit(2)
    elif args.stage == 'center':
        center_probe(out)
    elif args.stage == 'prepare':
        prepare(out, Path(args.smoke), Path(args.center))
    elif args.stage == 'formal':
        formal(out, Path(args.preparation), args.resume)
    else:
        from .report import generate
        generate(out)


if __name__ == '__main__':
    main()
