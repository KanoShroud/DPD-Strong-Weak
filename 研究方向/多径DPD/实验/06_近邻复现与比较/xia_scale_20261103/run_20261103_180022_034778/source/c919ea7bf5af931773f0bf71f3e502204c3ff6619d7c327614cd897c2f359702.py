"""One frozen public-resolution Xia adaptation; no truth in estimator."""
import argparse
from datetime import datetime
import hashlib
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits
from kernel import CrossSpectrumKernel, C

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE/'complete_neighbor_20261103'))
import native_xia as old  # noqa: E402
from storage import read, write, load, save, lock, stage, recover, cost  # noqa: E402

FS = 50e6
BOUND = 3000.


def design():
    radius = C/(4*old.B)
    probability = np.pi*radius**2/(4*(2*BOUND)**2)
    count = math.ceil(math.log(.05/2)/math.log1p(-probability))
    return dict(bandwidth_hz=old.B, c=C, bound_m=BOUND, radius_m=radius,
                delta=.05, phase_bound=np.pi/2, minimum_ball_probability=probability,
                particles=count, miss_bound=2*np.exp(count*np.log1p(-probability)),
                delta_h_m=400., std_m=10., rounds=3)


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def solve(y, f, n, seed, config, dst=None):
    start = time.perf_counter()
    kernel = CrossSpectrumKernel(y, f, n, FS)
    rng = np.random.default_rng(seed)
    rng_initial = rng.bit_generator.state
    count = config['particles']
    points = rng.uniform(-BOUND, BOUND, (count, 2))
    steps = []
    validation_seconds = 0.
    for iteration in range(3):
        values = kernel.scores(points, old.STATIONS)
        assert np.isfinite(values).all() and (values >= 0).all()
        t = time.perf_counter()
        indices = np.unique(np.r_[np.linspace(0, count-1, min(count, 64), dtype=int),
                                  np.argpartition(values, -min(count, 32))[-min(count, 32):]])
        direct = old.xia.scores(points[indices], old.STATIONS, y, f, n)
        absolute = float(np.max(abs(direct-values[indices]))/max(values.max(), np.finfo(float).tiny))
        assert absolute < 5e-11, absolute
        validation_seconds += time.perf_counter()-t
        answer = old.xia.decode(points, values, 2, config['delta_h_m'])
        mask = old.xia.threshold(values)['keep']
        retained = points[mask]
        # Storage reduction only: every generated candidate is scored/decoded.
        if dst is not None:
            np.savez_compressed(dst/f'round_{iteration+1}_retained.npz',
                                points=retained, scores=values[mask])
        row = dict(round=iteration+1, output=answer, points_sha=digest(points),
                   scores_sha=digest(values), particle_count=count,
                   rng_before_generation=rng_initial, numerical_direct_error=absolute,
                   saved_retained=len(retained), all_candidates_scored=True,
                   outside_domain_count=int((abs(points)>BOUND).any(axis=1).sum()))
        steps.append(row)
        if dst is not None:
            save(dst/f'round_{iteration+1}.json', row)
        print('  round', iteration+1, 'particles', count, answer['status'],
              'retained', len(retained), flush=True)
        if answer['positions'] is None:
            break
        if iteration < 2:
            rng_initial = rng.bit_generator.state
            owner = np.arange(count) % 2
            points = np.asarray(answer['positions'])[owner]+rng.normal(size=(count, 2))*config['std_m']
    full = [s['output']['positions'] for s in steps if s['output'].get('positions') is not None]
    return dict(positions=steps[-1]['output']['positions'], retained=full[-1] if full else None,
                steps=steps, seed=seed, config=config, kernel=kernel.metadata(),
                seconds=time.perf_counter()-start, validation_seconds=validation_seconds,
                truth_used=False, all_pools_reconstructible_from_rng=True)


def checks():
    d = design()
    assert d['miss_bound'] <= .05
    assert 2*np.exp((d['particles']-1)*np.log1p(-d['minimum_ball_probability'])) > .05
    assert np.isclose(2*np.pi*old.B*d['radius_m']/C, np.pi/2)
    case = old.cal.cases()[0]
    with np.load(old.FORMAL/'cases'/case['id']/'input.npz', allow_pickle=False) as z:
        f, y, n, _ = old.cal.spectral(z['iq'])
    seed = 20261029223+10000*20
    config = {**d, 'particles':1000}
    fast = solve(y.T, f, n, seed, config)
    direct = old.xia.solve(y.T, f, old.STATIONS, 2, n, BOUND, seed=seed,
                           particles=1000, delta_h=400., std=10.)
    assert fast['positions'] == direct['positions']
    assert len(fast['steps']) == len(direct['steps'])
    for a, b in zip(fast['steps'], direct['steps']):
        assert a['points_sha'] == digest(np.asarray(b['points']))
        for key in ('status', 'positions', 'retained', 'modal_bin', 'histogram', 'trace'):
            assert a['output'].get(key) == b['output'].get(key)
    return dict(passed=True, design=d, native_1000_flow_exact=True, truth_loaded=False)


def fingerprints():
    files = [HERE/'runner.py', HERE/'kernel.py', HERE/'执行协议.md', old.FORMAL/'contract.json',
             Path(old.__file__), Path(old.xia.__file__)]
    files += sorted((old.FORMAL/'cases').glob('*/input.npz'))
    files += sorted((old.FORMAL/'cases').glob('*/input.json'))
    files += sorted((old.FORMAL/'cases').glob('*/solve.json'))
    for module in tuple(sys.modules.values()):
        p = getattr(module, '__file__', None)
        if p and Path(p).suffix == '.py' and Path(p).resolve().is_relative_to(BASE):
            files.append(Path(p))
    return {str(p.resolve(strict=True)):old.sha(p) for p in files}


def summaries(out, rows):
    metrics = {}
    for cond in old.cal.CONDITIONS:
        group = [r for r in rows if r['case']['condition'] == cond]
        if not group:
            continue
        metrics[cond] = {}
        for name in ('Xia_scale_terminal', 'Xia_scale_lastcomplete', 'GEO', 'LOS', 'WIDE'):
            ee = [r['evaluation'][name] for r in group]
            complete = [e for e in ee if e['complete']]
            rmse = float(np.sqrt(np.mean([np.asarray(e['errors_m'])**2 for e in complete]))) if complete else None
            metrics[cond][name] = dict(n=len(group), complete=len(complete),
                success20=sum(max(e['errors_m']) <= 20 for e in complete),
                conditional_rmse_m=rmse, full_rmse_m=rmse if len(complete)==len(group) else None)
    write(out/'summary.json', dict(completed=len(rows), metrics=metrics, cost=cost(out),
                                   new_independent_data=False, design=design()))


def run_case(out, case, contract):
    folder = old.FORMAL/'cases'/case['id']
    dst = out/case['id']
    dst.mkdir(exist_ok=True)
    meta = load(folder/'input.json')
    with np.load(folder/'input.npz', allow_pickle=False) as z:
        assert set(z.files) == {'iq'}
        iq = z['iq']
    assert digest(iq) == meta['iq_sha']
    f, y, n, _ = old.cal.spectral(iq)
    seed = 20261029223+10000*20+case['repeat']
    result, _ = stage(out, case['id'], dst/'decision_before_truth.json',
        lambda: solve(y.T, f, n, seed, design(), dst))
    assert digest(iq) == meta['iq_sha']
    if (dst/'evaluation.json').exists():
        return load(dst/'evaluation.json')
    # Evaluation only, after final estimator decision saved.
    with np.load(contract['conditions'][case['condition']]['bank'], allow_pickle=False) as z:
        truth = z['p']
    old_result = load(folder/'solve.json')
    pred = dict(Xia_scale_terminal=result['positions'], Xia_scale_lastcomplete=result['retained'],
                GEO=old_result['initial'], LOS=old_result['los']['p'], WIDE=old_result['final']['p'])
    evaluation = {k:old.evaluate(np.asarray(p).reshape(-1,2) if p is not None else np.empty((0,2)), truth)
                  for k,p in pred.items()}
    # Rebuild first pool in blocks for offline coverage; no oracle feedback to solver.
    rng = np.random.default_rng(seed)
    nearest = np.full(2, np.inf)
    for start in range(0, design()['particles'], 65536):
        pp = rng.uniform(-BOUND, BOUND, (min(65536, design()['particles']-start), 2))
        nearest = np.minimum(nearest, np.linalg.norm(pp[:,None]-truth[None],axis=-1).min(axis=0))
    row = dict(case=case, evaluation=evaluation, initial_nearest_m=nearest,
               initial_coverage_radius_m=design()['radius_m'], seconds=result['seconds'],
               validation_seconds=result['validation_seconds'], iq_sha=meta['iq_sha'])
    save(dst/'evaluation.json', row)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--continue-long', action='store_true', help='User-run handoff beyond one hour only')
    args = parser.parse_args()
    test = checks()
    if args.check:
        write(HERE/'flow_check.json', test)
        print(test, flush=True)
        return
    kernel_tests = sorted(HERE.glob('test_*.json'))
    assert kernel_tests, 'Saved numeric kernel tests required'
    kt = read(kernel_tests[-1])
    assert kt['passed'] and kt['hashes_unchanged']
    assert all(old.sha(Path(p))==h for p,h in kt['hashes'].items())
    contract = load(old.FORMAL/'contract.json')['contract']
    old.cal.verify(contract)
    identity = fingerprints()
    pointer = HERE/'latest.json'
    if pointer.exists():
        out = Path(read(pointer)['path']).resolve(strict=True)
        assert read(out/'preflight.json')['hashes'] == identity, 'Frozen identity changed; no automatic resume'
    else:
        # Worst-case uncompressed retained arrays, plus modest metadata headroom.
        required = design()['particles']*24*3*32+1024**3
        if shutil.disk_usage(HERE).free < required:
            raise RuntimeError('Free space below conservative 32-case evidence budget; no run started')
        out = HERE/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        out.mkdir(exist_ok=False)
        write(pointer, dict(path=str(out)))
        write(out/'preflight.json', dict(hashes=identity, checks=test, design=design(), kernel_tests=str(kernel_tests[-1])))
        (out/'source').mkdir()
        for p,h in identity.items():
            if Path(p).suffix in ('.py','.md'):
                (out/'source'/(h+Path(p).suffix)).write_bytes(Path(p).read_bytes())
    print('结果目录',out,'粒子数',design()['particles'],flush=True)
    rows = []
    with lock(out):
        recover(out)
        for i, case in enumerate(old.cal.cases()):
            rows.append(run_case(out, case, contract))
            summaries(out, rows)
            print('完成', len(rows), '/32', case['id'], flush=True)
            if i==3:
                los = next(r for r in rows if r['case']['condition']=='L0')
                ee = los['evaluation']['Xia_scale_terminal']
                basic = ee['complete'] and max(ee['errors_m']) <= 20
                estimate = sum(r['seconds'] for r in rows)*8
                gate = dict(basic_los_pass=basic, projected_batch_seconds=estimate,
                            full_run_allowed=basic and (estimate<3600 or args.continue_long),
                            gate_is_development_diagnostic=True)
                write(out/'pilot_gate.json', gate)
                if not gate['full_run_allowed']:
                    print('停止在短测门禁',gate,flush=True)
                    break
        assert all(old.sha(Path(p)) == h for p,h in identity.items())
        old.cal.verify(contract)
        write(out/'postflight.json', dict(hashes_unchanged=True, completed=len(rows), all_cases=len(rows)==32))
    print('结束',len(rows),'/32',flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
