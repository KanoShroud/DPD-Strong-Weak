"""G11-P5 preparation and bounded native integration pilot. No formal auto-run."""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from 统一模型代码.gates.g11.model_integration.models import build, FS, BAND
from 统一模型代码.gates.g11.model_integration.solvers import run, CONFIG
from 统一模型代码.gates.g11.model_integration.checks import (
    all_checks, frozen_periodic_checks, pipeline_smoke, physical_checks, KINDS,
)
from 统一模型代码.gates.g10.r1.scenes import make
from 统一模型代码.gates.g10.r2.band.scenes import geometry_valid
from 统一模型代码.gates.g10.r4.paper_checks.assignment_metrics import evaluate
from 统一模型代码.gates.g10.physics import grid

BASE = ROOT/'outputs_e2e/unified/e2e_g11/p5_model_integration'
REVISION = 'G11_P5_NATIVE_MODEL_INTEGRATION_V1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix('.pending')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    for i in range(4):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == 3:
                raise
            time.sleep(.1*(i+1))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def layouts():
    rng, rows = np.random.default_rng(20261024052), []
    for category in range(4):
        for _ in range(100000):
            p = rng.uniform(-950, 950, (2, 2))
            if geometry_valid(p, category):
                rows.append(p.tolist())
                break
        else:
            raise RuntimeError('Geometry generation failed')
    return rows


def fresh(mode, lengths):
    out = BASE/(mode+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    assert not out.exists() and out.resolve().is_relative_to(ROOT/'outputs_e2e')
    out.mkdir(parents=True)
    sources = {str(p.relative_to(ROOT)): sha(p) for p in (ROOT/'统一模型代码/gates').rglob('*.py')}
    for name in sources:
        target = out/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    contract = dict(revision=REVISION, mode=mode, lengths=lengths, sources=sources,
        layouts=layouts(), waveform_seed=20261024053, realization=28000,
        generator_n=8192, fs=FS, band=BAND, snr_db=6, renormalize_crop=False,
        primary_metric='untruncated squared-assignment RMSE; no success-based parameter selection',
        python=sys.version, platform=platform.platform(), numpy=np.__version__, threads=2,
        models=list(KINDS), algorithms=['AP5', 'VP'], config=CONFIG,
        status='PREPARATION_ONLY_NOT_FORMAL_RELEASE', truth='generation and offline metrics only',
        comparison='model-specific native candidate search; not P4 shared finite frontend',
        regularization='finite bases whitened on common extended sample support, ridge 1e-6; periodic/Hann native LS')
    write(out/'contract.json', contract)
    scene = make(0, 28000, gap=0., n=8192, fs=FS, snr=6,
                 positions=contract['layouts'][0], seed=contract['waveform_seed'])
    np.savez_compressed(out/'input.npz', signal=scene['signal'], noise=scene['noise'], positions=scene['positions'])
    write(out/'input_identity.json', dict(sha256=sha(out/'input.npz'), meta=scene['meta']))
    return out, contract


def check(out, contract):
    if any(sha(ROOT/name) != h for name, h in contract['sources'].items()):
        raise RuntimeError('Source identity changed; create a new run')
    if sha(out/'input.npz') != read(out/'input_identity.json')['sha256']:
        raise RuntimeError('Input identity changed')


def observation(out, n, gap=15):
    with np.load(out/'input.npz', allow_pickle=False) as z:
        iq = (z['signal'][0]*10**(gap/20)+z['signal'][1]+z['noise'])[:, :n]
        return iq, z['positions']


def cost(out, lengths):
    rows = []
    # Three public off-grid probes; never selected using truth or method scores.
    candidates = np.array([[-340., 180.], [80., -220.], [640., 420.]])
    others = np.array([[180., 240.]])
    for n in lengths:
        iq, _ = observation(out, n)
        for kind in KINDS:
            print(f'成本短测：N={n}，{kind}，准备及原生搜索评分', flush=True)
            model = build(iq, kind)
            times = {}
            for label, kwargs in [('single', {}), ('conditional', {'others': others})]:
                tick = time.perf_counter()
                scores = model.scan(candidates, **kwargs)
                times[label] = (time.perf_counter()-tick)/len(candidates)
                assert np.isfinite(scores).all()
            tick = time.perf_counter()
            fit = model.fit(np.vstack([others, candidates[0]]), gradient=True)
            times['joint_value_gradient'] = time.perf_counter()-tick
            tick = time.perf_counter()
            _, h = model.hessian(candidates[0], model.z)
            times['profile_hessian'] = time.perf_counter()-tick
            assert np.isfinite(h).all()
            row = dict(n=n, kind=kind, times=times, preparation_seconds=model.preparation_seconds,
                metadata=model.metadata, gradient=fit['gradient'].tolist(),
                AP_two_grid_extrapolated_seconds=10201*(times['single']+times['conditional']),
                VP_two_grid_extrapolated_seconds=2*10201*times['single'],
                interpretation='three-point component extrapolation, NOT complete-run prediction or theoretical lower bound')
            rows.append(row)
            write(out/'cost.json', rows)
            print(f'  两次全域评分外推：AP {row["AP_two_grid_extrapolated_seconds"]:.1f}s；VP {row["VP_two_grid_extrapolated_seconds"]:.1f}s（未含精修）', flush=True)
    return rows


def pilot(out, contract, algorithms, kinds):
    """True 20 m whole-domain native scans, one paired development observation."""
    (out/'cases').mkdir(exist_ok=True)
    points = grid(20.)
    for n in contract['lengths']:
        iq, truth = observation(out, n)
        for kind in kinds:
            for algorithm in algorithms:
                dest = out/'cases'/f'{n}_{kind}_{algorithm}.json'
                if dest.exists():
                    row = read(dest)
                    assert row['revision'] == REVISION and row['input_sha'] == sha(out/'input.npz')
                    continue
                print(f'原生全域短测：N={n} {kind} {algorithm}', flush=True)
                model = build(iq, kind)
                tick, last = time.perf_counter(), [0.]

                def progress(i, total):
                    now = time.perf_counter()
                    if now-last[0] >= 30 or i == total:
                        print(f'\r  网格评分 {i}/{total} | 本例{(now-tick)/60:.1f} min', end='', flush=True)
                        last[0] = now

                result = run(model, algorithm, points, progress=progress)
                row = dict(revision=REVISION, n=n, kind=kind, algorithm=algorithm,
                    input_sha=sha(out/'input.npz'), gap=15, scope='one development observation, not formal performance evidence',
                    result=result, truth=truth.tolist(), metrics=evaluate(truth, result['positions']))
                write(dest, row)
                print(f'\n  完成：{result["seconds"]:.2f}s，{result["stop_reason"]}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('tests', 'cost', 'smoke', 'physical', 'pilot'))
    parser.add_argument('--lengths', type=int, nargs='+', default=[1024, 4096, 8192])
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--kinds', nargs='+', choices=KINDS, default=list(KINDS))
    parser.add_argument('--algorithms', nargs='+', choices=('AP5', 'VP'), default=['AP5', 'VP'])
    args = parser.parse_args(argv)
    if args.resume:
        out = args.resume.resolve(strict=True)
        assert out.is_relative_to(BASE)
        contract = read(out/'contract.json')
        assert contract['mode'] == args.mode and contract['lengths'] == args.lengths
    else:
        out, contract = fresh(args.mode, args.lengths)
    print('结果目录：', out, flush=True)
    check(out, contract)
    tick = time.perf_counter()
    try:
        with threadpool_limits(limits=2):
            if args.mode == 'tests':
                write(out/'math_checks.json', all_checks())
                iq, _ = observation(out, 1024)
                write(out/'frozen_checks.json', frozen_periodic_checks(iq))
            elif args.mode == 'smoke':
                iq, _ = observation(out, 1024)
                write(out/'smoke.json', pipeline_smoke(iq))
            elif args.mode == 'physical':
                with np.load(out/'input.npz', allow_pickle=False) as z:
                    write(out/'physical.json', physical_checks(z['signal'], z['noise'], z['positions']))
            elif args.mode == 'cost':
                cost(out, args.lengths)
            else:
                pilot(out, contract, args.algorithms, args.kinds)
        check(out, contract)
        write(out/'audit.json', dict(completed=True, seconds=time.perf_counter()-tick,
            sources_unchanged=True, input_unchanged=True, formal_released=False))
        print('本阶段完成；这不是正式大批次放行。', flush=True)
    except BaseException as exc:
        write(out/'failure.json', dict(error=repr(exc), traceback=traceback.format_exc(),
            seconds=time.perf_counter()-tick))
        raise


if __name__ == '__main__':
    main()
