"""Conventional-window falsification probe; no novelty or exact ML claim.

Three fixed windows share frozen AP primitives. Input-only solvers never receive
positions or clean components. Round 1 uses registered G11 cases; round 2 uses
fresh realizations and nested record lengths with unchanged waveform/noise power.
"""
import argparse
import copy
import hashlib
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.signal.windows import hann, tukey

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.g11.precision_diagnostic import P0, load_case, read, write, resolve, sha
from 统一模型代码.gates.g11.window_validation import locate as original_locate
from 统一模型代码.gates.g10.r1.scenes import make
from 统一模型代码.gates.g10.r3.solvers import Model, Trace, projected_gradient
from 统一模型代码.gates.g10.r4.solvers import ap
from 统一模型代码.gates.g10.r2.solvers import local
from 统一模型代码.gates.g10.physics import grid
from 统一模型代码.gates.g10.methods import peaks
from 统一模型代码.gates.g8.physics import C, RECEIVERS
from 统一模型代码.gates.g10.r4.paper_checks.assignment_metrics import evaluate, self_test

BASE = ROOT / 'outputs_e2e/research_discovery_20261022/experiments/A_window_baselines'
VARIANTS = ('rect', 'hann', 'tukey_geometry')
LAYOUTS = (13, 14, 15, 0, 4, 8, 12)


def weight(name, n, fs):
    baseline = np.linalg.norm(RECEIVERS[:, None] - RECEIVERS[None], axis=-1).max()
    # Fixed engineering comparator: each ramp = twice maximum station delay.
    alpha = min(1., 4 * baseline * fs / (C * (n - 1)))
    if name == 'rect':
        return np.ones(n)
    if name == 'hann':
        return hann(n, sym=False)
    if name == 'tukey_geometry':
        return tukey(n, alpha=alpha, sym=False)
    raise ValueError(name)


def locate(iq, k, cfg, name):
    """Same native primitives; permits explicitly registered shorter records."""
    torch.cuda.synchronize()
    tick = time.perf_counter()
    w = weight(name, iq.shape[1], cfg['sample_rate_hz'])
    processed = np.asarray(iq, np.complex128) * w
    trace = Trace(lambda: None, history=False)
    model = Model(processed, trace.check, sample_rate_hz=cfg['sample_rate_hz'],
                  half_band_hz=cfg['half_band_hz'])
    trace.model = model
    if k == 2:
        detail = ap(model, trace, 'AP_1_JOINT', copy.deepcopy(cfg))
        p = np.asarray(detail['output_positions'])
    elif k == 1:
        points = grid(cfg['grid'])
        first = points[peaks(points, model.conditional(points), cfg['top'])[0]]
        p, _, info = local(model, first[None], cfg)
        detail = dict(first_optimizer=info)
    else:
        raise ValueError(k)
    value, grad = model.joint(p, True)
    torch.cuda.synchronize()
    assert np.isfinite(p).all() and np.isfinite(value) and np.isfinite(grad).all()
    return dict(positions=p.tolist(), objective=float(value),
                projected_gradient=projected_gradient(p, grad),
                seconds=time.perf_counter()-tick, counts=model.counts,
                detail=detail, samples=len(w), retained_bins=model.n,
                mean_squared_window=float(np.mean(w*w)),
                criterion='approximate_windowed_profile_LS')


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def summary(rows):
    result = []
    for condition in sorted({(r['kind'], r['n'], r['gap'], r['group']) for r in rows}):
        kind, n, gap, group = condition
        part = [r for r in rows if (r['kind'], r['n'], r['gap'], r['group']) == condition]
        for name in VARIANTS:
            e = np.asarray([r['results'][name]['metrics']['errors_m'] for r in part])
            result.append(dict(kind=kind, n=n, gap=gap, group=group, window=name,
                               cases=len(part), weak_rmse=float(np.sqrt(np.mean(e[:, -1]**2))),
                               weak_p95=float(np.quantile(e[:, -1], .95)),
                               both20=int(np.all(e <= 20, axis=1).sum()),
                               mean_seconds=float(np.mean([r['results'][name]['seconds'] for r in part]))))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', type=int, choices=[1, 2], required=True)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    expected = read(P0 / 'source_manifest.json')
    pointer = ROOT/'outputs_e2e/unified/e2e_g10/r4/review_supplement_release.json'
    folder = resolve(read(pointer, expected[str(pointer)])['out'])
    initial = load_case(folder, expected, (13, 1000, 0))
    cfg = initial[-1]['config']
    tag = f'round{args.round}_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out = BASE / tag
    assert not out.exists() and out.resolve().is_relative_to(ROOT/'outputs_e2e')
    protected = {str(p): sha(p) for p in (ROOT/'统一模型代码/gates').rglob('*.py')}
    out.mkdir(parents=True)
    (out/'pairs').mkdir()
    (out/'inputs').mkdir()
    write(out/'protocol.json', dict(round=args.round, variants=VARIANTS, config=cfg,
          purpose='falsify simple new-window claim; these are conventional comparators',
          geometry_taper='each ramp=2*max station distance/c; clipped at full Hann ramp',
          windows_chosen_without_outcomes=True, exact_ML_claim=False,
          old_layouts=LAYOUTS, old_realization=1000,
          new_realizations=[22000, 22001, 22002], new_seed=2026102242,
          new_lengths=[1024, 8192], frozen_solver_hashes=protected, limit=args.limit,
          source_manifest_sha=sha(P0/'source_manifest.json'), python=sys.version,
          nature='development baseline checks, not a qualified candidate',
          timestamp=datetime.now().isoformat()))
    jobs = []
    if args.round == 1:
        for layout in LAYOUTS:
            for gap in (0, 15):
                case, sig, noise, truth, saved = load_case(folder, expected, (layout, 1000, gap))
                group = 'difficult' if layout in (13, 14, 15) else 'control'
                jobs.append((case, 'mixed', 8192, gap, group, sig.sum(0)+noise, truth, saved))
                if gap == 0:
                    jobs.append((case, 'weak', 8192, 0, group, sig[1]+noise, truth[1:2], None))
    else:
        for group in range(4):
            for rep in (22000, 22001, 22002):
                data = make(group, rep, 0, n=8192, fs=50e6, seed=2026102242)
                path = out/'inputs'/f'{group}_{rep}.npz'
                np.savez_compressed(path, **{k:data[k] for k in ('signal', 'noise', 'positions')})
                write(path.with_suffix('.json'), dict(**data['meta'], file_sha256=sha(path)))
                for n in (1024, 8192):
                    sig, noise, truth = data['signal'][..., :n], data['noise'][:, :n], data['positions']
                    for gap in (0, 15):
                        iq = sig[0]*10**(gap/20)+sig[1]+noise
                        jobs.append((f'new_{group}_{rep}', 'mixed', n, gap, 'fresh_realizations', iq, truth, None))
                    jobs.append((f'new_{group}_{rep}', 'weak', n, 0, 'fresh_realizations', sig[1]+noise, truth[1:2], None))
    if args.limit:
        jobs = jobs[:args.limit]
    checks = dict(metrics=self_test())
    for n in (1024, 8192):
        for name in VARIANTS:
            w = weight(name, n, 50e6)
            assert w.shape == (n,) and np.min(w) >= 0 and np.max(w) <= 1
            if name == 'tukey_geometry':
                assert w[n//2] == 1
    # Audit wrapper parity against the immutable original for both known windows.
    test_iq = initial[1].sum(0)+initial[2]
    for name in ('rect', 'hann'):
        reference = original_locate(test_iq, 2, cfg, name)
        current = locate(test_iq, 2, cfg, name)
        np.testing.assert_allclose(reference['positions'], current['positions'], atol=1e-8, rtol=0)
        np.testing.assert_allclose(reference['native_objective'], current['objective'], atol=1e-13)
    checks['frozen_8192_wrapper_parity'] = True
    write(out/'checks.json', checks)
    print(f'轮次{args.round}：预检通过，{len(jobs)}条观测×3种常规窗。', flush=True)
    rows = []
    tick = time.perf_counter()
    for index, (case, kind, n, gap, group, iq, truth, saved) in enumerate(jobs):
        order = VARIANTS[index % 3:] + VARIANTS[:index % 3]
        results = {}
        for name in order:
            result = locate(iq, len(truth), cfg, name)
            result['metrics'] = evaluate(truth, result['positions'])
            assert result['metrics']['valid_output']
            results[name] = result
        if saved is not None:
            np.testing.assert_allclose(saved['positions'], results['rect']['positions'], atol=1e-7, rtol=0)
        row = dict(case=case, kind=kind, n=n, gap=gap, group=group, iq_sha256=digest(iq),
                   truth=truth.tolist(), order=order, results=results)
        write(out/'pairs'/f'{index:04d}.json', row)
        rows.append(row)
        if (index+1) % 12 == 0 or index+1 == len(jobs):
            print(f'{index+1}/{len(jobs)} 条完成，累计 {time.perf_counter()-tick:.1f}s', flush=True)
    write(out/'summary.json', summary(rows))
    for path, identity in protected.items():
        assert sha(path) == identity, path
    write(out/'audit.json', dict(completed=True, observations=len(rows), solves=3*len(rows),
          protected_sources=len(protected), unchanged=True, wall_seconds=time.perf_counter()-tick,
          no_truth_in_locator=True, parity_checked=True))
    print(out, flush=True)


if __name__ == '__main__':
    main()
