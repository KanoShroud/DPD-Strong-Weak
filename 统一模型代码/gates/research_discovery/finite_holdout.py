"""Frozen conventional finite-record baseline on new geometries/realizations."""
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.research_discovery.finite_fft_blind import solve
from 统一模型代码.gates.research_discovery.window_baselines import weight, locate, VARIANTS
from 统一模型代码.gates.research_discovery.finite_extension_probe import SOURCE
from 统一模型代码.gates.g10.r1.scenes import make
from 统一模型代码.gates.g8.physics import RECEIVERS
from 统一模型代码.gates.g10.r3.solvers import Model, Trace
from 统一模型代码.gates.g10.r4.solvers import ap
from 统一模型代码.gates.g11.precision_diagnostic import sha, write
from 统一模型代码.gates.g10.r4.paper_checks.assignment_metrics import evaluate


def layouts():
    rng = np.random.default_rng(20261022014)
    answer = []
    for inside in ((True, True), (False, True), (True, False), (False, False)):
        for _ in range(100000):
            p = rng.uniform(-900, 900, (2, 2))
            if tuple(np.abs(p).sum(1) < 500) != inside:
                continue
            if np.linalg.norm(p[:, None]-RECEIVERS, axis=-1).min() < 150 or np.linalg.norm(p, axis=1).min() < 100:
                continue
            if np.linalg.norm(p[0]-p[1]) < 300:
                continue
            distances = []
            for i in range(4):
                for j in range(i+1, 4):
                    v = RECEIVERS[j]-RECEIVERS[i]
                    w = p-RECEIVERS[i]
                    distances.extend(abs(v[0]*w[:, 1]-v[1]*w[:, 0])/np.linalg.norm(v))
            if min(distances) < 50:
                continue
            answer.append(p)
            break
        else:
            raise RuntimeError('Geometry sampling exhausted')
    return answer


def baseline(iq, cfg, name):
    tick = time.perf_counter()
    trace = Trace(lambda: None, history=False)
    m = Model(iq*weight(name, 1024, 50e6), trace.check, sample_rate_hz=50e6, half_band_hz=7e6)
    trace.model = m
    detail = ap(m, trace, 'AP_5_JOINT', copy.deepcopy(cfg))
    return dict(positions=detail['output_positions'], detail=detail, seconds=time.perf_counter()-tick)


def main():
    out = SOURCE.parent/('round14_holdout_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    assert not out.exists() and out.resolve().is_relative_to(ROOT/'outputs_e2e')
    cfg = json.loads((SOURCE/'protocol.json').read_text(encoding='utf-8'))['config']
    protected = {str(p): sha(p) for p in (ROOT/'统一模型代码/gates').rglob('*.py')}
    pp = layouts()
    out.mkdir()
    (out/'inputs').mkdir()
    write(out/'protocol.json', dict(layouts=[p.tolist() for p in pp], geometry_seed=20261022014,
        waveform_seed=2026102244, repeats=[24000, 24001], gaps=[0, 15],
        generator_n=8192, observation_n=1024, no_rescaling_after_crop=True,
        finite_solver=dict(damp=1e-3, maxiter=40, inner_atol=1e-9, inner_maxiter=1000),
        reference='AP5 three windows, same data/initial grid', config=cfg, protected=protected,
        interpretation='new geometry and noise/waveform holdout for conventional baseline; no new algorithm claim'))
    rows, isolated = [], []
    with threadpool_limits(limits=2):
        for group, pos in enumerate(pp):
            for rep in (24000, 24001):
                data = make(group, rep, 0, n=8192, fs=50e6, positions=pos, seed=2026102244)
                path = out/'inputs'/f'{group}_{rep}.npz'
                np.savez_compressed(path, **{k: data[k] for k in ('signal', 'noise', 'positions')})
                write(path.with_suffix('.json'), dict(meta=data['meta'], sha=sha(path)))
                sig, noise = data['signal'][..., :1024], data['noise'][:, :1024]
                alone = {}
                for name in VARIANTS:
                    r = locate(sig[1]+noise, 1, cfg, name)
                    r['metrics'] = evaluate(pos[1:], r['positions'])
                    alone[name] = r
                isolated.append(dict(group=group, realization=rep, results=alone))
                write(out/'isolated.json', isolated)
                for gap in (0, 15):
                    iq = sig[0]*10**(gap/20)+sig[1]+noise
                    results = {name: baseline(iq, cfg, name) for name in VARIANTS}
                    first = results['rect']['detail']['first_position']
                    results['finite_ridge'] = solve(iq, first, damp=1e-3)
                    for r in results.values():
                        r['metrics'] = evaluate(pos, r['positions'])
                    rows.append(dict(group=group, realization=rep, gap=gap, results=results))
                    write(out/'results.json', rows)
                    print(f'{len(rows)}/16 布局{group} 实现{rep} 功率差{gap} 完成', flush=True)
    summary = []
    for gap in (0, 15):
        for name in (*VARIANTS, 'finite_ridge'):
            subset = [r['results'][name] for r in rows if r['gap'] == gap]
            e = np.asarray([r['metrics']['errors_m'] for r in subset])
            summary.append(dict(gap=gap, method=name, n=len(subset), both20=int(np.all(e <= 20, axis=1).sum()),
                weak_rmse=float(np.sqrt(np.mean(e[:, 1]**2))), mean_reported_seconds=float(np.mean([r['seconds'] for r in subset]))))
    write(out/'summary.json', summary)
    for path, identity in protected.items():
        assert sha(path) == identity, path
    for path in (out/'inputs').glob('*.npz'):
        assert sha(path) == json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))['sha']
    write(out/'audit.json', dict(completed=True, n=len(rows), isolated=len(isolated), sources_unchanged=True,
        inputs_unchanged=True, finite_prefix_cost_excluded=True, no_speed_comparison_claim=True))
    print(out, flush=True)


if __name__ == '__main__':
    main()
