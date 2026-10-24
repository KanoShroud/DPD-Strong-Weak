"""Frozen paired finite-record solvers on independent geometry/waveform seeds."""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.research_discovery.boundary_blind_pilot import solve, conventional
from 统一模型代码.gates.research_discovery.window_baselines import locate
from 统一模型代码.gates.research_discovery.residual_competition import HOLDOUT
from 统一模型代码.gates.g10.r1.scenes import make
from 统一模型代码.gates.g8.physics import RECEIVERS
from 统一模型代码.gates.g10.r4.paper_checks.assignment_metrics import evaluate
from 统一模型代码.gates.g11.precision_diagnostic import sha, write


def layouts():
    rng = np.random.default_rng(20261023023)
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
            distance = []
            for i in range(4):
                for j in range(i+1, 4):
                    v = RECEIVERS[j]-RECEIVERS[i]
                    w = p-RECEIVERS[i]
                    distance.extend(abs(v[0]*w[:, 1]-v[1]*w[:, 0])/np.linalg.norm(v))
            if min(distance) < 50:
                continue
            answer.append(p)
            break
        else:
            raise RuntimeError('Geometry rejection sampling exhausted')
    return answer


def main():
    out = HOLDOUT.parent/('round23_independent_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    assert not out.exists() and out.resolve().is_relative_to(ROOT/'outputs_e2e')
    protocol_path = (HOLDOUT/'protocol.json').resolve(strict=True)
    cfg = json.loads(protocol_path.read_text(encoding='utf-8'))['config']
    sources = {str(p): sha(p) for p in (ROOT/'统一模型代码/gates').rglob('*.py')}
    pp = layouts()
    out.mkdir()
    (out/'inputs').mkdir()
    write(out/'protocol.json', dict(config=cfg, config_source=str(protocol_path), config_sha=sha(protocol_path),
        sources=sources, layouts=[p.tolist() for p in pp], geometry_seed=20261023023,
        waveform_seed=20261023024, repeats=[25000, 25001], gaps=[0, 15, 30],
        generator_n=8192, observation_n=4096, input_power_not_rescaled=True,
        methods=['direct', 'boundary', 'rect', 'hann', 'tukey_geometry'],
        paired_expectations=dict(same_success_count=True, max_position_difference_m=0.1,
                                 max_normalized_objective_difference=1e-8, median_time_reduction=0.2),
        scope='independent small validation, not publication-wide generalization',
        no_outcome_based_exclusion=True, ridge=1e-6, compression_bound=1e-6))
    rows, isolated, identities = [], [], {}
    with threadpool_limits(limits=2):
        for group, positions in enumerate(pp):
            for rep in (25000, 25001):
                data = make(group, rep, 0, n=8192, fs=50e6, positions=positions, seed=20261023024)
                path = out/'inputs'/f'{group}_{rep}.npz'
                np.savez_compressed(path, **{k: data[k] for k in ('signal', 'noise', 'positions')})
                identities[str(path)] = sha(path)
                write(out/'input_identities.json', identities)
                signal, noise = data['signal'][..., :4096], data['noise'][:, :4096]
                alone = {}
                for window in ('rect', 'hann', 'tukey_geometry'):
                    result = locate(signal[1]+noise, 1, cfg, window)
                    result['metrics'] = evaluate(positions[1:], result['positions'])
                    alone[window] = result
                isolated.append(dict(group=group, realization=rep, results=alone))
                write(out/'isolated.json', isolated)
                for gap in (0, 15, 30):
                    iq = signal[0]*10**(gap/20)+signal[1]+noise
                    result = {}
                    for method in ['direct', 'boundary'] if len(rows) % 2 == 0 else ['boundary', 'direct']:
                        r = solve(iq, cfg, method)
                        r['metrics'] = evaluate(positions, r['positions'])
                        result[method] = r
                    np.testing.assert_allclose(result['direct']['starts'], result['boundary']['starts'], atol=1e-9, rtol=0)
                    for window in ('rect', 'hann', 'tukey_geometry'):
                        r = conventional(iq, cfg, window)
                        r['metrics'] = evaluate(positions, r['positions'])
                        result[window] = r
                    rows.append(dict(group=group, realization=rep, gap=gap, results=result))
                    write(out/'results.json', rows)
                    print(f'{len(rows)}/24 布局{group} 实现{rep} {gap}dB', {k: v['metrics']['both20'] for k, v in result.items()}, flush=True)
    for path, identity in {**sources, **identities, str(protocol_path): sha(protocol_path)}.items():
        assert sha(path) == identity
    write(out/'audit.json', dict(completed=True, cases=len(rows), isolated=len(isolated),
        sources_unchanged=True, inputs_unchanged=True, candidates_identical=True))
    print(out, flush=True)


if __name__ == '__main__':
    main()
