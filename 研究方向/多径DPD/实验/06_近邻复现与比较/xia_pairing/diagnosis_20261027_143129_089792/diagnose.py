"""Read-only-truth attribution after blind outputs; not a new locator."""
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

import inclusive_solver as xia
from preflight import STATIONS, N, sha, write
from xia_reference import sparse_threshold

ROOT = Path(__file__).resolve().parent
CONTROL = ROOT/'control_20261027_142859_457483'


def main():
    out = ROOT/('diagnosis_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    source = [ROOT/'diagnose.py', ROOT/'inclusive_solver.py', ROOT/'xia_reference.py']
    identity = {str(p): sha(p) for p in source}
    for p in source:
        (out/p.name).write_bytes(p.read_bytes())
    truth = np.array([[-260., -180.], [230., 120.]])
    with np.load(CONTROL/'dual_los_input.npz') as d:
        y, f = d['y'], d['frequency']
    start = time.perf_counter()
    original = json.loads((CONTROL/'dual_los_XIA.json').read_text(encoding='utf-8'))
    truth_score = xia.scores(truth, STATIONS, y, f, N)
    stages = []
    for s in original['output']['steps']:
        p, v = np.array(s['points']), np.array(s['scores'])
        threshold = sparse_threshold(v)
        mask = threshold['retained']
        nearest = np.min(np.linalg.norm(p[:, None]-truth, axis=-1), axis=0)
        stages.append({'round': s['round'], 'threshold': threshold['threshold'],
                       'truth_score': truth_score, 'truth_passes_threshold': truth_score >= threshold['threshold'],
                       'min_distance_all_m': nearest,
                       'min_distance_retained_m': np.min(np.linalg.norm(p[mask, None]-truth, axis=-1), axis=0),
                       'retained_near_A_100m': np.sum(np.linalg.norm(p[mask]-truth[0], axis=1) < 100),
                       'retained_near_B_100m': np.sum(np.linalg.norm(p[mask]-truth[1], axis=1) < 100)})
    axis = np.arange(-1000., 1001., 20.)
    grid = np.array(np.meshgrid(axis, axis, indexing='ij')).reshape(2, -1).T
    values = xia.scores(grid, STATIONS, y, f, N)
    decoder = xia.decode(grid, values, 2, 400/3)
    np.savez_compressed(out/'dense_score.npz', grid=grid, score=values, truth=truth, truth_score=truth_score)
    trials = []
    # Not best-of-seeds; all registered diagnostics are kept separately.
    for particles, seed in [(1000, 2702), (1000, 2703), (10000, 2701)]:
        result = xia.solve(y, f, STATIONS, 2, N, seed=seed, particles=particles)
        write(out/f'blind_{particles}_{seed}.json', result)
        trials.append({'particles': particles, 'seed': seed, 'positions': result['positions'],
                       'status': result['status'], 'seconds': result['seconds']})
    write(out/'result.json', {'source': identity, 'control_input': str(CONTROL/'dual_los_input.npz'),
                              'input_sha256': sha(CONTROL/'dual_los_input.npz'),
                              'stages': stages, 'dense_diagnostic_decoder': decoder,
                              'dense_max_position': grid[np.argmax(values)],
                              'truth_score': truth_score, 'dense_max_score': values.max(),
                              'registered_trials': trials, 'seconds': time.perf_counter()-start,
                              'role': 'OFFLINE_ATTRIBUTION_NOT_METHOD_RANKING',
                              'sources_unchanged': all(sha(Path(p)) == h for p, h in identity.items())})
    print('诊断完成：'+str(out), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
