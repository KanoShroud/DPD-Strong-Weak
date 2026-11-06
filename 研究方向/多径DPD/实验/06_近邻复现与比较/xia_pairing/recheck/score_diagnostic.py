"""Fixed off-node spatial grid: raw-score/output attribution, not a baseline."""
from pathlib import Path
from datetime import datetime
import json
import numpy as np
from scipy.ndimage import maximum_filter
from threadpoolctl import threadpool_limits
import xia_checked as xia
from run_exact import write, sha, evaluate

ROOT = Path(__file__).resolve().parent
EXACT = ROOT/'exact_20261027_153508_733933'


def main():
    data = EXACT/'components.npz'
    assert sha(data) == json.loads((EXACT/'result.json').read_text(encoding='utf-8'))['components_sha']
    out = ROOT/('score_grid_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    source = [Path(__file__), ROOT/'xia_checked.py', ROOT.parent/'xia_reference.py', ROOT/'执行协议.md']
    identity = {str(p): sha(p) for p in source}
    for p in source:
        (out/p.name).write_bytes(p.read_bytes())
    write(out/'preflight.json', {'source': identity, 'input_sha': sha(data),
          'grid_step_m': 100, 'grid_first_m': -2975, 'grid_last_m': 2925,
          'max_filter': '3x3', 'refinement': 'none', 'output': 'two largest local maxima',
          'scope': 'SCORING_DIAGNOSTIC_NOT_FAIR_TIMING_OR_FINAL_ALGORITHM'})
    z = np.load(data)
    axis = np.arange(-2975., 2926., 100.)
    p = np.array(np.meshgrid(axis, axis, indexing='ij')).reshape(2, -1).T
    rows = []
    for channel in ['los', 'mp']:
        y = z['direct'].sum(0)+(z['reflection'].sum(0) if channel == 'mp' else 0)
        v = xia.scores(p, z['stations'], y, z['frequency'], 2000)
        field = v.reshape(len(axis), len(axis))
        maxima = (field == maximum_filter(field, size=3, mode='constant', cval=-np.inf)).ravel()
        ids = np.flatnonzero(maxima)
        ids = ids[np.argsort(-v[ids], kind='stable')]
        selected = p[ids[:2]]
        np.savez_compressed(out/f'{channel}.npz', points=p, scores=v, local_max_indices=ids)
        rank = []
        for t in z['truth']:
            distances = np.linalg.norm(p[ids]-t, axis=1)
            j = int(distances.argmin())
            rank.append({'nearest_peak_rank': j+1, 'distance_m': float(distances[j]),
                         'position': p[ids[j]].tolist(), 'score': float(v[ids[j]])})
        row = {'channel': channel, 'positions': selected.tolist(),
               'evaluation': evaluate(selected, z['truth']), 'nearest_local_peak_posthoc': rank,
               'local_peak_count': len(ids), 'grid_points': len(p)}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    write(out/'result.json', {'rows': rows, 'input_unchanged': sha(data) == json.loads(
        (EXACT/'result.json').read_text(encoding='utf-8'))['components_sha'],
        'source_unchanged': all(sha(Path(p)) == h for p,h in identity.items())})
    print(str(out), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
