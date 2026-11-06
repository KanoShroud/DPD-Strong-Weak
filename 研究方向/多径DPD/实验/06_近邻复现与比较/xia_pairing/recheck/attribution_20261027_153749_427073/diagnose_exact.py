"""Bounded exact-model attribution; oracle probes never enter the solver."""
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import xia_checked as xia
from run_exact import write, sha, evaluate

ROOT = Path(__file__).resolve().parent
EXACT = ROOT/'exact_20261027_153508_733933'


def main():
    frozen = json.loads((EXACT/'result.json').read_text(encoding='utf-8'))
    assert sha(EXACT/'components.npz') == frozen['components_sha']
    out = ROOT/('attribution_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    src = [Path(__file__), ROOT/'xia_checked.py', ROOT/'run_exact.py',
           ROOT/'执行协议.md', ROOT.parent/'xia_reference.py']
    identity = {str(p): sha(p) for p in src}
    for p in src:
        (out/p.name).write_bytes(p.read_bytes())
    write(out/'preflight.json', {'sources': identity, 'input': str(EXACT/'components.npz'),
                                'input_sha': sha(EXACT/'components.npz'),
                                'extra_seeds': [2702, 2703], 'channels': ['los', 'mp'],
                                'scope': 'ONE_DATA_REALIZATION_TWO_SEED_CONTROLS_AND_POSTHOC_PROBES'})
    z = np.load(EXACT/'components.npz')
    f, stations, truth = z['frequency'], z['stations'], z['truth']
    rows, probes = [], []
    for channel in ['los', 'mp']:
        y = z['direct'].sum(0)+(z['reflection'].sum(0) if channel == 'mp' else 0)
        # True and local diagnostic positions are used ONLY after baseline solve.
        true_score = xia.scores(truth, stations, y, f, 2000)
        axis = np.arange(-100., 101., 25.)
        offsets = np.array(np.meshgrid(axis, axis, indexing='ij')).reshape(2, -1).T
        local = []
        for target in truth:
            scores = xia.scores(target+offsets, stations, y, f, 2000)
            i = int(scores.argmax())
            local.append({'offsets': offsets.tolist(), 'scores': scores.tolist(),
                          'best_offset_m': offsets[i].tolist(), 'best_score': float(scores[i])})
        for seed in [2701, 2702, 2703]:
            if seed == 2701:
                result = json.loads((EXACT/f'dual_{channel}.json').read_text(encoding='utf-8'))['output']
            else:
                result = xia.solve(y, f, stations, 2, 2000, 3000, seed=seed)
                write(out/f'{channel}_{seed}.json', {'output': result,
                      'evaluation': evaluate(result['positions'], truth)})
            first = result['steps'][0]
            threshold = first['output']['threshold']
            p = np.asarray(first['points'])
            v = np.asarray(first['scores'])
            keep = xia.threshold(v)['keep']
            rows.append({'channel': channel, 'seed': seed, 'status': result['status'],
                         'seconds': result['seconds'], 'threshold': threshold,
                         'true_scores': true_score.tolist(),
                         'true_below_threshold': (true_score < threshold).tolist(),
                         'local_best_below_threshold': [d['best_score'] < threshold for d in local],
                         'nearest_initial_distance': np.min(np.linalg.norm(p[:, None]-truth[None], axis=-1), axis=0).tolist(),
                         'retained_within_400m': [int((np.linalg.norm(p[keep]-t, axis=1)<=400).sum()) for t in truth],
                         'evaluation': evaluate(result['positions'], truth)})
        probes.append({'channel': channel, 'true_scores': true_score.tolist(), 'local': local})
    # Independent replay of the original rounding counterexample.
    old = json.loads((ROOT.parent/'pair_20261027_143405_945211/A_los_XIA.json').read_text(encoding='utf-8'))
    st = old['output']['steps'][-1]
    values, points = np.asarray(st['scores']), np.asarray(st['points'])
    corrected = xia.decode(points, values, 1, 400/3)
    rounding = {'scope': 'SAVED_STATE_REPLAY_NOT_NEW_LOCALIZATION',
                'output': corrected, 'evaluation': evaluate(corrected['positions'], np.array([[-260., -180.]]))}
    write(out/'probes.json', probes)
    write(out/'rounding_replay.json', rounding)
    write(out/'result.json', {'rows': rows, 'additional_solves': 4, 'rounding_replay': rounding,
                              'sources_unchanged': all(sha(Path(p)) == h for p,h in identity.items()),
                              'input_unchanged': sha(EXACT/'components.npz') == frozen['components_sha'],
                              'conclusion_limit': 'Fixed input; solver seeds are not independent signal/channel realizations'})
    for row in rows:
        print(json.dumps(row, ensure_ascii=False), flush=True)
    print(str(out), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
