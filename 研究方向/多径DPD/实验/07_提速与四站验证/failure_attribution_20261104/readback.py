"""Independent serialized-result audit; no optimizer or estimator imports."""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FORMAL = HERE.parent/'risk_calibration_20261102/formal_20261102_181736_622970'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def payload(path):
    obj = read(path)
    raw = json.dumps(obj['payload'], ensure_ascii=False, allow_nan=False, sort_keys=True)
    assert hashlib.sha256(raw.encode('utf-8')).hexdigest() == obj['sha256']
    return obj['payload']


def mse(p, truth):
    a = np.asarray(p)
    return float(min(np.sum((a-truth)**2), np.sum((a-truth[::-1])**2))/2)


def main():
    out = Path(read(HERE/'latest.json')['path'])
    result = read(out/'result.json')
    contract = payload(FORMAL/'contract.json')['contract']
    hashes = read(out/'preflight.json')['hashes']
    for filename, expected in hashes.items():
        h = hashlib.sha256()
        with Path(filename).open('rb') as f:
            for chunk in iter(lambda: f.read(1024*1024), b''):
                h.update(chunk)
        assert h.hexdigest() == expected, filename
    checks = {'all_frozen_hashes_unchanged': True}
    records = []
    for comp in result['comparisons']:
        tag = comp['case']
        a = payload(FORMAL/'cases'/tag/'solve.json')
        b = payload(out/(tag+'_alternative.json'))
        with np.load(contract['conditions']['HIGH']['bank'], allow_pickle=False) as z:
            truth = z['p']
        curves = []
        for name, obj in [('original', a), ('alternative', b)]:
            curve = [obj['los']]+[v['fit'] for v in obj['history'] if 'fit' in v and v['decrease']]
            assert len(curve) == len(obj['paths'])+1
            errs = np.sqrt([mse(v['p'], truth) for v in curve])
            costs = np.array([v['cost'] for v in curve])
            checks[tag+'/'+name+'_loss'] = bool(abs(errs[-1]-comp[name+'_rmse_m']) < 1e-10)
            checks[tag+'/'+name+'_decrease'] = bool(np.all(np.diff(costs) <= 1e-10))
            checks[tag+'/'+name+'_budget'] = all(v['iterations'] <= 300 for v in curve)
            curves.append((curve, errs, costs))
        c1, e1, j1 = curves[0]
        c2, e2, j2 = curves[1]
        k = min(len(c1), len(c2))-1
        below = np.flatnonzero(j2 < j1[-1])
        records.append(dict(case=tag, original_best_trajectory_rmse_m=float(min(e1)),
            alternative_final_rmse_m=float(e2[-1]), matched_total_K=k,
            original_matched_cost=float(j1[k]), alternative_matched_cost=float(j2[k]),
            first_K_below_original_final=int(below[0]) if len(below) else None,
            alternative_seconds=b['seconds']))
    if result['stage_B'] is not None:
        for row in result['stage_B']['rows']:
            chosen = 2 if row['second_cost'] < row['first_cost'] else 1
            checks[row['case']+'/screen_rule'] = chosen == row['selected']
    assert all(checks.values()), checks
    audit = dict(checks=checks, all_checks_pass=True, frozen_file_count=len(hashes), rows=records,
                 scope='serialized_identity_and_metric_replay_not_new_localization')
    (out/'readback.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
