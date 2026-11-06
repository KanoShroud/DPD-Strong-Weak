"""EVAL-20261010-v1；K<=2的精确部分匹配，不改历史指标。"""
from itertools import combinations, permutations

import numpy as np
from scipy.optimize import linear_sum_assignment


def partial(n, m):
    for k in range(min(n, m) + 1):
        for rows in combinations(range(n), k):
            for cols in permutations(range(m), k):
                yield tuple(zip(rows, cols))


def evaluate(truth, predicted):
    truth = np.asarray(truth, dtype=float).reshape(-1, 2)
    pred = np.asarray(predicted, dtype=float).reshape(-1, 2)
    if max(len(truth), len(pred)) > 2 or not np.isfinite(pred).all():
        raise ValueError('G9只接受有限的0/1/2个最终位置')
    if not np.isfinite(truth).all():
        raise ValueError('真值非有限')
    n, m = len(truth), len(pred)
    d = np.linalg.norm(truth[:, None] - pred[None], axis=-1)
    assignments = list(partial(n, m))
    rows, cols = linear_sum_assignment(d**2)
    errors = [None] * n
    for i, j in zip(rows, cols):
        errors[i] = float(d[i, j])
    success = {}
    for radius in (30, 50, 100):
        feasible = [a for a in assignments if all(d[i, j] <= radius for i, j in a)]
        a = min(feasible, key=lambda a: (-len(a), sum(d[i, j]**2 for i, j in a), a))
        flags = [False] * n
        for i, _ in a:
            flags[i] = True
        success[str(radius)] = flags
    c = 100.
    feasible = [a for a in assignments if all(d[i, j] < c for i, j in a)]
    a = min(feasible, key=lambda a: (sum(d[i, j]**2 for i, j in a)
                                   + c*c/2*(n+m-2*len(a)), a))
    localization = float(sum(d[i, j]**2 for i, j in a))
    missed, false = (n-len(a))*c*c/2, (m-len(a))*c*c/2
    return dict(protocol='EVAL-20261010-v1', errors_m=errors, success=success,
                gospa_m=float(np.sqrt(localization+missed+false)),
                localization_p=localization, missed_p=missed, false_p=false,
                matched=len(rows), targets=n, predictions=m)


def aggregate(rows):
    metrics = [r['metrics'] for r in rows]
    errors = [e for m in metrics for e in m['errors_m'] if e is not None]
    result = dict(cases=len(rows), gospa_m=float(np.mean([m['gospa_m'] for m in metrics])),
                  rmse_m=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                  p95_m=float(np.quantile(errors, .95)) if errors else None,
                  coverage=sum(m['matched'] for m in metrics)/max(1, sum(m['targets'] for m in metrics)),
                  no_output_fraction=float(np.mean([m['predictions'] == 0 for m in metrics])))
    for radius in (30, 50, 100):
        result[f'both_{radius}'] = float(np.mean([len(m['success'][str(radius)]) == 2
                                                and all(m['success'][str(radius)]) for m in metrics]))
        for source, name in ((0, 'strong'), (1, 'weak')):
            subset = [m for m in metrics if len(m['errors_m']) == 2]
            result[f'{name}_{radius}'] = float(np.mean([m['success'][str(radius)][source]
                                                       for m in subset])) if subset else None
    for source, name in ((0, 'strong'), (1, 'weak')):
        values = [m['errors_m'][source] for m in metrics
                  if len(m['errors_m']) == 2 and m['errors_m'][source] is not None]
        result[f'{name}_rmse_m'] = float(np.sqrt(np.mean(np.square(values)))) if values else None
    for key in ('seconds', 'score_points', 'response_points', 'updates'):
        values = [r[key] for r in rows if r.get(key) is not None]
        if values:
            result[key+'_mean'] = float(np.mean(values))
            result[key+'_p95'] = float(np.quantile(values, .95))
    return result
