"""EVAL-20261011-v2, independent from frozen G9's 30/50/100m aggregation."""
import numpy as np
from ..g9.metrics import evaluate as old_evaluate, partial


def evaluate(truth, predicted):
    result = old_evaluate(truth, predicted)
    d = np.linalg.norm(np.asarray(truth).reshape(-1, 2)[:, None]
                       -np.asarray(predicted).reshape(-1, 2)[None], axis=-1)
    assignments = [a for a in partial(*d.shape) if all(d[i, j] <= 20 for i, j in a)]
    matched = min(assignments, key=lambda a: (-len(a), sum(d[i, j]**2 for i, j in a), a))
    flags = [False]*len(truth)
    for i, _ in matched:
        flags[i] = True
    result['success']['20'] = flags
    result['protocol'] = 'EVAL-20261011-v2'
    return result


def aggregate(metrics):
    errors = [e for m in metrics for e in m['errors_m'] if e is not None]
    flags = [m['success']['20'] for m in metrics]
    return dict(cases=len(metrics), success20=float(np.mean([all(f) for f in flags])),
        strong20=float(np.mean([f[0] for f in flags])), weak20=float(np.mean([f[-1] for f in flags])),
        rmse_m=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
        p95_m=float(np.quantile(errors, .95)) if errors else None,
        gospa_m=float(np.mean([m['gospa_m'] for m in metrics])),
        coverage=sum(m['matched'] for m in metrics)/max(1, sum(m['targets'] for m in metrics)))
