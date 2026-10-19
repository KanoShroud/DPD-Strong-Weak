"""Paper evaluation v3: one squared-distance assignment for every metric.

Frozen v2 modules and evidence are intentionally left unchanged. Invalid or
missing outputs remain unsuccessful trials; their error coverage is explicit.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

REVISION = 'EVAL-20261019-v3-SINGLE-SQUARED-ASSIGNMENT'
RADIUS_M = 20.


def evaluate(truth, predicted):
    truth = np.asarray(truth, dtype=float).reshape(-1, 2)
    if not len(truth) or not np.isfinite(truth).all():
        raise ValueError('Finite nonempty truth required')
    errors, flags, pairs = [None]*len(truth), [False]*len(truth), []
    try:
        pred = np.asarray(predicted, dtype=float).reshape(-1, 2)
        valid = bool(np.isfinite(pred).all() and len(pred) <= len(truth))
    except (ValueError, TypeError):
        pred, valid = np.empty((0, 2)), False
    if valid and len(pred):
        # Canonical order makes exact-tie behaviour independent of output order.
        order = np.lexsort((pred[:, 1], pred[:, 0]))
        distance = np.linalg.norm(truth[:, None]-pred[order][None], axis=2)
        rows, cols = linear_sum_assignment(distance**2)
        for i, j in zip(rows, cols):
            errors[i] = float(distance[i, j])
            flags[i] = errors[i] <= RADIUS_M
            pairs.append([int(i), int(order[j])])
    return dict(protocol=REVISION, errors_m=errors, success20=flags,
                both20=bool(all(flags)), matched=len(pairs), targets=len(truth),
                predictions=len(pred), valid_output=valid, assignment=pairs)


def self_test():
    from ...metrics import evaluate as legacy
    # A realizable 2-D example: cross-match errors 19/19, squared-optimal 0/21.
    x = 281/38
    truth = [[0., 0.], [19., 0.]]
    pred = [[0., 0.], [x, float(np.sqrt(19**2-x*x))]]
    assert legacy(truth, pred)['success']['20'] == [True, True]
    new = evaluate(truth, pred)
    assert new['success20'] == [True, False]
    np.testing.assert_allclose(new['errors_m'], [0., 21.], atol=1e-12)
    assert evaluate(truth, pred[::-1])['errors_m'] == new['errors_m']
    assert evaluate([[0, 0]], [[20, 0]])['both20']
    assert not evaluate([[0, 0]], [[20.000001, 0]])['both20']
    assert evaluate(truth, [])['success20'] == [False, False]
    assert evaluate(truth, [[0, 0]])['success20'] == [True, False]
    assert not evaluate(truth, [[np.nan, 0]])['valid_output']
    assert not evaluate(truth, [[0, 0]]*3)['valid_output']
    return dict(passed=True, example_old=[19., 19.], example_new=new['errors_m'],
                checks=['exchange', '20m_boundary', 'missing', 'invalid', 'known_count'])
