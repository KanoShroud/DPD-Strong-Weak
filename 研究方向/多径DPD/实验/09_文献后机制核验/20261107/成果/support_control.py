"""Diagnostic only: frozen support, restart versus oracle-selected blind pool."""
import argparse
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits
from run_gate import api, cfg, errors, prepare, read, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    inp, out = args.input.resolve(), args.out.resolve()
    assert read(inp/'completion.json')['count'] == 18
    identity = prepare(out)
    extra = [Path(__file__), Path(__file__).with_name('固定支持控制.md')]
    identity.update({str(p): sha(p) for p in extra})
    source_rows = read(inp/'rows.json')
    rows = []
    start = time.perf_counter()
    for row in source_rows:
        if row['condition'] == 'L0':
            continue
        tag = row['tag']
        iqfile = inp/'inputs'/tag/'iq.npz'
        assert sha(iqfile) == row['input_sha']
        with np.load(iqfile, allow_pickle=False) as z:
            iq = z['iq']
        solved = read(inp/f'{tag}_MP_decision.json')
        if 'final' not in solved:
            rows.append(dict(tag=tag, skipped='no_finite_MP_state'))
            continue
        e = api.create_engine()
        f, y, n, sigma = e.fc.cal.spectral(iq)
        x0 = np.asarray(solved['final']['x'])
        paths = [tuple(p) for p in solved['paths']]
        old_cost, grad = e.objective(x0, f, y, paths)
        assert np.isclose(old_cost, solved['final']['cost'], rtol=1e-9, atol=1e-12)
        geometry = e.frontend(iq)
        pool = geometry['refined']
        bank = Path(row['truth_evaluation_file'])
        assert sha(bank) == row['truth_file_sha']
        with np.load(bank, allow_pickle=False) as z:
            truth = z['p']
        _, ids = linear_sum_assignment(np.sum((truth[:, None]-pool[None])**2, axis=2))
        pair = pool[ids]
        current = x0[:4].reshape(2, 2)*e.ref.UNIT
        _, assignment = linear_sum_assignment(np.sum((current[:, None]-pair[None])**2, axis=2))
        counterfactual = x0.copy()
        counterfactual[:4] = pair[assignment].ravel()/e.ref.UNIT
        result = dict(tag=tag, condition=row['condition'], original_cost=old_cost,
            original_errors_m=errors(current, truth), original_pg=float(np.max(abs(x0-e.ref.project(x0-grad, n)))),
            paths=paths, candidate_count=len(pool), oracle_pool_errors_m=errors(pair, truth), arms={})
        for arm, initial in [('RESTART', x0), ('ORACLE_POOL_START', counterfactual)]:
            print('CONTROL_START', tag, arm, flush=True)
            begin = time.perf_counter()
            xx, state = e.fit(initial, f, y, paths, n, 'GATED')
            cost, g = e.objective(xx, f, y, paths)
            pg = float(np.max(abs(xx-e.ref.project(xx-g, n))))
            feasible = float(np.linalg.norm(xx-e.ref.project(xx, n)))
            assert np.isfinite(xx).all() and np.isfinite(cost) and feasible < 1e-7
            assert np.isclose(cost, state['cost'], rtol=1e-9, atol=1e-12)
            cfg.write(out/f'{tag}_{arm}.json', state)
            result['arms'][arm] = dict(cost=cost, pg=pg, feasible=feasible, stationary=pg<=1e-7,
                p=xx[:4].reshape(2, 2)*e.ref.UNIT, errors_m=errors(xx[:4].reshape(2, 2)*e.ref.UNIT, truth),
                seconds=time.perf_counter()-begin, initial_cost=e.objective(initial, f, y, paths)[0],
                reason=state.get('reason'))
            print('CONTROL_END', tag, arm, result['arms'][arm]['seconds'], pg, flush=True)
        rows.append(result)
        cfg.write(out/'rows.json', rows)
    assert all(sha(p) == h for p, h in identity.items())
    cfg.write(out/'completion.json', dict(count=len(rows), elapsed_seconds=time.perf_counter()-start,
        source_unchanged=True, source_hashes=identity, scope='fixed_support_diagnostic_ORACLE_NOT_BLIND'))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
