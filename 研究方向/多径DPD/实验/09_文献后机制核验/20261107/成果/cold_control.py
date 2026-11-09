"""Oracle pair diagnostic with fresh support; never a blind estimator score."""
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
    identity = prepare(out)
    identity.update({str(p): sha(p) for p in [Path(__file__), Path(__file__).with_name('重建支持控制.md')]})
    rows = []
    start = time.perf_counter()
    for row in read(inp/'rows.json'):
        if row['condition'] == 'L0':
            continue
        tag = row['tag']
        iqfile = inp/'inputs'/tag/'iq.npz'
        assert sha(iqfile) == row['input_sha']
        with np.load(iqfile, allow_pickle=False) as z:
            iq = z['iq']
        e = api.create_engine()
        front = e.frontend(iq)
        bank = Path(row['truth_evaluation_file'])
        assert sha(bank) == row['truth_file_sha']
        with np.load(bank, allow_pickle=False) as z:
            truth = z['p']
        _, ids = linear_sum_assignment(np.sum((truth[:, None]-front['refined'][None])**2, axis=2))
        original = "pairs = [selection['first'], selection['second']] if two_regions else [fc.engine.joint_select(geom['reward'])[0]]"
        e.estimate_function = api.runtime().ad.clone(e.estimate_function,
            [(original, 'pairs = [diagnostic_pair]')], {'diagnostic_pair': ids.copy()})
        print('COLD_START', tag, ids.tolist(), flush=True)
        solved = e.mp(iq)
        cfg.write(out/f'{tag}_decision.json', solved)
        f, y, n, _ = e.fc.cal.spectral(iq)
        x = np.asarray(solved['final']['x'])
        cost, g = e.objective(x, f, y, solved['paths'])
        pg = float(np.max(abs(x-e.ref.project(x-g, n))))
        assert np.isclose(cost, solved['final']['cost'], rtol=1e-9, atol=1e-12)
        rows.append(dict(tag=tag, condition=row['condition'], pair=ids, normal=solved['normal'],
            stop=solved['stop'], pg=pg, paths=len(solved['paths']), cost=cost,
            errors_m=errors(solved['final']['p'], truth), seconds=solved['seconds'],
            scope='DIAGNOSTIC_ORACLE_PAIR_NOT_BLIND'))
        cfg.write(out/'rows.json', rows)
        print('COLD_END', tag, solved['stop'], solved['seconds'], flush=True)
    assert all(sha(p) == h for p, h in identity.items())
    cfg.write(out/'completion.json', dict(count=len(rows), elapsed_seconds=time.perf_counter()-start,
        source_hashes=identity, source_unchanged=True, scope='DIAGNOSTIC_ORACLE_PAIR_NOT_BLIND'))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
