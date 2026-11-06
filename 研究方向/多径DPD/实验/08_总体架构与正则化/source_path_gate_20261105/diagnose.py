"""Observation-pool / oracle-branch diagnostic, never a blind algorithm score."""
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'numerical_repair_20261104'))
import stable  # noqa: E402

cfg, ad = stable.cfg, stable.ad
INPUT = HERE.parent/'inside_pilot_20261104/batch_v2'


def forced_run(engine, iq, geom, pairs):
    """Only replace which observed candidate pair(s) reach the frozen solver."""
    class Regions:
        @staticmethod
        def frontend(unused):
            assert cfg.digest(unused) == cfg.digest(iq)
            return geom

    from types import SimpleNamespace
    fn = ad.clone(engine.estimate_function, [
        ("selection = bc.regions.alternate_pair(geom['refined'], geom['reward']) if two_regions else None", "selection = None"),
        ("pairs = [selection['first'], selection['second']] if two_regions else [fc.engine.joint_select(geom['reward'])[0]]", "pairs = supplied_pairs")],
        dict(bc=SimpleNamespace(regions=Regions), supplied_pairs=pairs))
    return fn(iq, 'M_GATED', True, True)


def data(tag):
    src = INPUT/'cases'/tag
    for name, digest in cfg.read(src/'receipt.json').items():
        assert cfg.sha(src/name) == digest
    with np.load(src/'iq.npz', allow_pickle=False) as z:
        iq = z['iq']
    return iq


def check():
    e = stable.Engine()
    iq = data('IN_103_R00_L0')
    native = e.mp(iq)
    geom = e.last_geometry
    select = e.estimate_function.__globals__['bc'].regions.alternate_pair(geom['refined'], geom['reward'])
    result = forced_run(e, iq, geom, [select['first'], select['second']])
    checks = dict(
        same_final=np.allclose(result['final']['x'], native['final']['x'], atol=1e-9, rtol=1e-10),
        same_objective=abs(result['final']['cost']-native['final']['cost']) < 1e-12,
        same_paths=result['paths'] == native['paths'],
        same_stop=result['stop'] == native['stop'],
        same_los_branches=all(np.allclose(a['x'], b['x'], atol=1e-9, rtol=1e-10) for a, b in zip(result['los_states'], native['los_states'])))
    cfg.write(HERE/'correctness.json', dict(passed=all(checks.values()), checks=checks,
        native=native, forced=result, iq_sha=cfg.digest(iq)))
    assert all(checks.values()), checks


def metrics(e, iq, value, truth):
    out = dict(status=value['status'], seconds=value['seconds'])
    if value['status'] != 'returned':
        return out
    result = value['result']
    state = result['final']
    f, y, n, _ = ad.b.fc.cal.spectral(iq)
    x = np.asarray(state['x'])
    cost, g = e.objective(x, f, y, result['paths'])
    r, j = e.evaluate(x, f, y, result['paths'])
    error = stable.inside.rr.evaluation(state['p'], truth)
    los_error = stable.inside.rr.evaluation(result['los']['p'], truth)
    out.update(cost=cost, pg=float(np.max(abs(x-e.ref.project(x-g, n)))),
        normal=result['normal'], stop=result['stop'], K=len(result['paths']),
        errors=error, rmse=float(np.sqrt(np.mean(np.square(error)))),
        los_rmse=float(np.sqrt(np.mean(np.square(los_error)))), los_cost=result['los']['cost'],
        feasible=bool(np.linalg.norm(x-e.ref.project(x, n)) < 1e-7),
        consistent=bool(np.allclose(g, j.T@r, atol=1e-9, rtol=1e-5) and abs(cost-state['cost']) < 1e-9))
    return out


def main():
    stable.inside.rr.check_hashes(cfg.read(INPUT/'identity.json'))
    out = HERE/'run_v1'
    assert not out.exists(), 'Fresh output required'
    out.mkdir()
    hashes = ad.source_hashes()
    hashes.update({str(p.resolve()): cfg.sha(p) for p in [Path(__file__), HERE/'protocol.md']})
    cfg.write(out/'identity.json', hashes)
    check()
    tags = [f'IN_{i}_R00_{c}' for i in range(100, 106) for c in ['L2_P075', 'L3_P150']]
    cfg.write(out/'contract.json', dict(tags=tags, threads=2, diagnostic_only=True,
        selection_uses_truth=True, no_true_channel_or_waveform=True, command=sys.argv))
    rows = []
    for index, tag in enumerate(tags):
        iq = data(tag)
        e = stable.Engine()
        t = time.perf_counter()
        geom = e.frontend(iq)
        front_seconds = time.perf_counter()-t
        first = np.asarray(e.geo_snapshot['selected'])
        cfg.write(out/(tag+'_blind_pool.json'), dict(geometry=geom, first=first, iq_sha=cfg.digest(iq), seconds=front_seconds))
        # Oracle is used to SELECT the control branch, never to generate a candidate.
        with np.load(INPUT/'banks'/f'{tag.split("_")[1]}_00.npz', allow_pickle=False) as z:
            truth = z['p']
        distance = np.linalg.norm(geom['refined'][:, None]-truth[None], axis=2)
        nearest = np.argmin(distance, axis=0)
        assert nearest[0] != nearest[1]
        cfg.write(out/(tag+'_diagnostic_selection.json'), dict(nearest_ids=nearest,
            nearest_distance_m=distance.min(axis=0), warning='Truth-selected diagnostic, not a blind estimator'))
        order = ['GEOMETRY_FIRST', 'DIAGNOSTIC_NEAREST']
        if index % 2:
            order.reverse()
        record = dict(tag=tag, nearest_distance_m=distance.min(axis=0), frontend_seconds=front_seconds)
        for arm in order:
            ids = first if arm == 'GEOMETRY_FIRST' else nearest
            print('START', tag, arm, flush=True)
            start = time.perf_counter()
            try:
                result = forced_run(e, iq, geom, [ids])
                value = dict(status='returned', result=result)
            except (ValueError, np.linalg.LinAlgError) as exc:
                value = dict(status='failed', exception=repr(exc), traceback=traceback.format_exc())
            value.update(seconds=time.perf_counter()-start, input_ids=ids, iq_sha=cfg.digest(iq), diagnostic_only=True)
            cfg.write(out/(tag+'_'+arm+'.json'), value)
            record[arm] = metrics(e, iq, value, truth)
            print('END', tag, arm, record[arm], flush=True)
        rows.append(record)
        cfg.write(out/'rows.json', rows)
    stable.inside.rr.check_hashes(hashes)
    stable.inside.rr.check_hashes(cfg.read(INPUT/'identity.json'))
    cfg.write(out/'summary.json', dict(completed=len(rows)*2, rows=rows, identity_unchanged=True,
        all_numerically_consistent=all(r[a].get('consistent', False) and r[a].get('feasible', False)
            for r in rows for a in ['GEOMETRY_FIRST', 'DIAGNOSTIC_NEAREST']), diagnostic_only=True))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
