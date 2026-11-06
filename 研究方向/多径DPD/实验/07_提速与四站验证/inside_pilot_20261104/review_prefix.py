"""Saved-case integrity, terminal numerics and stratified cost estimate."""
import sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_inside as run  # noqa: E402

cfg = run.cfg


def aggregate(rows, name):
    values = [x['methods'][name] for x in rows]
    errs = [x['errors_m'] for x in values if x['errors_m'] is not None]
    return dict(complete=len(errs), total=len(values), normal=sum(x['normal'] for x in values),
                rmse=float(np.sqrt(np.mean(np.square(errs)))) if errs else None,
                rmse_scope='all' if len(errs) == len(values) else 'complete_subset_only',
                mean_seconds=float(np.mean([x['seconds'] for x in values])),
                max_seconds=float(max(x['seconds'] for x in values)))


def main():
    out = HERE/'batch_v2'
    run.rr.check_hashes(cfg.read(out/'identity.json'))
    rows = []
    audit = []
    engine = run.Engine()
    for path in sorted((out/'cases').iterdir()):
        if not (path/'evaluation.json').exists():
            continue
        for name, digest in cfg.read(path/'receipt.json').items():
            assert cfg.sha(path/name) == digest
        row = cfg.read(path/'evaluation.json')
        rows.append(row)
        with np.load(path/'iq.npz') as z:
            iq = z['iq']
        assert cfg.digest(iq) == row['iq_sha']
        mm = cfg.read(path/'MP_decision.json')
        checks = dict(tag=row['tag'], decision_status=mm['status'], exception=mm.get('exception'))
        checks['all_returned_positions_in_domain'] = all(
            m['p'] is None or np.max(abs(np.asarray(m['p']))) <= run.BOUND+1e-6 for m in row['methods'].values())
        if mm['status'] == 'returned':
            result = mm['result']; final = result['final']
            x = np.array(final['x'])
            f, y, n, _ = run.ad.b.fc.cal.spectral(iq)
            r, j = engine.evaluate(x, f, y, result['paths'])
            cost, g = engine.objective(x, f, y, result['paths'])
            pg = float(np.max(abs(x-engine.ref.project(x-j.T@r, n))))
            gd = float(np.linalg.norm(g-j.T@r))
            checks.update(cost_difference=float(abs(cost-final['cost'])),
                original_pg_reproduced=bool(np.isclose(pg, final['projected_gradient'], rtol=1e-5, atol=1e-12)),
                gradient_difference=gd, gradient_consistent=bool(np.allclose(g, j.T@r, rtol=1e-5, atol=1e-9)),
                max_normalized_delay=float(max(x[engine.D+2::3], default=0)))
        audit.append(checks)
    methods = ('GEO', 'LOS', 'MP', 'FLOOD', 'XIA')
    grouped = {name: aggregate(rows, name) for name in methods}
    by_layout = {str(k): {m: aggregate([r for r in rows if r['layout'] == k], m) for m in methods}
                 for k in sorted({r['layout'] for r in rows})}
    by_condition = {k: {m: aggregate([r for r in rows if r['condition'] == k], m) for m in methods}
                    for k in sorted({r['condition'] for r in rows})}
    timing = {}
    weights = {'L0': 1, 'L1_P075': 2, 'L2_P075': 5, 'L3_P150': 5}
    for method in ('MP', 'FLOOD', 'XIA'):
        # Six equally weighted layouts, 10 worlds per layout, 13 conditions.
        assert all(k in by_condition and by_condition[k][method]['total'] == 6 for k in weights)
        timing[method] = sum(by_condition[k][method]['mean_seconds']*w*60 for k, w in weights.items())
    xcomplete = [r for r in rows if r['methods']['XIA']['errors_m'] is not None]
    common = [r for r in rows if all(r['methods'][m]['errors_m'] is not None for m in methods)]
    lay = run.layouts()[0]
    with np.load(out/'banks'/'100_00.npz') as z:
        first = dict(z)
    second = cfg.make_bank(lay, 'pilot', 1, run.ad.wave.synthesize, run.ad.wave.rrc)
    independence = {k: cfg.digest(first[k]) != cfg.digest(second[k]) for k in
                    ('bits', 'offsets', 'gains', 'delays', 'phases', 'weights', 'noise')}
    independence['same_position'] = bool(np.array_equal(first['p'], second['p']))
    independence['same_transmit_power'] = bool(np.array_equal(first['pt_w'], second['pt_w']))
    assert all(independence.values())
    summary = dict(completed=len(rows), independent_worlds=len(by_layout), target=780,
        methods=grouped, by_layout=by_layout, by_condition=by_condition,
        all_methods_common_subset={m: aggregate(common, m) for m in methods} if common else {},
        xia_complete_rows=len(xcomplete), audit=audit,
        cost_estimate=dict(seconds=timing, total_hours=sum(timing.values())/3600,
            caveat='Stratified 24-case pilot estimate; L2 middle power and L3 high power stand in for their five levels; failure costs not a repaired-solver budget'),
        remaining=780-len(rows),
        repeat_independence_check=independence, source_hash=cfg.sha(__file__))
    cfg.write(HERE/'prefix_review.json', summary)
    print(summary['methods'])
    print(summary['cost_estimate'])
    print('audit', sum(not a['all_returned_positions_in_domain'] for a in audit),
          sum(not a.get('gradient_consistent', True) for a in audit))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
