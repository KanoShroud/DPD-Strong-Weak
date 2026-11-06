"""Isolate a feasible initializer under unchanged legacy detection/stopping rules."""
import time
import numpy as np
from threadpoolctl import threadpool_limits
from local_cone import HERE, PREVIOUS, cfg, ad, stable, data, INPUT, score, tangent_constraints
from diagnose import forced_run, metrics


def make_seed(e, ledger):
    def initialize(x, xx, f, y, paths, n):
        zero = xx.copy(); zero[len(x):].reshape(-1, 3)[:, :2] = 0
        assert len(xx) == len(x)+3
        raw, tangent, _ = e.evaluate(zero, f, y, paths, True)
        c, labels = tangent_constraints(e, x, n)
        stable_active = all(tangent_constraints(e, x, n, t)[1] == labels for t in [1e-10, 1e-6])
        _, d, info = score(tangent[:, :len(x)], tangent[:, -3:-1], raw, c)
        d = np.r_[d, 0.]
        cost, g = e.objective(zero, f, y, paths)
        event = dict(K=len(paths), local=info, constraints=labels, stable_active=stable_active,
                     accepted=False, old_cost=cost, trials=[])
        if info['certified'] and stable_active and g@d < 0:
            for k in range(20):
                trial = e.ref.project(zero+(.5**k)*d, n)
                value, _ = e.objective(trial, f, y, paths)
                slope = float(g@(trial-zero))
                event['trials'].append(dict(alpha=.5**k, cost=value, slope=slope))
                if slope < 0 and value < cost and value <= cost+1e-4*slope and np.linalg.norm(trial[-3:-1]) > 0:
                    event.update(accepted=True, alpha=.5**k, actual_cost=value)
                    ledger.append(event)
                    return trial
        ledger.append(event)
        return zero
    return initialize


def main():
    assert cfg.read(HERE/'local_v1/summary.json')['local_gate']
    stable.inside.rr.check_hashes(cfg.read(HERE/'local_v1/identity.json'))
    oldroot = PREVIOUS/'single_full_v2'
    stable.inside.rr.check_hashes(cfg.read(oldroot/'identity.json'))
    out = HERE/'seed_control_v1'
    out.mkdir(exist_ok=False)
    hashes = ad.source_hashes()
    hashes.update({str(p.resolve()): cfg.sha(p) for p in [HERE/'local_cone.py', HERE/'seed_control.py', HERE/'seed_control_protocol.md']})
    cfg.write(out/'identity.json', hashes)
    rows = []; started = time.perf_counter()
    for oldrow in cfg.read(oldroot/'rows.json'):
        if time.perf_counter()-started > 3600:
            break
        tag = oldrow['tag']; iq = data(tag)
        poolfile = PREVIOUS/'run_v1'/(tag+'_blind_pool.json')
        saved = cfg.read(poolfile)
        assert cfg.digest(iq) == saved['iq_sha']
        geom = {k: np.asarray(v) if isinstance(v, list) else v for k, v in saved['geometry'].items()}
        ids = cfg.read(PREVIOUS/'run_v1'/(tag+'_diagnostic_selection.json'))['nearest_ids']
        e = stable.Engine(); ledger = []
        e.geo_snapshot = dict(p=geom['refined'][ids], selected=ids, cached_diagnostic_pool=True)
        e.estimate_function = ad.clone(e.estimate_function, [
            ('choices = sorted(choices, key=lambda a: -a[4])[:64-len(paths)]', 'choices = sorted(choices, key=lambda a: -a[4])[:1]'),
            ('xx[len(x):].reshape(-1, 3)[:, :2] = 0', 'xx = initialize(x, xx, f, y, pp, n)')],
            {'initialize': make_seed(e, ledger)})
        print('START', tag, flush=True); start = time.perf_counter()
        result = forced_run(e, iq, geom, [ids])
        value = dict(status='returned', result=result, seconds=time.perf_counter()-start,
            initializer_events=ledger, diagnostic_only=True, legacy_threshold_unchanged=True,
            iq_sha=cfg.digest(iq), blind_pool_sha=cfg.sha(poolfile))
        cfg.write(out/(tag+'.json'), value)
        with np.load(INPUT/'banks'/f'{tag.split("_")[1]}_00.npz', allow_pickle=False) as z:
            truth = z['p']
        row = dict(tag=tag, **metrics(e, iq, value, truth), initializer_calls=len(ledger),
            accepted_initializers=sum(ev['accepted'] for ev in ledger))
        rows.append(row); cfg.write(out/'rows.json', rows)
        print('END', row, flush=True)
    stable.inside.rr.check_hashes(hashes)
    oldrows = cfg.read(oldroot/'rows.json')
    oldmse = np.mean([r['rmse']**2 for r in oldrows]); newmse = np.mean([r['rmse']**2 for r in rows])
    gate = len(rows) == 12 and all(r['normal'] and r['consistent'] and r['feasible'] and r['initializer_calls'] == r['accepted_initializers'] for r in rows) and newmse <= oldmse
    cfg.write(out/'summary.json', dict(rows=rows, completed=len(rows), numerical_gate=bool(gate),
        old_rmse=float(np.sqrt(oldmse)), new_rmse=float(np.sqrt(newmse)),
        complete_detector_calibrated=False, identity_unchanged=True, elapsed=time.perf_counter()-started))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
