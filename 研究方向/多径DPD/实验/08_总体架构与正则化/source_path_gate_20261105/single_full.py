"""Full original single-increment control; oracle branch, not blind localization."""
import time
import numpy as np
from threadpoolctl import threadpool_limits
from diagnose import HERE, INPUT, cfg, ad, stable, data, forced_run, metrics


def main():
    assert cfg.read(HERE/'increment_control_v1/summary.json')['full_increment_comparison_gate']
    source = HERE/'run_v1'
    stable.inside.rr.check_hashes(cfg.read(source/'identity.json'))
    out = HERE/'single_full_v2'
    out.mkdir(exist_ok=False)
    hashes = ad.source_hashes()
    hashes.update({str(p.resolve()): cfg.sha(p) for p in
        [HERE/'diagnose.py', HERE/'single_full.py', HERE/'single_full_protocol.md']})
    cfg.write(out/'identity.json', hashes)
    rows = []
    start_all = time.perf_counter()
    tags = cfg.read(source/'contract.json')['tags']
    for tag in tags:
        if time.perf_counter()-start_all > 3600:
            break  # Administrative boundary only; never reports an incomplete solve as normal.
        iq = data(tag)
        pool_file = source/(tag+'_blind_pool.json')
        pool = cfg.read(pool_file)
        geom = {k: np.array(v) if isinstance(v, list) else v for k, v in pool['geometry'].items()}
        chosen = cfg.read(source/(tag+'_diagnostic_selection.json'))['nearest_ids']
        e = stable.Engine()
        e.geo_snapshot = dict(p=geom['refined'][chosen], selected=chosen,
            seconds=pool['seconds'], cached_diagnostic_pool=True)
        e.estimate_function = ad.clone(e.estimate_function, [
            ('choices = sorted(choices, key=lambda a: -a[4])[:64-len(paths)]',
             'choices = sorted(choices, key=lambda a: -a[4])[:1]')])
        print('START', tag, flush=True)
        start = time.perf_counter()
        result = forced_run(e, iq, geom, [chosen])
        seconds = time.perf_counter()-start
        assert all(h['added'] in (0, 1) for h in result['history'])
        value = dict(status='returned', result=result, seconds=seconds, diagnostic_only=True,
            iq_sha=cfg.digest(iq), blind_pool_sha=cfg.sha(pool_file), chosen_ids=chosen)
        cfg.write(out/(tag+'.json'), value)
        with np.load(INPUT/'banks'/f'{tag.split("_")[1]}_00.npz', allow_pickle=False) as z:
            truth = z['p']
        row = dict(tag=tag, **metrics(e, iq, value, truth))
        rows.append(row)
        cfg.write(out/'rows.json', rows)
        print('END', row, flush=True)
    stable.inside.rr.check_hashes(hashes)
    stable.inside.rr.check_hashes(cfg.read(INPUT/'identity.json'))
    old = cfg.read(source/'rows.json')
    old_mse = np.mean([r['DIAGNOSTIC_NEAREST']['rmse']**2 for r in old])
    new_mse = np.mean([r['rmse']**2 for r in rows])
    gate = len(rows) == 12 and all(r['normal'] and r['consistent'] and r['feasible'] for r in rows) and new_mse <= old_mse
    cfg.write(out/'summary.json', dict(completed=len(rows), rows=rows, identity_unchanged=True,
        diagnostic_only=True, next_gate=bool(gate), normal_count=sum(r['normal'] for r in rows),
        group_rmse=float(np.sqrt(old_mse)), single_rmse=float(np.sqrt(new_mse)),
        total_seconds=time.perf_counter()-start_all))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
