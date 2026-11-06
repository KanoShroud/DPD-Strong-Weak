"""Feasible Armijo initialization of new amplitudes; unchanged physical feasible set."""
import time
import numpy as np
from threadpoolctl import threadpool_limits
from diagnose import HERE, INPUT, cfg, ad, stable, data


def seed(e, x, xx, f, y, paths, n, cost):
    value, _ = e.objective(xx, f, y, paths)
    if value <= cost:
        return xx, dict(applied=False, accepted=True, cost=value)
    zero = xx.copy()
    zero[len(x):].reshape(-1, 3)[:, :2] = 0
    c0, g0 = e.objective(zero, f, y, paths)
    assert np.isclose(c0, cost, rtol=1e-10, atol=1e-12)
    direction = xx-zero
    slope = float(g0@direction)
    eps = 1e-5
    fd = (e.objective(zero+eps*direction, f, y, paths)[0]-e.objective(zero-eps*direction, f, y, paths)[0])/(2*eps)
    assert np.isclose(fd, slope, rtol=1e-4, atol=1e-9), (fd, slope)
    _, j = e.evaluate(zero, f, y, paths)
    delay_column = float(np.linalg.norm(j[:, len(x)+2]))
    assert delay_column < 1e-14
    for k in range(20):
        trial = zero+(.5**k)*direction
        vv, _ = e.objective(trial, f, y, paths)
        if slope < 0 and np.isfinite(vv) and vv <= c0+1e-4*(.5**k)*slope and vv < c0:
            assert np.linalg.norm(trial-e.ref.project(trial, n)) < 1e-7
            return trial, dict(applied=True, accepted=True, alpha=.5**k, slope=slope,
                finite_difference=fd, zero_delay_column_norm=delay_column, initial_cost=c0, cost=vv)
    return zero, dict(applied=True, accepted=False, slope=slope, finite_difference=fd,
        zero_delay_column_norm=delay_column, initial_cost=c0)


def main():
    src = HERE/'single_full_v2'
    oldrows = cfg.read(src/'rows.json')
    assert cfg.read(src/'summary.json')['completed'] == 12
    tags = [r['tag'] for r in oldrows if not r['normal']]+['IN_100_R00_L2_P075']
    assert len(tags) == 5
    out = HERE/'warmstart_control_v1'
    out.mkdir(exist_ok=False)
    hashes = ad.source_hashes()
    hashes.update({str(p.resolve()): cfg.sha(p) for p in [HERE/'diagnose.py', HERE/'warmstart.py', HERE/'warmstart_protocol.md']})
    cfg.write(out/'identity.json', hashes)
    rows = []
    for tag in tags:
        old = cfg.read(src/(tag+'.json'))['result']
        which = [i for i, h in enumerate(old['history']) if 'fit' in h][-1]
        event = old['history'][which]
        previous = old['los'] if which == 0 else old['history'][which-1]['fit']
        x = np.asarray(previous['x'])
        k = event['K']-1
        paths = old['paths'][:k]
        e = stable.Engine()
        iq = data(tag)
        f, y, n, sigma = ad.b.fc.cal.spectral(iq)
        use = abs(np.fft.fftfreq(n, 1/cfg.FS)) <= cfg.B/2
        pick, bank, meta = e.search(x, paths, f, y, n, use, sigma, which+1)
        assert pick is not None and all(meta['checks'].values())
        ids = np.flatnonzero((bank[:, 3] == 1)&(bank[:, 4] > meta['threshold']))
        a = bank[ids[np.argmax(bank[ids, 4])]]
        pp = paths+[(int(a[0]), int(a[1]))]
        xx = e.ref.project(np.r_[x, a[5], a[6], a[2]/cfg.FS*cfg.B], n)
        xx, diagnostic = seed(e, x, xx, f, y, pp, n, previous['cost'])
        print('START', tag, diagnostic, flush=True)
        start = time.perf_counter()
        xx, fit = e.fit(xx, f, y, pp, n, 'M_GATED')
        seconds = time.perf_counter()-start
        cfg.write(out/(tag+'.json'), dict(seed=diagnostic, fit=fit, paths=pp, seconds=seconds,
            iq_sha=cfg.digest(iq), source_sha=cfg.sha(src/(tag+'.json'))))
        c, g = e.objective(xx, f, y, pp)
        r, j = e.evaluate(xx, f, y, pp)
        assert np.allclose(g, j.T@r, rtol=1e-5, atol=1e-9)
        assert abs(c-fit['cost']) < 1e-9
        with np.load(INPUT/'banks'/f'{tag.split("_")[1]}_00.npz', allow_pickle=False) as z:
            truth = z['p']
        errors = stable.inside.rr.evaluation(fit['p'], truth)
        rows.append(dict(tag=tag, stationary=fit['stationary'], nonincrease=c <= previous['cost']+1e-10,
            pg=fit['projected_gradient'], iterations=fit['iterations'], seconds=seconds,
            rmse=float(np.sqrt(np.mean(np.square(errors)))), seed=diagnostic))
        cfg.write(out/'rows.json', rows)
        print('END', rows[-1], flush=True)
    stable.inside.rr.check_hashes(hashes)
    stable.inside.rr.check_hashes(cfg.read(src/'identity.json'))
    cfg.write(out/'summary.json', dict(rows=rows, complete_gate=all(r['stationary'] and r['nonincrease'] and r['seed']['accepted'] for r in rows),
        identity_unchanged=True, no_new_localization=True))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
