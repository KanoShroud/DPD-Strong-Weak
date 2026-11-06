"""One-step batch versus original single-component addition, fixed pre-failure states."""
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits
from diagnose import HERE, INPUT, cfg, ad, stable, data


def main():
    source = HERE/'run_v1'
    stable.inside.rr.check_hashes(cfg.read(source/'identity.json'))
    out = HERE/'increment_control_v1'
    out.mkdir(exist_ok=False)
    hashes = ad.source_hashes()
    hashes.update({str(p.resolve()): cfg.sha(p) for p in
        [HERE/'diagnose.py', HERE/'increment_control.py', HERE/'increment_control_protocol.md']})
    cfg.write(out/'identity.json', hashes)
    rows = []
    # All four failures, plus the first normally terminated L3 as fixed positive control.
    for number, index in enumerate([100, 102, 103, 104, 101]):
        tag = f'IN_{index}_R00_L3_P150'
        old = cfg.read(source/(tag+'_DIAGNOSTIC_NEAREST.json'))['result']
        fits = [i for i, h in enumerate(old['history']) if 'fit' in h]
        which = fits[-1]
        event = old['history'][which]
        previous = old['los'] if which == 0 else old['history'][which-1]['fit']
        assert previous['stationary']
        k = event['K']-event['added']
        x = np.asarray(previous['x'])
        paths = old['paths'][:k]
        iq = data(tag)
        e = stable.Engine()
        f, y, n, sigma = ad.b.fc.cal.spectral(iq)
        use = abs(np.fft.fftfreq(n, 1/cfg.FS)) <= cfg.B/2
        _, bank, meta = e.search(x, paths, f, y, n, use, sigma, which+1)
        assert all(meta['checks'].values())
        choices = []
        for m in range(e.M):
            for q in range(2):
                ids = np.flatnonzero((bank[:, 0] == m)&(bank[:, 1] == q)&(bank[:, 3] == 1)&(bank[:, 4] > meta['threshold']))
                if len(ids):
                    choices.append(bank[ids[np.argmax(bank[ids, 4])]])
        choices = sorted(choices, key=lambda a: -a[4])[:64-k]
        assert len(choices) == event['added']
        cfg.write(out/(tag+'_start.json'), dict(previous=previous, paths=paths, choices=choices,
            search=meta, step=which+1, iq_sha=cfg.digest(iq), old_source_sha=cfg.sha(source/(tag+'_DIAGNOSTIC_NEAREST.json'))))
        record = dict(tag=tag, previous_K=k, group_added=len(choices), positive_control=index == 101)
        for arm in (['GROUP', 'SINGLE'] if number % 2 == 0 else ['SINGLE', 'GROUP']):
            cc = choices if arm == 'GROUP' else choices[:1]
            pp = paths+[(int(a[0]), int(a[1])) for a in cc]
            xx = e.ref.project(np.r_[x, np.array([[a[5], a[6], a[2]/cfg.FS*cfg.B] for a in cc]).ravel()], n)
            if e.objective(xx, f, y, pp)[0] > previous['cost']:
                xx[len(x):].reshape(-1, 3)[:, :2] = 0
            print('START', tag, arm, flush=True)
            start = time.perf_counter()
            xx, result = e.fit(xx, f, y, pp, n, 'M_GATED')
            seconds = time.perf_counter()-start
            cfg.write(out/(tag+'_'+arm+'.json'), dict(state=result, paths=pp, seconds=seconds,
                input_sha=cfg.digest(iq), argv=sys.argv, control_only=True))
            value, g = e.objective(xx, f, y, pp)
            r, j = e.evaluate(xx, f, y, pp)
            assert np.allclose(g, j.T@r, atol=1e-9, rtol=1e-5)
            assert abs(value-result['cost']) < 1e-9
            assert np.linalg.norm(xx-e.ref.project(xx, n)) < 1e-7
            if arm == 'GROUP':
                assert np.allclose(result['x'], event['fit']['x'], rtol=1e-8, atol=1e-7)
                assert result['stationary'] == event['fit']['stationary']
            with np.load(INPUT/'banks'/f'{index}_00.npz', allow_pickle=False) as z:
                truth = z['p']
            errors = stable.inside.rr.evaluation(result['p'], truth)
            record[arm] = dict(stationary=result['stationary'], cost=result['cost'],
                pg=result['projected_gradient'], iterations=result['iterations'], seconds=seconds,
                errors=errors, rmse=float(np.sqrt(np.mean(np.square(errors)))),
                nonincrease=value <= previous['cost']+1e-10, numerical_consistent=True)
            print('END', tag, arm, record[arm], flush=True)
        rows.append(record)
        cfg.write(out/'rows.json', rows)
    gate = all(r['SINGLE']['stationary'] and not r['GROUP']['stationary'] for r in rows if not r['positive_control'])
    stable.inside.rr.check_hashes(hashes)
    cfg.write(out/'summary.json', dict(rows=rows, completed=10, full_increment_comparison_gate=gate,
        identity_unchanged=True, only_one_step=True))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
