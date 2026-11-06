"""Bounded counterfactual diagnosis; never modifies the frozen estimators."""
import argparse
import inspect
import sys
import textwrap
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
PHY = HERE.parent/'four_station_physical_20261104'
PILOT = PHY/'pilot_20261104_182457_012169'
sys.path.insert(0, str(PHY))
import scene as cfg  # noqa: E402
import adapters as ad  # noqa: E402


def errors(p, truth):
    p = np.asarray(p)
    a = np.linalg.norm(p-truth, axis=1)
    b = np.linalg.norm(p[::-1]-truth, axis=1)
    return a if a@a <= b@b else b


def rmse(p, truth):
    return float(np.sqrt(np.mean(errors(p, truth)**2)))


def five_engine():
    source = textwrap.dedent(inspect.getsource(ad.Engine.__init__))
    source = source.replace('super().__init__([1, 2, 3, 4], 0)', 'ad.b.Backend.__init__(self, range(5), 1)')
    source = source.replace('assert np.array_equal(self.stations, cfg.STATIONS)',
                            'assert np.array_equal(self.stations, np.vstack(([0., 0.], cfg.STATIONS)))')
    env = {**ad.Engine.__init__.__globals__, 'ad': ad}
    exec(compile(source, str(HERE/'five_init'), 'exec'), env)
    cls = type('CenterDiagnosticEngine', (ad.Engine,), {'__init__': env['__init__']})
    return cls()


def center_iq(bank, condition, layout):
    rng = np.random.default_rng(np.random.SeedSequence([2026110441, layout]))
    distance = np.linalg.norm(bank['p'], axis=1)
    gain = np.sqrt(bank['pt_w'])*cfg.C/(4*np.pi*cfg.FC*distance)*np.exp(1j*rng.uniform(-np.pi, np.pi, 2))
    delays = np.array([3., 7., 11., 15.])[None]+rng.uniform(0, .5, (2, 4))
    phase = np.exp(1j*rng.uniform(-np.pi, np.pi, (2, 4)))
    weight = rng.uniform(.8, 1.2, (2, 4))
    power = np.zeros_like(weight)
    ids = condition['ids']
    power[:, ids] = condition['rho']*weight[:, ids]/weight[:, ids].sum(axis=1, keepdims=True)
    assert (power < 1).all()
    t = np.arange(cfg.N)/cfg.FS
    iq = np.zeros(cfg.N, complex)
    for q in range(2):
        def signal(extra):
            return ad.wave.synthesize(bank['bits'][q], int(bank['first']), bank['offsets'][q],
                                      t-distance[q]/cfg.C-extra)/np.sqrt(bank['pulse_norm2'])
        iq += gain[q]*signal(0.)
        for k in ids:
            iq += gain[q]*phase[q, k]*np.sqrt(power[q, k])*signal(delays[q, k]/cfg.B)
    iq += np.sqrt(cfg.NOISE_VARIANCE/2)*(rng.normal(size=cfg.N)+1j*rng.normal(size=cfg.N))
    return iq


def preflight(engine5):
    rng = np.random.default_rng(10442)
    f = np.linspace(-cfg.B/2, cfg.B/2, 35)
    y = rng.normal(size=(35, 5))+1j*rng.normal(size=(35, 5))
    x = np.r_[np.array([3600., 1100., -700., -1300.])/ad.b.ref.UNIT,
              np.ones(8), np.full(8, .1), .1, .04, 3.3, -.15, .1, 8.1]
    paths = [(0, 0), (4, 1)]
    r, j = engine5.evaluate(x, f, y, paths)
    d = rng.normal(size=len(x)); d /= np.linalg.norm(d)
    step = 1e-5
    fd = (engine5.evaluate(x+step*d, f, y, paths)[0]-engine5.evaluate(x-step*d, f, y, paths)[0])/(2*step)
    err = np.linalg.norm(fd-j@d)/np.linalg.norm(j@d)
    cost, grad = engine5.objective(x, f, y, paths)
    checks = dict(stations=engine5.M == 5, pairs=len(engine5.pairs) == 10,
                  jacobian=float(err), objective=float(abs(cost-.5*r@r)),
                  gradient=float(np.linalg.norm(grad-j.T@r)),
                  boundary=bool(np.array_equal(engine5.ref.project(x, cfg.N)[:4], x[:4])))
    assert err < 3e-6 and checks['gradient'] < 1e-10 and checks['boundary']
    return checks


def offline(engine, banks):
    records = []
    for path in sorted((PILOT/'cases').iterdir()):
        for name, digest in cfg.read(path/'receipt.json').items():
            assert cfg.sha(path/name) == digest, path/name
        row = cfg.read(path/'evaluation.json')
        bank = banks[row['layout']]
        condition = next(c for c in cfg.CONDITIONS if c['name'] == row['condition'])
        iq, _ = cfg.observation(bank, condition)
        assert cfg.digest(iq) == row['iq_sha']
        front = engine.frontend(iq)
        true = bank['p']
        pool_error = np.linalg.norm(front['pool'][:, None]-true[None], axis=2).min(axis=0)
        refined_error = np.linalg.norm(front['refined'][:, None]-true[None], axis=2).min(axis=0)
        coverage = []
        for point in true:
            prediction = engine.predicted(point[None])[0][0]
            coverage.append(np.min(abs(np.asarray(prediction)[:, None]-front['lags']), axis=1))
        decision = cfg.read(path/'MP_decision.json')['result']
        final = decision['final']; x = np.array(final['x']); paths = decision['paths']
        f, y, n, _ = ad.b.fc.cal.spectral(iq)
        h = engine.core(x, f, y, paths)[0]
        colnorm = np.linalg.norm(h, axis=1)
        hn = h/colnorm[:, None, :]
        rawcond = np.linalg.cond(h); normcond = np.linalg.cond(hn)
        r, j = engine.evaluate(x, f, y, paths)
        _, grad = engine.objective(x, f, y, paths)
        pg = lambda g: float(np.max(abs(x-engine.ref.project(x-g, n))))
        stages = {m: float(np.sqrt(np.mean(np.array(row['methods'][m]['errors_m'])**2))) for m in ('GEO', 'LOS', 'MP')}
        rec = dict(tag=row['tag'], layout=row['layout'], condition=row['condition'], stages=stages,
                   pool_nearest_m=pool_error, refined_nearest_m=refined_error,
                   true_tdoa_nearest_m=coverage, los_branches=[dict(rmse=rmse(s['p'], true), cost=s['cost']) for s in decision['los_states']],
                   normal=decision['normal'], stop=decision['stop'], K=len(paths), seconds=decision['seconds'],
                   raw_cond_max=float(rawcond.max()), normalized_cond_max=float(normcond.max()),
                   relative_gain_max=float(np.max(abs(x[4:engine.D]))),
                   delay_B_max=float(max(x[engine.D+2::3], default=0)),
                   pg_jtr=pg(j.T@r), pg_envelope=pg(grad), gradient_difference=float(np.linalg.norm(grad-j.T@r)),
                   feasibility=final.get('feasible_acceptance'), source_snr_db=bank['realized_los_snr_db'])
        records.append(rec)
    return records


def jobs():
    result = []
    for i in range(4):
        for l in (2, 3):
            result.append(dict(kind='weak_reflection', layout=i, condition=f'L{l}_P050', rho=10**(-.6)))
        result.append(dict(kind='equal_tx', layout=i, condition='L2_P150'))
        for rho in ('050', '150'):
            result.append(dict(kind='add_center', layout=i, condition='L2_P'+rho))
    result.extend([dict(kind='scale', layout=0, condition='L1_P075'),
                   dict(kind='scale', layout=2, condition='L2_P100')])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=22)
    args = parser.parse_args()
    out = HERE/'run_v3'
    identities = ad.source_hashes()
    identities[str(Path(__file__).resolve())] = cfg.sha(__file__)
    identities[str(HERE/'protocol.md')] = cfg.sha(HERE/'protocol.md')
    banks = {}
    for i in range(4):
        p = PILOT/'banks'/f'{i:02d}_00.npz'
        assert cfg.sha(p) == cfg.read(p.with_suffix('.json'))['sha']
        identities[str(p)] = cfg.sha(p)
        with np.load(p, allow_pickle=False) as a:
            banks[i] = dict(a)
    if out.exists():
        assert cfg.read(out/'identity.json') == identities, 'Resume identity mismatch'
    else:
        out.mkdir()
        cfg.write(out/'identity.json', identities)
        cfg.write(out/'jobs.json', jobs())
    with threadpool_limits(limits=2):
        e4 = ad.Engine(); e5 = five_engine()
        cfg.write(out/'preflight.json', preflight(e5))
        if not (out/'offline.json').exists():
            cfg.write(out/'offline.json', offline(e4, banks))
        for idx, job in enumerate(jobs()[:args.limit]):
            dst = out/f'pair_{idx:02d}.json'
            if dst.exists():
                assert cfg.sha(dst) == cfg.read(dst.with_suffix('.sha.json'))['sha']
                continue
            bank = banks[job['layout']]
            condition = dict(next(c for c in cfg.CONDITIONS if c['name'] == job['condition']))
            if job['kind'] == 'weak_reflection':
                condition['rho'] = job['rho']
            if job['kind'] == 'equal_tx':
                bank = dict(bank)
                factor = np.sqrt(bank['pt_w'][1]/bank['pt_w'][0])
                bank['los'] = bank['los'].copy(); bank['los'][0] *= factor
                bank['reflected'] = bank['reflected'].copy(); bank['reflected'][0] *= factor
            iq, _ = cfg.observation(bank, condition)
            original_sha = cfg.digest(iq)
            engine = e4
            if job['kind'] == 'add_center':
                iq = np.vstack((center_iq(bank, condition, job['layout']), iq))
                assert cfg.digest(iq[1:]) == original_sha
                engine = e5
            elif job['kind'] == 'scale':
                iq *= 1e6
            print('START', idx, job, flush=True)
            solved = engine.mp(iq)
            # Save the estimator output before reading the truth for evaluation.
            cfg.write(out/f'decision_{idx:02d}.json', dict(job=job, iq_sha=cfg.digest(iq), result=solved))
            measured = dict(job=job, errors_m=errors(solved['final']['p'], bank['p']),
                            geo_rmse=rmse(solved['geo']['p'], bank['p']), los_rmse=rmse(solved['los']['p'], bank['p']),
                            rmse=rmse(solved['final']['p'], bank['p']), normal=solved['normal'],
                            stop=solved['stop'], seconds=solved['seconds'], K=len(solved['paths']),
                            pg=solved['final']['projected_gradient'], original_four_iq_sha=original_sha)
            cfg.write(dst, measured)
            cfg.write(dst.with_suffix('.sha.json'), dict(sha=cfg.sha(dst)))
            print('END', idx, measured['rmse'], measured['normal'], measured['seconds'], flush=True)
    assert all(cfg.sha(p) == h for p, h in identities.items())
    cfg.write(out/'postflight.json', dict(unchanged=True, completed=len(list(out.glob('pair_[0-9][0-9].json')))))


if __name__ == '__main__':
    main()
