"""Scoped all-inside pilot using frozen four-station generators and solvers."""
import argparse
import shutil
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
PHY = HERE.parent/'four_station_physical_20261104'
sys.path.insert(0, str(PHY))
import scene as cfg  # noqa: E402
import adapters as ad  # noqa: E402

BOUND = 3000.
BOUND_CFG = SimpleNamespace(**{**cfg.__dict__, 'BOUND': BOUND})
rr = ad.load('inside_frozen_runner', PHY/'runner.py')


def layouts():
    templates = [([-1600, -1200], [1300, 1700]), ([-1600, 1300], [-600, -1000]),
                 ([100, -300], [2100, 1700]), ([-500, -100], [600, 400]),
                 ([2000, -1300], [2100, 1200]), ([-1800, 150], [400, -1900])]
    labels = ['异侧分布', '同半平面', '中心与边缘', '中心附近双源', '同边缘两端', '不同轴向']
    rng = np.random.default_rng(2026110451)
    out = []
    for i, template in enumerate(templates):
        p = np.array(template, float)+rng.uniform(-200, 200, (2, 2))
        p = p[rng.permutation(2)]
        assert np.max(abs(p)) <= 2400 and cfg.geometry_ok(p)
        distance = np.linalg.norm(cfg.STATIONS[:, None]-p[None], axis=-1)
        gain = (cfg.C/(4*np.pi*cfg.FC*distance))**2
        pt = 10**.6*cfg.NOISE_PSD*cfg.B/gain.mean(axis=0)
        out.append(dict(index=100+i, kind='IN', description=labels[i], p=p,
                        pt_w=pt, pt_dbm=10*np.log10(pt*1000)))
    return out


class Engine(ad.Engine):
    def __init__(self):
        super().__init__()
        self.bound_functions = [self.ref.project, self.base_fit, self.los_fit,
                                self.geometry_fit, self.triples, self.flood]
        for fn in self.bound_functions:
            assert 'BOUND' in fn.__globals__
            fn.__globals__['BOUND'] = BOUND


REJECTS = []


def bounded_resample(centers, count, rng, std):
    target = centers[np.arange(count) % 2]
    points = target+rng.normal(size=(count, 2))*std
    rejected = 0
    bad = (abs(points)>BOUND).any(axis=1)
    while bad.any():
        rejected += int(bad.sum())
        points[bad] = target[bad]+rng.normal(size=(bad.sum(), 2))*std
        bad = (abs(points)>BOUND).any(axis=1)
    REJECTS.append(rejected)
    return points


xia_design = ad.clone(ad.xia_design, bindings={'cfg': BOUND_CFG})
xia_impl = ad.clone(ad.xia_solve,
    [('points = centers[np.arange(count) % 2]+rng.normal(size=(count, 2))*config[\'std_m\']',
      "points = bounded_resample(centers, count, rng, config['std_m'])")],
    dict(cfg=BOUND_CFG, xia_design=xia_design, bounded_resample=bounded_resample))


def xia_solve(iq, seed):
    REJECTS.clear()
    result = xia_impl(iq, seed)
    result['resampling_domain_rejections'] = list(REJECTS)
    assert all(x['outside_domain_count'] == 0 for x in result['steps'])
    return result


saved_decision = ad.clone(rr.saved_decision, bindings={'ad': SimpleNamespace(xia_solve=xia_solve)})
run_case = ad.clone(rr.run_case, bindings={'saved_decision': saved_decision})


def preflight(engine):
    oldqa = cfg.read(PHY/'correctness.json')
    # The two post-run changes are documented report/release-only changes.
    for p, digest in oldqa['hashes'].items():
        if Path(p).name not in ('report.py', 'release.py'):
            assert cfg.sha(p) == digest, p
    tests = {}
    tests['stations_pairs_dimension'] = engine.M == 4 and len(engine.pairs) == 6 and engine.D == 16
    tests['all_bounds'] = all(fn.__globals__['BOUND'] == BOUND for fn in engine.bound_functions)
    x = np.r_[np.array([3400., -3200., -1200., 1400.])/ad.b.ref.UNIT, np.ones(6), np.zeros(6)]
    projected = engine.ref.project(x, cfg.N)
    tests['project_boundary'] = bool(np.allclose(projected[:2]*ad.b.ref.UNIT, [3000, -3000]))
    lags = engine.predicted(np.array([[1000., 1700.]]))[0][0]
    pool = engine.triples(np.tile(lags[:, None], (1, 6)))
    tests['analytic_true_root'] = bool(np.linalg.norm(pool-[1000, 1700], axis=1).min() < 1e-6)
    tests['candidate_bounds'] = bool(np.max(abs(pool)) <= BOUND+1e-6)
    rng = np.random.default_rng(2026110452)
    f = np.linspace(-cfg.B/2, cfg.B/2, 31)
    y = rng.normal(size=(31, 4))+1j*rng.normal(size=(31, 4))
    x = projected; x[:2] *= .5
    # Avoid the rank-changing test point of identical source gains at f=0.
    x[4:] = rng.normal(size=len(x)-4)
    r, j = engine.evaluate(x, f, y, [])
    cost, g = engine.objective(x, f, y, [])
    tests['objective_gradient'] = bool(np.isclose(cost, .5*r@r) and np.allclose(g, j.T@r, atol=1e-10))
    d = rng.normal(size=len(x)); d /= np.linalg.norm(d)
    fd = (engine.evaluate(x+1e-5*d, f, y, [])[0]-engine.evaluate(x-1e-5*d, f, y, [])[0])/2e-5
    fd_error = float(np.linalg.norm(fd-j@d)/np.linalg.norm(j@d))
    tests['jacobian'] = fd_error < 3e-6
    a = bounded_resample(np.array([[3000., 3000.], [-3000., -3000.]]), 1000, rng, 10.)
    tests['xia_truncated_gaussian'] = bool(np.max(abs(a)) <= BOUND and REJECTS[-1] > 0)
    tests['xia_coverage_rule'] = xia_design()['miss_bound'] <= .05 and xia_design()['bound'] == BOUND
    tests['generator_unchanged_search_only'] = cfg.BOUND == 4500
    assert all(tests.values()), tests
    return dict(passed=True, checks=tests, fd_relative=fd_error, xia=xia_design())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=78)
    args = ap.parse_args()
    out = HERE/'pilot_v1'
    engine = Engine()
    qa = preflight(engine)
    contract = dict(version='INSIDE_PILOT_V1', layouts=layouts(), conditions=cfg.CONDITIONS,
                    search_bound=BOUND, source_contract=cfg.read(PHY/'contract.json'),
                    note='source_contract contains historical layouts; only new six layouts used',
                    independent_worlds=6, repeats=1, expected=78, numerical_solver='unchanged_frozen')
    identity = ad.source_hashes()
    identity[str(HERE/'protocol.md')] = cfg.sha(HERE/'protocol.md')
    if out.exists():
        assert cfg.read(out/'identity.json') == identity, 'Source identity mismatch'
    else:
        out.mkdir(exist_ok=False)
        cfg.write(out/'identity.json', identity)
        cfg.write(out/'contract.json', contract)
        cfg.write(out/'preflight.json', qa)
        (out/'source').mkdir()
        for p, h in identity.items():
            shutil.copy2(p, out/'source'/(h+Path(p).suffix))
    count = 0
    start = time.perf_counter()
    with rr.lock(out), threadpool_limits(limits=2):
        try:
            for lay in contract['layouts']:
                bp = out/'banks'/f"{lay['index']:02d}_00.npz"
                if bp.exists():
                    assert cfg.sha(bp) == cfg.read(bp.with_suffix('.json'))['sha']
                    with np.load(bp, allow_pickle=False) as z:
                        bank = dict(z)
                    generation = cfg.read(bp.with_suffix('.json'))['generation_seconds']
                else:
                    t = time.perf_counter()
                    bank = cfg.make_bank(lay, 'pilot', 0, ad.wave.synthesize, ad.wave.rrc)
                    generation = time.perf_counter()-t
                    bp.parent.mkdir(exist_ok=True)
                    rr.atomic_npz(bp, **bank)
                    cfg.write(bp.with_suffix('.json'), dict(sha=cfg.sha(bp), generation_seconds=generation))
                for cond in contract['conditions']:
                    if count >= args.limit:
                        break
                    run_case(out, 'pilot', lay, 0, cond, bank, engine, count, generation/13)
                    count += 1
                    cfg.write(out/'progress.json', dict(completed=count, expected=78,
                        seconds_this_invocation=time.perf_counter()-start, last=f"{lay['index']}_{cond['name']}"))
                    print(f'PROGRESS {count}/78', flush=True)
                if count >= args.limit:
                    break
            rr.check_hashes(identity)
            cfg.write(out/'postflight.json', dict(unchanged=True, completed=count, finished=count == 78,
                                                seconds_this_invocation=time.perf_counter()-start))
        except Exception:
            cfg.write(out/f'failure_{time.time_ns()}.json', dict(traceback=traceback.format_exc(), completed=count))
            raise


if __name__ == '__main__':
    main()
