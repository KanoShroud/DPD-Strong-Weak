"""Oracle-subspace gate, not an observation-only DPD algorithm."""
from pathlib import Path
from datetime import datetime
import argparse
import json
import time
import itertools

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

from structural_probe import ROOT, B, C, SEED, STATIONS, LAYOUTS, response, wave, sha, write
from alternative_fit import SOURCE
from blind_initialization import GRID, peaks, errors, signature

PROTOCOL = ROOT/'联合结构准入协议.md'
UNIT = C/B
EPS = 1e-10


def observations():
    old = dict(np.load(SOURCE/'inputs.npz'))
    gen = np.random.default_rng(SEED)
    rng = np.random.default_rng(20261029071)
    f = leggauss(96)[0]*.49*B
    for li, p in enumerate(LAYOUTS):
        for rep in range(10):
            rr = gen if rep == 0 else rng
            bits = rr.choice([-1., 1.], (2, 128))
            offset = rr.uniform(0, 1/(B/1.25), 2)
            d = np.zeros((5, 2, 3))
            d[:, :, 1] = rr.uniform(3, 5, (5, 2))
            d[:, :, 2] = rr.uniform(8, 10, (5, 2))
            dist = np.linalg.norm(p[None]-STATIONS[:, None], axis=2)
            g = np.empty((5, 2, 3), complex)
            g[:, :, 0] = 1000/dist*np.exp(1j*rr.uniform(-np.pi, np.pi, (5, 2)))
            split = rr.uniform(.3, .7, (5, 2))
            g[:, :, 1] = abs(g[:, :, 0])*np.sqrt(10**(-.6)*split)*np.exp(1j*rr.uniform(-np.pi, np.pi, (5, 2)))
            g[:, :, 2] = abs(g[:, :, 0])*np.sqrt(10**(-.6)*(1-split))*np.exp(1j*rr.uniform(-np.pi, np.pi, (5, 2)))
            if rep == 0:
                assert np.array_equal(bits, old[f'layout{li}_bits'])
                assert np.array_equal(g, old[f'layout{li}_g'])
                assert np.array_equal(d, old[f'layout{li}_delta'])
            s = np.stack([wave(f, bits[q], offset[q]) for q in range(2)], axis=1)
            for paths in [1, 3]:
                gg = g.copy()
                if paths == 1:
                    gg[:, :, 1:] = 0
                h, _ = response(p, d, gg, f)
                y = np.einsum('fmq,fq->fm', h, s)
                yield dict(tag=f'L{li}_R{rep}_P{paths-1}', layout=li, realization=rep,
                           paths=paths, p=p.copy(), f=f, g=gg, d=d.copy(), s=s, h=h, y=y,
                           bits=bits, offset=offset)


def projector(h):
    u, ss, _ = np.linalg.svd(h, full_matrices=False)
    assert np.min(ss[:, 1]/ss[:, 0]) > 1e-10
    return np.eye(5)[None]-u@u.conj().transpose(0, 2, 1)


def channel(x, f):
    """40 actual coordinates: p2, LOS8, ratios20, gaps10. Reference LOS=1."""
    p = x[:2]*UNIT
    los = np.r_[1.+0j, x[2:6]+1j*x[6:10]]
    rat = x[10:20].reshape(5, 2)+1j*x[20:30].reshape(5, 2)
    delays = np.cumsum(x[30:40].reshape(5, 2), axis=1)
    dist = np.linalg.norm(p[None]-STATIONS, axis=1)
    e0 = np.exp(-2j*np.pi*f[:, None]*dist[None]/C)
    ee = np.exp(-2j*np.pi*f[:, None, None]*delays[None]/B)
    shape = 1+np.sum(ee*rat[None], axis=2)
    h = e0*los[None]*shape
    j = np.zeros((len(f), 5, 40), complex)
    direction = (p[None]-STATIONS)/np.maximum(dist[:, None], 1e-12)
    j[:, :, :2] = h[:, :, None]*(-2j*np.pi*f[:, None, None]/B)*direction[None]
    for m in range(5):
        if m:
            j[:, m, 1+m] = e0[:, m]*shape[:, m]
            j[:, m, 5+m] = 1j*j[:, m, 1+m]
        for k in range(2):
            i = 2*m+k
            j[:, m, 10+i] = e0[:, m]*los[m]*ee[:, m, k]
            j[:, m, 20+i] = 1j*j[:, m, 10+i]
            j[:, m, 30+i] = e0[:, m]*los[m]*(-2j*np.pi*f/B)*np.sum(ee[:, m, k:]*rat[m, k:][None], axis=1)
    return h, j


def score(x, f, pn):
    h, j = channel(x, f)
    r = np.einsum('fmn,fn->fm', pn, h)
    den = float(np.vdot(h, h).real)
    value = float(np.vdot(h, r).real)/den
    grad = 2*np.einsum('fmj,fm->j', j.conj(), r-value*h).real/den
    return value, grad


def constraint(x):
    return 1-EPS-np.sum(x[10:20].reshape(5, 2)**2+x[20:30].reshape(5, 2)**2, axis=1)


def constraint_jac(x):
    j = np.zeros((5, 40))
    for m in range(5):
        j[m, 10+2*m:12+2*m] = -2*x[10+2*m:12+2*m]
        j[m, 20+2*m:22+2*m] = -2*x[20+2*m:22+2*m]
    return j


def project(x):
    z = x.copy()
    z[:2] = np.clip(z[:2], -3000/UNIT, 3000/UNIT)
    z[30:] = np.clip(z[30:], 3, 20)
    rat = z[10:20].reshape(5, 2)+1j*z[20:30].reshape(5, 2)
    rat /= np.maximum(1., np.linalg.norm(rat, axis=1)/np.sqrt(1-EPS))[:, None]
    z[10:20] = rat.real.ravel()
    z[20:30] = rat.imag.ravel()
    return z


def optimize(x, f, pn, multipath):
    active = np.arange(40 if multipath else 10)
    initial = x.copy()
    bounds = [(-3000/UNIT, 3000/UNIT)]*2+[(None, None)]*28+[(3., 20.)]*10
    cache = [None, None, None]

    def unpack(z):
        xx = initial.copy()
        xx[active] = z
        return xx

    def fun(z):
        if cache[0] is None or not np.array_equal(z, cache[0]):
            val, grad = score(unpack(z), f, pn)
            cache[:] = [z.copy(), val, grad[active]]
        return cache[1], cache[2]

    cons = [] if not multipath else [dict(type='ineq', fun=lambda z: constraint(unpack(z)),
                                         jac=lambda z: constraint_jac(unpack(z))[:, active])]
    begin = time.perf_counter()
    opt = minimize(fun, initial[active], method='SLSQP', jac=True,
                   bounds=[bounds[i] for i in active], constraints=cons,
                   options=dict(maxiter=200, ftol=1e-12))
    xx = unpack(opt.x)
    val, grad = score(xx, f, pn)
    los = np.r_[1.+0j, xx[2:6]+1j*xx[6:10]]
    feasible = (constraint(xx).min()+EPS > 0 and np.min(abs(los)) > 1e-12 and
                np.max(abs(xx[:2])) <= 3000/UNIT+1e-8 and
                np.min(xx[30:]) >= 3-1e-8 and np.max(xx[30:]) <= 20+1e-8)
    # SLSQP can stop at a nonstationary point; retain that distinction.
    mapping = project(xx-grad)-xx
    pg = float(np.max(abs(mapping[active])))
    return dict(x=xx, p=xx[:2]*UNIT, value=val, feasible=bool(feasible),
                min_power_margin=float(constraint(xx).min()+EPS),
                projected_gradient=pg, stationary=pg <= 1e-7,
                scipy_success=bool(opt.success), status=int(opt.status), message=str(opt.message),
                iterations=int(opt.nit), nfev=int(opt.nfev), seconds=time.perf_counter()-begin)


def los_map(f, pn, points):
    vals = []
    for begin in range(0, len(points), 256):
        pp = points[begin:begin+256]
        dist = np.linalg.norm(pp[:, None]-STATIONS[None], axis=2)
        e = np.exp(-2j*np.pi*f[None, :, None]*dist[:, None]/C)
        a = np.einsum('bfm,fmn,bfn->bmn', e.conj(), pn, e)/len(f)
        vals.extend(np.linalg.eigvalsh(a)[:, 0].real)
    return np.asarray(vals)


def los_start(p, f, pn, gap=8.):
    e = np.exp(-2j*np.pi*f[:, None]*np.linalg.norm(p-STATIONS, axis=1)[None]/C)
    a = np.einsum('fm,fmn,fn->mn', e.conj(), pn, e)/len(f)
    _, v = np.linalg.eigh(a)
    g = v[:, 0]
    if abs(g[0]) < 1e-10:
        return None
    g /= g[0]
    return np.r_[p/UNIT, g[1:].real, g[1:].imag, np.zeros(20), np.full(10, gap)]


def truth_x(case, q):
    g = case['g'][:, q]/case['g'][0, q, 0]
    r = g[:, 1:]/g[:, :1]
    return np.r_[case['p'][q]/UNIT, g[1:, 0].real, g[1:, 0].imag,
                 r.real.ravel(), r.imag.ravel(), np.diff(case['d'][:, q], axis=1).ravel()]


def pair_score(a, b, f, pn):
    h = np.stack([channel(a['x'], f)[0], channel(b['x'], f)[0]], axis=2)
    u, s, _ = np.linalg.svd(h, full_matrices=False)
    ratio = float(np.min(s[:, 1]/s[:, 0]))
    if ratio < 1e-8:
        return None
    # Both columns must jointly span the ideal space, not merely share one source.
    cost = float(np.einsum('fmq,fmn,fnq->', u.conj(), pn, u).real/(2*len(f)))
    return cost, ratio


def earliest_geometry(arrival_b):
    """Exact necessary squared-range equations, including a common time gauge.

    Unique full-rank linear solution is checked in the unsquared equations.
    Failure is a rejection of this exact support, not of arbitrary filters.
    """
    a = arrival_b*UNIT
    station_norm = np.sum(STATIONS**2, axis=1)
    matrix = np.column_stack([-2*(STATIONS[1:]-STATIONS[0]), 2*(a[1:]-a[0])])
    rhs = a[1:]**2-a[0]**2-(station_norm[1:]-station_norm[0])
    solution, _, rank, _ = np.linalg.lstsq(matrix, rhs, rcond=1e-12)
    p, clock = solution[:2], solution[2]
    distance = np.linalg.norm(p-STATIONS, axis=1)
    err = float(np.max(abs(distance-(a-clock))))
    return dict(position=p, rank=int(rank), unsquared_max_error_m=err,
                positive_ranges=bool(np.min(a-clock) > 0),
                in_domain=bool(np.max(abs(p)) <= 3000),
                exact_feasible=bool(rank == 3 and err < 1e-6 and np.min(a-clock) > 0 and np.max(abs(p)) <= 3000))


def mix_audit(case, pn):
    h, s, f = case['h'], case['s'], case['f']
    tau = np.linalg.norm(case['p'][None]-STATIONS[:, None], axis=2)/C
    rows = []
    for deg, phi, lag in itertools.product([15, 30, 45], [0., np.pi/2], [-4., 0., 4.]):
        t = np.deg2rad(deg)
        u = np.array([[np.cos(t), -np.sin(t)*np.exp(-1j*phi)],
                      [np.sin(t)*np.exp(1j*phi), np.cos(t)]])
        transform = np.broadcast_to(u, (len(f), 2, 2)).copy()
        transform[:, 1, :] *= np.exp(-2j*np.pi*f*lag/B)[:, None]
        hm = h@transform
        sm = np.einsum('fij,fi->fj', transform.conj(), s)
        yy = np.einsum('fmq,fq->fm', hm, sm)
        # Analytic support count; exact continuous-frequency exponential model.
        counts = []
        minimum_gaps = []
        maximum_gaps = []
        first_times = []
        first_powers = [[], []]
        later_powers = [[], []]
        for m in range(5):
            terms = []
            coeffs = []
            for q in range(2):
                for k in range(case['paths']):
                    terms.append(tau[m, q]*B+case['d'][m, q, k]+(lag if q == 1 else 0))
                    coeffs.append(case['g'][m, q, k]*u[q])
            order = np.argsort(terms)
            terms = np.asarray(terms)[order]
            coeffs = np.asarray(coeffs)[order]
            assert np.min(np.diff(terms)) > 1e-8, 'Unexpected delay collision; needs explicit grouping.'
            counts.append(len(terms))
            minimum_gaps.append(float(np.min(np.diff(terms))))
            maximum_gaps.append(float(np.max(np.diff(terms))))
            first_times.append(terms[0])
            for q in range(2):
                first_powers[q].append(float(abs(coeffs[0, q])**2))
                later_powers[q].append(float(np.sum(abs(coeffs[1:, q])**2)))
        geometry = earliest_geometry(np.array(first_times))
        column_physical = []
        for q in range(2):
            dominance = bool(np.all(np.array(first_powers[q]) > np.array(later_powers[q])))
            column_physical.append(dict(dominance=dominance,
                                        all_constraints=bool(max(counts) <= 3 and min(minimum_gaps) >= 3 and max(maximum_gaps) <= 20
                                                             and geometry['exact_feasible'] and dominance)))
        rows.append(dict(angle=deg, phase=phi, lag=lag, max_path_count=max(counts),
                         min_gap=min(minimum_gaps), max_gap=max(maximum_gaps), exceeds_three_paths=max(counts) > 3,
                         earliest_geometry=geometry, columns=column_physical,
                         observation_error=float(np.linalg.norm(yy-case['y'])/np.linalg.norm(case['y'])),
                         subspace_error=float(np.linalg.norm(projector(hm)-pn)/np.linalg.norm(pn))))
    return rows


def selftest():
    case = next(observations())
    # Test derivatives away from the zero-reflection degeneracy.
    x = truth_x(case, 0)
    x[10:30] = np.random.default_rng(8831).normal(0, .1, 20)
    pn = projector(case['h'])
    value, jac = score(x, case['f'], pn)
    fd = []
    for i in range(40):
        xx, yy = x.copy(), x.copy()
        xx[i] += 1e-6
        yy[i] -= 1e-6
        fd.append((score(xx, case['f'], pn)[0]-score(yy, case['f'], pn)[0])/2e-6)
    check = dict(gradient=np.linalg.norm(fd-jac)/max(np.linalg.norm(jac), 1e-12) < 1e-6,
                 normalized_score=0 <= value <= 1,
                 true_score=abs(score(truth_x(case, 0), case['f'], pn)[0]) < 1e-12,
                 power_projection=bool(constraint(project(x)).min() > -1e-12))
    points = np.array([[143., 256.], [1234., 1456.]])
    a = los_map(case['f'], pn, points)
    b = np.array([score(los_start(p, case['f'], pn), case['f'], pn)[0] for p in points])
    check['map_equals_explicit'] = bool(np.allclose(a, b, atol=1e-12, rtol=1e-10))
    x0 = truth_x(case, 0)
    h0, _ = channel(x0, case['f'])
    check['channel_equals_generator'] = bool(np.allclose(h0, case['h'][:, :, 0]/case['g'][0, 0, 0], atol=1e-12))
    mixes = mix_audit(case, pn)
    check['mixing_invariance'] = max(r['observation_error'] for r in mixes) < 1e-12
    check['subspace_invariance'] = max(r['subspace_error'] for r in mixes) < 1e-12
    arrival = np.linalg.norm(case['p'][0]-STATIONS, axis=1)/UNIT+7.3
    geo = earliest_geometry(arrival)
    check['geometry_common_time_gauge'] = bool(geo['exact_feasible'] and np.linalg.norm(geo['position']-case['p'][0]) < 1e-6)
    swapped = case['h'][:, :, ::-1]
    check['source_permutation'] = bool(np.allclose(projector(swapped), pn, atol=1e-12))
    scaled = case['h']*np.array([2+3j, -.4+.7j])[None, None]
    check['complex_scale'] = bool(np.allclose(projector(scaled), pn, atol=1e-12))
    assert all(check.values()), check
    return check


def solve_oracle(pn, f, multipath):
    """Only oracle projector and public constants; never channel/source/truth."""
    begin = time.perf_counter()
    values = los_map(f, pn, GRID)
    candidates = peaks(-values, count=5)
    branches = []
    for i, p in enumerate(candidates):
        for gap in ([8., 12.] if multipath else [8.]):
            x = los_start(p, f, pn, gap)
            if x is None:
                branches.append(dict(candidate=i, gap=gap, excluded='reference_gauge'))
                continue
            b = optimize(x, f, pn, multipath)
            b.update(candidate=i, gap=gap, initial_p=p)
            branches.append(b)
    pairs = []
    for i, a in enumerate(branches):
        if 'x' not in a or not a['feasible']:
            continue
        for j in range(i+1, len(branches)):
            b = branches[j]
            if 'x' not in b or not b['feasible'] or np.linalg.norm(a['p']-b['p']) < 100:
                continue
            ps = pair_score(a, b, f, pn)
            if ps is not None:
                pairs.append(dict(indices=[i, j], cost=ps[0], min_rank_ratio=ps[1]))
    pairs.sort(key=lambda t: t['cost'])
    selected = pairs[0] if pairs else None
    output = None if selected is None else np.array([branches[i]['p'] for i in selected['indices']])
    return dict(candidates=candidates, branches=branches, pairs=pairs, selected=selected,
                output=output, seconds=time.perf_counter()-begin), values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=40)
    args = parser.parse_args()
    checks = selftest()
    src = SOURCE/'inputs.npz'
    assert src.resolve().is_relative_to(ROOT.resolve())
    assert sha(src) == json.loads((SOURCE/'result.json').read_text(encoding='utf-8'))['input_sha']
    files = [Path(__file__), PROTOCOL, src, ROOT/'structural_probe.py', ROOT/'blind_initialization.py', ROOT/'alternative_fit.py']
    hashes = {str(p): sha(p) for p in files}
    out = ROOT/('joint_structure_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    (out/'source').mkdir()
    for p in files:
        if p.suffix in ['.py', '.md']:
            (out/'source'/p.name).write_bytes(p.read_bytes())
    write(out/'protocol.json', dict(kind='ORACLE_SUBSPACE_NOT_BLIND_IQ', seed=20261029071,
                                  cases=40, limit=args.limit, hashes=hashes, selftest=checks))
    rows = []
    start = time.perf_counter()
    print('OUTPUT', out, flush=True)
    for index, case in enumerate(observations()):
        if index >= args.limit:
            break
        pn = projector(case['h'])
        identity = signature(case['y'], case['h'], case['s'], case['g'], case['d'])
        mixes = mix_audit(case, pn)
        # Public prior is AT MOST TWO reflections, not the realized path count.
        # The no-reflection input must not silently receive a simpler estimator.
        fitted, values = solve_oracle(pn, case['f'], True)
        local = []
        for q in range(2):
            xx = truth_x(case, q)
            trueval = score(xx, case['f'], pn)[0]
            xx[:2] += np.array([20., -10.])/UNIT
            opt = optimize(xx, case['f'], pn, True)
            opt.update(source=q, truth_score=trueval,
                       position_error=float(np.linalg.norm(opt['p']-case['p'][q])))
            local.append(opt)
        metrics = None if fitted['output'] is None else errors(fitted['output'], case['p'])
        row = dict(tag=case['tag'], layout=case['layout'], realization=case['realization'],
                   paths=case['paths'], truth=case['p'], identity=identity,
                   oracle=fitted, oracle_local=local, mixing=mixes, metrics=metrics,
                   coarse_nearest_m=[min(np.linalg.norm(p-q) for p in fitted['candidates']) for q in case['p']])
        check = dict(identity=identity == signature(case['y'], case['h'], case['s'], case['g'], case['d']),
                     true_score=max(abs(t['truth_score']) for t in local) < 1e-12,
                     mixing=max(t['observation_error'] for t in mixes) < 1e-12,
                     subspace=max(t['subspace_error'] for t in mixes) < 1e-12,
                     map_finite=bool(np.isfinite(values).all()),
                     scores_range=bool(values.min() > -1e-10 and values.max() < 1+1e-10))
        assert all(check.values()), check
        row['checks'] = check
        np.savez_compressed(out/(case['tag']+'.npz'), **{k: case[k] for k in ['p', 'f', 'g', 'd', 's', 'h', 'y', 'bits', 'offset']}, map=values)
        write(out/(case['tag']+'.json'), row)
        rows.append(row)
        print(f'{index+1}/{args.limit} {case["tag"]} RMSE={None if metrics is None else metrics["rmse_m"]} sec={fitted["seconds"]:.1f}', flush=True)
    checks.update({'unchanged/'+p: sha(p) == value for p, value in hashes.items()})
    assert all(checks.values()), checks
    write(out/'result.json', dict(rows=rows, checks=checks, seconds=time.perf_counter()-start,
                                 kind='ORACLE_SUBSPACE_NOT_BLIND_IQ', completed=len(rows)))
    print('COMPLETE', out, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
