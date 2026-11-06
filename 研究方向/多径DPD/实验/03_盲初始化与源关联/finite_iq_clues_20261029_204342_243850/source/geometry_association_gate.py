"""Exhaustive ideal-lag geometry diagnostics, not a blind-IQ/EDM performance run."""
from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import itertools
import json
import time

import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'joint_structure_20261029_132004_058171'
PROTOCOL = ROOT / '几何关联核验协议.md'
C = 299792458.
B = 12.5e6
SCALE = 3000.
STATIONS = np.array([[0., 0.], [3000., 3000.], [-3000., 3000.],
                     [-3000., -3000.], [3000., -3000.]])
EXACT_M = 1e-5
DEDUP_M = 1e-3
PAIRS = list(itertools.combinations(range(5), 2))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    def default(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                              default=default, allow_nan=False), encoding='utf-8')


def signature(*arrays):
    hh = hashlib.sha256()
    for a in arrays:
        hh.update(np.ascontiguousarray(a).tobytes())
    return hh.hexdigest()


def unique_lags(u):
    # Fixed numerical resolution only; multiplicity and ordering carry no labels.
    return np.unique(np.round(np.asarray(u), 9))


def generate_lags(p, d, g):
    arrival = np.linalg.norm(STATIONS[:, None] - p[None], axis=2)[:, :, None] + C / B * d
    sets = {}
    for m, n in PAIRS:
        values = []
        for q in range(2):
            am = arrival[m, q, abs(g[m, q]) > 0]
            an = arrival[n, q, abs(g[n, q]) > 0]
            values.extend((an[None, :] - am[:, None]).ravel())
        sets[(m, n)] = unique_lags(values)
    return sets


def quadratic_roots(a, b, c):
    if abs(a) < 1e-12:
        return [] if abs(b) < 1e-12 else [-c / b]
    disc = b*b - 4*a*c
    if disc < -1e-12:
        return []
    sd = np.sqrt(max(disc, 0.))
    q = -.5 * (b + np.copysign(sd, b))
    if abs(q) < 1e-15:
        return [-b / (2*a)]
    return sorted(set([q/a, c/q]))


def solve_linear(u, stations):
    """u is distance difference to station 0, in meters. Enumerate rank-2 roots."""
    rr = (stations[1:] - stations[0]) / SCALE
    uu = u / SCALE
    aa = np.empty((len(u), len(rr), 3))
    aa[:, :, :2] = 2*rr
    aa[:, :, 2] = 2*uu
    bb = np.sum(rr*rr, axis=1)[None] - uu*uu
    left, ss, vh = np.linalg.svd(aa, full_matrices=False)
    rank = np.sum(ss > np.maximum(ss[:, :1], 1e-30)*1e-11, axis=1)
    inv = np.divide(1., ss, out=np.zeros_like(ss), where=ss > np.maximum(ss[:, :1], 1e-30)*1e-11)
    xx = np.einsum('nki,nk->ni', vh, inv*np.einsum('nmk,nm->nk', left, bb))
    orig = np.flatnonzero(rank == 3)
    candidates = list(xx[orig])
    ids = list(orig)
    for i in np.flatnonzero(rank == 2):
        if np.max(abs(aa[i]@xx[i]-bb[i])) > 1e-10:
            continue
        v, z = vh[i, -1], xx[i]
        roots = quadratic_roots(v[:2]@v[:2]-v[2]**2,
                                2*(z[:2]@v[:2]-z[2]*v[2]), z[:2]@z[:2]-z[2]**2)
        for t in roots:
            candidates.append(z+t*v)
            ids.append(i)
    xx = np.asarray(candidates, dtype=float).reshape(-1, 3)
    return (xx[:, :2]*SCALE + stations[0], xx[:, 2]*SCALE,
            np.asarray(ids, int), dict(rank3=int(sum(rank == 3)), rank2=int(sum(rank == 2)),
                                      unresolved_rank_lt2=int(sum(rank < 2))))


def edm_scores(u, alpha, stations):
    """Original centered Gram Eq.8/22, at supplied alpha (not minimized alpha)."""
    m = len(stations)
    a = np.r_[np.ones(m)/m, 0.]
    center = np.eye(m+1) - np.ones((m+1, 1))*a[None]
    dmm = cdist(stations / SCALE, stations / SCALE)**2
    values, mineig = [], []
    for start in range(0, len(u), 2048):
        uu, al = u[start:start+2048] / SCALE, alpha[start:start+2048] / SCALE
        ranges = al[:, None] + np.column_stack([np.zeros(len(uu)), uu])
        dd = np.zeros((len(uu), m+1, m+1))
        dd[:, :m, :m] = dmm
        dd[:, :m, m] = ranges*ranges
        dd[:, m, :m] = ranges*ranges
        gram = -.5 * (center @ dd @ center.T)
        eig = np.linalg.eigvalsh(gram)
        values.extend(np.sum(abs(eig[:, :-2]), axis=1))
        mineig.extend(eig[:, 0])
    return np.asarray(values), np.asarray(mineig)


def distinct_indices(p, ordered, limit=None):
    accepted = []
    for i in ordered:
        if all(np.linalg.norm(p[i]-p[j]) > DEDUP_M for j in accepted):
            accepted.append(int(i))
            if limit is not None and len(accepted) == limit:
                break
    return accepted


def associate(sets, stations):
    """No access to truth, gains, path count, source identifiers or waveform."""
    begin = time.perf_counter()
    public = [unique_lags(sets[(0, m)]) for m in range(1, len(stations))]
    # A LOS TDOA always obeys the receiver-pair baseline bound.
    filtered = [x[abs(x) <= np.linalg.norm(stations[m]-stations[0])+EXACT_M]
                for m, x in enumerate(public, 1)]
    u = np.array(list(itertools.product(*filtered)), dtype=float).reshape(-1, 4)
    if len(u) == 0:
        raise ValueError('Empty lag product: input lacks any candidate for a reference edge')
    p, alpha, ids, ranks = solve_linear(u, stations)
    assigned = u[ids]
    ranges = np.linalg.norm(p[:, None]-stations[None], axis=2)
    legal = ((abs(p).max(axis=1) <= 3000+EXACT_M) & (alpha >= -EXACT_M)
             & (alpha[:, None]+assigned >= -EXACT_M).all(axis=1))
    eqerr = np.max(abs(ranges - (alpha[:, None]+np.column_stack([np.zeros(len(alpha)), assigned]))), axis=1)
    tdoa_rms = np.sqrt(np.mean((ranges[:, 1:]-ranges[:, :1]-assigned)**2, axis=1))
    edm, mineig = edm_scores(assigned, alpha, stations)
    full = np.zeros(len(p))
    for m, n in PAIRS:
        residual = np.min(abs((ranges[:, n]-ranges[:, m])[:, None]-sets[(m, n)][None]), axis=1)
        full += residual**2
    full = np.sqrt(full/len(PAIRS))
    exact = np.flatnonzero(legal & (eqerr <= EXACT_M))
    # Round only to resolve floating-point ordering of numerical zeros.
    exact = sorted(exact, key=lambda i: (round(float(eqerr[i]), 6), *np.round(p[i], 7)))
    exact_unique = distinct_indices(p, exact)
    strict_all = [i for i in exact_unique if full[i] <= EXACT_M]
    geometric_order = np.flatnonzero(legal)
    geometric_order = geometric_order[np.argsort(tdoa_rms[geometric_order], kind='stable')]
    edm_order = np.flatnonzero(legal)
    edm_order = edm_order[np.argsort(edm[edm_order], kind='stable')]
    full_order = np.flatnonzero(legal)
    full_order = full_order[np.argsort(full[full_order], kind='stable')]
    top = {name: distinct_indices(p, order, 50)
           for name, order in [('geometric_at_linear_solution', geometric_order),
                               ('edm_at_linear_alpha', edm_order), ('all_pair_diagnostic', full_order)]}
    metadata = dict(raw_reference_sizes=[len(s) for s in public],
                    feasible_reference_sizes=[len(s) for s in filtered], combinations=len(u),
                    candidates=len(p), legal=int(sum(legal)), ranks=ranks,
                    exact=exact_unique, exact_all_pairs=strict_all, top=top,
                    seconds=time.perf_counter()-begin)
    arrays = dict(p=p, alpha=alpha, combination_ids=ids, assigned=assigned,
                  legal=legal, equation_max_m=eqerr, tdoa_rms_m=tdoa_rms,
                  edm=edm, min_gram_eig=mineig, all_pair_rms_m=full)
    return metadata, arrays


def coverage(p, truth):
    if not len(p):
        return dict(nearest_m=[None, None], both20=False)
    near = cdist(p, truth).min(axis=0)
    return dict(nearest_m=near, both20=bool(max(near) <= 20))


def evaluate(meta, arrays, truth):
    p = arrays['p']
    exact = p[meta['exact']]
    all_exact = p[meta['exact_all_pairs']]
    errors = None
    if len(exact) == 2:
        distance = cdist(exact, truth)
        a = np.diag(distance)
        b = np.diag(distance[:, ::-1])
        errors = a if a@a <= b@b else b
    top = {}
    for name, inds in meta['top'].items():
        top[name] = {str(k): coverage(p[inds[:k]], truth) for k in [2, 10, 50]}
    return dict(exact_coverage=coverage(exact, truth), exact_count=len(exact),
                exact_false20=int(sum(cdist(exact, truth).min(axis=1) > 20)) if len(exact) else 0,
                all_pairs_exact_count=len(all_exact), all_pairs_coverage=coverage(all_exact, truth),
                raw_legal_coverage=coverage(p[arrays['legal']], truth), top=top,
                two_output_errors_m=errors,
                two_output_rmse_m=None if errors is None else float(np.sqrt(np.mean(errors**2))))


def selftest():
    checks = {}
    points = np.array([[700., 500.], [-900., -650.], [2100., 1700.], [2500., 2150.],
                       [143.123, -1198.456], [0., 0.]])
    rr = cdist(points, STATIONS)
    u = rr[:, 1:]-rr[:, :1]
    pp, al, ids, _ = solve_linear(u, STATIONS)
    checks['linear_geometry_offgrid'] = bool(np.max(abs(pp-points[ids])) < 1e-7)
    checks['linear_range_offgrid'] = bool(np.max(abs(al-rr[ids, 0])) < 1e-7)
    ee, ev = edm_scores(u, rr[:, 0], STATIONS)
    checks['edm_true_zero_psd'] = bool(max(ee) < 1e-12 and min(ev) > -1e-12)
    shifted = STATIONS + [123., -67.]
    pp, _, _, _ = solve_linear(u, shifted)
    checks['coordinate_translation'] = bool(np.max(abs(pp-(points+[123., -67.]))) < 1e-7)
    checks['quadratic_two_roots'] = bool(np.allclose(quadratic_roots(1., -3., 2.), [1., 2.]))
    checks['quadratic_linear_root'] = quadratic_roots(0., 2., -4.) == [2.]
    # Collinear receivers give two mirror positions: preserve the rank-2 roots.
    line = np.column_stack([np.arange(5)*500., np.zeros(5)])
    r = cdist([[800., 400.]], line)
    ps, _, _, rank = solve_linear(r[:, 1:]-r[:, :1], line)
    checks['rank2_both_mirror_roots'] = bool(rank['rank2'] == 1 and len(ps) == 2
                                          and np.max(np.min(cdist(ps, [[800., 400.], [800., -400.]]), axis=0)) < 1e-6)
    ss = {(m, n): unique_lags([rr[0, n]-rr[0, m], rr[1, n]-rr[1, m]]) for m, n in PAIRS}
    meta, ar = associate(ss, STATIONS)
    checks['two_source_exact_no_labels'] = bool(len(meta['exact']) == 2 and
                                               coverage(ar['p'][meta['exact']], points[:2])['both20'])
    meta2, ar2 = associate({k: v[::-1] for k, v in ss.items()}, STATIONS)
    checks['lag_order_invariance'] = bool(np.array_equal(ar['p'], ar2['p']) and meta['exact'] == meta2['exact'])
    checks['time_units'] = bool(np.allclose((u/C)*C, u, atol=1e-10))
    assert all(checks.values()), checks
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=40)
    parser.add_argument('--stress', action='store_true')
    args = parser.parse_args()
    checks = selftest()
    original = json.loads((SOURCE/'result.json').read_text(encoding='utf-8'))
    entries = original['rows'][:args.limit]
    files = [SOURCE/'result.json', PROTOCOL, Path(__file__)] + [SOURCE/(r['tag']+'.npz') for r in entries]
    hashes = {str(p.resolve(strict=True)): sha(p) for p in files}
    assert all(Path(p).is_relative_to(ROOT.resolve()) for p in hashes)
    out = ROOT/('geometry_association_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    (out/'source').mkdir()
    for p in [PROTOCOL, Path(__file__)]:
        (out/'source'/p.name).write_bytes(p.read_bytes())
    write(out/'protocol.json', dict(kind='IDEAL_UNLABELLED_LAGS_NOT_BLIND_IQ', hashes=hashes,
                                   seed=20261029197, limit=args.limit, stress=args.stress,
                                   selftest=checks, numpy=np.__version__, stations=STATIONS,
                                   c=C, B=B, strict_m=EXACT_M, dedup_m=DEDUP_M,
                                   edm='exact feasibility cross-check; not optimized noisy EDM'))
    rows = []
    rng = np.random.default_rng(20261029197)
    begin = time.perf_counter()
    print('OUTPUT', out, flush=True)
    for row in entries:
        z = dict(np.load(SOURCE/(row['tag']+'.npz')))
        assert signature(z['y'], z['h'], z['s'], z['g'], z['d']) == row['identity']
        assert np.array_equal(z['p'], row['truth'])
        base = generate_lags(z['p'], z['d'], z['g'])
        assert all(np.array_equal(base[k], v) for k, v in generate_lags(z['p'][::-1], z['d'][:, ::-1], z['g'][:, ::-1]).items())
        phase_g = z['g']*np.exp(1j*np.arange(5))[:, None, None]
        assert all(np.array_equal(base[k], v) for k, v in generate_lags(z['p'], z['d'], phase_g).items())
        variants = [('exact', base)]
        if args.stress and row['realization'] == 0 and row['paths'] == 3:
            for sigma in [.01, .05]:
                variants.append((f'jitter_{sigma}', {k: unique_lags(v+rng.normal(0., sigma*C/B, len(v))) for k, v in base.items()}))
            variants.append(('drop_one', {k: np.delete(v, rng.integers(len(v))) for k, v in base.items()}))
        for mode, sets in variants:
            meta, arrays = associate(sets, STATIONS)
            metrics = evaluate(meta, arrays, z['p'])
            strict = meta['exact']
            if strict:
                assert max(arrays['edm'][strict]) < 1e-9
            tag = row['tag']+'_'+mode
            record = dict(tag=tag, layout=row['layout'], realization=row['realization'],
                          reflections=row['paths']-1, mode=mode, association=meta, metrics=metrics)
            write(out/(tag+'.json'), record)
            np.savez_compressed(out/(tag+'.npz'), **arrays, **{f'lags_{m}_{n}': sets[m, n] for m, n in PAIRS})
            rows.append(record)
            print(f'{len(rows)} {tag}: combos={meta["combinations"]} exact={len(strict)} '
                  f'false20={metrics["exact_false20"]} both20={metrics["exact_coverage"]["both20"]} '
                  f'sec={meta["seconds"]:.2f}', flush=True)
    checks.update({p: sha(p) == h for p, h in hashes.items()})
    assert all(checks.values())
    write(out/'result.json', dict(rows=rows, checks=checks, seconds=time.perf_counter()-begin,
                                 completed=len(rows), kind='IDEAL_UNLABELLED_LAGS_NOT_BLIND_IQ'))
    print('COMPLETE', out, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
