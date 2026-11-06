"""Bounded conventional conditional rescoring; not full unknown-gain AP."""
from pathlib import Path
from datetime import datetime
import argparse
import json
import time
import warnings
import numpy as np
from threadpoolctl import threadpool_limits
from structural_probe import ROOT, STATIONS, C, B, sha, write
from blind_initialization import (GRID, rank_one, score_positions, peaks, los_vector,
                                 local_los, seed_paths, residual, errors, signature)
from boundary_update import run as backend
from first_refinement import OLD, BASE
from alternative_fit import SOURCE
from position_reduction import position_jac
from solver_reliability import evaluate

PROTOCOL = ROOT/'条件评分一致性核验协议.md'


def response_los(p, gain, f):
    return np.exp(-2j*np.pi*f[:, None]*np.linalg.norm(p-STATIONS, axis=1)[None]/C)*gain


def conditional_map(y, f, p0, g0, points):
    """Keep original residual rank-one gain proposal; only change its score."""
    h0 = response_los(p0, g0, f)
    h0 /= np.linalg.norm(h0, axis=1)[:, None]
    r = y-h0*np.sum(h0.conj()*y, axis=1)[:, None]
    new, old, mins = [], [], []
    for start in range(0, len(points), 256):
        pp = points[start:start+256]
        dist = np.linalg.norm(pp[:, None]-STATIONS[None], axis=2)
        phase = np.exp(-2j*np.pi*f[None, :, None]*dist[:, None]/C)
        z = phase.conj()*r[None]
        cov = np.einsum('bfm,bfn->bmn', z, z.conj())
        vals, vec = np.linalg.eigh(cov)
        gain = vec[:, :, -1]
        h1 = phase*gain[:, None]
        projected = h1-h0[None]*np.sum(h0.conj()[None]*h1, axis=2)[:, :, None]
        denom = np.sum(abs(projected)**2, axis=2)
        # Use the projected vector also in the numerator to avoid cancellation.
        numer = abs(np.sum(projected.conj()*r[None], axis=2))**2
        term = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 1e-24)
        new.extend(term.sum(axis=1)); old.extend(vals[:, -1].real)
        mins.extend(denom.min(axis=1))
    return np.array(new), np.array(old), np.array(mins)


def direct_check(y, f, p0, g0, p1):
    h0 = response_los(p0, g0, f)
    r = y-h0*(np.sum(h0.conj()*y, axis=1)/np.sum(abs(h0)**2, axis=1))[:, None]
    g1, _, _ = rank_one(r, f, p1)
    h1 = response_los(p1, g1, f)
    projected = h1-h0*(np.sum(h0.conj()*h1, axis=1)/np.sum(abs(h0)**2, axis=1))[:, None]
    denom = np.sum(abs(projected)**2, axis=1)
    numer = abs(np.sum(projected.conj()*r, axis=1))**2
    gain = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 1e-24).sum()
    direct = []
    for a, b, yy in zip(h0, h1, y):
        h = np.column_stack([a, b])
        direct.append(yy-h@np.linalg.lstsq(h, yy, rcond=1e-12)[0])
    direct_cost = np.linalg.norm(direct)**2
    predicted = np.linalg.norm(r)**2-gain
    return dict(error=abs(direct_cost-predicted)/np.linalg.norm(y)**2,
                conditional_gain=float(gain), direct_cost=float(direct_cost),
                min_projected_norm_fraction=float(np.min(denom/np.sum(abs(h1)**2, axis=1))))


def selftest():
    rng = np.random.default_rng(2026102901)
    f = np.linspace(-.49*B, .49*B, 57)
    y = rng.normal(size=(57, 5))+1j*rng.normal(size=(57, 5))
    p0 = np.array([432., -765.]); p1 = np.array([-679., 876.])
    g0, _, fit = rank_one(y, f, p0)
    c, s, _ = conditional_map(y, f, p0, g0, p1[None])
    d = direct_check(y, f, p0, g0, p1)
    checks = dict(direct_LS=d['error'] < 1e-12,
                  map_vs_direct=np.isclose(c[0], d['conditional_gain'], rtol=1e-12),
                  old_score=np.allclose(s, score_positions(y-fit, f, p1[None]), rtol=1e-12),
                  conditional_not_smaller=bool(c[0] >= s[0]-1e-12))
    # Independent full 20-coordinate LOS Jacobian check, not just position.
    g1, _, _ = rank_one(y-fit, f, p1)
    v = los_vector(np.column_stack([g0, g1])); p = np.array([p0, p1])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        _, j, _ = evaluate(v, p, f, y, 'physical')
    analytic = np.column_stack([position_jac(v, p, f, y), j[:, :16]])
    numeric = []
    for k in range(20):
        eps = 1e-6
        pa, pb, va, vb = p.copy(), p.copy(), v.copy(), v.copy()
        if k < 4:
            pa.flat[k] += eps*(C/B); pb.flat[k] -= eps*(C/B)
        else:
            va[k-4] += eps; vb[k-4] -= eps
        numeric.append((residual(va, pa, f, y)[0]-residual(vb, pb, f, y)[0])/(2*eps))
    checks['LOS_all20_J'] = np.linalg.norm(np.column_stack(numeric)-analytic)/np.linalg.norm(analytic) < 1e-6
    assert all(checks.values()), checks
    return {k: bool(v) for k, v in checks.items()}


def solve(y, f, first):
    identity = signature(y, f, first); begin = time.perf_counter()
    pairs, maps, checks = [], [], {}
    for i, p0 in enumerate(first):
        g0, _, fit = rank_one(y, f, p0); rest = y-fit
        new, old, norm = conditional_map(y, f, p0, g0, GRID)
        checks[f'{i}/old_replay'] = bool(np.allclose(old, score_positions(rest, f, GRID), rtol=1e-11, atol=1e-12))
        second = peaks(new, exclude=[p0])
        maps.append(dict(index=i, first=p0, second=second, min_projected_norm_fraction=float(norm.min())))
        for k, p1 in enumerate(second):
            g1, _, _ = rank_one(rest, f, p1)
            p = np.array([p0, p1]); v = los_vector(np.column_stack([g0, g1]))
            diag = direct_check(y, f, p0, g0, p1)
            checks[f'{i}_{k}/joint_identity'] = diag['error'] < 1e-12
            pp, vv, meta = local_los(p, v, f, y)
            pairs.append(dict(id=f'{i}_{k}', coarse=p, los_p=pp, los_v=vv, los=meta, identity=diag))
    order = sorted(range(9), key=lambda i: pairs[i]['los']['cost'])[:3]
    branches = []
    for index in order:
        item = pairs[index]; p, v = item['los_p'], item['los_v']
        vv, seed = seed_paths(p, v, f, y)
        r, _ = residual(vv, p, f, y)
        final = backend(p, vv, f, y)
        branches.append(dict(pair_id=item['id'], initial_p=p, initial_v=vv, seed=seed,
                             seed_cost=.5*float(r@r), final=final))
    chosen = min(range(3), key=lambda i: branches[i]['final']['cost'])
    checks['input_unchanged'] = signature(y, f, first) == identity
    return dict(input_signature=identity, first=maps, pairs=pairs, branches=branches,
                chosen=chosen, output=branches[chosen]['final']['p'], checks=checks,
                seconds=time.perf_counter()-begin)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--limit', type=int, default=4)
    ap.add_argument('--test-only', action='store_true'); args = ap.parse_args()
    checks = selftest()
    if args.test_only:
        print(checks); return
    assert 1 <= args.limit <= 4
    old = json.loads((OLD/'result.json').read_text(encoding='utf-8'))
    baseline = json.loads((BASE/'result.json').read_text(encoding='utf-8'))
    protocols = [json.loads((d/'protocol.json').read_text(encoding='utf-8')) for d in [OLD, BASE]]
    for protocol in protocols:
        assert all(sha(p) == h for p, h in protocol['hashes'].items())
    paths = [Path(__file__), PROTOCOL, OLD/'result.json', BASE/'result.json', SOURCE/'inputs.npz',
             *[OLD/(r['case']+'_input.npz') for r in old['rows']],
             *[Path(p) for p in protocols[1]['hashes'] if p.endswith('.py')], ROOT/'first_refinement.py']
    assert all(p.resolve(strict=True).is_relative_to(ROOT.resolve()) for p in paths)
    hashes = {str(p): sha(p) for p in paths}
    out = ROOT/('conditional_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False); (out/'source').mkdir()
    for p in paths:
        if p.suffix in ['.py', '.md']: (out/'source'/p.name).write_bytes(p.read_bytes())
    write(out/'protocol.json', dict(hashes=hashes, limit=args.limit, input_root=str(OLD),
        baseline=str(BASE), scope='Fixed original gain proposals; conditional re-score only, not fully optimized conditional ML.',
        no_truth_in_solve=True, timing='Includes old-map replay diagnostic overhead; not a speed benchmark.'))
    rows = []
    for previous, control in zip(old['rows'][:args.limit], baseline['rows']):
        key = previous['case']; z = np.load(OLD/(key+'_input.npz')); y, f = z['y'], z['f']
        assert signature(y, f) == previous['input_signature']
        first = np.array([next(p for p in previous['pairs'] if p['id'] == f'{i}_0')['coarse'][0] for i in range(3)])
        checks[key+'/first'] = np.array_equal(first, np.array(peaks(score_positions(y, f, GRID))))
        print('START', key, flush=True); row = solve(y, f, first)
        truth = np.load(SOURCE/'inputs.npz')['layouts'][int(key[1])]
        row.update(case=key, evaluation=errors(row['output'], truth), control_evaluation=control['evaluation'])
        for pair in row['pairs']: pair['evaluation'] = errors(pair['los_p'], truth)
        for b in row['branches']:
            b['evaluation'] = errors(b['final']['p'], truth)
            checks[key+'/'+b['pair_id']+'/feasible'] = b['final']['feasible']
            checks[key+'/'+b['pair_id']+'/descent'] = all(h['next_cost'] < h['cost'] for h in b['final']['history'] if h.get('accepted'))
        checks.update({key+'/'+k: v for k, v in row['checks'].items()})
        rows.append(row); write(out/(key+'.json'), row)
        print('DONE', key, control['evaluation']['rmse_m'], row['evaluation']['rmse_m'], flush=True)
    checks.update({'hash/'+p: sha(p) == h for p, h in hashes.items()})
    write(out/'result.json', dict(rows=rows, checks=checks, passed=bool(all(checks.values())), complete=args.limit == 4))
    print('REPORT', out, flush=True)
    assert all(checks.values())


if __name__ == '__main__':
    with threadpool_limits(limits=2): main()
