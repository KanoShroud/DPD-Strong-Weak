"""Auditable exact-kernel and full-IQ solver gates; no time-based termination."""
import argparse
from datetime import datetime
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits
import solver as s

fc, rg, ref, HERE = s.fc, s.rg, s.ref, s.HERE


def audit(result, iq):
    f, y, n, _ = fc.cal.spectral(iq)
    checks = {}
    count = 0
    pp = result['paths']
    for state, paths in [(result['los'], [])]+[(h['fit'], pp[:h['K']]) for h in result['history'] if 'fit' in h]:
        x = np.array(state['x'])
        r, j = ref.evaluate(x, f, y, paths)
        pg = float(np.max(abs(x-ref.project(x-j.T@r, n))))
        checks[f'{count}/cost'] = bool(abs(.5*r@r-state['cost']) < 1e-10)
        checks[f'{count}/PG'] = bool(abs(pg-state['projected_gradient']) < 1e-9)
        checks[f'{count}/feasible'] = bool(np.linalg.norm(x-ref.project(x, n)) < 1e-7)
        count += 1
    checks['monotone'] = all(h.get('decrease', True) for h in result['history'])
    return checks


def kernel(out, tags):
    checks, rows = {}, []
    for tag in tags:
        iq = rg.inputs(tag)
        f, y, n, sigma = fc.cal.spectral(iq)
        old = fc.load(rg.FORMAL/'cases'/tag/'solve.json')
        x, pp = np.array(old['final']['x']), [tuple(p) for p in old['paths']]
        x[:4] += [.01, -.02, .015, .01]
        r, j = ref.evaluate(x, f, y, pp)
        rr, jj = s.evaluate(x, f, y, pp)
        cost, g = s.objective(x, f, y, pp)
        checks[tag+'/residual'] = bool(np.allclose(r, rr, rtol=1e-9, atol=1e-11))
        checks[tag+'/jacobian'] = bool(np.allclose(j, jj, rtol=1e-7, atol=1e-10))
        checks[tag+'/gradient'] = bool(np.allclose(g, j.T@r, rtol=1e-7, atol=1e-10))
        checks[tag+'/cost'] = bool(abs(cost-.5*r@r) < 1e-11)
        direction = np.random.default_rng(441).normal(size=len(x)); direction /= np.linalg.norm(direction)
        eps = 1e-5
        fd = (s.objective(x+eps*direction, f, y, pp)[0]-s.objective(x-eps*direction, f, y, pp)[0])/(2*eps)
        checks[tag+'/finite_difference'] = bool(np.isclose(fd, g@direction, rtol=2e-4, atol=2e-8))
        use = abs(np.fft.fftfreq(n, 1/ref.FS)) <= ref.B/2
        _, b1, m1 = fc.search(x, pp, f, y, n, use, sigma, 1)
        _, b2, m2 = s.search(x, pp, f, y, n, use, sigma, 1)
        checks[tag+'/search_rank'] = m1['rank'] == m2['rank']
        checks[tag+'/search_valid'] = bool(np.array_equal(b1[:, 3], b2[:, 3]))
        ids = b1[:, 3] == 1
        checks[tag+'/search_scores'] = bool(np.allclose(b1[ids, 4], b2[ids, 4], rtol=1e-5, atol=1e-5))
        times = {}
        for name, fun in [('previous_objective', fc.objective), ('guarded_objective', s.objective), ('previous_jacobian', fc.evaluate), ('guarded_jacobian', s.evaluate)]:
            tt = []
            for _ in range(5):
                start = time.perf_counter(); fun(x, f, y, pp); tt.append(time.perf_counter()-start)
            times[name] = float(np.median(tt))
        rows.append(dict(case=tag, times=times))
        print(tag, times, flush=True)
    rng = np.random.default_rng(42)
    h = rng.normal(size=(30, 5, 2))+1j*rng.normal(size=(30, 5, 2))
    h[10:, :, 1] = h[10:, :, 0]+1e-8*h[10:, :, 1]
    h[20:, :, 1] = h[20:, :, 0]
    checks['ill_conditioned_fallback'] = bool(np.allclose(s.GuardedLinalg().pinv(h), np.linalg.pinv(h, rcond=1e-12)))
    fc.write(out/'kernel.json', dict(checks=checks, rows=rows, all_pass=all(checks.values())))
    assert all(checks.values()), [k for k, v in checks.items() if not v]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['kernel', 'pilot', 'followup'], required=True)
    ap.add_argument('--methods', nargs='+', default=['GN', 'QN'])
    ap.add_argument('--two-regions', action='store_true')
    ap.add_argument('--tags', nargs='+')
    args = ap.parse_args()
    tags = args.tags or ([f'{c}_R{i:02d}' for i in range(1, 8) for c in ['L0', 'L2', 'L3', 'HIGH']] if args.mode == 'followup' else [f'{c}_R00' for c in ['L0', 'L2', 'L3', 'HIGH']])
    out = HERE/(args.mode+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    print('结果目录', out, flush=True)
    hashes = rg.preflight(out, tags)
    for p in list(HERE.glob('*.py'))+list(HERE.glob('*.md')):
        hashes[str(p.resolve())] = ref.sha(p)
    snap = out/'source'; snap.mkdir()
    for i, p in enumerate(hashes):
        if Path(p).suffix in ('.py', '.md'):
            shutil.copy2(p, snap/f'{i:03d}_{Path(p).name}')
    fc.write(out/'identity.json', dict(hashes=hashes, args=vars(args), command=sys.argv))
    rows = []
    try:
        if args.mode == 'kernel':
            kernel(out, tags)
        else:
            contract = fc.load(rg.FORMAL/'contract.json')['contract']
            for i, tag in enumerate(tags):
                iq = rg.inputs(tag)
                for method in (args.methods if i % 2 == 0 else args.methods[::-1]):
                    print('START', tag, method, flush=True)
                    result = s.estimate(iq, method, args.two_regions)
                    fc.write(out/f'{tag}_{method}.json', result)
                    check = audit(result, iq)
                    fc.write(out/f'{tag}_{method}_audit.json', check)
                    with np.load(contract['conditions'][tag.split('_')[0]]['bank'], allow_pickle=False) as z:
                        truth = z['p']
                    error = np.asarray(ref.error(result['final']['p'], truth))
                    row = dict(case=tag, method=method, seconds=result['seconds'], front_seconds=result['frontend_seconds'],
                        rmse_m=float(np.sqrt(np.mean(error**2))), errors_m=error, K=len(result['paths']),
                        PG=result['final']['projected_gradient'], J=result['final']['cost'], normal=result['normal'],
                        stop=result['stop'], audit=all(check.values()),
                        fit_seconds=sum(h['fit']['seconds'] for h in result['history'] if 'fit' in h),
                        search_seconds=sum(h['search']['seconds'] for h in result['history']))
                    rows.append(row)
                    fc.write(out/'results.json', dict(rows=rows, complete=False))
                    print(row, flush=True)
                    assert all(check.values()), check
            fc.write(out/'results.json', dict(rows=rows, complete=True))
    finally:
        unchanged = all(ref.sha(p) == h for p, h in hashes.items())
        fc.write(out/'postflight.json', dict(unchanged=unchanged))
        assert unchanged


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
