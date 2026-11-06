"""Read-only trajectory instrumentation and fixed-support numerical-position audit."""
import argparse
import ast
from datetime import datetime
import inspect
import json
from pathlib import Path
import shutil
import sys
import time
import types

import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
NAT = HERE.parent / 'natural_stop_20261104'
sys.path.insert(0, str(NAT))
import hybrid_gate as h  # noqa: E402

s, fc, ref, rg = h.s, h.fc, h.ref, h.s.rg
TOL = .005
TAGS = ['L0_R00', 'L2_R00', 'L3_R00', 'HIGH_R00', 'L3_R05', 'HIGH_R04']


def plain(path):
    return json.loads(path.read_text(encoding='utf-8'))


def max_position(dx):
    return float(np.max(np.linalg.norm(np.asarray(dx)[:4].reshape(2, 2), axis=1)) * ref.UNIT)


def clearance(x, n):
    margins = [float(np.min(3000 / ref.UNIT - abs(x[:4])))]
    if len(x) > 20:
        a = x[20:].reshape(-1, 3)
        margins += [float(np.min(1 - np.linalg.norm(a[:, :2], axis=1))),
                    float(np.min(a[:, 2])), float(np.min(n / ref.FS * ref.B - a[:, 2]))]
    return min(margins)


def gn_step(r, j):
    scale = np.maximum(np.linalg.norm(j, axis=0), 1e-15)
    z = j / scale
    # Numerical rank threshold, not a fitted regularizer; all weak modes retained above it.
    cond = np.finfo(float).eps * max(z.shape)
    dz, _, rank, sv = lstsq(z, -r, cond=cond, lapack_driver='gelsd')
    q, rr = np.linalg.qr(z[:, 4:], mode='reduced')
    u, v, _ = np.linalg.svd(rr, full_matrices=False)
    nr = int(np.sum(v > cond * v[0])) if len(v) else 0
    q = q @ u[:, :nr]
    a = z[:, :4] - q @ (q.T @ z[:, :4])
    b = r - q @ (q.T @ r)
    dp, _, pr, _ = lstsq(a, -b, cond=cond, lapack_driver='gelsd')
    return dz / scale, dict(rank=int(rank), dimension=j.shape[1], nuisance_rank=nr,
        position_rank=int(pr), scaled_condition=float(sv[0] / sv[-1]),
        schur_agreement_m=max_position(np.r_[dp / scale[:4] - dz[:4] / scale[:4], np.zeros(len(scale)-4)]))


def fd_hessian(x, f, y, paths, relative_step):
    mat = np.empty((len(x), len(x)))
    for i in range(len(x)):
        delta = relative_step * max(1., abs(x[i]))
        plus, minus = x.copy(), x.copy()
        plus[i] += delta
        minus[i] -= delta
        mat[:, i] = (s.objective(plus, f, y, paths)[1] - s.objective(minus, f, y, paths)[1]) / (2 * delta)
    symmetry = float(np.linalg.norm(mat-mat.T) / max(np.linalg.norm(mat), 1e-30))
    return (mat+mat.T)/2, symmetry


def newton_check(x, f, y, paths, n):
    start = time.perf_counter()
    r, j = s.evaluate(x, f, y, paths)
    g = s.objective(x, f, y, paths)[1]
    ha, sym = fd_hessian(x, f, y, paths, 1e-5)
    hb, sym2 = fd_hessian(x, f, y, paths, .5e-5)
    scale = np.maximum(np.linalg.norm(j, axis=0), 1e-15)
    hs = hb / scale[:, None] / scale[None, :]
    eig = np.linalg.eigvalsh(hs)
    d, _, rank, _ = lstsq(hs, -g/scale, cond=np.finfo(float).eps*len(x))
    d = d/scale
    stable = np.linalg.norm(ha-hb)/max(np.linalg.norm(hb), 1e-30)
    positive = bool(eig[0] > np.finfo(float).eps * len(x) * max(abs(eig)))
    eligible = bool(positive and rank == len(x) and clearance(x, n) > 1e-8
                    and np.linalg.norm(x+d-ref.project(x+d, n)) < 1e-8 and stable < 1e-3)
    return dict(predicted_m=max_position(d), delta=d, positive_definite=positive,
        eligible=eligible, min_eigenvalue=float(eig[0]), max_eigenvalue=float(eig[-1]),
        rank=int(rank), hessian_halving_relative=float(stable), symmetry_relative=[sym, sym2],
        residual_curvature_relative=float(np.linalg.norm(hb-j.T@j)/max(np.linalg.norm(hb), 1e-30)),
        seconds=time.perf_counter()-start)


class Capture:
    def __init__(self):
        self.records = []
        self.active = None
        self.start = 0.
        # Preserve optimizer statements; only insert observation at QN loop entry.
        tree = ast.parse(inspect.getsource(h.fit))
        loops = [a for a in ast.walk(tree) if isinstance(a, ast.For)
                 and isinstance(a.iter, ast.Call) and ast.unparse(a.iter) == 'range(300)']
        assert len(loops) == 1
        loops[0].body.insert(0, ast.parse("observe(x, 'QN')").body[0])
        ast.fix_missing_locations(tree)
        scope = {**h.scope, 'minimize': self.minimize}
        bf = types.FunctionType(fc.fit_base.__code__, scope, argdefs=fc.fit_base.__defaults__)
        scope['fit_base'] = bf
        env = {**h.fit.__globals__, 'base_fit': bf, 'observe': self.observe}
        exec(compile(tree, str(HERE / 'instrumented_fit'), 'exec'), env)
        self.inner = env['fit']

    def observe(self, x, phase):
        if self.active is None:
            return
        points = self.active['points']
        if not points or not np.array_equal(points[-1]['x'], x):
            points.append(dict(x=np.asarray(x).copy(), phase=phase,
                               elapsed=time.perf_counter()-self.start))

    def minimize(self, fun, x0, *args, **kwargs):
        unpack = inspect.getclosurevars(fun).nonlocals['unpack']
        assert 'callback' not in kwargs
        kwargs['callback'] = lambda z: self.observe(unpack(z), 'SLSQP')
        return minimize(fun, x0, *args, **kwargs)

    def fit(self, x0, f, y, paths, n, method='HYBRID'):
        self.start = time.perf_counter()
        self.active = dict(paths=list(paths), points=[])
        self.observe(ref.project(x0, n), 'initial')
        x, state = self.inner(x0, f, y, paths, n, method)
        self.observe(x, 'terminal')
        self.active['terminal'] = state
        self.records.append(self.active)
        self.active = None
        return x, state


def unit_checks():
    rng = np.random.default_rng(42)
    j = rng.normal(size=(200, 15))
    d0 = rng.normal(size=15) * .01
    step, meta = gn_step(-j @ d0, j)
    checks = dict(linear_full_step=bool(np.allclose(step, d0, atol=1e-12)),
                  linear_profile_agreement=meta['schur_agreement_m'] < 1e-10)
    # Explicit nonlinear residual: Hessian of .5*(x^2-a)^2 is 6*x^2-2*a.
    x, a = .8, .5
    eps = 1e-5
    g = lambda t: 2*t*(t*t-a)
    hh = (g(x+eps)-g(x-eps))/(2*eps)
    checks['nonzero_residual_hessian'] = bool(abs(hh-(6*x*x-2*a)) < 1e-8)
    checks['nonzero_not_GN'] = bool(abs(hh-4*x*x) > .1)
    checks['zero_residual_GN'] = bool(abs((6*a-2*a)-4*a) < 1e-12)
    return checks


def trace(out, tags):
    rows = []
    for tag in tags:
        print('TRACE', tag, flush=True)
        cap = Capture()
        estimate = types.FunctionType(s.estimate.__code__, {**s.estimate.__globals__, 'fit': cap.fit},
                                      argdefs=s.estimate.__defaults__)
        result = estimate(rg.inputs(tag), 'HYBRID', True, True)
        olddir = 'pilot_20261104_135733_008285' if tag.endswith('R00') else 'followup_20261104_140420_955153'
        oldpath = NAT / olddir / (tag+'_HYBRID.json')
        old = plain(oldpath)
        checks = h.runner.audit(result, rg.inputs(tag))
        checks.update(final_x=bool(np.array_equal(result['final']['x'], old['final']['x'])),
                      same_paths=result['paths'] == [tuple(p) for p in old['paths']],
                      same_stop=result['stop'] == old['stop'])
        fc.write(out/(tag+'_trace.json'), dict(result=result, records=cap.records, checks=checks,
                                               reference_path=str(oldpath)))
        rows.append(dict(case=tag, supports=len(cap.records), points=sum(len(r['points']) for r in cap.records),
                         audit=all(checks.values())))
        fc.write(out/'trace_summary.json', dict(rows=rows, complete=False))
        print(rows[-1], checks, flush=True)
        assert all(checks.values()), checks
    fc.write(out/'trace_summary.json', dict(rows=rows, complete=True))


def diagnose(out, source, tags):
    rows, checks = [], unit_checks()
    for tag in tags:
        f, y, n, _ = fc.cal.spectral(rg.inputs(tag))
        saved = plain(source/(tag+'_trace.json'))
        for model, rec in enumerate(saved['records']):
            paths = [tuple(p) for p in rec['paths']]
            terminal = rec['terminal']
            xt = np.asarray(terminal['x'])
            if not terminal['stationary']:
                rows.append(dict(case=tag, model=model, K=len(paths), valid_reference=False))
                continue
            chosen = {}
            for index, point in enumerate(rec['points']):
                x = np.asarray(point['x'])
                _, g = s.objective(x, f, y, paths)
                pg = float(np.max(abs(x-ref.project(x-g, n))))
                for threshold in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
                    if pg <= threshold and threshold not in chosen:
                        chosen[threshold] = (index, pg)
            selected = sorted(set(chosen.values()))
            local = []
            for index, pg in selected:
                point = rec['points'][index]
                x = np.asarray(point['x'])
                start = time.perf_counter()
                r, j = s.evaluate(x, f, y, paths)
                delta, meta = gn_step(r, j)
                min_margin = clearance(x, n)
                eligible = bool(meta['rank'] == len(x) and min_margin > 1e-8
                                and np.linalg.norm(x+delta-ref.project(x+delta, n)) < 1e-8)
                predicted = max_position(delta)
                actual = max_position(xt-x)
                row = dict(case=tag, model=model, K=len(paths), index=index, PG=pg,
                    valid_reference=True, predicted_m=predicted, actual_continuation_m=actual,
                    eligible=eligible, declares_safe=bool(eligible and predicted <= TOL),
                    false_safe=bool(eligible and predicted <= TOL and actual > TOL),
                    margin=min_margin, meta=meta, seconds=time.perf_counter()-start,
                    observed_remaining_seconds=max(0., rec['points'][-1]['elapsed']-point['elapsed']),
                    remaining_callbacks=len(rec['points'])-index-1)
                local.append(row)
            if local:
                pick = next((v for v in local if v['declares_safe']), local[-1])
                x = np.asarray(rec['points'][pick['index']]['x'])
                nr = newton_check(x, f, y, paths, n)
                nr['declares_safe'] = bool(nr['eligible'] and nr['predicted_m'] <= TOL)
                nr['false_safe'] = bool(nr['declares_safe'] and pick['actual_continuation_m'] > TOL)
                pick['newton'] = nr
                # Independent old formula checks at the same preselected point.
                rr, jj = ref.evaluate(x, f, y, paths)
                r, j = s.evaluate(x, f, y, paths)
                checks[f'{tag}/{model}/r'] = bool(np.allclose(r, rr, rtol=1e-9, atol=1e-11))
                checks[f'{tag}/{model}/J'] = bool(np.allclose(j, jj, rtol=1e-7, atol=1e-10))
                checks[f'{tag}/{model}/g'] = bool(np.allclose(s.objective(x, f, y, paths)[1], jj.T@rr, rtol=1e-7, atol=1e-10))
                checks[f'{tag}/{model}/profile'] = all(v['meta']['schur_agreement_m'] < 1e-5 for v in local)
                print(tag, model, len(paths), 'GN safe/false', sum(v['declares_safe'] for v in local),
                      sum(v['false_safe'] for v in local), 'Newton', nr['eligible'], nr['predicted_m'], flush=True)
            rows += local
            fc.write(out/'diagnostics.json', dict(rows=rows, checks=checks, complete=False))
    fc.write(out/'diagnostics.json', dict(rows=rows, checks=checks, complete=True))
    assert all(checks.values()), {k:v for k,v in checks.items() if not v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['unit', 'trace', 'diagnose'], required=True)
    ap.add_argument('--source', type=Path)
    ap.add_argument('--tags', nargs='+', default=TAGS)
    args = ap.parse_args()
    out = HERE/(args.mode+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    print('结果目录', out, flush=True)
    hashes = rg.preflight(out, args.tags)
    extra = list(HERE.glob('*.py'))+list(HERE.glob('*.md'))+list(NAT.glob('*.py'))
    for tag in args.tags:
        olddir = 'pilot_20261104_135733_008285' if tag.endswith('R00') else 'followup_20261104_140420_955153'
        extra.append(NAT/olddir/(tag+'_HYBRID.json'))
        if args.source:
            extra.append(args.source/(tag+'_trace.json'))
    hashes.update({str(p.resolve(strict=True)): ref.sha(p) for p in extra})
    snap = out/'source'
    snap.mkdir()
    for i, p in enumerate(hashes):
        if Path(p).suffix in ('.py', '.md'):
            shutil.copy2(p, snap/f'{i:03d}_{Path(p).name}')
    fc.write(out/'identity.json', dict(hashes=hashes, command=sys.argv, tolerance_m=TOL))
    try:
        ck = unit_checks()
        fc.write(out/'unit.json', ck)
        assert all(ck.values())
        if args.mode == 'trace':
            trace(out, args.tags)
        elif args.mode == 'diagnose':
            assert args.source is not None
            diagnose(out, args.source.resolve(strict=True), args.tags)
    finally:
        unchanged = all(ref.sha(p) == value for p, value in hashes.items())
        fc.write(out/'postflight.json', dict(unchanged=unchanged))
        assert unchanged


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
