"""Frozen-start constrained GN audit; no candidate search or truth input."""
from pathlib import Path
from datetime import datetime
import argparse
import json
import time
import warnings
import numpy as np
import scipy
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
from structural_probe import ROOT, C, B, sha, write
from solver_reliability import evaluate, project
from position_reduction import position_jac
from blind_initialization import feasible, residual, errors, signature

OLD = ROOT/'blind_init_20261028_200514_640818'


def closure(x):
    v = x.copy()
    v[:4] = np.clip(v[:4], -3000/(C/B), 3000/(C/B))
    v[4:] = project(v[4:])
    return v


def balls(x):
    r = x[20:40].reshape(10, 2)
    i = x[40:60].reshape(10, 2)
    return 1-np.sum(r*r+i*i, axis=1)


def ball_jac(x):
    j = np.zeros((10, 80))
    for k in range(10):
        j[k, 20+2*k:22+2*k] = -2*x[20+2*k:22+2*k]
        j[k, 40+2*k:42+2*k] = -2*x[40+2*k:42+2*k]
    return j


def step_problem(x, r, j, damping):
    scale = np.maximum(np.linalg.norm(j, axis=0), 1e-10)
    a = j/scale
    gram = a.T@a+damping*np.eye(80)
    gradient = a.T@r
    lower = np.r_[np.full(4, -3000/(C/B)), np.full(56, -np.inf), np.full(20, 3.)]
    upper = np.r_[np.full(4, 3000/(C/B)), np.full(56, np.inf), np.full(20, 20.)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', RuntimeWarning)
        fit = minimize(lambda z: .5*z@gram@z+gradient@z, np.zeros(80),
                       jac=lambda z: gram@z+gradient, method='SLSQP',
                       bounds=list(zip((lower-x)*scale, (upper-x)*scale)),
                       constraints=[dict(type='ineq', fun=lambda z: balls(x+z/scale),
                                         jac=lambda z: ball_jac(x+z/scale)/scale)],
                       options=dict(maxiter=100, ftol=1e-10, disp=False))
    # Numerical closure repair is not a stricter physical prior. Actual
    # accepted states must be strictly feasible, via interior line search.
    target = closure(x+fit.x/scale)
    direction = target-x
    fallback = False
    if not np.all(np.isfinite(direction)) or (j.T@r)@direction >= 0:
        direction = closure(x-j.T@r)-x
        fallback = True
    return direction, dict(success=bool(fit.success), status=int(fit.status),
                           warnings=[str(w.message) for w in caught],
                           iterations=int(fit.nit), fallback=fallback,
                           closure_repair_norm=float(np.linalg.norm(target-(x+fit.x/scale))),
                           message=fit.message)


def state(x, f, y):
    p = x[:4].reshape(2, 2)*(C/B)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        r, jn, _ = evaluate(x[4:], p, f, y, 'physical')
    j = np.column_stack([position_jac(x[4:], p, f, y), jn])
    mapping = closure(x-j.T@r)-x
    return r, j, float(np.max(abs(mapping)))


def run(p, v, f, y):
    start = time.perf_counter()
    x = np.r_[p.ravel()/(C/B), v]
    assert feasible(p, v)
    initial_signature = signature(p, v, f, y)
    damping = 1e-3
    history = []
    reason = 'iteration_limit'
    for iteration in range(60):
        r, j, mapping = state(x, f, y)
        value = .5*float(r@r)
        row = dict(iteration=iteration, cost=value, mapping=mapping, positions=x[:4]*(C/B))
        history.append(row)
        if mapping <= 1e-7:
            reason = 'physical_projected_gradient'
            break
        direction, info = step_problem(x, r, j, damping)
        slope = float((j.T@r)@direction)
        row.update(subproblem=info, directional_derivative=slope)
        accepted = False
        for k in range(12):
            alpha = .99*(.5**k)
            xx = x+alpha*direction
            pp = xx[:4].reshape(2, 2)*(C/B)
            if not feasible(pp, xx[4:]):
                continue
            rr, _ = residual(xx[4:], pp, f, y)
            cost = .5*float(rr@rr)
            if cost < value and cost <= value+1e-4*alpha*slope:
                x = xx
                accepted = True
                row.update(alpha=alpha, next_cost=cost)
                break
        row['accepted'] = accepted
        if accepted:
            damping = max(1e-12, damping/3)
        else:
            damping *= 10
            if damping > 1e9:
                reason = 'no_feasible_decrease'
                break
    r, j, mapping = state(x, f, y)
    cost = .5*float(r@r)
    direction = closure(x-j.T@r)-x
    probes = []
    for alpha in [.5, .1, .01, .001]:
        xx = x+alpha*direction
        pp = xx[:4].reshape(2, 2)*(C/B)
        okay = feasible(pp, xx[4:])
        cc = None
        if okay:
            rr, _ = residual(xx[4:], pp, f, y)
            cc = .5*float(rr@rr)
        probes.append(dict(alpha=alpha, feasible=okay, cost=cc))
    return dict(p=x[:4].reshape(2, 2)*(C/B), v=x[4:], cost=cost, mapping=mapping,
                reason=reason, history=history, seconds=time.perf_counter()-start,
                feasible=feasible(x[:4].reshape(2, 2)*(C/B), x[4:]), probes=probes,
                feasible_descent=any(t['feasible'] and t['cost'] < cost-1e-12 for t in probes),
                initial_signature=initial_signature)


def selftest():
    rng = np.random.default_rng(20261028021)
    x = np.zeros(80)
    x[:4] = [1., 2., -3., -4.]  # Keep toy sources away from station zero.
    x[4:12] = 1
    x[60:] = 8
    x[20:60] = rng.normal(0, .1, 40)
    eps = 1e-6
    fd = np.column_stack([(balls(x+eps*np.eye(80)[k])-balls(x-eps*np.eye(80)[k]))/(2*eps) for k in range(80)])
    checks = dict(constraint_jacobian=bool(np.allclose(fd, ball_jac(x), rtol=1e-6, atol=1e-9)),
                  projection_idempotent=bool(np.allclose(closure(closure(x)), closure(x))))
    target = x.copy()
    target[20], target[40], target[60], target[0] = 3., 4., 1., 200.
    damping = .001
    direction, info = step_problem(x, x-target, np.eye(80), damping)
    exact = closure((target+damping*x)/(1+damping))
    checks['quadratic_ball_box_solution'] = np.linalg.norm(x+direction-exact) < 1e-5
    checks['subproblem_normal_stop'] = info['success']
    trial = x+.99*direction
    checks['interior_strict_feasibility'] = feasible(trial[:4].reshape(2, 2)*(C/B), trial[4:])
    checks['actual_descent'] = np.linalg.norm(trial-target) < np.linalg.norm(x-target)
    assert all(checks.values()), checks
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--limit', type=int, default=4)
    args = parser.parse_args()
    checks = selftest()
    if args.test_only:
        print(checks)
        return
    assert 1 <= args.limit <= 4
    old = json.loads((OLD/'result.json').read_text(encoding='utf-8'))
    oldprotocol = json.loads((OLD/'protocol.json').read_text(encoding='utf-8'))
    assert all(sha(p) == h for p, h in oldprotocol['hashes'].items())
    paths = [Path(__file__), OLD/'result.json', *[OLD/(r['case']+'_input.npz') for r in old['rows']],
             *[Path(p) for p in oldprotocol['hashes'] if p.endswith('.py')]]
    hashes = {str(p): sha(p) for p in paths}
    out = ROOT/('boundary_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    assert out.resolve().is_relative_to(ROOT.resolve()) and out.resolve() != OLD.resolve()
    (out/'source').mkdir()
    for p in paths:
        if p.suffix == '.py':
            (out/'source'/p.name).write_bytes(p.read_bytes())
    write(out/'protocol.json', dict(hashes=hashes, source=str(OLD), scipy=scipy.__version__,
        kind='FIXED_BLIND_START_CONSTRAINED_STEP_PILOT', limit=args.limit,
        changed='Constrained damped GN step with SLSQP inner solve; same original target and feasible set.',
        max_outer=60, max_inner=100, inner_ftol=1e-10, line_search_trials=12,
        armijo=1e-4, fraction_to_boundary=.99, mapping_stop=1e-7,
        open_set='Closure only used inside subproblem; accept strict original feasibility. No fixed positive DRR margin.',
        caveat='Same outer count does not mean same operations; no runtime speedup claim. No new initializations or truth-based selection.',
        source_reference='https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html'))
    rows = []
    for previous in old['rows'][:args.limit]:
        case = previous['case']
        z = np.load(OLD/(case+'_input.npz'))
        y, f = z['y'], z['f']
        assert signature(y, f) == previous['input_signature']
        results = []
        for branch in previous['branches']:
            p, v = np.array(branch['initial_p']), np.array(branch['initial_v'])
            print('START', case, branch['pair_id'], flush=True)
            new = run(p, v, f, y)
            assert signature(p, v, f, y) == new['initial_signature']
            new['pair_id'] = branch['pair_id']
            checks[case+'/'+branch['pair_id']+'/valid'] = new['feasible']
            checks[case+'/'+branch['pair_id']+'/monotone'] = all(h['next_cost'] < h['cost'] for h in new['history'] if h.get('accepted'))
            checks[case+'/'+branch['pair_id']+'/same_start'] = bool(np.isclose(new['history'][0]['cost'], branch['seed_cost'], rtol=1e-10, atol=1e-14))
            results.append(new)
            write(out/(case+'_'+branch['pair_id']+'.json'), new)
            print('DONE', case, branch['pair_id'], new['reason'], 'cost', new['cost'], 'mapping', new['mapping'], flush=True)
        chosen = min(range(len(results)), key=lambda i: results[i]['cost'])
        # The evaluator, never run(), reads positions after all outputs exist.
        from alternative_fit import SOURCE
        truth = np.load(SOURCE/'inputs.npz')['layouts'][int(case[1])]
        for r in results:
            r['evaluation'] = errors(r['p'], truth)
        rows.append(dict(case=case, branches=results, chosen=chosen,
                         evaluation=results[chosen]['evaluation'], previous=previous['evaluation']))
        print('EVALUATED', case, rows[-1]['evaluation'], flush=True)
    checks.update({'hash/'+p: sha(p) == h for p, h in hashes.items()})
    write(out/'result.json', dict(rows=rows, checks=checks, passed=all(checks.values()), complete=args.limit == 4))
    print('REPORT', out, flush=True)
    assert all(checks.values())


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
