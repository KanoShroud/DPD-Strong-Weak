"""Same objective and seeds: joint vs source-block constrained VP updates."""
import argparse
import time
import warnings

import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

from common import (HERE, B, UNIT, STATIONS, cases, preflight, load, write,
                    verify, spectra, source_indices, regularity)
from boundary_update import closure, state, step_problem, balls, ball_jac, run as frozen_run
from blind_initialization import residual, feasible, signature
from position_reduction import channel
from structural_probe import response
from blind_backend_pilot import errors


def active_state(x, f, y, ids):
    """Only requested Jacobian columns, still profile BOTH source waveforms."""
    p = x[:4].reshape(2, 2)*UNIT
    gain, delays = channel(x[4:])
    h, phase = response(p, delays, gain, f)
    uu, ss, vh = np.linalg.svd(h, full_matrices=False)
    invs = np.divide(1, ss, out=np.zeros_like(ss), where=ss > ss[:, :1]*1e-12)
    inv = np.einsum('fqi,fi,fmi->fqm', vh.conj().transpose(0, 2, 1), invs, uu.conj())
    coef = np.einsum('fqm,fm->fq', inv, y)
    res = y-np.einsum('fmq,fq->fm', h, coef)
    rho = x[20:40].reshape(5, 2, 2)+1j*x[40:60].reshape(5, 2, 2)
    los = gain[:, :, 0]
    unit = p[None]-STATIONS[:, None]
    unit /= np.linalg.norm(unit, axis=2)[:, :, None]
    dh = np.zeros((len(f), 5, 2, len(ids)), complex)
    for col, k in enumerate(ids):
        if k < 4:
            q, xy = divmod(int(k), 2)
            dh[:, :, q, col] = h[:, :, q]*(-2j*np.pi*f[:, None]/B)*unit[None, :, q, xy]
        elif k < 20:
            i = k-4 if k < 12 else k-12
            m, q = 1+i//2, i % 2
            dh[:, m, q, col] = (phase[:, m, q, 0]+phase[:, m, q, 1:]@rho[m, q])*(1 if k < 12 else 1j)
        else:
            offset = 20 if k < 40 else 40 if k < 60 else 60
            i = k-offset
            m, q, ell = i//4, (i % 4)//2, i % 2
            if k < 60:
                dh[:, m, q, col] = los[m, q]*phase[:, m, q, ell+1]*(1 if k < 40 else 1j)
            else:
                dh[:, m, q, col] = los[m, q]*(-2j*np.pi*f/B)*(phase[:, m, q, ell+1:]@rho[m, q, ell:])
    ds = np.einsum('fmqv,fq->fmv', dh, coef)
    j = -ds+np.einsum('fmq,fqv->fmv', h, np.einsum('fqm,fmv->fqv', inv, ds))
    j -= np.einsum('fqm,fqv->fmv', inv.conj(), np.einsum('fmqv,fm->fqv', dh.conj(), res))
    scale = np.sqrt(2)/np.linalg.norm(y)
    return scale*np.r_[res.real.ravel(), res.imag.ravel()], scale*np.concatenate(
        [j.real.reshape(-1, len(ids)), j.imag.reshape(-1, len(ids))])


def block_step(x, r, j, ids, damping):
    scale = np.maximum(np.linalg.norm(j, axis=0), 1e-10)
    a = j/scale
    gram = a.T@a+damping*np.eye(len(ids))
    grad = a.T@r
    lower = np.r_[np.full(4, -3000/UNIT), np.full(56, -np.inf), np.full(20, 3.)]
    upper = np.r_[np.full(4, 3000/UNIT), np.full(56, np.inf), np.full(20, 20.)]
    def unpack(z):
        xx = x.copy()
        xx[ids] += z/scale
        return xx
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', RuntimeWarning)
        opt = minimize(lambda z: .5*z@gram@z+grad@z, np.zeros(len(ids)),
                       jac=lambda z: gram@z+grad, method='SLSQP',
                       bounds=list(zip((lower[ids]-x[ids])*scale, (upper[ids]-x[ids])*scale)),
                       constraints=[dict(type='ineq', fun=lambda z: balls(unpack(z)),
                                         jac=lambda z: ball_jac(unpack(z))[:, ids]/scale)],
                       options=dict(maxiter=100, ftol=1e-10, disp=False))
    target = closure(unpack(opt.x))
    direction = target-x
    inactive = np.setdiff1d(np.arange(80), ids)
    assert np.max(abs(direction[inactive])) < 1e-12
    direction[inactive] = 0
    fallback = False
    if not np.all(np.isfinite(direction)) or (j.T@r)@direction[ids] >= 0:
        trial = x.copy()
        trial[ids] -= j.T@r
        direction = closure(trial)-x
        direction[inactive] = 0
        fallback = True
    return direction, dict(success=bool(opt.success), status=int(opt.status), iterations=int(opt.nit),
          fallback=fallback, warnings=[str(v.message) for v in caught], message=opt.message,
          closure_repair_norm=float(np.linalg.norm(target-unpack(opt.x))))


def solve(x0, f, y, method):
    x = x0.copy()
    start = time.perf_counter()
    damping = 1e-3
    events, history = [], []
    counts = dict(full_jacobians=0, block_jacobians=0, residual_calls=0, subproblem_iterations=0)
    reason = 'iteration_limit'
    for cycle in range(60):
        r, j, mapping = state(x, f, y)
        counts['full_jacobians'] += 1
        events.append(dict(seconds=time.perf_counter()-start, cycle=cycle, cost=.5*float(r@r),
                           mapping=mapping, p=x[:4].reshape(2, 2)*UNIT))
        if mapping <= 1e-7:
            reason = 'physical_projected_gradient'
            break
        groups = [np.arange(80)] if method == 'JOINT' else [source_indices(0), source_indices(1)]
        abort = False
        for bi, ids in enumerate(groups):
            if bi:
                r, ja = active_state(x, f, y, ids)
                counts['block_jacobians'] += 1
            else:
                ja = j[:, ids]
            value = .5*float(r@r)
            if method == 'JOINT':
                direction, info = step_problem(x, r, ja, damping)
            else:
                direction, info = block_step(x, r, ja, ids, damping)
            counts['subproblem_iterations'] += info['iterations']
            slope = float((ja.T@r)@direction[ids])
            row = dict(cycle=cycle, block=bi, cost=value, directional_derivative=slope, subproblem=info)
            accepted = False
            for k in range(12):
                alpha = .99*.5**k
                xx = x+alpha*direction
                pp = xx[:4].reshape(2, 2)*UNIT
                if not feasible(pp, xx[4:]):
                    continue
                rr, _ = residual(xx[4:], pp, f, y)
                counts['residual_calls'] += 1
                cost = .5*float(rr@rr)
                if cost < value and cost <= value+1e-4*alpha*slope:
                    x = xx
                    accepted = True
                    row.update(alpha=alpha, next_cost=cost)
                    events.append(dict(seconds=time.perf_counter()-start, cycle=cycle, block=bi,
                                       cost=cost, p=pp, mapping=None))
                    break
            row['accepted'] = accepted
            history.append(row)
            if accepted:
                damping = max(1e-12, damping/3)
            else:
                damping *= 10
                if damping > 1e9:
                    reason = 'no_feasible_decrease'
                    abort = True
                    break
        if abort:
            break
    r, j, mapping = state(x, f, y)
    counts['full_jacobians'] += 1
    cost = .5*float(r@r)
    events.append(dict(seconds=time.perf_counter()-start, cycle=cycle+1, cost=cost,
                       mapping=mapping, p=x[:4].reshape(2, 2)*UNIT))
    probes = []
    direction = closure(x-j.T@r)-x
    for alpha in [.5, .1, .01, .001]:
        trial = x+alpha*direction
        pp = trial[:4].reshape(2, 2)*UNIT
        okay = feasible(pp, trial[4:])
        cc = None
        if okay:
            rr, _ = residual(trial[4:], pp, f, y)
            counts['residual_calls'] += 1
            cc = .5*float(rr@rr)
        probes.append(dict(alpha=alpha, feasible=okay, cost=cc))
    seconds = time.perf_counter()-start
    return dict(method=method, p=x[:4].reshape(2, 2)*UNIT, v=x[4:], cost=cost, mapping=mapping,
                reason=reason, seconds=seconds, counts=counts, history=history, events=events,
                regularity=regularity(x, f), probes=probes,
                initial_signature=signature(x0, f, y),
                feasible_descent=any(t['feasible'] and t['cost'] < cost-1e-12 for t in probes))


def selftest(x, f, y):
    r, j, _ = state(x, f, y)
    checks = {}
    for q in range(2):
        ids = source_indices(q)
        rr, jj = active_state(x, f, y, ids)
        checks[f'block{q}/residual'] = bool(np.allclose(rr, r, atol=1e-12, rtol=1e-10))
        checks[f'block{q}/jacobian'] = bool(np.allclose(jj, j[:, ids], atol=1e-11, rtol=1e-8))
        direction, _ = block_step(x, rr, jj, ids, .001)
        inactive = np.setdiff1d(np.arange(80), ids)
        checks[f'block{q}/unchanged_other'] = np.max(abs(direction[inactive])) == 0
        checks[f'block{q}/descent_direction'] = float((jj.T@rr)@direction[ids]) <= 1e-14
    # Relabeling source columns must preserve the profiled objective and Jacobian permutation.
    perm = np.arange(80)
    a, b = source_indices(0), source_indices(1)
    perm[a], perm[b] = b, a
    swapped = x[perm]
    rs, js, _ = state(swapped, f, y)
    checks['source_swap_residual'] = bool(np.allclose(r, rs, rtol=1e-9, atol=1e-11))
    checks['source_swap_jacobian'] = bool(np.allclose(j, js[:, perm], rtol=1e-7, atol=1e-10))
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['short', 'full'], required=True)
    args = parser.parse_args()
    prerequisite = load(HERE/'task1_full_latest.json')
    assert all(load(prerequisite['path']+'/result.json')['checks'].values())
    if args.phase == 'full':
        prior = load(HERE/'task2_short_latest.json')
        assert all(load(prior['path']+'/result.json')['checks'].values())
    short = args.phase == 'short'
    out, hashes = preflight('task2_'+args.phase, short)
    checks, rows = {}, []
    begin = time.perf_counter()
    serial = 0
    for tag, inp, folder in cases(short):
        data = dict(np.load(inp))
        for path in ['LOS', 'MP']:
            record = load(folder/(tag+'_'+path+'.json'))
            f, y, _ = spectra(data, noisy=True, mp=path == 'MP')
            # Preserve the frozen generator's addition order, not merely algebraic equality.
            iq = (data['los']+(data['reflected'] if path == 'MP' else 0)).sum(axis=0)+data['noise']
            full_f = np.fft.fftfreq(8192, 1/50e6)
            y = np.fft.fft(iq, axis=1, norm='ortho')[:, abs(full_f) <= B/2].T
            # Existing initial signature uses (p,frequencies,observations).
            checks[tag+'/'+path+'/input'] = signature(np.array(record['initial']), f, y) == record['input_signature']
            assert checks[tag+'/'+path+'/input'], 'Frozen input identity mismatch: '+tag+'/'+path
            x = np.r_[np.asarray(record['los_p']).ravel()/UNIT, record['seeded_v']]
            if serial == 0:
                checks.update(selftest(x, f, y))
                assert all(checks.values()), checks
            order = ['JOINT', 'BLOCK'] if serial % 2 == 0 else ['BLOCK', 'JOINT']
            pair = {}
            for method in order:
                ans = solve(x, f, y, method)
                ans['errors_m'] = errors(ans['p'], data['p'])
                ans.update(tag=tag+'_'+path, phase='development' if tag.startswith('L2') else 'validation',
                           layout=int(tag.split('_')[0][1:]), observation=path,
                           common_frontend_seconds=record['frontend_seconds'],
                           common_los_seconds=record['los_seconds'], common_seed_seconds=record['seed_seconds'])
                pair[method] = ans
                checks[ans['tag']+'/'+method+'/feasible'] = ans['regularity']['feasible']
                checks[ans['tag']+'/'+method+'/monotone'] = all(e.get('next_cost', e['cost']) <= e['cost'] for e in ans['history'])
                checks[ans['tag']+'/'+method+'/rank'] = ans['regularity']['min_rank_ratio'] > 1e-10
                write(out/(ans['tag']+'_'+method+'.json'), ans)
                rows.append(ans)
                print(len(rows), ans['tag'], method, 'RMSE', round(float(np.sqrt(np.mean(ans['errors_m']**2))), 4),
                      ans['reason'], 's', round(ans['seconds'], 2), flush=True)
            checks[tag+'/'+path+'/same_start'] = pair['JOINT']['initial_signature'] == pair['BLOCK']['initial_signature']
            if serial == 0 and short:
                frozen = frozen_run(x[:4].reshape(2, 2)*UNIT, x[4:], f, y)
                checks['joint_frozen_p_parity'] = np.linalg.norm(pair['JOINT']['p']-frozen['p']) < 1e-9
                checks['joint_frozen_objective_parity'] = abs(pair['JOINT']['cost']-frozen['cost']) < 1e-12
                checks['joint_frozen_stop_parity'] = pair['JOINT']['reason'] == frozen['reason']
                write(out/'frozen_joint_control.json', frozen)
            serial += 1
    identities = verify(hashes)
    checks['identities'] = all(identities.values())
    write(out/'result.json', dict(rows=rows, checks=checks, identities=identities,
                                 seconds=time.perf_counter()-begin, comparisons=serial,
                                 scope='FIXED_BUDGET_SOLVER_COMPARISON_NOT_GLOBAL_OPTIMALITY'))
    write(HERE/('task2_'+args.phase+'_latest.json'), dict(path=str(out)))
    assert all(checks.values()), checks
    print('任务二完成', serial, '配对 检查', len(checks), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2), warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='divide by zero encountered in log10', category=RuntimeWarning)
        main()
