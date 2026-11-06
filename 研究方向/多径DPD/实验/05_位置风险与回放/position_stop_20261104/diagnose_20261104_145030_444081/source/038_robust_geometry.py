"""Frozen observed-lag geometry pilot; no truth in estimator interfaces."""
from pathlib import Path
from datetime import datetime
from itertools import combinations, product
import argparse
import json
import time

import numpy as np
from scipy.optimize import minimize_scalar, linear_sum_assignment
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from geometry_association_gate import C, B, STATIONS, PAIRS, sha, write

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'finite_iq_clues_20261029_204430_528859'
PROTOCOL = ROOT/'容错几何关联协议.md'
SCALE = 3000.
RADIUS = C/B


def predicted(p, stations=STATIONS):
    delta = p[:, None, :]-stations[None]
    ranges = np.linalg.norm(delta, axis=2)
    unit = delta/np.maximum(ranges[..., None], 1e-12)
    ii, jj = np.array(PAIRS).T
    return ranges[:, jj]-ranges[:, ii], unit[:, jj]-unit[:, ii]


def triple_candidates(lags, stations=STATIONS):
    result = []
    for a, b, c in combinations(range(5), 3):
        rr = (stations[[b, c]]-stations[a])/SCALE
        if abs(np.linalg.det(rr)) < 1e-10:
            continue
        u = np.array(list(product(lags[PAIRS.index((a, b))], lags[PAIRS.index((a, c))])))/SCALE
        inverse = np.linalg.inv(2*rr)
        x0 = (.5*0+np.sum(rr*rr, axis=1)[None]-u*u)@inverse.T
        v = (-2*u)@inverse.T
        aa = np.sum(v*v, axis=1)-1
        bb = 2*np.sum(x0*v, axis=1)
        cc = np.sum(x0*x0, axis=1)
        for i in range(len(u)):
            from geometry_association_gate import quadratic_roots
            for alpha in quadratic_roots(aa[i], bb[i], cc[i]):
                if alpha < 0 or np.min(alpha+u[i]) < 0:
                    continue
                p = (x0[i]+alpha*v[i])*SCALE+stations[a]
                if max(abs(p)) > SCALE+1e-6:
                    continue
                ranges = np.linalg.norm(p-stations[[a, b, c]], axis=1)
                if max(abs(ranges-ranges[0]-np.r_[0.,u[i]*SCALE])) < 1e-5:
                    result.append(p)
    if not result:
        return np.empty((0, 2))
    return np.unique(np.round(result, 7), axis=0)


def residual(p, lags, keep):
    pred, jac = predicted(p)
    errors = pred[..., None]-lags[None]
    nearest = np.argmin(abs(errors), axis=2)
    res = np.take_along_axis(errors, nearest[..., None], axis=2)[..., 0]
    order = np.argsort(abs(res), axis=1, kind='stable')[:, :keep]
    return (np.take_along_axis(res, order, axis=1),
            np.take_along_axis(jac, order[..., None], axis=1))


def refine(pool, lags, keep):
    p = pool.copy()
    count = len(p)
    status = np.full(count, 'iteration_cap', dtype='<U24')
    nit = np.zeros(count, int)
    active = np.ones(count, bool)
    for it in range(100):
        ids = np.flatnonzero(active)
        if not len(ids):
            break
        res, jac = residual(p[ids], lags, keep)
        cost = np.sum(res*res, axis=1)
        grad = np.einsum('nki,nk->ni', jac, res)
        stable = np.max(abs(grad), axis=1) < 1e-6
        status[ids[stable]] = 'gradient'
        active[ids[stable]] = False
        hess = np.einsum('nki,nkj->nij', jac, jac)+1e-6*np.eye(2)[None]
        step = -np.linalg.solve(hess, grad[..., None])[..., 0]
        accepted = stable.copy()
        nextp = p[ids].copy()
        for power in range(20):
            todo = ~accepted
            if not np.any(todo):
                break
            trial = np.clip(p[ids[todo]]+step[todo]*(.5**power), -SCALE, SCALE)
            rr, _ = residual(trial, lags, keep)
            newcost = np.sum(rr*rr, axis=1)
            ok = newcost < cost[todo]
            local = np.flatnonzero(todo)[ok]
            nextp[local] = trial[ok]
            accepted[local] = True
        movement = np.linalg.norm(nextp-p[ids], axis=1)
        stalled = ~accepted
        status[ids[stalled]] = 'line_search_stall'
        active[ids[stalled]] = False
        small = accepted & ~stable & (movement < 1e-4)
        status[ids[small]] = 'step'
        active[ids[small]] = False
        p[ids] = nextp
        nit[ids] = it+1
    res, _ = residual(p, lags, keep)
    return p, np.mean(res*res, axis=1), status, nit


def select(p, scores, k):
    chosen = []
    for i in np.argsort(scores, kind='stable'):
        if all(np.linalg.norm(p[i]-p[j]) > RADIUS for j in chosen):
            chosen.append(int(i))
            if len(chosen) == k:
                break
    return chosen


def gram(u, alpha, stations=STATIONS):
    # Receiver-centroid centering, Eq.(8), (22); all quantities normalized.
    n = len(alpha)
    distances = (alpha[:, None]+np.column_stack([np.zeros(n), u]))/SCALE
    dd = np.zeros((n, 6, 6))
    dd[:, :5, :5] = cdist(stations/SCALE, stations/SCALE)**2
    dd[:, :5, 5] = distances**2
    dd[:, 5, :5] = distances**2
    center = np.eye(6)-np.ones((6, 1))*np.r_[np.ones(5)/5, 0.][None]
    return -.5*(center@dd@center.T)


def edm_value(u, alpha):
    eig = np.linalg.eigvalsh(gram(u, alpha))
    return np.sum(abs(eig[:, :-2]), axis=1)


def edm(lags, step=10.):
    u = np.array(list(product(*lags[:4])))
    corners = np.array(list(product([-SCALE, SCALE], repeat=2)))
    maxdist = cdist(STATIONS, corners).max(axis=1)
    lo = np.maximum(0., -u.min(axis=1))
    hi = np.minimum(maxdist[0], np.min(maxdist[None, 1:]-u, axis=1))
    good = hi > lo
    u, lo, hi = u[good], lo[good], hi[good]
    grid = np.arange(0, maxdist[0], step)
    costs = np.full((len(u), len(grid)), np.inf)
    for start in range(0, len(u), 32):
        batch = u[start:start+32]
        values = edm_value(np.repeat(batch, len(grid), axis=0), np.tile(grid, len(batch)))
        costs[start:start+len(batch)] = values.reshape(len(batch), -1)
    costs[(grid[None] < lo[:, None]) | (grid[None] > hi[:, None])] = np.inf
    best_a, best_c, evaluations = [], [], 0
    for i, curve in enumerate(costs):
        def fun(a):
            return float(edm_value(u[i:i+1], np.array([a]))[0])
        # Include both bounds and every local grid minimum, not just one seed.
        local = np.flatnonzero(np.isfinite(curve) & (curve <= np.r_[np.inf,curve[:-1]])
                               & (curve <= np.r_[curve[1:],np.inf]))
        intervals = [(lo[i], min(lo[i]+step, hi[i])), (max(lo[i],hi[i]-step), hi[i])]
        intervals += [(max(lo[i],grid[j]-step), min(hi[i],grid[j]+step)) for j in local]
        options = [(fun(lo[i]),lo[i]), (fun(hi[i]),hi[i])]
        for left, right in intervals:
            opt = minimize_scalar(fun, bounds=(left,right), method='bounded', options={'xatol':1e-4})
            assert opt.success and np.isfinite(opt.fun)
            options.append((float(opt.fun),float(opt.x)))
            evaluations += opt.nfev
        value, alpha = min(options)
        best_a.append(alpha)
        best_c.append(value)
    eig, vec = np.linalg.eigh(gram(u, np.array(best_a)))
    relative = vec[:, :, -2:]*np.sqrt(np.maximum(eig[:, None, -2:],0))
    target = (STATIONS-STATIONS.mean(axis=0))/SCALE
    left, _, right = np.linalg.svd(np.transpose(relative[:, :5], (0,2,1))@target)
    rotation = left@right
    p = np.einsum('ni,nij->nj',relative[:,5],rotation)*SCALE+STATIONS.mean(axis=0)
    valid = np.max(abs(p),axis=1) <= SCALE+1e-6
    return p[valid], np.array(best_c)[valid], dict(combinations=len(u), valid=int(sum(valid)),
              alpha_evaluations=int(costs.size+evaluations+2*len(u)), alpha=np.array(best_a)[valid], u=u[valid])


def estimate(extracted, k, include_edm=True, edm_step=10.):
    # Sole data ingress: public station-pair keys and observed peak delays.
    lags = np.array([np.sort(np.array(extracted[f'{m}_{n}']['lag_s'][:6])*C) for m,n in PAIRS])
    if lags.shape != (10,6) or not np.isfinite(lags).all():
        raise ValueError('Expected six finite observed candidates per pair')
    records, arrays = {}, {}
    begin = time.perf_counter()
    pool = triple_candidates(lags)
    generation = time.perf_counter()-begin
    arrays['initial'] = pool
    for name, keep in [('GEO_ALL',10),('GEO_TRIM8',8)]:
        begin = time.perf_counter()
        p, scores, reasons, iterations = refine(pool,lags,keep)
        chosen = select(p,scores,k)
        records[name] = dict(p=p[chosen], selected=chosen, seconds=generation+time.perf_counter()-begin,
             pool=len(pool), stopped={str(s):int(sum(reasons==s)) for s in np.unique(reasons)},
             iterations=int(sum(iterations)), chosen_stop=reasons[chosen].tolist())
        arrays[name+'_p'], arrays[name+'_scores'] = p,scores
        arrays[name+'_iterations'], arrays[name+'_stop'] = iterations,reasons
    if include_edm:
        begin = time.perf_counter()
        p,scores,meta = edm(lags,edm_step)
        chosen = select(p,scores,k)
        records['EDM_RF'] = dict(p=p[chosen], selected=chosen, seconds=time.perf_counter()-begin,
                 combinations=meta['combinations'], valid=meta['valid'], alpha_evaluations=meta['alpha_evaluations'])
        arrays['EDM_RF_p'], arrays['EDM_RF_scores'] = p,scores
        arrays['EDM_RF_alpha'], arrays['EDM_RF_u'] = meta['alpha'],meta['u']
    return records,arrays


def evaluate(p, truth):
    if len(p) != len(truth):
        return dict(complete=False, errors_m=None, rmse_m=None, success20=False)
    d = cdist(p,truth)
    i,j = linear_sum_assignment(d*d)
    errors = np.zeros(len(truth))
    errors[j] = d[i,j]
    return dict(complete=True, errors_m=errors, rmse_m=float(np.sqrt(np.mean(errors**2))),
                success20=bool(max(errors)<=20))


def selftest():
    truth = np.array([[700.,500.],[-900.,-650.]])
    pr,_ = predicted(truth)
    lags = np.sort(pr.T,axis=1)
    pool = triple_candidates(lags)
    checks = {'triple_true_roots': bool(cdist(pool,truth).min(axis=0).max()<1e-5)}
    for keep in [8,10]:
        p,s,_,_ = refine(pool,lags,keep)
        checks[f'exact_two_{keep}'] = evaluate(p[select(p,s,2)],truth)['success20']
    checks['edm_zero'] = bool(max(edm_value(pr[:,:4],np.linalg.norm(truth-STATIONS[0],axis=1)))<1e-12)
    p,s,_ = edm(lags,40.)
    checks['edm_procrustes'] = bool(cdist(p,truth).min(axis=0).max()<.01)
    testp=truth+[[3.,5.],[-2.,4.]]
    _,jac=predicted(testp)
    fd=np.stack([(predicted(testp+np.eye(2)[i]*.001)[0]-predicted(testp-np.eye(2)[i]*.001)[0])/.002 for i in range(2)],axis=2)
    checks['analytic_jacobian'] = bool(np.max(abs(fd-jac))<1e-8)
    missing=lags.copy()
    missing[2]+=300.
    pp=triple_candidates(missing)
    fitted,s,_,_=refine(pp,missing,8)
    checks['one_missing_pair'] = evaluate(fitted[select(fitted,s,2)],truth)['success20']
    checks['shared_pair_not_exclusive'] = bool(len(select(np.array([[0.,0.],[100.,100.]]),np.array([0.,1.]),2))==2)
    checks['boundary20'] = evaluate(np.array([[20.,0.]]),np.zeros((1,2)))['success20']
    checks['missing_output'] = not evaluate(np.empty((0,2)),truth)['complete']
    assert all(checks.values()),checks
    return checks


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prefix',action='store_true')
    args=parser.parse_args()
    checks=selftest()
    source=json.loads((SOURCE/'result.json').read_text(encoding='utf-8'))
    rows=[r for r in source['rows'] if r['noise']=='snr6' and
          (r['realization']==0 or (not args.prefix and r['signal']=='AB'))]
    files=[Path(__file__),PROTOCOL,ROOT/'geometry_association_gate.py',SOURCE/'result.json',SOURCE/'protocol.json']
    files += [SOURCE/(r['tag']+'.json') for r in rows]
    files += sorted({SOURCE/f'L{r["layout"]}_R{r["realization"]}_components.npz' for r in rows})
    hashes={str(p.resolve(strict=True)):sha(p) for p in files}
    assert all(Path(p).is_relative_to(ROOT) for p in hashes)
    out=ROOT/('robust_geometry_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    (out/'source').mkdir()
    for file in files[:3]:
        (out/'source'/file.name).write_bytes(file.read_bytes())
    write(out/'protocol.json',dict(hashes=hashes,checks=checks,rows=[r['tag'] for r in rows],radius_m=RADIUS,
          kind='OBSERVED_TDOA_GEOMETRY_NOT_DPD',edm_alpha_step_m=10.,trim_keep=8))
    print('OUTPUT',out,flush=True)
    records=[]
    begin=time.perf_counter()
    for index,row in enumerate(rows):
        truthfile=SOURCE/f'L{row["layout"]}_R{row["realization"]}_components.npz'
        truth=np.load(truthfile)['truth'][row['active_sources']]
        for front in ['CC_ABS','PHAT_ABS']:
            methods,arrays=estimate(row['extraction'][front],len(row['active_sources']))
            for name,rec in methods.items():
                rec['evaluation']=evaluate(rec['p'],truth)
                candidate=arrays[name+'_p']
                rec['coverage_m']=cdist(candidate,truth).min(axis=0) if len(candidate) else None
            raw=arrays['initial']
            record=dict(tag=row['tag'],front=front,layout=row['layout'],realization=row['realization'],
                 paths=row['paths'],signal=row['signal'],iq_sha=row['iq_sha'],methods=methods,
                 initial_coverage_m=cdist(raw,truth).min(axis=0) if len(raw) else None)
            # Finer alpha grid checks optimization adequacy, not success-driven tuning.
            if index==0 and front=='CC_ABS':
                smaller,_=estimate(row['extraction'][front],len(truth),edm_step=5.)
                record['alpha_grid_check']=dict(p5=smaller['EDM_RF']['p'],
                    paired_distance_m=cdist(methods['EDM_RF']['p'],smaller['EDM_RF']['p']).min(axis=1))
                checks['alpha_grid_stability']=bool(max(record['alpha_grid_check']['paired_distance_m'])<.01)
                assert checks['alpha_grid_stability']
            write(out/(row['tag']+'_'+front+'.json'),record)
            np.savez_compressed(out/(row['tag']+'_'+front+'.npz'),**arrays)
            records.append(record)
        print(f'{index+1}/{len(rows)} {row["tag"]}: {time.perf_counter()-begin:.1f}s',flush=True)
    checks.update({p:sha(p)==h for p,h in hashes.items()})
    assert all(checks.values())
    write(out/'result.json',dict(rows=records,checks=checks,seconds=time.perf_counter()-begin))
    print('COMPLETE',out,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
