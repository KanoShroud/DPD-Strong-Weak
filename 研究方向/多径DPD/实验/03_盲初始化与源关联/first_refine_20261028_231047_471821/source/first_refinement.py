"""One-change pilot: continuous rank-one refinement before subtraction."""
from pathlib import Path
from datetime import datetime
import argparse
import json
import time
import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
from structural_probe import ROOT, STATIONS, C, B, sha, write
from blind_initialization import (GRID, rank_one, score_positions, peaks, los_vector,
                                 local_los, seed_paths, residual, errors, signature)
from boundary_update import run as backend

OLD = ROOT/'blind_init_20261028_200514_640818'
BASE = ROOT/'boundary_20261028_214016_411172'


def single_objective(x, f, y):
    """Exact scalar eigenvalue derivative; x uses C/B metre coordinates."""
    p=x*(C/B)
    vec=p[None]-STATIONS
    dist=np.linalg.norm(vec,axis=1)
    unit=vec/np.maximum(dist[:,None],1e-15)
    z=y*np.exp(2j*np.pi*f[:,None]*dist[None]/C)
    vals,vectors=np.linalg.eigh(z.T@z.conj())
    g=vectors[:,-1]
    t=z@g.conj()
    energy=float(np.linalg.norm(y)**2)
    grad=[]
    for xy in range(2):
        dz=z*(2j*np.pi*f[:,None]/B)*unit[None,:,xy]
        grad.append(-2*np.vdot(t,dz@g.conj()).real/energy)
    return 1-float(vals[-1])/energy,np.array(grad)


def first_refine(p,f,y):
    begin=time.perf_counter()
    x0=p/(C/B)
    initial=single_objective(x0,f,y)[0]
    best=[initial,x0.copy()]
    def fun(x):
        value,gradient=single_objective(x,f,y)
        if value<best[0]:best[:]=[value,x.copy()]
        return value,gradient
    opt=minimize(fun,x0,method='L-BFGS-B',jac=True,
                 bounds=[(-3000/(C/B),3000/(C/B))]*2,
                 options=dict(maxiter=100,maxfun=1000,ftol=1e-10,gtol=1e-7,maxls=30))
    pp=best[1]*(C/B)
    g,s,fit=rank_one(y,f,pp)
    value,gradient=single_objective(best[1],f,y)
    return pp,g,s,fit,dict(initial_cost=initial,cost=value,gradient_inf=float(np.max(abs(gradient))),
        success=bool(opt.success),status=int(opt.status),message=str(opt.message),
        iterations=int(opt.nit),evaluations=int(opt.nfev),seconds=time.perf_counter()-begin,
        raw_final_cost=float(opt.fun),selected_best_observed=True)


def solve(y,f,first_points):
    before=signature(y,f,first_points)
    begin=time.perf_counter()
    first_records=[];pairs=[]
    for i,p0 in enumerate(first_points):
        pp,g0,_,fit0,meta=first_refine(p0,f,y)
        rest=y-fit0
        second=peaks(score_positions(rest,f,GRID),exclude=[pp])
        first_records.append(dict(index=i,coarse=p0,refined=pp,fit=meta,second_candidates=second))
        for k,p1 in enumerate(second):
            g1,_,_=rank_one(rest,f,p1)
            p=np.array([pp,p1]);v=los_vector(np.column_stack([g0,g1]))
            p2,v2,los=local_los(p,v,f,y)
            pairs.append(dict(id=f'{i}_{k}',coarse=p,los_p=p2,los_v=v2,los=los))
    order=sorted(range(9),key=lambda i:pairs[i]['los']['cost'])[:3]
    branches=[]
    for index in order:
        item=pairs[index]
        p,v=item['los_p'],item['los_v']
        vv,seed=seed_paths(p,v,f,y)
        r,_=residual(vv,p,f,y)
        final=backend(p,vv,f,y)
        branches.append(dict(pair_id=item['id'],initial_p=p,initial_v=vv,seed=seed,
                             seed_cost=.5*float(r@r),final=final))
    chosen=min(range(3),key=lambda i:branches[i]['final']['cost'])
    assert signature(y,f,first_points)==before
    return dict(first=first_records,pairs=pairs,branches=branches,chosen=chosen,
                output=branches[chosen]['final']['p'],seconds=time.perf_counter()-begin,input_signature=before)


def selftest():
    rng=np.random.default_rng(20261028041)
    f=np.linspace(-.49*B,.49*B,57)
    y=rng.normal(size=(57,5))+1j*rng.normal(size=(57,5))
    x=np.array([457.,-876.])/(C/B)
    val,gradient=single_objective(x,f,y)
    eps=1e-5
    numeric=np.array([(single_objective(x+eps*np.eye(2)[i],f,y)[0]-single_objective(x-eps*np.eye(2)[i],f,y)[0])/(2*eps) for i in range(2)])
    _,_,fit=rank_one(y,f,x*(C/B))
    checks=dict(gradient=bool(np.allclose(gradient,numeric,rtol=1e-5,atol=1e-9)),
        objective=bool(np.isclose(val,np.linalg.norm(y-fit)**2/np.linalg.norm(y)**2,rtol=1e-12)))
    pp,_,_,_,meta=first_refine(x*(C/B),f,y)
    checks['nonincreasing']=meta['cost']<=meta['initial_cost']
    checks['public_bounds']=bool(np.max(abs(pp))<=3000)
    checks['repeat']=np.array_equal(pp,first_refine(x*(C/B),f,y)[0])
    assert all(checks.values()),checks
    return checks


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--test-only',action='store_true')
    parser.add_argument('--limit',type=int,default=4)
    args=parser.parse_args();checks=selftest()
    if args.test_only:
        print(checks);return
    assert 1<=args.limit<=4
    old=json.loads((OLD/'result.json').read_text(encoding='utf-8'))
    baseline=json.loads((BASE/'result.json').read_text(encoding='utf-8'))
    baseprotocol=json.loads((BASE/'protocol.json').read_text(encoding='utf-8'))
    assert baseline['passed'] and baseline['complete']
    assert all(sha(p)==h for p,h in baseprotocol['hashes'].items())
    paths=[Path(__file__),BASE/'result.json',OLD/'result.json',
           *[Path(p) for p in baseprotocol['hashes'] if p.endswith('.py')],
           *[OLD/(r['case']+'_input.npz') for r in old['rows']]]
    hashes={str(p):sha(p) for p in paths}
    out=ROOT/('first_refine_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False);(out/'source').mkdir()
    for p in paths:
        if p.suffix=='.py':(out/'source'/p.name).write_bytes(p.read_bytes())
    write(out/'protocol.json',dict(hashes=hashes,baseline=str(BASE),input_root=str(OLD),
        kind='SINGLE_CHANGE_FIRST_SOURCE_CONTINUOUS_REFINEMENT',limit=args.limit,
        fixed='Observations, first three grid candidates, grid and NMS rule, 9 pairs/3 branches, LOS/path initializer, constrained backend and budgets.',
        changed='Before subtraction, maximize original single-source rank-one score continuously in 2D, refitting unknown gains/waveform.',
        first_options=dict(method='L-BFGS-B',maxiter=100,maxfun=1000,gtol=1e-7,ftol=1e-10,maxls=30),
        oracle='Truth evaluation after output only; no truth-derived component data used.',
        timing='Saved baseline reused for quality, not wall-clock speed comparison. New timing excludes unchanged original first-map generation.',
        decision='Four paired cases only; no changes to grid, seeds, candidate count, paths or solver after results.'))
    rows=[]
    for previous,control in zip(old['rows'][:args.limit],baseline['rows']):
        case=previous['case'];z=np.load(OLD/(case+'_input.npz'));y,f=z['y'],z['f']
        assert signature(y,f)==previous['input_signature']
        first=np.array([next(p for p in previous['pairs'] if p['id']==f'{i}_0')['coarse'][0] for i in range(3)])
        checks[case+'/first_replay']=np.array_equal(first,np.array(peaks(score_positions(y,f,GRID))))
        print('START',case,flush=True)
        result=solve(y,f,first)
        # No true component or position is used until all decisions are final.
        from alternative_fit import SOURCE
        truth=np.load(SOURCE/'inputs.npz')['layouts'][int(case[1])]
        result['case']=case;result['evaluation']=errors(result['output'],truth)
        result['control_evaluation']=control['evaluation']
        for item in result['first']:
            item['coarse_nearest_error_m']=float(np.min(np.linalg.norm(truth-item['coarse'],axis=1)))
            item['refined_nearest_error_m']=float(np.min(np.linalg.norm(truth-item['refined'],axis=1)))
            checks[case+f'/first{item["index"]}/cost']=item['fit']['cost']<=item['fit']['initial_cost']
        for item in result['pairs']:item['evaluation']=errors(item['los_p'],truth)
        for item in result['branches']:
            item['evaluation']=errors(item['final']['p'],truth)
            checks[case+'/'+item['pair_id']+'/feasible']=item['final']['feasible']
            checks[case+'/'+item['pair_id']+'/descent']=all(h['next_cost']<h['cost'] for h in item['final']['history'] if h.get('accepted'))
        rows.append(result);write(out/(case+'.json'),result)
        print('DONE',case,'old/new RMSE',control['evaluation']['rmse_m'],result['evaluation']['rmse_m'],flush=True)
    checks.update({'hash/'+p:sha(p)==h for p,h in hashes.items()})
    write(out/'result.json',dict(rows=rows,checks=checks,passed=bool(all(checks.values())),complete=args.limit==4))
    print('REPORT',out,flush=True)
    assert all(checks.values())


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
