"""Flood Eq.(5), zero-entropy LP reference; not the authors' entropy solver."""
from datetime import datetime
import argparse
import json
import time
import warnings

import numpy as np
from scipy.optimize import linprog, least_squares
from scipy.sparse import coo_matrix, vstack
from threadpoolctl import threadpool_limits

from geometry_association_gate import ROOT, C, B, PAIRS, sha, write, signature
from robust_geometry import predicted, evaluate
from blind_backend_pilot import solve, errors
from flood_lp_check import solve_lp

DEV=ROOT/'consensus_development_20261029_223235_334946'
VAL=ROOT/'consensus_validation_20261029_223520_495993'
INPUT=ROOT/'independent_backend_20261029_222111_026721'


def pruned_lp(cost,trash=1.,eta=1.,capacity=10.,known_k=None):
    n,j=cost.shape
    ii,jj=np.where(cost<trash)
    count=len(ii);total=count+n+j
    eqrow=np.r_[ii,np.arange(n)]
    eqcol=np.r_[np.arange(count),count+np.arange(n)]
    aeq=coo_matrix((np.ones(count+n),(eqrow,eqcol)),shape=(n,total)).tocsr()
    beq=np.ones(n)
    if known_k is not None:
        activation=coo_matrix((np.ones(j),(np.zeros(j,dtype=int),count+n+np.arange(j))),shape=(1,total)).tocsr()
        aeq=vstack([aeq,activation],format='csr')
        beq=np.r_[beq,known_k]
    acol=coo_matrix((np.ones(count),(jj,np.arange(count))),shape=(j,total)).tocsr()
    link=coo_matrix((np.r_[np.ones(count),-np.ones(count)],
         (np.tile(np.arange(count),2),np.r_[np.arange(count),count+n+jj])),shape=(count,total)).tocsr()
    obj=np.r_[cost[ii,jj],np.full(n,trash),np.full(j,eta)]
    begin=time.perf_counter()
    res=linprog(obj,A_ub=vstack([acol,link],format='csr'),b_ub=np.r_[np.full(j,capacity),np.zeros(count)],
                A_eq=aeq,b_eq=beq,bounds=(0,1),method='highs',
                options={'dual_feasibility_tolerance':1e-8,'primal_feasibility_tolerance':1e-8})
    if not res.success:
        raise RuntimeError(res.message)
    m=np.zeros((n,j));m[ii,jj]=res.x[:count]
    t=res.x[count:count+n];q=res.x[count+n:]
    dual=float(res.eqlin.marginals@beq+res.ineqlin.marginals[:j].sum()*capacity+res.upper.marginals.sum())
    checks=dict(row=float(max(abs(m.sum(1)+t-1))),column=float(max(0,max(m.sum(0)-capacity))),
        epigraph=float(max(0,np.max(m-q[None]))),gap=abs(res.fun-dual)/max(1,abs(res.fun)))
    if known_k is not None:
        checks['known_k']=abs(q.sum()-known_k)
    assert max(checks.values())<1e-7,checks
    return m,t,q,dict(objective=float(res.fun),dual=dual,checks=checks,
          iterations=res.nit,seconds=time.perf_counter()-begin,variables=total,full_variables=n*j+n+j)


def reestimate(cost,trash=1.,capacity=10.):
    # No activation penalty after selecting two candidate locations, as in author reestimate routine.
    m,t,q,meta=pruned_lp(cost,trash,0.,capacity)
    return m,t,meta


def estimate(pool,lags,known_k=None):
    begin=time.perf_counter()
    pred=predicted(pool)[0]
    costs=((pred[:,:,None]-lags[None])/(C/B))**2
    cost=costs.reshape(len(pool),-1).T
    m,t,q,meta=pruned_lp(cost,known_k=known_k)
    code_ids=np.argsort(-m.sum(0),kind='stable')[:2]
    paper_ids=np.argsort(-m.max(0),kind='stable')[:2]
    association,trash,assocmeta=reestimate(cost[:,code_ids])
    refined=[];details=[]
    for col,idx in enumerate(code_ids):
        rows=np.flatnonzero(association[:,col]>.7)
        edge=rows//lags.shape[1];obs=lags.ravel()[rows]
        start=pool[idx]
        if not len(rows):
            refined.append(start);details.append(dict(status='no_associated_measurement',rows=[]))
            continue
        def fun(p):
            return (predicted(p[None])[0][0,edge]-obs)/(C/B)
        def jac(p):
            return predicted(p[None])[1][0,edge]/(C/B)
        result=least_squares(fun,start,jac=jac,bounds=(-3000,3000),
               ftol=1e-10,xtol=1e-10,gtol=1e-10,max_nfev=100)
        refined.append(result.x)
        details.append(dict(status=int(result.status),rows=rows.tolist(),nfev=result.nfev,
            initial_cost=float(fun(start)@fun(start)),final_cost=float(result.fun@result.fun)))
    return np.array(refined),dict(cost=cost,M=m,trash=t,q=q,code_ids=code_ids,paper_ids=paper_ids,
        association=association,association_trash=trash,lp=meta,reestimate=assocmeta,
        refinement=details,seconds=time.perf_counter()-begin)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['development','validation'],required=True)
    parser.add_argument('--fixed-k',action='store_true')
    args=parser.parse_args()
    source=DEV if args.phase=='development' else VAL
    data_source=INPUT if args.phase=='development' else VAL
    files=[ROOT/n for n in ['flood_band_pair.py','Flood数值参照与带宽适配.md','flood_lp_check.py',
       'blind_backend_pilot.py','robust_geometry.py','geometry_association_gate.py','boundary_update.py',
       'blind_initialization.py','solver_reliability.py','position_reduction.py','structural_probe.py']]
    files+=[source/'result.json',source/'protocol.json']+sorted(source.glob('*_geometry.npz'))+sorted(data_source.glob('*_input.npz'))
    hashes={str(p.resolve(strict=True)):sha(p) for p in files}
    previous=json.loads((source/'protocol.json').read_text(encoding='utf-8'))['hashes']
    assert all(h==previous[p] for p,h in hashes.items() if p in previous)
    out=ROOT/(('flood_fixed_' if args.fixed_k else 'flood_band_')+args.phase+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False);(out/'source').mkdir()
    for p in files[:11]:
        (out/'source'/p.name).write_bytes(p.read_bytes())
    write(out/'protocol.json',dict(hashes=hashes,phase=args.phase,eta=1.,trash=1.,epsilon=0.,scale_m=C/B,
       known_k_constraint=2 if args.fixed_k else None,
       decoder='author column mass',secondary_decoder='paper column max diagnostic only',truth_in_solver=False))
    small=np.random.default_rng(792).uniform(0,3,(8,6))
    mm,tt,qq,meta=pruned_lp(small,1,1,3)
    full=solve_lp(small,1,1,3)
    checks={'exact_pruning':abs(meta['objective']-full['objective'])<1e-8}
    assert all(checks.values())
    rows=[];begin=time.perf_counter()
    f=np.fft.fftfreq(8192,1/50e6);mask=abs(f)<=B/2
    inputs=json.loads((source/'result.json').read_text(encoding='utf-8'))['rows']
    print('OUTPUT',out,flush=True)
    for row in inputs:
        tag=row['tag'];base=tag.rsplit('_',1)[0]
        data=dict(np.load(data_source/(base+'_input.npz')))
        pool=np.load(source/(tag+'_geometry.npz'))['initial_candidates']
        lags=np.array([sorted(np.array(row['peaks'][f'{m}_{n}']['lag_s'][:6])*C) for m,n in PAIRS])
        p,assoc=estimate(pool,lags,2 if args.fixed_k else None)
        iq=(data['los']+(data['reflected'] if row['path']=='MP' else 0)).sum(axis=0)+data['noise']
        checks[tag+'/IQ']=signature(iq)==row['iq_sha']
        y=np.fft.fft(iq,axis=1,norm='ortho')[:,mask].T
        fitted=solve(p,f[mask],y)
        for name,positions in [('geometry',p),('los',fitted['los_p']),('mp',fitted['mp']['p'])]:
            fitted[name+'_errors_m']=errors(positions,data['p'])
        fitted.update(tag=tag,path=row['path'],layout=row['layout'],rep=row['rep'],iq_sha=signature(iq),
            geometry_seconds=assoc['seconds'],association={k:v for k,v in assoc.items() if k not in ['cost','M','q','trash','association','association_trash']},
            paper_raw=evaluate(pool[assoc['paper_ids']],data['p']),code_raw=evaluate(pool[assoc['code_ids']],data['p']))
        checks[tag+'/feasible']=bool(fitted['mp']['feasible'])
        checks[tag+'/descent']=fitted['mp']['cost']<=fitted['seed_cost']+1e-10
        rows.append(fitted)
        write(out/(tag+'.json'),fitted)
        np.savez_compressed(out/(tag+'_association.npz'),**{k:assoc[k] for k in ['cost','M','q','trash','association','association_trash','code_ids','paper_ids']})
        print(len(rows),tag,'GEO',np.round(fitted['geometry_errors_m'],2),'MP',np.round(fitted['mp_errors_m'],2),flush=True)
        assert all(checks.values())
    checks.update({p:sha(p)==h for p,h in hashes.items()})
    assert all(checks.values())
    write(out/'result.json',dict(rows=rows,checks=checks,seconds=time.perf_counter()-begin))
    print('COMPLETE',out,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2),warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='divide by zero encountered in log10',category=RuntimeWarning)
        main()
