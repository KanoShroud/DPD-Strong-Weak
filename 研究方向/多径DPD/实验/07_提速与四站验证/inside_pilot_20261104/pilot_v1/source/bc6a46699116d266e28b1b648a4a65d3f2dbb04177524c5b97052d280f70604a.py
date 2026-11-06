"""Independent LP certificate for the zero-entropy limit of Flood Eq.(5)."""
from datetime import datetime
import json
import time

import numpy as np
from scipy.io import loadmat
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, hstack, eye, vstack
from scipy.special import xlogy

from geometry_association_gate import ROOT, sha, write

SOURCE=ROOT/'flood_audit_20261029_233351_308298'


def solve_lp(cost,trash,eta,capacity):
    n,j=cost.shape
    nm=n*j
    total=nm+n+j
    matrix=coo_matrix((np.ones(nm),(np.repeat(np.arange(n),j),np.arange(nm))),shape=(n,nm)).tocsr()
    aeq=hstack([matrix,eye(n),coo_matrix((n,j))],format='csr')
    acol=coo_matrix((np.ones(nm),(np.tile(np.arange(j),n),np.arange(nm))),shape=(j,total)).tocsr()
    row=np.r_[np.arange(nm),np.arange(nm)]
    col=np.r_[np.arange(nm),nm+n+np.tile(np.arange(j),n)]
    vals=np.r_[np.ones(nm),-np.ones(nm)]
    link=coo_matrix((vals,(row,col)),shape=(nm,total)).tocsr()
    objective=np.r_[cost.ravel(),np.full(n,trash),np.full(j,eta)]
    begin=time.perf_counter()
    sol=linprog(objective,A_ub=vstack([acol,link],format='csr'),
       b_ub=np.r_[np.full(j,capacity),np.zeros(nm)],A_eq=aeq,b_eq=np.ones(n),
       bounds=(0,1),method='highs',options={'dual_feasibility_tolerance':1e-8,'primal_feasibility_tolerance':1e-8})
    seconds=time.perf_counter()-begin
    if not sol.success:
        raise RuntimeError(sol.message)
    m=sol.x[:nm].reshape(n,j)
    t=sol.x[nm:nm+n]
    q=sol.x[nm+n:]
    dual=float(sol.eqlin.marginals@np.ones(n)+sol.ineqlin.marginals@np.r_[np.full(j,capacity),np.zeros(nm)]+sol.upper.marginals.sum())
    return dict(M=m,trash=t,q=q,objective=float(sol.fun),dual=dual,
       relative_gap=abs(sol.fun-dual)/max(1,abs(sol.fun)),seconds=seconds,iterations=sol.nit,
       row_residual=float(max(abs(m.sum(1)+t-1))),column_excess=float(max(0,max(m.sum(0)-capacity))),
       epigraph_excess=float(max(0,np.max(m-q[None]))))


def main():
    protocol=json.loads((SOURCE/'protocol.json').read_text(encoding='utf-8'))
    for p,h in protocol['hashes'].items():
        assert sha(p)==h
    out=SOURCE/('lp_check_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    (out/'flood_lp_check.py').write_bytes((ROOT/'flood_lp_check.py').read_bytes())
    rows=[]
    for name in protocol['files']:
        data=loadmat(SOURCE/(name+'.mat'))
        cost=data['C'];trash=float(data['trash'].item());eta=float(data['eta'].item())
        r=float(data['R'].item());eps=float(data['epsilon'].item())
        sol=solve_lp(cost,trash,eta,r*(r-1)/2)
        record=dict(name=name,input_sha=sha(SOURCE/(name+'.mat')),lp=sol)
        path=SOURCE/(name+'_author.mat')
        if path.exists():
            author=loadmat(path)
            m=author['M'];t=author['trash_mass'].ravel()
            objective=float(np.sum(cost*m)+trash*sum(t)+eta*np.max(m,axis=0).sum())
            entropy=float(xlogy(m,m).sum()-m.sum()+xlogy(t,t).sum()-t.sum())
            record['author']=dict(objective_no_entropy=objective,objective_entropy=objective+eps*entropy,
                 row_residual=float(author['mass_residual'].item()),column_excess=float(author['capacity_residual'].item()),
                 excess_over_lp=objective-sol['objective'],epsilon_objective_range=eps*len(cost)*np.log(cost.shape[1]+1),
                 paper_best=np.argsort(-m.max(0),kind='stable')[:2],code_best=np.argsort(-m.sum(0),kind='stable')[:2],
                 seconds=float(author['seconds'].item()))
        write(out/(name+'.json'),record)
        rows.append(record)
        print(name,'LP',sol['objective'],'gap',sol['relative_gap'],'s',sol['seconds'],
              'author',record.get('author'),flush=True)
    checks={r['name']:r['lp']['relative_gap']<1e-7 and r['lp']['row_residual']<1e-7 and r['lp']['epigraph_excess']<1e-7 for r in rows}
    assert all(checks.values())
    write(out/'result.json',dict(rows=rows,checks=checks))
    print('COMPLETE',out)


if __name__=='__main__':
    main()
