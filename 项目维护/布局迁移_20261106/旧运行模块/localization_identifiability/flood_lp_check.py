# ruff: noqa: F401
# Preserve imported globals for legacy function cloning and reexports.
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


# Historical main retained only in the evidence archive.


# Historical batch entry disabled; use the public mp_dpd API.
