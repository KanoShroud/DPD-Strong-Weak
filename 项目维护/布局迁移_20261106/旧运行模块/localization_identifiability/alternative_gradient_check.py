# ruff: noqa: F401
# Preserve imported globals for legacy function cloning and reexports.
"""One bounded analytic-gradient follow-up; no new data or search positions."""
from pathlib import Path
from datetime import datetime
import time
import json
import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
from alternative_fit import SOURCE, encode, decode, fit_cost
from structural_probe import ROOT, LAYOUTS, B, SEED, wave, response, write, sha
from numpy.polynomial.legendre import leggauss


def objective(x,p,f,y):
    g,d=decode(x);h,phase=response(p,d,g,f)
    uu,ss,vh=np.linalg.svd(h,full_matrices=False)
    inv=np.divide(1,ss,out=np.zeros_like(ss),where=ss>ss[:,:1]*1e-12)
    coef=np.einsum('fqi,fi->fq',vh.conj().transpose(0,2,1),inv*np.einsum('fmi,fm->fi',uu.conj(),y))
    residual=y-np.einsum('fmq,fq->fm',h,coef);den=np.linalg.norm(y)**2
    jac=np.zeros((len(f),5,76),complex)
    u=x[16:36].reshape(5,2,2)+1j*x[36:56].reshape(5,2,2)
    for m in range(5):
        for q in range(2):
            t=np.sqrt(1+np.sum(abs(u[m,q])**2));rat=u[m,q]/t
            e=phase[:,m,q,1:];es=e@u[m,q];signal=coef[:,q];g0=g[m,q,0]
            if m:
                z=signal*(phase[:,m,q,0]+e@rat);i=(m-1)*2+q
                jac[:,m,i]=z;jac[:,m,i+8]=1j*z
            for ell in range(2):
                i=(m*2+q)*2+ell
                jac[:,m,16+i]=signal*g0*(e[:,ell]/t-es*u[m,q,ell].real/t**3)
                jac[:,m,36+i]=signal*g0*(1j*e[:,ell]/t-es*u[m,q,ell].imag/t**3)
                affected=slice(ell,None)
                jac[:,m,56+i]=signal*g0*(-2j*np.pi*f/B)*(e[:,affected]@rat[affected])
    grad=-2*np.einsum('fm,fmv->v',residual.conj(),jac).real/den
    return float(np.linalg.norm(residual)**2/den),grad


# Historical main retained only in the evidence archive.


# Historical batch entry disabled; use the public mp_dpd API.
