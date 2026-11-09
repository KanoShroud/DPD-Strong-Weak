# ruff: noqa: F401
# Preserve imported globals for legacy function cloning and reexports.
"""Fixed wrong-position explanations in the ideal frequency model.

This is a truth-assisted diagnostic, not blind localization. Failure to find
an alternative is not a lower bound on the achievable residual.
"""
from datetime import datetime
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
from 研究方向.多径DPD.代码.mp_dpd.models.frequency_operator import ROOT, LAYOUTS, B, SEED, wave, response, write, sha
from numpy.polynomial.legendre import leggauss

SOURCE=ROOT/'structure_20261027_202324_816706'


def encode(g,d):
    g=g/g[:1,:,0,None]
    los=g[:,:,0];ratio=g[:,:,1:]/los[:,:,None]
    u=ratio/np.sqrt(1-np.sum(abs(ratio)**2,axis=2))[:,:,None]
    # All paths have Bdelta gaps >=3; diagnostic's allowed gap upper bound is20.
    gaps=np.diff(d,axis=2)
    x=np.r_[los[1:].real.ravel(),los[1:].imag.ravel(),u.real.ravel(),u.imag.ravel(),gaps.ravel()]
    return x


def decode(x):
    los=np.ones((5,2),complex);los[1:]=x[:8].reshape(4,2)+1j*x[8:16].reshape(4,2)
    u=x[16:36].reshape(5,2,2)+1j*x[36:56].reshape(5,2,2)
    ratios=u/np.sqrt(1+np.sum(abs(u)**2,axis=2))[:,:,None]
    g=np.concatenate([los[:,:,None],los[:,:,None]*ratios],axis=2)
    d=np.concatenate([np.zeros((5,2,1)),np.cumsum(x[56:].reshape(5,2,2),axis=2)],axis=2)
    return g,d


def fit_cost(x,p,f,y):
    g,d=decode(x);h,_=response(p,d,g,f)
    # Batched SVD avoids normal equations and determines rank explicitly.
    u,s,vh=np.linalg.svd(h,full_matrices=False)
    inv=np.divide(1,s,out=np.zeros_like(s),where=s>s[:,:1]*1e-12)
    coef=np.einsum('fqi,fi->fq',vh.conj().transpose(0,2,1),inv*np.einsum('fmi,fm->fi',u.conj(),y))
    residual=y-np.einsum('fmq,fq->fm',h,coef)
    return float(np.linalg.norm(residual)**2/np.linalg.norm(y)**2)


# Historical main retained only in the evidence archive.


# Historical batch entry disabled; use the public mp_dpd API.
