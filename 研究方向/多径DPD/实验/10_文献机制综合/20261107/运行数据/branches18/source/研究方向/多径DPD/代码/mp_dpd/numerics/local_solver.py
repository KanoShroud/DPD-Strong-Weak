# ruff: noqa: F401
# Preserve imported globals for legacy function cloning and reexports.
"""Bounded solver audit: original objective/feasible set, no new prior.

Audit the radial ratio map and original-coordinate stationarity. Compare two
TRF scaling choices on four preselected diagnostic cases (not localization).
The closed-ball projection below is only a diagnostic direction construction;
all evaluated trial steps are strictly inside the original open ball.
"""
from pathlib import Path
from datetime import datetime
import json
import time
import scipy
import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits
from numpy.polynomial.legendre import leggauss
from 研究方向.多径DPD.代码.mp_dpd.models.frequency_operator import ROOT, LAYOUTS, B, SEED, wave, response, sha, write
from 研究方向.多径DPD.代码.mp_dpd.numerics.parameterization import SOURCE, encode, fit_cost
from 研究方向.多径DPD.代码.mp_dpd.numerics.derivatives import objective as old_objective

PREVIOUS=ROOT/'gradient_20261027_203059_256888'


def physical(x):
    v=np.array(x,copy=True);u=x[16:36].reshape(5,2,2)+1j*x[36:56].reshape(5,2,2)
    r=u/np.sqrt(1+np.sum(abs(u)**2,axis=2))[:,:,None]
    v[16:36]=r.real.ravel();v[36:56]=r.imag.ravel()
    return v


def evaluate(x,p,f,y,coordinate='internal'):
    los=np.ones((5,2),complex);los[1:]=x[:8].reshape(4,2)+1j*x[8:16].reshape(4,2)
    u=x[16:36].reshape(5,2,2)+1j*x[36:56].reshape(5,2,2)
    if coordinate=='internal':
        t=np.sqrt(1+np.sum(abs(u)**2,axis=2));rat=u/t[:,:,None]
    else:rat=u;t=np.ones((5,2))
    g=np.concatenate([los[:,:,None],los[:,:,None]*rat],axis=2)
    d=np.concatenate([np.zeros((5,2,1)),np.cumsum(x[56:].reshape(5,2,2),axis=2)],axis=2)
    h,phase=response(p,d,g,f)
    uu,ss,vh=np.linalg.svd(h,full_matrices=False)
    inv=np.divide(1,ss,out=np.zeros_like(ss),where=ss>ss[:,:1]*1e-12)
    pinv=np.einsum('fqi,fi,fmi->fqm',vh.conj().transpose(0,2,1),inv,uu.conj())
    coef=np.einsum('fqm,fm->fq',pinv,y);res=y-np.einsum('fmq,fq->fm',h,coef)
    dh=np.zeros((len(f),5,2,76),complex)
    for m in range(5):
        for q in range(2):
            e=phase[:,m,q,1:]
            if m:
                z=phase[:,m,q,0]+e@rat[m,q];i=(m-1)*2+q
                dh[:,m,q,i]=z;dh[:,m,q,8+i]=1j*z
            for ell in range(2):
                i=(m*2+q)*2+ell
                if coordinate=='internal':
                    es=e@u[m,q]
                    dr=e[:,ell]/t[m,q]-es*u[m,q,ell].real/t[m,q]**3
                    di=1j*e[:,ell]/t[m,q]-es*u[m,q,ell].imag/t[m,q]**3
                else:dr=e[:,ell];di=1j*e[:,ell]
                dh[:,m,q,16+i]=los[m,q]*dr;dh[:,m,q,36+i]=los[m,q]*di
                dh[:,m,q,56+i]=los[m,q]*(-2j*np.pi*f/B)*(e[:,ell:]@rat[m,q,ell:])
    # Exact variable-projection residual Jacobian at constant full column rank:
    # dR = -P_perp(dH)S_hat - (H^+)^H(dH)^H R.
    ds=np.einsum('fmqv,fq->fmv',dh,coef)
    j=-ds+np.einsum('fmq,fqv->fmv',h,np.einsum('fqm,fmv->fqv',pinv,ds))
    j-=np.einsum('fqm,fqv->fmv',pinv.conj(),np.einsum('fmqv,fm->fqv',dh.conj(),res))
    scale=np.sqrt(2)/np.linalg.norm(y)
    rr=scale*np.r_[res.real.ravel(),res.imag.ravel()]
    jj=scale*np.concatenate([j.real.reshape(-1,76),j.imag.reshape(-1,76)])
    metrics=dict(min_rank_ratio=float(np.min(ss[:,1]/ss[:,0])),
        min_DRR_db=float(-10*np.log10(np.max(np.sum(abs(rat)**2,axis=2)))),
        max_reflection_power_ratio=float(np.max(np.sum(abs(rat)**2,axis=2))),
        min_LOS_amplitude=float(abs(los).min()),max_LOS_amplitude=float(abs(los).max()))
    return rr,jj,metrics


def project(v):
    out=v.copy();r=v[16:36].reshape(5,2,2)+1j*v[36:56].reshape(5,2,2)
    r=r/np.maximum(1.,np.linalg.norm(r,axis=2))[:,:,None]
    out[16:36]=r.real.ravel();out[36:56]=r.imag.ravel();out[56:]=np.clip(v[56:],3,20)
    return out


def audit_point(x,p,f,y):
    r,j,metrics=evaluate(x,p,f,y);v=physical(x);rr,jj,mm=evaluate(v,p,f,y,'physical')
    grad=j.T@r;gp=jj.T@rr;direction=project(v-gp)-v
    pg=grad.copy();pg[56:][(x[56:]<=3+1e-8)&(pg[56:]>0)]=0;pg[56:][(x[56:]>=20-1e-8)&(pg[56:]<0)]=0
    value=.5*float(r@r);trials=[]
    for step in [.1,.01,.001]:
        vv=v+step*direction;r2,_,m2=evaluate(vv,p,f,y,'physical')
        valid=m2['max_reflection_power_ratio']<1 and m2['min_LOS_amplitude']>0 and np.min(vv[56:])>=3-1e-12 and np.max(vv[56:])<=20+1e-12
        trials.append(dict(step=step,value=.5*float(r2@r2),valid=bool(valid)))
    u=x[16:36].reshape(5,2,2)+1j*x[36:56].reshape(5,2,2);norm2=np.sum(abs(u)**2,axis=2)
    # Tangential/radial eigenvalues of the real radial-map Jacobian.
    saturation=(1+norm2)**1.5
    sj=np.linalg.svd(j,compute_uv=False);sp=np.linalg.svd(jj,compute_uv=False)
    return dict(value=value,internal_projected_gradient_inf=float(max(abs(pg))),
        physical_projected_mapping_inf=float(max(abs(direction))),
        equivalent_residual_error=float(np.linalg.norm(r-rr)),
        radial_gradient_compression_max=float(saturation.max()),
        internal_jacobian_singular_ratio=float(sj[-1]/sj[0]),physical_jacobian_singular_ratio=float(sp[-1]/sp[0]),
        metrics=metrics,feasible_descent_trials=trials,
        feasible_descent_found=bool(any(t['valid'] and t['value']<value-1e-12 for t in trials)))


def datasets():
    z=dict(np.load(SOURCE/'inputs.npz'));gen=np.random.default_rng(SEED);pert=np.random.default_rng(SEED+1)
    f=leggauss(96)[0]*.49*B
    for layout,p in enumerate(LAYOUTS):
        bits=gen.choice([-1.,1.],(2,128));offset=gen.uniform(0,1/(B/1.25),2)
        gen.uniform(3,5,(5,2));gen.uniform(8,10,(5,2));gen.uniform(-np.pi,np.pi,(5,2));gen.uniform(.3,.7,(5,2));gen.uniform(-np.pi,np.pi,(5,2));gen.uniform(-np.pi,np.pi,(5,2))
        assert np.array_equal(bits,z[f'layout{layout}_bits'])
        g=z[f'layout{layout}_g'];d=z[f'layout{layout}_delta'];s=np.stack([wave(f,bits[k],offset[k]) for k in range(2)],axis=1)
        h,_=response(p,d,g,f);y=np.einsum('fmq,fq->fm',h,s);x0=encode(g,d)
        for q in range(2):
            for shift in [20,200]:
                wrong=p.copy();wrong[q,q]+=shift
                for init in range(2):
                    x=x0.copy()
                    if init:x[:56]+=pert.normal(0,.1,56);x[56:]=np.clip(x[56:]+pert.normal(0,.3,20),3,20)
                    yield f'L{layout}_q{q}_d{shift}_i{init}',x,wrong,f,y


class CachedResidual:
    def __init__(self,p,f,y):self.p=p;self.f=f;self.y=y;self.x=None;self.count=0
    def update(self,x):
        if self.x is None or not np.array_equal(x,self.x):
            self.x=x.copy();self.r,self.j,self.metrics=evaluate(x,self.p,self.f,self.y);self.count+=1
        return self.r,self.j
    def fun(self,x):return self.update(x)[0]
    def jac(self,x):return self.update(x)[1]


# Historical main retained only in the evidence archive.


# Historical batch entry disabled; use the public mp_dpd API.
