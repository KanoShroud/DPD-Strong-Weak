# ruff: noqa: F401
# Preserve imported globals for legacy function cloning and reexports.
"""Ideal frequency-operator diagnostics, NOT finite-record IQ localization.

Independent BPSK symbol trains with an RRC spectrum generate source values.
Unknown waveforms are eliminated freely per frequency for this local test.
No position estimator, performance ranking, or global uniqueness claim.
"""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import svd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
C=299792458.
B=12.5e6
SCALE=1000.
SEED=20261027091
STATIONS=np.array([[0.,0.],[3000.,3000.],[-3000.,3000.],[-3000.,-3000.],[3000.,-3000.]])
LAYOUTS=np.array([[[700.,500.],[-900.,-650.]],[[2100.,1700.],[2500.,2150.]]])


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,data):
    def default(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=default,allow_nan=False),encoding='utf-8')


def wave(f,bits,offset):
    rate=B/1.25;af=abs(f);lo=.75*rate/2;hi=1.25*rate/2
    power=np.ones(len(f));middle=af>lo
    power[middle]=.5*(1+np.cos(np.pi*(af[middle]-lo)/(hi-lo)))
    power[af>=hi]=0
    return np.sqrt(np.maximum(power,0))*(np.exp(-2j*np.pi*f[:,None]*(np.arange(len(bits))[None]/rate+offset))@bits)/np.sqrt(len(bits))


def response(p,delta,g,f):
    dist=np.linalg.norm(p[None]-STATIONS[:,None],axis=2)
    phase=np.exp(-2j*np.pi*f[:,None,None,None]*(dist[None,:,:,None]/C+delta[None]/B))
    return np.sum(phase*g[None],axis=3),phase


def realify(v):
    return np.concatenate([v.real.reshape(-1,v.shape[-1]),v.imag.reshape(-1,v.shape[-1])],axis=0)


def project_wave(h,v):
    # Per-frequency complex orthogonal projection, no squaring of condition number.
    out=[]
    for hf,vf in zip(h,v):
        u,s,_=svd(hf,full_matrices=False)
        rank=int(np.sum(s>s[0]*1e-12))
        out.append(vf-u[:,:rank]@(u[:,:rank].conj().T@vf))
    return np.array(out)


def residualize(a,j,tol=1e-8):
    norms=np.linalg.norm(a,axis=0)
    keep=norms>max(np.max(norms),1.)*1e-12
    a=a[:,keep]/norms[keep]
    u,s,_=svd(a,full_matrices=False)
    rank=int(np.sum(s>s[0]*tol)) if len(s) else 0
    return j-u[:,:rank]@(u[:,:rank].T@j),rank


def columns(p,delta,g,f,source):
    h,phase=response(p,delta,g,f);q=len(p);fcount=len(f);paths=g.shape[2]
    jp=np.zeros((fcount,5,2*q),complex);nuis=[];losdelay=[]
    vec=p[None]-STATIONS[:,None];unit=vec/np.linalg.norm(vec,axis=2)[:,:,None]
    for k in range(q):
        for xy in range(2):
            jp[:,:,2*k+xy]=source[:,k,None]*h[:,:,k]*(-2j*np.pi*f[:,None])*unit[None,:,k,xy]*SCALE/C
        for m in range(5):
            for ell in range(paths):
                z=np.zeros((fcount,5),complex);z[:,m]=source[:,k]*phase[:,m,k,ell]
                nuis.extend([z,1j*z])
                dz=z*g[m,k,ell]*(-2j*np.pi*f[:,None]/B)
                if ell: nuis.append(dz)
                else:losdelay.append(dz)
    return h,jp,np.stack(nuis,axis=2),np.stack(losdelay,axis=2)


# Historical main retained only in the evidence archive.


# Historical batch entry disabled; use the public mp_dpd API.
