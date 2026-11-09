# ruff: noqa: F401
# Preserve imported globals for legacy function cloning and reexports.
"""Conditional fresh-IQ order transfer; physical truth never enters solve_iq."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

from 研究方向.多径DPD.代码.mp_dpd.solvers.reference import HERE, BASE, B, C, FS, UNIT, STATIONS, search, los_fit, sha, write, error
from 研究方向.多径DPD.代码.mp_dpd.solvers.projected import fit
pass  # Explicit package imports replace historical search-path mutation.
from 研究方向.多径DPD.代码.mp_dpd.models.waveform_iq import synthesize, extract, RS, SPAN  # noqa: E402
from 研究方向.多径DPD.代码.mp_dpd.initialization.tdoa_candidates import PAIRS  # noqa: E402
from 研究方向.多径DPD.代码.mp_dpd.initialization.robust_geometry import triple_candidates  # noqa: E402
from 研究方向.多径DPD.代码.mp_dpd.initialization.consensus import fit as geometry_fit, joint_select  # noqa: E402
from 研究方向.多径DPD.代码.mp_dpd.initialization.blind_initialization import rank_one  # noqa: E402

N=8192


def generate_bank(p, li, unseen):
    rng=np.random.default_rng((2026103023 if unseen else 2026103022)+10000*li)
    first=-int(np.ceil((np.sqrt(2)*6000/C+40/B)*RS))-2*SPAN-4
    last=int(np.ceil(N/FS*RS))+2*SPAN+4
    bits=rng.choice([-1.,1.],(2,last-first+1))
    offsets=rng.uniform(0,1/RS,2)
    distance=np.linalg.norm(STATIONS[:,None]-p[None],axis=2)
    gains=1000/distance*np.exp(1j*rng.uniform(-np.pi,np.pi,(5,2)))
    delays=np.array([3.,7.,11.,15.])[None,None]+rng.uniform(0,.5,(5,2,4))
    phases=np.exp(1j*rng.uniform(-np.pi,np.pi,(5,2,4)))
    weights=rng.uniform(.8,1.2,(5,2,4))
    single=rng.integers(0,4,(5,2))
    t=np.arange(N)/FS
    los=np.empty((2,5,N),complex)
    bank=np.empty((2,5,4,N),complex)
    for q in range(2):
        for m in range(5):
            tau=distance[m,q]/C
            los[q,m]=gains[m,q]*synthesize(bits[q],first,offsets[q],t-tau)
            for ell in range(4):
                bank[q,m,ell]=gains[m,q]*phases[m,q,ell]*synthesize(bits[q],first,offsets[q],t-tau-delays[m,q,ell]/B)
    variance=np.mean(abs(los[1])**2)/(10**.6*(B/FS))
    noise=np.sqrt(variance/2)*(rng.normal(size=(5,N))+1j*rng.normal(size=(5,N)))
    los_snr_db=10*np.log10(np.mean(abs(los)**2,axis=2)/(variance*(B/FS)))
    return dict(los=los,bank=bank,noise=noise,bits=bits,first=first,offsets=offsets,los_snr_db=los_snr_db,
                gains=gains,delays=delays,phases=phases,weights=weights,single=single,variance=variance,p=p)


def observation(bank, condition):
    counts=np.full((5,2),int(condition[1:]) if condition.startswith('L') else 4)
    if condition=='MIX':
        counts=np.resize([0,1,2,4],10).reshape(5,2)
    ratio=1.2 if condition=='HIGH' else 10**(-.6)
    reflected=np.zeros_like(bank['los'])
    coefficients=np.zeros((5,2,4),complex)
    for m in range(5):
        for q in range(2):
            k=counts[m,q]
            ids={0:[],1:[int(bank['single'][m,q])],2:[0,3],3:[0,1,3],4:[0,1,2,3]}[k]
            if ids:
                amp=np.sqrt(ratio*bank['weights'][m,q,ids]/sum(bank['weights'][m,q,ids]))
                for ii,ell in enumerate(ids):
                    reflected[q,m]+=amp[ii]*bank['bank'][q,m,ell]
                    coefficients[m,q,ell]=amp[ii]*bank['phases'][m,q,ell]
    assert np.max(abs(coefficients))<1
    total=np.sum(abs(coefficients)**2,axis=2)
    assert np.allclose(total[counts>0],ratio) and np.all(total[counts==0]==0)
    iq=(bank['los']+reflected).sum(axis=0)+bank['noise']
    return iq,dict(counts=counts,relative_coefficients=coefficients,total_reflection_power_ratio=total)


def solve_iq(iq, out, label):
    """label is used only for filenames; no physical condition accepted."""
    start=time.perf_counter()
    extracted,_,_=extract(iq)
    peaks=extracted['CC_ABS']
    lags=np.array([sorted(np.array(peaks[f'{m}_{nn}']['lag_s'][:6])*C) for m,nn in PAIRS])
    pool=triple_candidates(lags)
    pp,_,reward,_,_=geometry_fit(pool,lags)
    ids,_=joint_select(reward)
    initial=pp[ids]
    f0=np.fft.fftfreq(N,1/FS)
    use=abs(f0)<=B/2
    ft=np.fft.fft(iq,axis=1,norm='ortho')
    f,y=f0[use],ft[:,use].T
    sigma2=float(np.mean(abs(ft[:,~use])**2))
    gains=np.column_stack([rank_one(y,f,p)[0] for p in initial])
    x0=np.r_[initial.ravel()/UNIT,gains[1:].real.ravel(),gains[1:].imag.ravel()]
    frontend=time.perf_counter()-start
    x,los=los_fit(x0,f,y,N)
    paths,history=[],[]
    state=los
    checks={}
    stop='capacity'
    for step in range(1,65):
        pick,bank,meta=search(x,paths,f,y,N,use,sigma2,step)
        checks.update({f'{step}/'+k:v for k,v in meta['checks'].items()})
        np.savez_compressed(out/f'{label}_step{step}_candidates.npz',bank=bank)
        ev=dict(step=step,pick=pick,search=meta)
        history.append(ev)
        if pick is None:
            stop='no_detected_increment'
            break
        trial=np.r_[x,pick['rho'],pick['delay']]
        nextpaths=paths+[(pick['m'],pick['q'])]
        trial,new=fit(trial,f,y,nextpaths,N)
        ev['fit']=new
        ev['decrease']=new['cost']<=state['cost']+1e-10
        checks[f'{step}/finite']=bool(np.isfinite(trial).all())
        checks[f'{step}/closure']=new['closure_error']<1e-7
        checks[f'{step}/budget']=new['iterations']<=300
        if not ev['decrease']:
            stop='numerical_non_decrease'
            break
        x,paths,state=trial,nextpaths,new
        write(out/f'{label}_progress.json',dict(paths=paths,state=state,history=history))
        if not state['stationary']:
            stop='numerical_nonstationary'
            break
        if not all(checks.values()):
            stop='correctness_failure'
            break
    return dict(initial=initial,los=los,final=state,paths=paths,history=history,stop=stop,
                frontend_seconds=frontend,seconds=time.perf_counter()-start,checks=checks,
                delay_alias=sum(v>N/FS*B/2 for v in x[22::3]),
                power_boundary=sum(x[k:k+2]@x[k:k+2]>=1-1e-10 for k in range(20,len(x),3)))


# Historical main retained only in the evidence archive.


# Historical batch entry disabled; use the public mp_dpd API.
