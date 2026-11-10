#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pre-registered, non-oracle feasibility *diagnostic*, NOT a DPD locator.

Four synchronized single-antenna stations, 2 same-rate independent BPSK+RRC
sources, and one unknown per-link reflection. Diagnostic estimates a *band-
integrated* fourth cumulant, not the exact four-frequency trispectrum K_off.
It compares observed and ideal band-flat mixture proxies and a multichannel
phase-randomized second-order-matched null. All model truth is evaluator-only.

Run: python stat_probe.py --out results.json
Dependencies: numpy (only).
"""
from __future__ import annotations
import argparse, json, math, time, pathlib
import numpy as np
C=299792458.; FS=50e6; B=12.5e6; RS=10e6; BETA=.25; SPAN=32
STATIONS=np.array([[3000.,3000.],[-3000.,3000.],[-3000.,-3000.],[3000.,-3000.]])
N=32768
SOURCES=np.array([[550.,-800.],[-1100.,1100.]])
CENTERS=np.array([-3.2e6,1.4e6,-2.75e6,.95e6]) # a+b=c+d, P-off
FWHM=np.array([50e3,150e3,450e3]) # prespecified, *no* tuning after evidence
SEED=2026111027
# Deterministic ensemble pulse-power calibration, independent of sample length.
_PHASE=(np.arange(2048)+.5)/2048
# PULSE_NORM2 set after definition of rrc; source values are nested across N.


def rrc(x):
    x=np.asarray(x,dtype=np.float64)
    v=np.empty_like(x); z=np.isclose(x,0,atol=1e-11)
    s=np.isclose(abs(x),1/(4*BETA),atol=1e-11)
    mask=~(z|s)
    u=x[mask]; beta=BETA
    v[mask]=(np.sin(np.pi*(1-beta)*u)+4*beta*u*np.cos(np.pi*(1+beta)*u))/(np.pi*u*(1-(4*beta*u)**2))
    v[z]=1+beta*(4/np.pi-1)
    v[s]=beta/np.sqrt(2)*((1+2/np.pi)*np.sin(np.pi/(4*beta))+(1-2/np.pi)*np.cos(np.pi/(4*beta)))
    return v


PULSE_NORM2=float(np.mean(np.sum(rrc(_PHASE[:,None]-np.arange(-31,33)[None,:])**2,axis=1)))

def pulse_stream(t,offset,rng,law="bpsk"):
    u=(t-offset)*RS
    ind=np.floor(u).astype(np.int64)[:,None]+np.arange(-SPAN+1,SPAN+1)[None,:]
    lo=int(ind.min()); hi=int(ind.max())
    if law=="bpsk":
        bits=rng.choice([-1.,1.],hi-lo+1)
    elif law=="complex_gaussian":
        v=rng.standard_normal((hi-lo+1,2))
        bits=(v[:,0]+1j*v[:,1])/np.sqrt(2)
    else:raise ValueError(law)
    s=np.sum(bits[ind-lo]*rrc(u[:,None]-ind),axis=1)
    return s / np.sqrt(PULSE_NORM2)


def generate(seed=SEED,n=N,law="bpsk"):
    t0=time.perf_counter()
    # Independent RNG streams: N=8192/32768/65536 share the same physical
    # channel and a nested source/noise sample prefix. No N-specific redraw.
    rscene=np.random.default_rng(np.random.SeedSequence([seed,0]))
    tau=np.linalg.norm(STATIONS[:,None,:]-SOURCES[None,:,:],axis=2)/C
    extra=rscene.uniform(.25e-6,1.5e-6,(4,2))
    offset=rscene.uniform(0,1/RS,2)
    delaymax=float(np.max(tau+extra))
    guard=int(np.ceil(FS*(delaymax+15e-6)))
    ts=(np.arange(n+2*guard)-guard)/FS
    s=np.stack([pulse_stream(ts,offset[q],np.random.default_rng(np.random.SeedSequence([seed,1,q])),law=law) for q in range(2)])
    freqs=np.fft.fftfreq(len(ts),1/FS)
    sf=np.fft.fft(s,axis=-1)
    dist=np.linalg.norm(STATIONS[:,None,:]-SOURCES[None,:,:],axis=2)
    avgpowers=(1/dist**2).mean(axis=0)
    target_power=(B/FS)*10**(.6)
    g0=(1/dist)/np.sqrt(avgpowers)[None,:]*np.sqrt(target_power)
    g0=g0*np.exp(1j*rscene.uniform(-np.pi,np.pi,(4,2)))
    g1=.5*g0*np.exp(1j*rscene.uniform(-np.pi,np.pi,(4,2)))
    yf=np.zeros((4,len(ts)),complex)
    for m in range(4):
        for q in range(2):
            H=g0[m,q]*np.exp(-2j*np.pi*freqs*tau[m,q])+g1[m,q]*np.exp(-2j*np.pi*freqs*(tau[m,q]+extra[m,q]))
            yf[m]+=sf[q]*H
    yi=np.fft.ifft(yf,axis=-1)[:,guard:guard+n]
    noise=np.empty((4,n),complex)
    for m in range(4):
        rnoise=np.random.default_rng(np.random.SeedSequence([seed,2,m]))
        ri=rnoise.standard_normal((n,2))
        noise[m]=(ri[:,0]+1j*ri[:,1])/np.sqrt(2)
    y=yi+noise
    return dict(y=y,noise=noise,source=s[:,guard:guard+n],tau=tau,delta=extra,g0=g0,g1=g1,
                generation_s=time.perf_counter()-t0)


def banddata(x,centers,width,lo,hi):
    x=np.asarray(x)
    freq=np.fft.fftfreq(x.shape[-1],1/FS)
    xf=np.fft.fft(x,axis=-1)
    sigma=width/(2*np.sqrt(2*np.log(2)))
    t=np.arange(lo,hi)/FS
    out=[]
    for c in centers:
        weights=np.exp(-.5*((freq-c)/sigma)**2)
        z=np.fft.ifft(xf*weights[None,:],axis=-1)[...,lo:hi]
        z*=np.exp(-2j*np.pi*c*t)[None,:]
        out.append(z)
    return np.stack(out) # (bands,channels,time)


def normalize(z):
    rms=np.sqrt(np.mean(np.abs(z)**2,axis=(1,2)))
    return z/rms[:,None,None]


def cumulant_matrix(z):
    # 4 bands x 4 stations x usable time. z channel means removed per band.
    z=z-z.mean(axis=-1,keepdims=True)
    a,b,c,d=z
    T=a.shape[1]
    L=np.einsum('it,jt->ijt',a,b).reshape(16,T)
    R=np.einsum('it,jt->ijt',c,d).reshape(16,T)
    fourth=L@R.conj().T/T
    ac=a@c.conj().T/T; bd=b@d.conj().T/T
    ad=a@d.conj().T/T; bc=b@c.conj().T/T
    ab=a@b.T/T; cd=c@d.T/T
    remove=(np.einsum('ir,js->ijrs',ac,bd)+np.einsum('is,jr->ijrs',ad,bc)
            +np.einsum('ij,rs->ijrs',ab,cd.conj())).reshape(16,16)
    return fourth-remove


def oracle_projector(data):
    g0,g1,tau,delta=data['g0'],data['g1'],data['tau'],data['delta']
    Z=[]
    for q in range(2):
        H=[]
        for fc in CENTERS[:2]:
            h=g0[:,q]*np.exp(-2j*np.pi*fc*tau[:,q])+g1[:,q]*np.exp(-2j*np.pi*fc*(tau[:,q]+delta[:,q]))
            H.append(h)
        Z.append(np.kron(H[0],H[1]))
    Z=np.column_stack(Z)
    U=np.linalg.svd(Z,full_matrices=False)[0][:,:2]
    return U@U.conj().T


def metrics(K,P):
    u,s,_=np.linalg.svd(K,full_matrices=False)
    Ph=u[:,:2]@u[:,:2].conj().T
    err=float(np.linalg.norm(Ph-P,'fro'))
    return dict(singular_first4=[float(z) for z in s[:4]],sigma2=float(s[1]),sigma2_over_sigma3=float(s[1]/max(s[2],1e-30)),
                oracle_projector_error=err)


def population_cumulant(data,width,rms,ncalc=32768):
    """Deterministic exact *time-averaged* fourth cumulant of the band-filtered
    independent BPSK impulse model. Uses only ground truth inside evaluator.
    Source innovations are iid +/-1 (kappa4=-2), period FS/RS=5 samples.
    Physical multipath is retained *inside* each band, so K need not rank 2.
    """
    t=(np.arange(ncalc)-ncalc//2)/FS
    pulse=rrc(t*RS)
    pulse[np.abs(t*RS)>SPAN]=0
    pulse=pulse/np.sqrt(PULSE_NORM2)
    pf=np.fft.fft(pulse)
    f=np.fft.fftfreq(ncalc,1/FS)
    freq_masks=[]
    sigma=width/(2*np.sqrt(2*np.log(2)))
    for center in CENTERS:
        freq_masks.append(np.exp(-.5*((f-center)/sigma)**2))
    K=np.zeros((16,16),complex)
    for q in range(2):
        v=np.empty((4,4,ncalc),complex)  # bands,m,t
        for j in range(4):
            for m in range(4):
                H=(data['g0'][m,q]*np.exp(-2j*np.pi*f*data['tau'][m,q])
                   +data['g1'][m,q]*np.exp(-2j*np.pi*f*(data['tau'][m,q]+data['delta'][m,q])))
                v[j,m]=np.fft.ifft(pf*freq_masks[j]*H)
        la=np.einsum('it,jt->ijt',v[0],v[1]).reshape(16,ncalc)
        rb=np.einsum('it,jt->ijt',v[2],v[3]).reshape(16,ncalc)
        K+=(-2*RS/FS)*(la@rb.conj().T)
    return K/np.prod(rms)


def population_flat(data,width,rms,ncalc=32768):
    t=(np.arange(ncalc)-ncalc//2)/FS
    pulse=rrc(t*RS)
    pulse[np.abs(t*RS)>SPAN]=0
    pulse=pulse/np.sqrt(PULSE_NORM2)
    pf=np.fft.fft(pulse)
    f=np.fft.fftfreq(ncalc,1/FS)
    sigma=width/(2*np.sqrt(2*np.log(2)))
    pp=[]
    for center in CENTERS:
        resp=np.fft.ifft(pf*np.exp(-.5*((f-center)/sigma)**2))
        pp.append(resp)
    gamma=(-2*RS/FS)*np.sum(pp[0]*pp[1]*pp[2].conj()*pp[3].conj())
    ZL=[];ZR=[]
    for q in range(2):
        H=[]
        for center in CENTERS:
            H.append(data['g0'][:,q]*np.exp(-2j*np.pi*center*data['tau'][:,q])
                 +data['g1'][:,q]*np.exp(-2j*np.pi*center*(data['tau'][:,q]+data['delta'][:,q])))
        ZL.append(np.kron(H[0],H[1]));ZR.append(np.kron(H[2],H[3]))
    ZL=np.column_stack(ZL);ZR=np.column_stack(ZR)
    return (gamma/np.prod(rms))*(ZL@ZR.conj().T)


def population_metrics(Kpop, Khat, P):
    u,s,_=np.linalg.svd(Kpop,full_matrices=False)
    ph=u[:,:2]@u[:,:2].conj().T
    return dict(population_singular_first4=[float(i) for i in s[:4]],
        population_nonrank2_sigma3_over_sigma2=float(s[2]/max(s[1],1e-15)),
        population_projector_distance_to_center_frequency_oracle=float(np.linalg.norm(ph-P,'fro')),
        finite_sample_spectral_error=float(np.linalg.norm(Khat-Kpop,2)),
        error_over_population_sigma2=float(np.linalg.norm(Khat-Kpop,2)/max(s[1],1e-15)))


def do_case(n=N,seed=SEED,law="bpsk"):
    t0=time.perf_counter();data=generate(seed,n,law=law)
    y=data['y'];src=data['source'];noise=data['noise']
    P=oracle_projector(data)
    results=[]
    for width in FWHM:
        # Trim ~2.5 Gaussian effective correlation lengths on both sides.
        trim=max(100,int(np.ceil(FS*2.5/width)))
        left,right=trim,n-trim
        if right-left<1000:raise ValueError('Too few samples after edge trim')
        zy=banddata(y,CENTERS,width,left,right)
        scale_y=np.sqrt(np.mean(abs(zy)**2,axis=(1,2)))
        zy=zy/scale_y[:,None,None]
        Kobs=cumulant_matrix(zy)
        observed=metrics(Kobs,P)
        Kpop=population_cumulant(data,width,scale_y) if law=="bpsk" else np.zeros((16,16),complex)
        pop=population_metrics(Kpop,Kobs,P) if law=="bpsk" else {"gaussian_null_spectral_norm":float(np.linalg.norm(Kobs,2))}
        # Band-flat idealization with the same source realization & noise.
        sf=banddata(src,CENTERS,width,left,right)
        wn=banddata(noise,CENTERS,width,left,right)
        flat=np.empty((4,4,right-left),complex)
        for j,fc in enumerate(CENTERS):
            for m in range(4):
                h=np.array([data['g0'][m,q]*np.exp(-2j*np.pi*fc*data['tau'][m,q])
                    +data['g1'][m,q]*np.exp(-2j*np.pi*fc*(data['tau'][m,q]+data['delta'][m,q]))
                    for q in range(2)])
                flat[j,m]=h@sf[j]+wn[j,m]
        scale_flat=np.sqrt(np.mean(abs(flat)**2,axis=(1,2)))
        flat=flat/scale_flat[:,None,None]
        Kflat=cumulant_matrix(flat)
        idealized=metrics(Kflat,P)
        Kpopflat=population_flat(data,width,scale_flat) if law=="bpsk" else np.zeros((16,16),complex)
        popflat=population_metrics(Kpopflat,Kflat,P) if law=="bpsk" else {"gaussian_null_spectral_norm":float(np.linalg.norm(Kflat,2))}
        gaussian=[]
        spectrum=np.fft.fft(y,axis=-1)
        for k in range(3):
            rr=np.random.default_rng(seed+701+k)
            ph=np.exp(1j*rr.uniform(-np.pi,np.pi,n))
            surrogate=np.fft.ifft(spectrum*ph[None,:],axis=-1)
            zn=normalize(banddata(surrogate,CENTERS,width,left,right))
            gaussian.append(metrics(cumulant_matrix(zn),P))
        results.append(dict(fwhm_hz=float(width),usable_samples=right-left,nominal_time_bandwidth=float((right-left)*width/FS),
                            observed=observed,bandflat_population_proxy=idealized,
                            physical_population_oracle=pop,bandflat_population_oracle=popflat,
                            phase_randomized_nulls=gaussian,
                            observed_sigma2_above_null_max=observed['sigma2']>max(q['sigma2'] for q in gaussian)))
    return dict(protocol='PRE_REGISTERED_1A_STATISTIC_DIAGNOSTIC_V1',num_stations=4,num_sources=2,max_unknown_reflections_per_link=1,
                n=n,law=law,fs=FS,rs=RS,beta=BETA,nominal_source_avg_LOS_snr_db=6,
                source_geometry_evaluator_only=SOURCES.tolist(),
                band_centers_hz=CENTERS.tolist(),bandwidths_fwhm_hz=FWHM.tolist(),
                multipath_excess_delay_range_us=[.25,1.5],reflection_los_power_ratio=.25,
                non_oracle_estimator=True,truth_used_for_evaluation_only=True,
                rank2_subspace_measured_not_trispectrum=True,
                gaussian_control='shared spectral phase randomization preserves all second-order cross-periodograms',
                scene_generation_seconds=data['generation_s'],elapsed_seconds=time.perf_counter()-t0,
                rows=results,
                result_is_not_location_performance=True,
                caveat='Narrowband-filtered cumulant is a frequency-integrated proxy: its exact population tensor need not be rank 2 with physical frequency-dispersive multipath. A failure does NOT prove all fourth-order DPD impossible.')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='stat_probe_results.json')
    ap.add_argument('--n',type=int,default=N,choices=[8192,32768,65536])
    ap.add_argument('--law',default='bpsk',choices=['bpsk','complex_gaussian'])
    args=ap.parse_args()
    t=time.perf_counter();res=do_case(args.n,law=args.law)
    path=pathlib.Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('JSON:',path.resolve(),'Elapsed_s:',round(time.perf_counter()-t,2))
    for r in res['rows']:
        o=r['observed'];p=r['bandflat_population_proxy']; nulls=r['phase_randomized_nulls']
        print('FWHM_kHz=',r['fwhm_hz']/1000,'T*B=',round(r['nominal_time_bandwidth'],1),
              'obs_projector_error=',round(o['oracle_projector_error'],3),
              'flat_error=',round(p['oracle_projector_error'],3),
              'sigma2=',round(o['sigma2'],4),'nullmax=',round(max(n['sigma2'] for n in nulls),4))
    print('WARNING: These are *statistical feasibility proxies*, not localization RMSE or a formal trispectrum estimator.')


if __name__=='__main__':
    main()
