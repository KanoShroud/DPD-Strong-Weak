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


def main():
    start=time.perf_counter();stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out=ROOT/('structure_'+stamp);out.mkdir(exist_ok=False)
    design=ROOT/'理论论证与实验设计.md';hashes={str(p):sha(p) for p in [Path(__file__),design]}
    protocol=dict(kind='IDEAL_FREQUENCY_OPERATOR_LOCAL_DIAGNOSTIC_NOT_IQ_EXPERIMENT',seed=SEED,
        stations_m=STATIONS,layouts_m=LAYOUTS,B_hz=B,coordinate_scale_m=SCALE,frequencies=[48,96,192],
        frequency_rule='Gauss-Legendre nodes within +/-0.49B; avoids artificial periodic delay aliasing. Unweighted Euclidean operator diagnostic, not a physical CRB.',
        source='Independent 128-symbol BPSK trains, RRC rolloff .25, unknown freely eliminated spectral coefficients in diagnostic',
        gain='1/d normalized by 1km for generator, random phase; estimator nuisance gains free complex constants per path',
        channel='Two reflections with Bdelta in [3,5] and [8,10]; total reflection power / LOS power = 10^(-6/10); 1-path subset drops second reflection',
        source_info='Truth is used only to evaluate local derivatives. No blind estimates or RMSE produced.',
        checks='Analytic vs finite difference; direct scalar vs vector forward; sequential vs full nuisance projection; nuisance-rank tolerances; no per-case tuning',
        hashes=hashes)
    write(out/'protocol.json',protocol);(out/'structural_probe.py').write_bytes(Path(__file__).read_bytes())
    rng=np.random.default_rng(SEED);rows=[];checks={};inputs={}
    for layout,pair in enumerate(LAYOUTS):
        bits=rng.choice([-1.,1.],(2,128));offset=rng.uniform(0,1/(B/1.25),2)
        delta=np.zeros((5,2,3));delta[:,:,1]=rng.uniform(3,5,(5,2));delta[:,:,2]=rng.uniform(8,10,(5,2))
        dist=np.linalg.norm(pair[None]-STATIONS[:,None],axis=2)
        g=np.empty((5,2,3),complex);g[:,:,0]=(1000/dist)*np.exp(1j*rng.uniform(-np.pi,np.pi,(5,2)))
        split=rng.uniform(.3,.7,(5,2));rho=10**(-6/10)
        g[:,:,1]=abs(g[:,:,0])*np.sqrt(rho*split)*np.exp(1j*rng.uniform(-np.pi,np.pi,(5,2)))
        g[:,:,2]=abs(g[:,:,0])*np.sqrt(rho*(1-split))*np.exp(1j*rng.uniform(-np.pi,np.pi,(5,2)))
        inputs[f'layout{layout}_bits']=bits;inputs[f'layout{layout}_delta']=delta;inputs[f'layout{layout}_g']=g
        for ids in [[0],[1],[0,1]]:
            for paths in [1,2,3]:
                p=pair[ids];d=delta[:,ids,:paths];gain=g[:,ids,:paths];q=len(ids)
                tag=f'L{layout}_Q'+''.join(map(str,ids))+f'_P{paths}'
                checks[tag+'/power']=bool(np.all(abs(gain[:,:,0])**2>np.sum(abs(gain[:,:,1:])**2,axis=2)))
                if paths>1:checks[tag+'/delay']=bool(np.min(np.diff(d,axis=2))>=3)
                for nf in [48,96,192]:
                    f=leggauss(nf)[0]*.49*B;s=np.stack([wave(f,bits[k],offset[k]) for k in ids],axis=1)
                    h,jp,jn,jd=columns(p,d,gain,f,s);y=np.einsum('fmq,fq->fm',h,s)
                    # Scalar evaluation independent of phase broadcasting.
                    scalar=np.zeros_like(y)
                    for m in range(5):
                        for k in range(q):
                            for ell in range(paths):
                                tau=np.linalg.norm(p[k]-STATIONS[m])/C+d[m,k,ell]/B
                                scalar[:,m]+=gain[m,k,ell]*s[:,k]*np.exp(-2j*np.pi*f*tau)
                    checks[tag+f'/{nf}/forward']=bool(np.allclose(y,scalar,rtol=1e-11,atol=1e-11))
                    eps=1e-6;errs=[]
                    for j in range(2*q):
                        dp=np.zeros_like(p);dp.flat[j]=eps*SCALE
                        hp,_=response(p+dp,d,gain,f);hm,_=response(p-dp,d,gain,f)
                        num=np.einsum('fmq,fq->fm',(hp-hm)/(2*eps),s)
                        errs.append(float(np.linalg.norm(num-jp[:,:,j])/np.linalg.norm(jp[:,:,j])))
                    checks[tag+f'/{nf}/gradient']=max(errs)<1e-6
                    wp=realify(project_wave(h,jp));wn=realify(project_wave(h,jn));wd=realify(project_wave(h,jd))
                    spectra=[]
                    for tol in [1e-6,1e-8,1e-10]:
                        effective,nrank=residualize(wn,wp,tol)
                        singular=svd(effective,compute_uv=False)/np.linalg.norm(realify(jp))
                        spectra.append(dict(tolerance=tol,nuisance_rank=nrank,singular=singular))
                    if nf==48:
                        # Explicit full real nuisance Jacobian, including waveform coefficients.
                        w=np.zeros((nf,5,nf*q*2),complex)
                        for fi in range(nf):
                            for k in range(q):
                                w[fi,:,2*(fi*q+k)]=h[fi,:,k];w[fi,:,2*(fi*q+k)+1]=1j*h[fi,:,k]
                        full=np.column_stack([realify(w),realify(jn)])
                        ef,_=residualize(full,realify(jp),1e-8)
                        seq,_=residualize(wn,wp,1e-8)
                        error=float(np.linalg.norm(ef.T@ef-seq.T@seq)/max(np.linalg.norm(ef.T@ef),1e-30))
                        checks[tag+'/projection']=error<1e-7
                    no_geometry,_=residualize(np.column_stack([wn,wd]),wp)
                    nullratio=float(np.linalg.norm(no_geometry)/np.linalg.norm(realify(jp)))
                    checks[tag+f'/{nf}/free_LOS_delay']=nullratio<1e-7
                    row=dict(case=tag,nf=nf,positions=p,paths=paths,source_ids=ids,spectra=spectra,
                        finite_difference_max=max(errs),free_LOS_delay_remaining=nullratio)
                    rows.append(row)
                print(tag,'完成',flush=True)
                assert all(checks.values()),[k for k,v in checks.items() if not v]
        # Exact common-filter ambiguity: different waveform/channel, same positions.
        f=leggauss(96)[0]*.49*B;s=np.stack([wave(f,bits[k],offset[k]) for k in range(2)],axis=1)
        h0,_=response(pair,delta[:,:,:1],g[:,:,:1],f)
        a=.3*np.exp(.4j);b=.2*np.exp(-.9j)
        common=1+a*np.exp(-2j*np.pi*f*4/B)+b*np.exp(-2j*np.pi*f*9/B)
        one=np.einsum('fmq,fq->fm',h0*common[:,None,None],s)
        two=np.einsum('fmq,fq->fm',h0,s*common[:,None])
        checks[f'L{layout}/same_position_common_filter']=bool(np.allclose(one,two,rtol=1e-12,atol=1e-12))
    np.savez_compressed(out/'inputs.npz',stations=STATIONS,layouts=LAYOUTS,**inputs)
    checks.update({'unchanged/'+p:sha(p)==v for p,v in hashes.items()})
    assert all(checks.values())
    write(out/'result.json',dict(rows=rows,checks=checks,passed=True,seconds=time.perf_counter()-start,
        input_sha=sha(out/'inputs.npz'),note='54 ideal-operator cases; no finite time records or localization estimates. Full local rank is not global uniqueness.'))
    print('REPORT',out,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
