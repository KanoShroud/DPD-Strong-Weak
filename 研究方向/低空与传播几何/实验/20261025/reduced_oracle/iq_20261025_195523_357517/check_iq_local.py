"""Local finite-record IQ identifiability; unknown waveforms and complex gains."""
import hashlib
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.linalg import svd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
PRIOR=ROOT.parent/'unknown_environment/run_20261025_193552_712855/result.json'
C=299792458.
SCALE=1000.


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def realify(matrix):
    return np.concatenate([matrix.real,matrix.imag],axis=0)


def main():
    start=time.perf_counter()
    prior=json.loads(PRIOR.read_text(encoding='utf-8'))
    if not prior['all_checks_pass'] or not all(sha(Path(p))==h for p,h in prior['identity'].items()):
        raise RuntimeError('Prior identity mismatch')
    geometry=next(Path(p) for p in prior['identity'] if p.endswith('result.json'))
    iq_path=next(Path(p) for p in prior['identity'] if p.endswith('.npz'))
    data=json.loads(geometry.read_text(encoding='utf-8'))
    with np.load(iq_path) as iq:
        freq,t,coef,observed=[iq[k] for k in ['frequency_hz','t_s','coefficients','mixture']]
    stations=np.array(data['stations_m'])
    positions=np.array(data['positions_m'])[[9,12]]
    rays=[]
    for q,sid in enumerate([9,12]):
        for r in sorted([r for r in data['records'] if r['scene']=='two_AB' and r['source_id']==sid],key=lambda r:r['station_id']):
            for p in r['paths']:
                label=-1 if p['kind']=='direct' else {'A':0,'B':1}[p['panel']]
                rays.append((q,r['station_id'],label,(1. if label==-1 else -.5)/p['length_m']))
    out=ROOT/('iq_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    identity={str(p.resolve(strict=True)):sha(p) for p in [Path(__file__),ROOT/'执行合同.md',PRIOR,geometry,iq_path]}
    (out/'preflight.json').write_text(json.dumps({'sha256':identity,'python':platform.python_version(),'numpy':np.__version__,'threads':2},indent=2),encoding='utf-8')
    for p in [Path(__file__),ROOT/'执行合同.md']:
        (out/p.name).write_bytes(p.read_bytes())
    V=np.exp(2j*np.pi*freq[:,None]*t[None,:])
    x=np.r_[positions.ravel()/SCALE,0.,-.25,np.pi/2,-.55]
    def components(v,derivatives=False):
        channels=np.zeros((2,5,len(freq)),complex)
        components=[];length_derivatives=[]
        for q,m,label,gain in rays:
            p=v[3*q:3*q+3]*SCALE
            grad=np.zeros(10)
            if label==-1:
                im=p;transform=np.eye(3)
            else:
                theta,offset=v[6+2*label:8+2*label]
                offset*=SCALE
                n=np.array([np.cos(theta),np.sin(theta),0.])
                dn=np.array([-np.sin(theta),np.cos(theta),0.])
                im=p-2*(p@n-offset)*n
                transform=np.eye(3)-2*np.outer(n,n)
            length=np.linalg.norm(im-stations[m]);u=(im-stations[m])/length
            grad[3*q:3*q+3]=u@transform*SCALE
            if label!=-1:
                grad[6+2*label]=u@(-2*((p@dn)*n+(p@n-offset)*dn))
                grad[7+2*label]=u@(2*n)*SCALE
            contribution=gain*np.exp(-2j*np.pi*freq*length/C)
            channels[q,m]+=contribution
            components.append(contribution);length_derivatives.append(grad)
        return (channels,components,np.array(length_derivatives)) if derivatives else channels
    H,parts,dl=components(x,True)
    synthesis=np.einsum('qmf,qf,fn->mn',H,coef,V,optimize=True)
    reconstruction=float(np.linalg.norm(synthesis-observed)/np.linalg.norm(observed))
    # Complex waveform tangent, stacked time samples, including exact synthesis Gram.
    W=np.concatenate([(H[q,:,:,None]*V[None,:,:]).transpose(0,2,1).reshape(5*len(t),len(freq)) for q in range(2)],axis=1)
    W/=np.linalg.norm(W,axis=0)
    U,values,_=svd(W,full_matrices=False,check_finite=False)
    wave_singular=values.tolist()
    variants=[('original',coef)]
    for seed in [781,782]:
        rng=np.random.default_rng(seed)
        variants.append((f'phase_{seed}',coef*np.exp(2j*np.pi*rng.random(coef.shape))))
    def tangents(c):
        G=np.zeros((5,len(t),10),complex)
        gains=np.zeros((5,len(t),len(rays)),complex)
        for k,(q,m,_,_) in enumerate(rays):
            gains[m,:,k]=(parts[k]*c[q])@V
            timing=(-2j*np.pi*freq/C*parts[k]*c[q])@V
            G[m]+=timing[:,None]*dl[k][None,:]
        return G.reshape(-1,10),gains.reshape(-1,len(rays))
    results=[]; fd_errors=[]
    for name,c in variants:
        G,gains=tangents(c)
        norm=np.linalg.norm(np.einsum('qmf,qf,fn->mn',H,c,V,optimize=True))
        if name=='original':
            h=1e-7
            fd=np.column_stack([((np.einsum('qmf,qf,fn->mn',components(x+np.eye(10)[i]*h),c,V,optimize=True)
                                -np.einsum('qmf,qf,fn->mn',components(x-np.eye(10)[i]*h),c,V,optimize=True))/(2*h)).ravel() for i in range(10)])
            fd_errors=(np.linalg.norm(fd-G,axis=0)/np.maximum(np.linalg.norm(G,axis=0),1e-30)).tolist()
        for cutoff in [1e-8,1e-10,1e-12]:
            r=int(np.sum(values>values[0]*cutoff));Q=U[:,:r]
            projected=np.column_stack([G,gains])/norm
            projected-=Q@(Q.conj().T@projected)
            position=realify(projected[:,:6])
            nuisance=np.column_stack([realify(projected[:,6:10]),realify(projected[:,10:]),realify(1j*projected[:,10:])])
            scales=np.linalg.norm(nuisance,axis=0)
            nuisance/=np.maximum(scales,1e-30)
            Nu,ns,_=svd(nuisance,full_matrices=False,check_finite=False)
            nr=int(np.sum(ns>ns[0]*cutoff))
            final=position-Nu[:,:nr]@(Nu[:,:nr].T@position)
            sv=np.linalg.svd(final,compute_uv=False)
            rank=int(np.sum(sv>sv[0]*1e-8))
            results.append({'waveform':name,'cutoff':cutoff,'waveform_complex_rank':r,'other_nuisance_real_rank':nr,
                            'position_singular_values':sv.tolist(),'position_rank':rank,'relative_smallest':float(sv[-1]/sv[0])})
    # Negative control: frequency-dependent gains absorb arbitrary delay changes exactly.
    shifted=x.copy();shifted[:3]+=np.array([1.,-2.,1.])/SCALE
    _,newparts,newdl=components(shifted,True)
    del newdl
    recovered=np.zeros_like(observed)
    for k,(q,m,_,_) in enumerate(rays):
        free_ratio=parts[k]/newparts[k]
        recovered[m]+=(newparts[k]*free_ratio*coef[q])@V
    free_error=float(np.linalg.norm(recovered-observed)/np.linalg.norm(observed))
    checks={'exact_time_synthesis':reconstruction<1e-10,'finite_difference':max(fd_errors)<1e-5,
            'free_frequency_gain_negative_control':free_error<1e-10,
            'input_unchanged':all(sha(Path(p))==h for p,h in identity.items())}
    result={'identity':identity,'information':'LOCAL_TRUE_GEOMETRY_AND_SUPPORT_FINITE_KNOWN_FREQUENCY_BASIS_UNKNOWN_COEFFICIENTS_AND_GAINS',
            'synthesis_error':reconstruction,'waveform_singular_values':wave_singular,'finite_difference_relative_errors':fd_errors,
            'results':results,'free_gain_exact_replay_error':free_error,'checks':checks,'all_checks_pass':all(checks.values()),
            'wall_seconds':time.perf_counter()-start}
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'out':str(out),'checks':checks,'fd_max':max(fd_errors),'results':results,'seconds':result['wall_seconds']}))
    if not result['all_checks_pass']:
        raise RuntimeError('IQ numerical verification failed')


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
