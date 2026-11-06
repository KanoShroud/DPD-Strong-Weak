"""Independent raw-time sensitivity and algebraic root stress checks."""
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.linalg import svd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
SCALE=1000.
C=299792458.


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def re(matrix):
    return np.r_[matrix.real,matrix.imag]


def main():
    start=time.perf_counter()
    first=ROOT/'support_20261025_195234_243209/result.json'
    second=ROOT/'iq_20261025_195523_357517/result.json'
    r1=json.loads(first.read_text(encoding='utf-8'))
    r2=json.loads(second.read_text(encoding='utf-8'))
    for r in [r1,r2]:
        if not r['all_checks_pass'] or not all(sha(Path(p))==h for p,h in r['identity'].items()):
            raise RuntimeError('Prior identity failed')
    geometry=next(Path(p) for p in r2['identity'] if 'finite_scene' in p and p.endswith('result.json'))
    iq_path=next(Path(p) for p in r2['identity'] if p.endswith('.npz'))
    data=json.loads(geometry.read_text(encoding='utf-8'))
    with np.load(iq_path) as iq:
        freq,t,coeff,observed=[iq[k] for k in ['frequency_hz','t_s','coefficients','mixture']]
    stations=np.array(data['stations_m'])
    truepos=np.array(data['positions_m'])[[9,12]]
    rays=[]
    for q,sid in enumerate([9,12]):
        for r in sorted([r for r in data['records'] if r['scene']=='two_AB' and r['source_id']==sid],key=lambda r:r['station_id']):
            for p in r['paths']:
                label=-1 if p['kind']=='direct' else {'A':0,'B':1}[p['panel']]
                rays.append((q,r['station_id'],label,(1. if label==-1 else -.5)/p['length_m']))
    out=ROOT/('stress_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    files=[Path(__file__),ROOT/'压力核验补充合同.md',ROOT/'enumerate_support.py',first,second,geometry,iq_path]
    identity={str(p.resolve(strict=True)):sha(p) for p in files}
    (out/'preflight.json').write_text(json.dumps({'sha256':identity,'numpy':np.__version__,'threads':2},indent=2),encoding='utf-8')
    for p in files[:2]:
        (out/p.name).write_bytes(p.read_bytes())
    x=np.r_[truepos.ravel()/SCALE,0.,-.25,np.pi/2,-.55]
    V=np.exp(2j*np.pi*freq[:,None]*t[None,:])
    def channels(v,f,sources):
        H=np.zeros((len(sources),5,len(f)),complex)
        parts=[]
        for q,m,label,gain in rays:
            if q not in sources:
                continue
            receiver=stations[m].copy()
            if label>=0:
                theta,offset=v[6+2*label:8+2*label]
                n=np.array([np.cos(theta),np.sin(theta),0.])
                receiver-=2*(receiver@n-offset*SCALE)*n
            length=np.linalg.norm(v[3*q:3*q+3]*SCALE-receiver)
            a=gain*np.exp(-2j*np.pi*f*length/C)
            H[sources.index(q),m]+=a
            parts.append((q,m,label,a))
        return H,parts
    def predicted(v,sources):
        H,_=channels(v,freq,sources)
        return np.einsum('qmf,qf,fn->mn',H,coeff[sources],V,optimize=True)
    replay_error=float(np.linalg.norm(predicted(x,[0,1])-observed)/np.linalg.norm(observed))
    cases=[('A_only',[0],0,False),('B_only',[1],0,False),('joint_shared',[0,1],0,False),
           ('joint_independent_delays',[0,1],0,True),('joint_plus64',[0,1],64,False),('joint_plus256',[0,1],256,False)]
    results=[]
    for name,sources,extra,independent in cases:
        begin=time.perf_counter()
        f=np.r_[freq,np.linspace(-6.25e6,6.25e6,extra)] if extra else freq
        v=np.exp(2j*np.pi*f[:,None]*t[None,:])
        H,parts=channels(x,f,sources)
        W=np.concatenate([(H[q,:,:,None]*v[None,:,:]).transpose(0,2,1).reshape(5*len(t),len(f)) for q in range(len(sources))],axis=1)
        W/=np.linalg.norm(W,axis=0)
        U,s,_=svd(W,full_matrices=False,check_finite=False)
        r=int(np.sum(s>s[0]*1e-12));Q=U[:,:r]
        h=3e-7
        geom=np.column_stack([((predicted(x+np.eye(10)[i]*h,sources)-predicted(x-np.eye(10)[i]*h,sources))/(2*h)).ravel() for i in range(10)])
        gains=np.zeros((5,len(t),len(parts)),complex)
        delays=[]
        for k,(q,m,label,a) in enumerate(parts):
            cc=np.r_[coeff[q],np.zeros(extra)]
            gains[m,:,k]=(a*cc)@v
            if label>=0:
                d=np.zeros((5,len(t)),complex)
                d[m]=(-2j*np.pi*f/C*SCALE*a*cc)@v
                delays.append(d.ravel())
        posids=[3*q+j for q in sources for j in range(3)]
        nuisance_geometry=np.column_stack(delays) if independent else geom[:,6:]
        joined=np.column_stack([geom[:,posids],nuisance_geometry,gains.reshape(5*len(t),-1)])
        norm=np.linalg.norm(predicted(x,sources));joined/=norm
        reference=float(np.linalg.svd(re(joined[:,:len(posids)]),compute_uv=False)[0])
        joined-=Q@(Q.conj().T@joined)
        npos=len(posids);ng=nuisance_geometry.shape[1]
        P=re(joined[:,:npos])
        N=np.column_stack([re(joined[:,npos:npos+ng]),re(joined[:,npos+ng:]),re(1j*joined[:,npos+ng:])])
        norms=np.linalg.norm(N,axis=0);N=N[:,norms>1e-10]/norms[norms>1e-10]
        Nu,ns,_=svd(N,full_matrices=False,check_finite=False)
        nr=int(np.sum(ns>ns[0]*1e-10))
        P-=Nu[:,:nr]@(Nu[:,:nr].T@P)
        vals=np.linalg.svd(P,compute_uv=False)
        results.append({'case':name,'waveform_complex_rank':r,'position_rank':int(np.sum(vals>reference*1e-8)),
                        'singular_values':vals.tolist(),'rank_absolute_threshold':reference*1e-8,
                        'seconds':time.perf_counter()-begin})
        print(json.dumps(results[-1]),flush=True)
    # Positive algebraic tests deliberately with three LOS, not four.
    spec=importlib.util.spec_from_file_location('support_enum',ROOT/'enumerate_support.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    tests=[]
    for seed in range(910,918):
        rng=np.random.default_rng(seed)
        p=np.r_[rng.uniform(-150,150,2),rng.uniform(40,110)]
        planes=[(np.array([1.,0.,0.]),-250.),(np.array([0.,1.,0.]),-550.)]
        virtual=[p]+[module.previous.mirror(p,plane) for plane in planes]
        events=[];labels=[];observations=[]
        for label,ids in zip([-1,0,1],[[0,1,2],[1,2,4],[0,2,3]]):
            for m in ids:
                events.append((1,m,0));labels.append(label)
                observations.append(np.linalg.norm(virtual[label+1]-stations[m]))
        observations=np.array(observations)-np.min(observations)
        roots,status=module.virtual_roots(stations,events,observations,labels)
        err=min((np.linalg.norm(r['position']-p) for r in roots),default=1e99)
        tests.append({'seed':seed,'status':status,'roots':len(roots),'positive_error_m':float(err),'pass':bool(err<1e-6)})
    joint=next(r for r in results if r['case']=='joint_shared')
    earlier=r2['results'][2]['position_singular_values']
    consistency=float(np.linalg.norm(np.array(joint['singular_values'])-earlier)/np.linalg.norm(earlier))
    checks={'replay':replay_error<1e-10,'independent_implementation':consistency<1e-5,
            'three_LOS_algebra_positive_controls':all(t['pass'] for t in tests),
            'frozen_inputs':all(sha(Path(p))==h for p,h in identity.items())}
    result={'identity':identity,'cases':results,'algebra_tests':tests,'time_replay_error':replay_error,
            'independent_singular_value_relative_difference':consistency,'checks':checks,'all_checks_pass':all(checks.values()),'seconds':time.perf_counter()-start}
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'out':str(out),'checks':checks,'seconds':result['seconds']}))
    if not result['all_checks_pass']:
        raise RuntimeError('Stress checks failed')


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
