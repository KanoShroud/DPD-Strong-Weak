"""Frozen paired generation, tests and bounded rounds; truth never enters solve."""
import argparse
import hashlib
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize._numdiff import approx_derivative
from threadpoolctl import threadpool_limits

from model import C, STATIONS, Model, pack, solve, evaluate

ROOT=Path(__file__).resolve().parent
LAYOUTS=np.array([[[-260.,-180.,65.],[230.,120.,100.]],[[-520.,120.,90.],[180.,-260.,55.]]])
FS=50e6
B=12.5e6


def write(path,value):
    def default(x):
        if isinstance(x,np.ndarray):
            return x.tolist()
        if isinstance(x,np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value,default=default,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(layout,seed,n,nonperiodic=False,noiseless=False):
    rng=np.random.default_rng(seed+10000*layout)
    p=LAYOUTS[layout]
    full=n*4 if nonperiodic else n
    f=np.fft.fftfreq(full,1/FS)
    mask=abs(f)<=B/2
    spectrum=np.zeros((2,full),complex)
    spectrum[:,mask]=rng.normal(size=(2,mask.sum()))+1j*rng.normal(size=(2,mask.sum()))
    spectrum/=np.sqrt(np.sum(abs(spectrum)**2,axis=1,keepdims=True))
    phase=np.exp(1j*rng.uniform(-np.pi,np.pi,5))
    distance=np.linalg.norm(p[:,None,:]-STATIONS[None,:,:],axis=-1)
    virtual=p.copy();virtual[:,0]=1800-p[:,0]
    reflected=np.linalg.norm(virtual[:,None,:]-STATIONS[None,:,:],axis=-1)
    los=[];echo=[]
    cut=slice(n,2*n) if nonperiodic else slice(None)
    for q in range(2):
        signal=[];reflection=[]
        for m in range(5):
            direct=spectrum[q]*np.exp(-2j*np.pi*f*distance[q,m]/C)*(500/distance[q,m])*phase[m]
            bounce=spectrum[q]*np.exp(-2j*np.pi*f*reflected[q,m]/C)*(-.7*500/reflected[q,m])*phase[m]
            signal.append((np.fft.ifft(direct)*full)[cut])
            reflection.append((np.fft.ifft(bounce)*full)[cut])
        los.append(signal);echo.append(reflection)
    los=np.array(los);echo=np.array(echo)
    base_power=np.mean(abs(los[1])**2)
    sigma2=base_power/(10**(10/10))/(B/FS)
    noise=np.sqrt(sigma2/2)*(rng.normal(size=(5,n))+1j*rng.normal(size=(5,n)))
    if noiseless:
        noise*=0
    return los,echo,noise,{'truth':p,'distance':distance,'reflected_length':reflected,
                         'direct_gain':500/distance*phase[None,:],
                         'sigma2':sigma2,'snr_A_db':10*np.log10(np.mean(abs(los[0])**2)/(sigma2*B/FS)),
                         'snr_B_db':10.,'reflection_to_direct_db':10*np.log10(np.mean(abs(echo)**2,axis=(1,2))/np.mean(abs(los)**2,axis=(1,2)))}


def preprocess(iq,hann=False):
    n=iq.shape[-1]
    win=np.hanning(n) if hann else np.ones(n)
    f=np.fft.fftfreq(n,1/FS)
    mask=abs(f)<=7e6
    return (np.fft.fft(iq*win,axis=-1)/n)[:,mask],f[mask]


def tests():
    los,echo,noise,meta=generate(0,77,1024,noiseless=True)
    y,f=preprocess(los.sum(axis=0))
    g=meta['direct_gain']/meta['direct_gain'][:,:1]
    x=pack(meta['truth'],g)
    model=Model(y,f)
    loss,_=model.value_grad(x,2)
    shifted=x.copy();shifted[:6]+=np.array([.007,-.004,.003,-.002,.006,.001])
    value,analytic=model.value_grad(shifted,2)
    numerical=approx_derivative(lambda z:model.value_grad(z,2)[0],shifted,method='3-point').ravel()
    grad_error=float(np.linalg.norm(analytic-numerical)/np.linalg.norm(numerical))
    a,_,_,coef,r=model.components(shifted,2)
    direct=np.stack([np.linalg.lstsq(aa,yy,rcond=None)[0] for aa,yy in zip(a,model.y)])
    direct_res=model.y-np.einsum('fmk,fk->fm',a,direct)
    replay=float(np.linalg.norm(direct_res-r)/np.linalg.norm(model.y))
    sy,sf=preprocess(los[0])
    single=Model(sy,sf)
    sp=meta['truth'][0]+[6,-4,3]
    _,sg=single.single(sp)
    sn=approx_derivative(lambda z:single.single(z*1000)[0],sp/1000,method='3-point').ravel()
    single_grad=float(np.linalg.norm(sg-sn)/np.linalg.norm(sn))
    score,gain=single.single(sp,True)
    eig_ls=abs(1-score-single.value_grad(pack(sp[None,:],gain[None,:]),1)[0])
    swapped=pack(meta['truth'][::-1],g[::-1])
    swap=abs(model.value_grad(swapped,2)[0]-loss)
    # Independent direct FFT relation and correct subtraction pairing.
    spec=np.fft.fft(los[0],axis=-1)/1024
    energy=abs(np.sum(abs(spec)**2)-np.mean(abs(los[0])**2,axis=-1).sum())
    checks={'noiseless_los_exact':loss<1e-20,'joint_gradient':grad_error<1e-5,
            'single_gradient':single_grad<1e-5,'ls_replay':replay<1e-10,
            'eigenvalue_vs_ls':eig_ls<1e-10,'source_exchange':swap<1e-12,
            'parseval':energy<1e-10,'reflection_later':bool(np.all(meta['reflected_length']>meta['distance'])),
            'pairing':bool(np.allclose((los.sum(axis=0)+echo.sum(axis=0)+noise)-(los[0]+echo[0]+noise),los[1]+echo[1])),
            'matching_swap':evaluate(meta['truth'][::-1],meta['truth'])['rmse_m']==0}
    return {'checks':checks,'all_checks_pass':all(checks.values()),'joint_gradient_relative':grad_error,
            'single_gradient_relative':single_grad,'direct_ls_replay':replay,'true_loss':loss,'test_loss':value}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['tests','control','periodic','nonperiodic'])
    args=parser.parse_args()
    out=ROOT/(args.stage+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    source=[ROOT/'model.py',ROOT/'experiment.py',ROOT/'执行合同.md']
    identity={str(p):sha(p) for p in source}
    for p in source:
        (out/p.name).write_bytes(p.read_bytes())
    write(out/'preflight.json',{'identity':identity,'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'threads':2})
    test=tests();write(out/'tests.json',test)
    if not test['all_checks_pass']:
        raise RuntimeError(test)
    if args.stage=='tests':
        print(json.dumps({'out':str(out),**test}));return
    rows=[];started=time.perf_counter()
    if args.stage=='control':
        jobs=[(0,77)]
    else:
        jobs=[(layout,seed) for layout in range(2) for seed in ([200,201] if args.stage=='nonperiodic' else [100,101])]
    for layout,seed in jobs:
        nonperiodic=args.stage=='nonperiodic'
        n=4096 if nonperiodic else 2048
        los,echo,noise,meta=generate(layout,seed,n,nonperiodic,args.stage=='control')
        truth=meta.pop('truth');gain=meta.pop('direct_gain')
        filename=out/f'input_{layout}_{seed}.npz'
        np.savez_compressed(filename,los=los,echo=echo,noise=noise,truth=truth,direct_gain=gain)
        input_sha=sha(filename)
        conditions=[('dual_los',[0,1],False)] if args.stage=='control' else [
            ('A_los',[0],False),('B_los',[1],False),('dual_los',[0,1],False),
            ('A_mp',[0],True),('B_mp',[1],True),('dual_mp',[0,1],True)]
        for name,ids,mp in conditions:
            iq=los[ids].sum(axis=0)+noise
            if mp:
                iq=iq+echo[ids].sum(axis=0)
            y,f=preprocess(iq,nonperiodic)
            output=solve(y,f,len(ids))
            result={'layout':layout,'seed':seed,'condition':name,'stage':args.stage,'n':n,
                    'input_sha256':input_sha,'public':{'stations_m':STATIONS,'fs':FS,'B':B,'k':len(ids),'hann':nonperiodic},
                    'generator_metadata':meta,'output':output,
                    'evaluation':evaluate(output['selected']['positions'],truth[ids]),
                    'spectrum_evaluation':evaluate(output['spectrum_peaks'],truth[ids])}
            write(out/f'{layout}_{seed}_{name}.json',result);rows.append(result)
            print(json.dumps({'case':[layout,seed,name],'rmse':result['evaluation']['rmse_m'],
                  'seconds':output['seconds'],'stop':output['selected']['message']},ensure_ascii=False),flush=True)
        if sha(filename)!=input_sha:
            raise RuntimeError('Input identity changed')
    result={'stage':args.stage,'count':len(rows),'seconds':time.perf_counter()-started,
            'identity':identity,'source_unchanged':all(sha(Path(p))==h for p,h in identity.items()),
            'numerical_tests_pass':test['all_checks_pass'],
            'rows':[{'layout':r['layout'],'seed':r['seed'],'condition':r['condition'],
                     **r['evaluation'],'seconds':r['output']['seconds'],
                     'loss':r['output']['selected']['loss'],'gradient':r['output']['selected']['gradient_inf'],
                     'iterations':r['output']['selected']['iterations'],'stop':r['output']['selected']['message']} for r in rows]}
    write(out/'result.json',result)
    print('COMPLETED '+str(out),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
