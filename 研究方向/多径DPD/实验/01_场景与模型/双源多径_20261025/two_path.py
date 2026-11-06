"""Conventional map-free two-tap profiled LS, not a claimed new algorithm."""
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.optimize._numdiff import approx_derivative
from threadpoolctl import threadpool_limits

from experiment import ROOT, B, preprocess, write, sha
from model import C, STATIONS, BOUNDS, Model, pack, unpack, delays, evaluate


class TwoPath:
    def __init__(self,y,f,k):
        self.y=y.T;self.f=f;self.k=k;self.m=y.shape[0]
        self.energy=float(np.vdot(y,y).real)
        self.base=3*k+2*k*(self.m-1)

    def residual_jac(self,x):
        k,m=self.k,self.m
        p,g=unpack(x[:self.base],k,m)
        rho,theta,delay=x[self.base:].reshape(3,k,m)
        t,dt=delays(p,STATIONS)
        e=np.exp(-2j*np.pi*self.f[:,None,None]*t.T[None,:,:])
        bounce=np.exp(1j*theta.T[None,:,:]-2j*np.pi*self.f[:,None,None]*delay.T[None,:,:]*1e-6)
        h=rho.T[None,:,:]*bounce
        a=e*g.T[None,:,:]*(1+h)
        da=np.zeros((*a.shape,len(x)),complex)
        for q in range(k):
            da[:,:,q,3*q:3*q+3]=a[:,:,q,None]*(-2j*np.pi*self.f[:,None,None])*dt[q][None,:,:]
            for station in range(m):
                if station:
                    index=3*k+q*(m-1)+station-1
                    da[:,station,q,index]=e[:,station,q]*(1+h[:,station,q])
                    da[:,station,q,index+k*(m-1)]=1j*e[:,station,q]*(1+h[:,station,q])
                index=self.base+q*m+station
                da[:,station,q,index]=e[:,station,q]*g[q,station]*bounce[:,station,q]
                da[:,station,q,index+k*m]=1j*e[:,station,q]*g[q,station]*h[:,station,q]
                da[:,station,q,index+2*k*m]=e[:,station,q]*g[q,station]*h[:,station,q]*(-2j*np.pi*self.f*1e-6)
        gram=np.einsum('fmk,fml->fkl',a.conj(),a)
        s=np.linalg.solve(gram,np.einsum('fmk,fm->fk',a.conj(),self.y)[...,None])[...,0]
        r=self.y-np.einsum('fmk,fk->fm',a,s)
        das=np.einsum('fmkj,fk->fmj',da,s)
        ds=np.linalg.solve(gram,np.einsum('fmkj,fm->fkj',da.conj(),r)-np.einsum('fmk,fmj->fkj',a.conj(),das))
        dr=-das-np.einsum('fmk,fkj->fmj',a,ds)
        z=np.r_[r.real.ravel(),r.imag.ravel()]/np.sqrt(self.energy)
        jac=np.concatenate([dr.real.reshape(-1,len(x)),dr.imag.reshape(-1,len(x))])/np.sqrt(self.energy)
        return z,jac

    def fit(self,x):
        begin=time.perf_counter();cache={};trace=[]
        def compute(z):
            if 'x' not in cache or not np.array_equal(cache['x'],z):
                r,j=self.residual_jac(z);cache.update(x=z.copy(),r=r,j=j)
                trace.append(float(r@r))
            return cache['r'],cache['j']
        bounds=BOUNDS*self.k+[(-np.inf,np.inf)]*(self.base-3*self.k)+[(0,.9)]*(self.k*self.m)+[(-np.inf,np.inf)]*(self.k*self.m)+[(0,10)]*(self.k*self.m)
        opt=least_squares(lambda z:compute(z)[0],x,jac=lambda z:compute(z)[1],bounds=np.array(bounds).T,
                          method='trf',x_scale='jac',gtol=1e-7,ftol=1e-12,xtol=1e-12,max_nfev=300)
        r,j=compute(opt.x)
        p,_=unpack(opt.x[:self.base],self.k,self.m)
        return {'positions':p.tolist(),'x':opt.x.tolist(),'loss':float(r@r),'gradient_inf':float(np.max(abs(2*j.T@r))),
                'optimality':float(opt.optimality),'optimizer_success':bool(opt.success),'message':str(opt.message),
                'evaluations':int(opt.nfev),'seconds':time.perf_counter()-begin,'trace':trace}

    def initialize(self,base_x):
        base=Model(self.y.T,self.f)
        a,_,_,s,res=base.components(base_x,self.k)
        rho=np.zeros((self.k,self.m));theta=rho.copy();delay=rho.copy()
        scales=np.array([.25,1,4,16,64])/B
        for q in range(self.k):
            for m in range(self.m):
                atom=a[:,m,q]*s[:,q]
                atoms=atom[:,None]*np.exp(-2j*np.pi*self.f[:,None]*scales[None,:])
                norm=np.sum(abs(atoms)**2,axis=0)
                h=np.einsum('fj,f->j',atoms.conj(),res[:,m])/norm
                index=int(np.argmax(abs(h)**2*norm))
                rho[q,m]=np.clip(abs(h[index]),.03,.8);theta[q,m]=np.angle(h[index]);delay[q,m]=scales[index]*1e6
        return np.r_[base_x,rho.ravel(),theta.ravel(),delay.ravel()]


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('ground_dir');args=parser.parse_args()
    folder=Path(args.ground_dir).resolve(strict=True)
    out=ROOT/('two_path_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir(exist_ok=False)
    files=[Path(__file__),ROOT/'model.py',ROOT/'experiment.py',ROOT/'常规双径检查合同.md']
    identity={str(p):sha(p) for p in files}
    for p in files:
        (out/p.name).write_bytes(p.read_bytes())
    write(out/'preflight.json',{'identity':identity,'input':str(folder),'threads':2})
    rows=[]
    for file in sorted(folder.glob('*_dual_ground.json')):
        prior=json.loads(file.read_text(encoding='utf-8'));layout,seed=prior['layout'],prior['seed']
        inputfile=folder/f'input_{layout}_{seed}.npz'
        if sha(inputfile)!=prior['input_sha256']:
            raise RuntimeError('Frozen input mismatch')
        identity[str(file)]=sha(file);identity[str(inputfile)]=sha(inputfile)
        with np.load(inputfile) as data:
            iq=(data['los']+data['echo']).sum(axis=0)+data['noise'];truth=data['truth'];gain=data['direct_gain']
        y,f=preprocess(iq,True);model=TwoPath(y,f,2)
        init_start=time.perf_counter()
        x=model.initialize(np.array(prior['output']['selected']['x']))
        initialization_seconds=time.perf_counter()-init_start
        # Test all derivative columns on a subset of the SAME public frequencies.
        small=TwoPath(y[:,::9],f[::9],2)
        test=x.copy();test[model.base:model.base+10]=.2;test[-10:]=np.linspace(.03,.3,10)
        r,j=small.residual_jac(test)
        fd=approx_derivative(lambda z:small.residual_jac(z)[0],test,method='3-point')
        error=float(np.linalg.norm(j-fd)/np.linalg.norm(fd))
        zero=test.copy();zero[model.base:model.base+10]=0
        rz,_=small.residual_jac(zero)
        rl,_=Model(y[:,::9],f[::9]).residual_jac(zero[:model.base],2)
        reduction=float(np.linalg.norm(rz-rl))
        pp,gg=unpack(test[:model.base],2,5)
        rr,tt,dd=test[model.base:].reshape(3,2,5)
        distances=np.linalg.norm(pp[:,None,:]-STATIONS[None,:,:],axis=-1)
        trial_f=f[::9]
        aa=np.empty((len(trial_f),5,2),complex)
        for q in range(2):
            for m in range(5):
                aa[:,m,q]=gg[q,m]*np.exp(-2j*np.pi*trial_f*(distances[q,m]-distances[q,0])/C)*(1+rr[q,m]*np.exp(1j*tt[q,m]-2j*np.pi*trial_f*dd[q,m]*1e-6))
        rng=np.random.default_rng(909)
        coeff=rng.normal(size=(len(trial_f),2))+1j*rng.normal(size=(len(trial_f),2))
        exact_y=np.einsum('fmk,fk->fm',aa,coeff).T
        er,_=TwoPath(exact_y,trial_f,2).residual_jac(test)
        capacity=float(np.linalg.norm(er))
        checks={'jacobian':error<1e-5,'zero_echo_matches_los':reduction<1e-10,'independent_synthetic_replay':capacity<1e-10}
        write(out/f'jacobian_{layout}_{seed}.json',{'relative_error':error,'reduction_error':reduction,'capacity_error':capacity,'checks':checks,'pass':all(checks.values())})
        if not all(checks.values()):
            raise RuntimeError('Two-path derivative mismatch')
        blind=model.fit(x)
        # Extra-information diagnostic, never a candidate for the blind output.
        direct=np.linalg.norm(truth[:,None,:]-STATIONS[None,:,:],axis=-1)
        virtual=truth.copy();virtual[:,2]*=-1
        reflected=np.linalg.norm(virtual[:,None,:]-STATIONS[None,:,:],axis=-1)
        oracle_x=np.r_[pack(truth,gain/gain[:,:1]),(.7*direct/reflected).ravel(),np.full(10,np.pi),((reflected-direct)/C*1e6).ravel()]
        oracle=model.fit(oracle_x)
        row={'layout':layout,'seed':seed,'input_sha256':sha(inputfile),'jacobian_relative_error':error,
             'blind':blind,'oracle':oracle,'direct_evaluation':prior['evaluation'],
             'tap_initialization_seconds':initialization_seconds,
             'solver_seconds_including_los_initialization':prior['output']['seconds']+initialization_seconds+blind['seconds'],
             'timing_scope':'saved_LOS_solver_plus_echo_initialization_and_fit_excludes_IQ_FFT_and_development_tests',
             'blind_evaluation':evaluate(blind['positions'],truth),'oracle_evaluation':evaluate(oracle['positions'],truth),
             'additional_prior':'all_LOS_and_reflection_amplitude_ratio_at_most_0.9'}
        rows.append(row);write(out/f'{layout}_{seed}.json',row)
        print(json.dumps({'layout':layout,'direct':prior['evaluation']['rmse_m'],'two_path_blind':row['blind_evaluation']['rmse_m'],
                          'oracle':row['oracle_evaluation']['rmse_m'],'blind_stop':blind['message'],'oracle_stop':oracle['message'],
                          'seconds':blind['seconds']+oracle['seconds']}),flush=True)
    write(out/'result.json',{'identity':identity,'input_source_unchanged':all(sha(Path(p))==h for p,h in identity.items()),'rows':rows})
    print('COMPLETED '+str(out),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
