"""Conventional unknown-gain LOS DPD, not a multipath-aware new method."""
import time

import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

C = 299792458.
STATIONS = np.array([[-400.,-300.,4.],[450.,-250.,18.],[380.,400.,8.],[-350.,350.,28.],[20.,-460.,12.]])
BOUNDS = [(-1.,1.),(-1.,1.),(.02,.15)]


def delays(p, stations):
    delta = p[:,None,:]-stations[None,:,:]
    d = np.linalg.norm(delta,axis=-1)
    t = (d-d[:,:1])/C
    deriv = (delta/d[:,:,None]-delta[:,:1,:]/d[:,:1,None])*1000/C
    return t, deriv


def pack(p, g):
    return np.r_[p.ravel()/1000, g[:,1:].real.ravel(), g[:,1:].imag.ravel()]


def unpack(x, k, m):
    p = x[:3*k].reshape(k,3)*1000
    d = k*(m-1)
    g = np.ones((k,m),complex)
    g[:,1:] = (x[3*k:3*k+d]+1j*x[3*k+d:]).reshape(k,m-1)
    return p,g


class Model:
    def __init__(self, y, f, stations=STATIONS):
        self.y = np.asarray(y).T.copy()  # frequency, station
        self.f = f
        self.stations = stations.copy()
        self.energy = float(np.vdot(y,y).real)
        self.calls = 0

    def components(self, x, k):
        p,g = unpack(x,k,len(self.stations))
        t,dt = delays(p,self.stations)
        e = np.exp(-2j*np.pi*self.f[:,None,None]*t.T[None,:,:])
        a = e*g.T[None,:,:]
        gram = np.einsum('fmk,fml->fkl',a.conj(),a)
        rhs = np.einsum('fmk,fm->fk',a.conj(),self.y)
        # Exact small LS, no ridge or truth-dependent regularization.
        coeff = np.linalg.solve(gram,rhs[...,None])[...,0]
        residual = self.y-np.einsum('fmk,fk->fm',a,coeff)
        return a,e,dt,coeff,residual

    def value_grad(self, x, k):
        self.calls += 1
        a,e,dt,s,r = self.components(x,k)
        common = r.conj()[:,:,None]*s[:,None,:]
        spatial = (-2/self.energy*np.real(np.einsum('fmk,fmk,kmd->kd',
                    common, a*(-2j*np.pi*self.f[:,None,None]),dt)))
        gains = -2/self.energy*np.einsum('fmk,fmk->km',common,e)
        grad = np.r_[spatial.ravel(),gains[:,1:].real.ravel(),(-gains[:,1:].imag).ravel()]
        return float(np.vdot(r,r).real/self.energy),grad

    def fit(self, p, g):
        k = len(p)
        x = pack(p,g)
        trace = []
        started = time.perf_counter()
        def callback(z):
            value,grad = self.value_grad(z,k)
            trace.append([value,float(np.max(abs(grad)))])
        initial = self.value_grad(x,k)[0]
        result = minimize(lambda z:self.value_grad(z,k),x,jac=True,method='L-BFGS-B',
            bounds=BOUNDS*k+[(None,None)]*(2*k*(len(self.stations)-1)),callback=callback,
            options={'maxiter':300,'maxfun':1200,'gtol':1e-7,'ftol':1e-12,'maxls':30})
        value,grad = self.value_grad(result.x,k)
        pp,gg = unpack(result.x,k,len(self.stations))
        return {'positions':pp.tolist(),'g_real':gg.real.tolist(),'g_imag':gg.imag.tolist(),
                'loss':value,'initial_loss':initial,'gradient_inf':float(np.max(abs(grad))),
                'optimizer_success':bool(result.success),'message':str(result.message),
                'iterations':int(result.nit),'trace':trace,'seconds':time.perf_counter()-started,
                'x':result.x.tolist()}

    def single(self,p,return_gain=False):
        t,dt = delays(np.atleast_2d(p),self.stations)
        phase = np.exp(2j*np.pi*self.f[:,None]*t[0][None,:])
        aligned = self.y*phase
        covariance = aligned.T@aligned.conj()
        w,v = np.linalg.eigh(covariance)
        g = v[:,-1]
        score = float(w[-1]/self.energy)
        if return_gain:
            return score,g/g[0]
        a = phase.conj()*g[None,:]
        s = np.einsum('fm,fm->f',a.conj(),self.y)
        r = self.y-a*s[:,None]
        deriv = -2/self.energy*np.real(np.einsum('fm,fm,f,md->d',
             r.conj(),a*(-2j*np.pi*self.f[:,None]),s,dt[0]))
        return 1-score,deriv

    def grid_scores(self, grid):
        scores=[]
        for start in range(0,len(grid),64):
            pp=grid[start:start+64]
            t,_=delays(pp,self.stations)
            aligned=self.y[None,:,:]*np.exp(2j*np.pi*self.f[None,:,None]*t[:,None,:])
            cov=np.einsum('bfm,bfn->bmn',aligned,aligned.conj())
            scores.extend((np.linalg.eigvalsh(cov)[:,-1]/self.energy).tolist())
        return np.array(scores)

    def candidates(self):
        axis=np.arange(-1000.,1001.,20.)
        grid=np.array(np.meshgrid(axis,axis,[20.,40.,60.,80.,100.,120.,140.,150.],indexing='ij')).reshape(3,-1).T
        scores=self.grid_scores(grid)
        chosen=[]
        for i in np.argsort(-scores,kind='stable'):
            if all(np.linalg.norm(grid[i]-q)>=60 for q in chosen):
                chosen.append(grid[i])
            if len(chosen)==6:
                break
        result=[]
        for p in chosen:
            opt=minimize(lambda z:self.single(z*1000),p/1000,jac=True,method='L-BFGS-B',
               bounds=BOUNDS,options={'maxiter':300,'gtol':1e-7,'ftol':1e-12,'maxls':30})
            pos=opt.x*1000
            score,g=self.single(pos,True)
            if all(np.linalg.norm(pos-r['p'])>1 for r in result):
                result.append({'p':pos,'g':g,'score':score,'status':str(opt.message)})
        return sorted(result,key=lambda r:-r['score'])


def solve(y,f,k,stations=STATIONS):
    started=time.perf_counter()
    model=Model(y,f,stations)
    candidates=model.candidates()
    first=candidates[0]
    naive=[r['p'].tolist() for r in candidates[:k]]
    if k==1:
        branches=[model.fit(np.array([first['p']]),np.array([first['g']]))]
    else:
        _,_,_,_,res=model.components(pack(np.array([first['p']]),np.array([first['g']])),1)
        second=Model(res.T,f,stations).candidates()
        branches=[]
        for r in second[:3]:
            if np.linalg.norm(first['p']-r['p'])<1:
                continue
            branches.append(model.fit(np.array([first['p'],r['p']]),np.array([first['g'],r['g']])))
    best=min(branches,key=lambda r:r['loss'])
    return {'selected':best,'branches':branches,'spectrum_peaks':naive,
            'seconds':time.perf_counter()-started,'objective_calls':model.calls}


def evaluate(pred,truth):
    pred=np.asarray(pred);truth=np.asarray(truth)
    cost=np.sum((pred[:,None,:]-truth[None,:,:])**2,axis=-1)
    i,j=linear_sum_assignment(cost)
    if len(i)!=len(truth):
        return {'missing':True}
    error=np.zeros_like(truth)
    error[j]=pred[i]-truth[j]
    return {'errors_m':np.linalg.norm(error,axis=1).tolist(),
            'horizontal_m':np.linalg.norm(error[:,:2],axis=1).tolist(),
            'vertical_m':abs(error[:,2]).tolist(),'rmse_m':float(np.sqrt(np.mean(np.sum(error**2,axis=1)))),
            'all_within_20m':bool(np.all(np.linalg.norm(error,axis=1)<=20))}
