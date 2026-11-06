"""Budgeted blind block LS with complete finite-panel path generation.

No truth loading here. Nonsmooth visibility and finite initialization remain limitations.
"""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import minimize

MODULE=Path(__file__).resolve().parent.parent/'finite_scene/finite_scene.py'
spec=importlib.util.spec_from_file_location('generic_finite_geometry',MODULE)
geo=importlib.util.module_from_spec(spec);sys.modules[spec.name]=geo;spec.loader.exec_module(geo)
C=299792458.
SCALE=1000.


def initial_pairs(blind,public):
    version=blind['ordinary']['versions']['4']
    fours=[p for p in version['candidate_points'] if p['size']==4][:2]
    pairs=[]
    midpoint=(public['bounds_m'][0][2]+public['bounds_m'][1][2])/2
    for four in fours:
        eligible=[p for p in version['candidate_points'] if p['size']==3 and not set(p['nodes']).issubset(four['nodes'])]
        if not eligible:
            continue
        first=eligible[0]['graph_id']
        curve=[p for p in eligible if p['graph_id']==first]
        p=min(curve,key=lambda p:abs(p['position_m'][2]-midpoint))
        pairs.append({'positions_m':[p['position_m'],four['position_m']],
                      'graph_ids':[p['graph_id'],four['graph_id']],'nodes':[p['nodes'],four['nodes']]})
    return pairs


class Model:
    def __init__(self,y,public):
        self.scale=np.linalg.norm(y)
        self.y=y/self.scale
        self.stations=np.array(public['stations_m'])
        self.bounds=np.array(public['bounds_m'])/SCALE
        n=y.shape[1];t=np.arange(n)/public['fs_hz'];period=3*n/public['fs_hz']
        k=int(np.floor(public['bandwidth_hz']/2*period+1e-9))
        self.freq=np.arange(-k,k+1)/period
        self.V=np.exp(2j*np.pi*self.freq[:,None]*t[None,:])
        _,s,vh=np.linalg.svd(self.y,full_matrices=False)
        init=(s[:2,None]*vh[:2])
        self.initial_coeff=lstsq(self.V.T,init.T,cond=1e-8,lapack_driver='gelsd')[0].T

    def paths(self,x,extents):
        panels=[]
        for k in range(2):
            theta,b=x[6+2*k:8+2*k];middle,width,height=extents[k]
            n=np.array([np.cos(theta),np.sin(theta),0.]);tangent=np.cross([0.,0.,1.],n)
            center=SCALE*(b*n+middle*tangent+np.array([0.,0.,height/2]))
            panels.append(geo.Panel(str(k),center,n,width*SCALE,height*SCALE))
        result=[];counts=[]
        for q in range(2):
            p=x[3*q:3*q+3]*SCALE;los=0
            for m,s in enumerate(self.stations):
                traced=geo.trace(p,s,panels)
                if traced['degenerate']:
                    return None,None
                los+=int(any(r['kind']=='direct' for r in traced['paths']))
                for r in traced['paths']:
                    label=-1 if r['kind']=='direct' else int(r['panel'])
                    grad=np.zeros(10)
                    if label==-1:
                        image=p;transform=np.eye(3)
                    else:
                        theta,b=x[6+2*label:8+2*label];b*=SCALE
                        normal=np.array([np.cos(theta),np.sin(theta),0.]);dn=np.array([-np.sin(theta),np.cos(theta),0.])
                        image=p-2*(p@normal-b)*normal;transform=np.eye(3)-2*np.outer(normal,normal)
                    u=(image-s)/np.linalg.norm(image-s)
                    grad[3*q:3*q+3]=u@transform*SCALE
                    if label>=0:
                        grad[6+2*label]=u@(-2*((p@dn)*normal+(p@normal-b)*dn))
                        grad[7+2*label]=u@(2*normal)*SCALE
                    result.append((q,m,label,r['length_m'],grad))
            counts.append(los)
        if min(counts)<3:
            return None,None
        return result,counts

    def signals(self,paths,coeff):
        coefficients=np.array([coeff[q]*np.exp(-2j*np.pi*self.freq*length/C) for q,_,_,length,_ in paths])
        values=coefficients@self.V
        derivative=(coefficients*(-2j*np.pi*self.freq/C))@self.V
        return values,derivative

    def gain_fit(self,x,extents,coeff):
        paths,counts=self.paths(x,extents)
        if paths is None:
            return None
        values,_=self.signals(paths,coeff)
        gains=np.zeros((2,5,3),complex);prediction=np.zeros_like(self.y)
        for m in range(5):
            ids=[i for i,p in enumerate(paths) if p[1]==m]
            if not ids:
                continue
            A=values[ids].T
            g=lstsq(A,self.y[m],cond=1e-8,lapack_driver='gelsd')[0]
            prediction[m]=A@g
            for i,value in zip(ids,g):
                q,_,label,_,_=paths[i];gains[q,m,label+1]=value
        return float(np.linalg.norm(prediction-self.y)**2),gains,prediction,counts

    def predict(self,x,extents,coeff,gains,gradient=False):
        paths,counts=self.paths(x,extents)
        if paths is None:
            return (1e3,np.zeros(10)) if gradient else None
        values,derivatives=self.signals(paths,coeff)
        pred=np.zeros_like(self.y);J=np.zeros((*self.y.shape,10),complex) if gradient else None
        for i,(q,m,label,_,grad) in enumerate(paths):
            g=gains[q,m,label+1];pred[m]+=g*values[i]
            if gradient:
                J[m]+=g*derivatives[i,:,None]*grad[None,:]
        if not gradient:
            return pred
        residual=pred-self.y
        return float(np.linalg.norm(residual)**2),2*np.real(np.einsum('mn,mni->i',residual.conj(),J))

    def wave_fit(self,x,extents,gains):
        paths,_=self.paths(x,extents)
        H=np.zeros((2,5,len(self.freq)),complex)
        for q,m,label,length,_ in paths:
            H[q,m]+=gains[q,m,label+1]*np.exp(-2j*np.pi*self.freq*length/C)
        A=np.concatenate([(H[q,:,:,None]*self.V[None,:,:]).transpose(0,2,1).reshape(self.y.size,-1) for q in range(2)],axis=1)
        coeff,_,rank,_=lstsq(A,self.y.ravel(),cond=1e-8,lapack_driver='gelsd')
        return coeff.reshape(2,-1),int(rank)

    def initialize(self,positions,seed):
        rng=np.random.default_rng(seed)
        for attempt in range(2000):
            x=np.r_[np.array(positions).ravel()/SCALE,rng.uniform(-np.pi,np.pi),rng.uniform(-.8,.8),rng.uniform(-np.pi,np.pi),rng.uniform(-.8,.8)]
            extents=np.column_stack([rng.uniform(-.7,.7,2),rng.uniform(.3,1.3,2),rng.uniform(.05,.18,2)])
            paths,counts=self.paths(x,extents)
            if paths is not None and sum(p[2]>=0 for p in paths)>=3:
                return x,extents,attempt+1
        return None

    def run(self,positions,seed):
        begin=time.perf_counter()
        initialization=self.initialize(positions,seed)
        if initialization is None:
            return {'status':'NO_FEASIBLE_INITIAL_GEOMETRY','seed':seed,'seconds':time.perf_counter()-begin}
        x,extents,attempts=initialization
        coeff=self.initial_coeff.copy()
        loss,gains,_,counts=self.gain_fit(x,extents,coeff)
        initial={'positions_m':(x[:6].reshape(2,3)*SCALE).tolist(),'geometry_parameters':x.tolist(),
                 'extents_normalized':extents.tolist(),'loss':loss,'proposal_attempts':attempts}
        h=1e-7
        value,gradient=self.predict(x,extents,coeff,gains,True)
        fd=np.array([(self.predict(x+np.eye(10)[i]*h,extents,coeff,gains,True)[0]-self.predict(x-np.eye(10)[i]*h,extents,coeff,gains,True)[0])/(2*h) for i in range(10)])
        gradient_error=float(np.linalg.norm(fd-gradient)/max(1.,np.linalg.norm(gradient)))
        if gradient_error>1e-5:
            return {'status':'GRADIENT_CHECK_FAILED','seed':seed,'initial':initial,'gradient_error':gradient_error}
        trace=[];monotone=True
        bounds=[tuple(self.bounds[:,i%3]) for i in range(6)]+[(-np.pi,np.pi),(-1.,1.)]*2
        for iteration in range(3):
            previous=loss
            newcoeff,rank=self.wave_fit(x,extents,gains)
            newpred=self.predict(x,extents,newcoeff,gains)
            newloss=float(np.linalg.norm(newpred-self.y)**2)
            wave_accepted=newloss<=loss+1e-10
            if wave_accepted:
                coeff=newcoeff;loss=newloss
            fitted=self.gain_fit(x,extents,coeff)
            if fitted[0]<=loss+1e-10:
                loss,gains,_,counts=fitted
            opt=minimize(lambda v:self.predict(v,extents,coeff,gains,True),x,jac=True,method='L-BFGS-B',bounds=bounds,
                         options={'maxiter':40,'maxfun':80,'ftol':1e-10,'gtol':1e-6,'maxls':15})
            if np.isfinite(opt.fun) and opt.fun<=loss+1e-10 and self.paths(opt.x,extents)[0] is not None:
                x=opt.x;loss=float(opt.fun)
            steps=np.array([.1,.1,.02])/(2**iteration)
            extent_moves=0
            for k in range(2):
                for j in range(3):
                    best=None
                    for direction in [-1,1]:
                        candidate=extents.copy();candidate[k,j]+=direction*steps[j]
                        lo,hi=[(-1.,1.),(.1,1.5),(.02,.2)][j]
                        if not lo<=candidate[k,j]<=hi:
                            continue
                        fit=self.gain_fit(x,candidate,coeff)
                        if fit is not None and fit[0]<loss-1e-12 and (best is None or fit[0]<best[0][0]):
                            best=fit,candidate
                    if best is not None:
                        (loss,gains,_,counts),extents=best;extent_moves+=1
            fitted=self.gain_fit(x,extents,coeff)
            if fitted[0]<=loss+1e-10:
                loss,gains,_,counts=fitted
            monotone &= loss<=previous+1e-10
            trace.append({'iteration':iteration+1,'loss':loss,'wave_rank':rank,'wave_accepted':wave_accepted,
                          'optimizer_message':str(opt.message),'optimizer_success':bool(opt.success),'optimizer_nit':int(opt.nit),
                          'extent_moves':extent_moves,'LOS_counts':counts,'elapsed':time.perf_counter()-begin})
        pred=self.predict(x,extents,coeff,gains)
        recheck=float(np.linalg.norm(pred-self.y)**2)
        paths,counts=self.paths(x,extents)
        small=sum(abs(gains[q,m,label+1])<1e-4*np.max(abs(gains)) for q,m,label,_,_ in paths)
        return {'status':'PROTOTYPE_BUDGET','seed':seed,'initial':initial,'trace':trace,
                'positions_m':(x[:6].reshape(2,3)*SCALE).tolist(),'geometry_parameters':x.tolist(),
                'extents_normalized':extents.tolist(),'loss':recheck,'relative_residual':float(np.sqrt(recheck)),
                'gradient_error':gradient_error,'monotone':bool(monotone),'loss_replay_error':abs(recheck-loss),
                'geometric_LOS_counts':counts,'visible_paths':len(paths),'near_zero_gains':int(small),
                'coefficients':coeff,'gains':gains,'prediction':pred*self.scale,'seconds':time.perf_counter()-begin}
