from 研究方向.多径DPD.代码.mp_dpd import code_path
# ruff: noqa: F401
# Preserve imported globals for legacy function cloning and reexports.
"""Exact VP objective/gradient via channel-local derivatives; frozen ref unchanged."""
import importlib.util
from pathlib import Path
import time
import types

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
CAL = BASE/'risk_calibration_20261102'
spec = importlib.util.spec_from_file_location('speed_cal', code_path('io/spectral_pipeline.py'))
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)
ref, engine = cal.ref, cal.engine
from 研究方向.多径DPD.代码.mp_dpd.io.storage import load as load, write as write  # noqa: E402
import 研究方向.多径DPD.代码.mp_dpd.solvers.feasible as feasible  # noqa: E402


def core(x,f,y,paths):
    p=x[:4].reshape(2,2)*ref.UNIT
    v=p[None]-ref.STATIONS[:,None]
    dist=np.linalg.norm(v,axis=2)
    if dist.min()<1e-8: raise ValueError('station coincidence')
    g=np.ones((5,2),complex)
    g[1:]=(x[4:12]+1j*x[12:20]).reshape(4,2)
    w=-2j*np.pi*f/ref.B
    phase=np.exp(-2j*np.pi*f[:,None,None]*dist[None]/ref.C)
    base=phase*g[None]
    fac=np.ones_like(base)
    ee=[]
    for i,(m,q) in enumerate(paths):
        k=20+3*i
        e=np.exp(w*x[k+2]);ee.append(e)
        fac[:,m,q]+=(x[k]+1j*x[k+1])*e
    h=base*fac
    inv=np.linalg.pinv(h,rcond=1e-12)
    s=(inv@y[:,:,None])[:,:,0]
    r=y-(h@s[:,:,None])[:,:,0]
    return h,inv,s,r,base,phase,fac,ee,w,v/dist[:,:,None]


def derivatives(x,paths,data):
    h,inv,s,r,base,phase,fac,ee,w,direction=data
    for q in range(2):
        for a in range(2):
            yield 2*q+a,q,None,h[:,:,q]*w[:,None]*direction[None,:,q,a]
        for m in range(1,5):
            k=4+2*(m-1)+q
            z=phase[:,m,q]*fac[:,m,q]
            yield k,q,m,z
            yield k+8,q,m,1j*z
    for i,(m,q) in enumerate(paths):
        k=20+3*i;z=base[:,m,q]*ee[i]
        yield k,q,m,z
        yield k+1,q,m,1j*z
        yield k+2,q,m,z*(x[k]+1j*x[k+1])*w


def objective(x,f,y,paths):
    d=core(x,f,y,paths);s,r=d[2:4]
    den=float(np.vdot(y,y).real)
    grad=np.zeros(len(x))
    for k,q,m,z in derivatives(x,paths,d):
        a=z*s[:,q,None] if m is None else z*s[:,q]
        rr=r if m is None else r[:,m]
        grad[k]=-2*np.vdot(rr,a).real/den
    return float(np.vdot(r,r).real/den),grad


def evaluate(x,f,y,paths,tangent=False):
    d=core(x,f,y,paths);h,inv,s,r,base=d[:5]
    ds=np.zeros((len(f),5,len(x)),complex)
    t=np.zeros((len(f),2,len(x)),complex)
    for k,q,m,z in derivatives(x,paths,d):
        if m is None:
            ds[:,:,k]=z*s[:,q,None]
            t[:,q,k]=np.sum(z.conj()*r,axis=1)
        else:
            ds[:,m,k]=z*s[:,q]
            t[:,q,k]=z.conj()*r[:,m]
    jp=ds-h@(inv@ds)
    if tangent:return ref.real(r),ref.cols(jp),(h,inv,s,base)
    j=-jp-inv.conj().transpose(0,2,1)@t
    norm=np.linalg.norm(y)
    return ref.real(r)/norm,ref.cols(j)/norm


def fit_base(x0,f,y,paths,n,budget=300,recenter=True):
    start=time.perf_counter();x0=ref.project(x0,n)
    r,j=evaluate(x0,f,y,paths)
    scale=np.maximum(np.linalg.norm(j,axis=0),1e-10)
    cost_scale=max(.5*r@r,1e-12)
    cache=[None,None];calls=0
    def unpack(z):return x0+z/scale
    def obj(z):
        nonlocal calls
        if cache[0] is None or not np.array_equal(z,cache[0]):
            cache[:]=[z.copy(),objective(unpack(z),f,y,paths)];calls+=1
        cost,grad=cache[1]
        return cost/cost_scale,grad/scale/cost_scale
    def cons(z):return 1-np.sum(unpack(z)[20:].reshape(-1,3)[:,:2]**2,axis=1)
    def jac(z):
        x=unpack(z);a=np.zeros((len(paths),len(x)))
        for i in range(len(paths)):
            k=20+3*i;a[i,k:k+2]=-2*x[k:k+2]/scale[k:k+2]
        return a
    low=np.full(len(x0),-np.inf);high=-low.copy()
    low[:4],high[:4]=-3000/ref.UNIT,3000/ref.UNIT
    low[22::3],high[22::3]=0,n/ref.FS*ref.B
    opt=minimize(obj,np.zeros(len(x0)),jac=True,method='SLSQP',bounds=list(zip((low-x0)*scale,(high-x0)*scale)),constraints=[dict(type='ineq',fun=cons,jac=jac)] if paths else [],options=dict(maxiter=budget,ftol=1e-12))
    x=unpack(opt.x);cost,grad=objective(x,f,y,paths)
    mapping=float(np.max(abs(x-ref.project(x-grad,n))))
    result=dict(x=x,p=x[:4].reshape(2,2)*ref.UNIT,cost=cost,projected_gradient=mapping,stationary=mapping<=1e-7,iterations=int(opt.nit),success=bool(opt.success),message=str(opt.message),calls=calls,seconds=time.perf_counter()-start,closure_error=float(np.linalg.norm(x-ref.project(x,n))))
    if recenter and opt.success and not result['stationary'] and opt.nit<budget:
        xx,other=fit_base(x,f,y,paths,n,budget-int(opt.nit),False)
        attempts=[result.copy(),other.copy()]
        if other['cost']<=result['cost']:x,result=xx,other
        result['attempts']=attempts;result['iterations']=sum(a['iterations'] for a in attempts)
        result['calls']=sum(a['calls'] for a in attempts);result['seconds']=time.perf_counter()-start
    return x,result


fast_ref=types.SimpleNamespace(project=ref.project,evaluate=evaluate,UNIT=ref.UNIT)
finish=types.FunctionType(feasible.finish.__code__,{**feasible.finish.__globals__,'ref':fast_ref})


def fit(x0,f,y,paths,n):
    start=time.perf_counter();x,res=fit_base(x0,f,y,paths,n)
    x,res=finish(x,res,x0,f,y,paths,n)
    res['seconds']=time.perf_counter()-start
    return x,res


search=types.FunctionType(ref.search.__code__,{**ref.search.__globals__,'evaluate':evaluate})
