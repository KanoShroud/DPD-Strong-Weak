"""Existing PG/feasibility acceptance during SLSQP; no new numerical tolerance."""
import importlib.util
import inspect
from pathlib import Path
import sys
import time
import types

import numpy as np
from scipy.optimize import OptimizeResult, minimize

HERE=Path(__file__).resolve().parent
NAT=HERE.parent/'natural_stop_20261104'
# Load under unique names: natural_stop's own imports expect its solver as 'solver'.
spec=importlib.util.spec_from_file_location('natural_solver',NAT/'solver.py')
s=importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
sys.modules['solver']=s
sys.path.insert(0,str(NAT))
import hybrid_gate as h  # noqa: E402

fc,ref,rg=s.fc,s.ref,s.rg


class GateReturn(Exception):
    def __init__(self,z,cost):
        self.z,self.cost=z,cost


class Gate:
    def __init__(self,x0,f,y,paths,n):
        self.f,self.y,self.paths,self.n=f,y,paths,n
        self.initial_cost=s.objective(ref.project(x0,n),f,y,paths)[0]
        self.checks,self.extra,self.reused,self.returned=0,1,0,0
        scope={**h.scope,'minimize':self.minimize}
        base=types.FunctionType(fc.fit_base.__code__,scope,argdefs=fc.fit_base.__defaults__)
        scope['fit_base']=base
        self.inner=types.FunctionType(h.fit.__code__,{**h.fit.__globals__,'base_fit':base},argdefs=h.fit.__defaults__)

    def acceptable(self,x,cost,g):
        pg=float(np.max(abs(x-ref.project(x-g,self.n))))
        return bool(np.isfinite(x).all() and np.isfinite(cost) and np.isfinite(g).all()
                    and np.linalg.norm(x-ref.project(x,self.n))<1e-7
                    and cost<=self.initial_cost+1e-10 and pg<=1e-7)

    def minimize(self,fun,x0,*args,**kwargs):
        cl=inspect.getclosurevars(fun).nonlocals
        unpack,cache,scale=cl['unpack'],cl['cache'],cl['scale']
        origin=unpack(np.zeros_like(x0))
        count=0

        def callback(z):
            nonlocal count
            count+=1;self.checks+=1
            x=unpack(z);p=ref.project(x,self.n)
            if cache[0] is not None and np.array_equal(cache[0],z) and np.array_equal(x,p):
                cost,g=cache[1];self.reused+=1
            else:
                cost,g=s.objective(p,self.f,self.y,self.paths);self.extra+=1
            if self.acceptable(p,cost,g):
                # fc.fit_base will unpack this exact scaled coordinate again.
                zz=(p-origin)*scale
                actual=unpack(zz)
                cc,gg=s.objective(actual,self.f,self.y,self.paths);self.extra+=1
                if self.acceptable(actual,cc,gg):
                    self.returned+=1
                    raise GateReturn(zz,cc)

        assert 'callback' not in kwargs
        kwargs['callback']=callback
        try:
            return minimize(fun,x0,*args,**kwargs)
        except GateReturn as done:
            return OptimizeResult(x=done.z,success=True,nit=count,message='existing_original_PG_gate',
                                  fun=done.cost/cl['cost_scale'])


def fit(x0,f,y,paths,n,method='GATED'):
    start=time.perf_counter()
    gate=Gate(x0,f,y,paths,n)
    x,state=gate.inner(x0,f,y,paths,n,method)
    state.update(solver='GATED',gate_checks=gate.checks,gate_cached=gate.reused,
                 gate_extra_calls=gate.extra,gate_returns=gate.returned,
                 seconds=time.perf_counter()-start,calls=state['calls']+gate.extra)
    return x,state


def estimate(iq,method='GATED'):
    fun=types.FunctionType(s.estimate.__code__,{**s.estimate.__globals__,'fit':fit if method=='GATED' else h.fit},argdefs=s.estimate.__defaults__)
    return fun(iq,method,True,True)
