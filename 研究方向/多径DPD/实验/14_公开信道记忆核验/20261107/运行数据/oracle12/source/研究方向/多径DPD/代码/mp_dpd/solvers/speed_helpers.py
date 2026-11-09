# ruff: noqa: F401
# Preserve imported globals for legacy function cloning and reexports.
"""Bounded speed gates; outputs isolated, full estimators receive IQ only."""
import argparse
import cProfile
from datetime import datetime
import io
from pathlib import Path
import pstats
import sys
import time
import types

import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
import 研究方向.多径DPD.代码.mp_dpd.solvers.profile_core as fc

BASE,HERE,cal,ref,engine=fc.BASE,fc.HERE,fc.cal,fc.ref,fc.engine
FORMAL=BASE/'risk_calibration_20261102/formal_20261102_181736_622970'


def inputs(tag):
    folder=FORMAL/'cases'/tag;info=fc.load(folder/'input.json')
    with np.load(folder/'input.npz',allow_pickle=False) as z:
        assert z.files==['iq'];iq=z['iq']
    assert cal.digest(iq)==info['iq_sha']
    return iq


def preflight(out,tags):
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.md'))+[FORMAL/'contract.json']
    for tag in tags:files += [FORMAL/'cases'/tag/n for n in ('input.json','input.npz','solve.json')]
    for mod in tuple(sys.modules.values()):
        p=getattr(mod,'__file__',None)
        if p and Path(p).suffix=='.py' and Path(p).resolve().is_relative_to(BASE):files.append(Path(p))
    hashes={str(p.resolve(strict=True)):ref.sha(p) for p in files}
    fc.write(out/'preflight.json',dict(hashes=hashes,threads=2,hardware=threadpool_info(),tags=tags))
    return hashes


def kernel(out):
    rows=[];checks={}
    for tag in ('L0_R00','L2_R00','L3_R00','HIGH_R00'):
        iq=inputs(tag);f,y,n,_=cal.spectral(iq)
        solved=fc.load(FORMAL/'cases'/tag/'solve.json')
        paths=[tuple(a) for a in solved['paths']];x=np.array(solved['final']['x'])
        # Off-stationary test avoids testing an almost-zero gradient only.
        xx=x.copy();xx[:4]+=[.01,-.02,.015,.01]
        oldr,oldj=ref.evaluate(xx,f,y,paths)
        r,j=fc.evaluate(xx,f,y,paths);value,g=fc.objective(xx,f,y,paths)
        checks[tag+'/residual']=bool(np.allclose(r,oldr,rtol=1e-10,atol=1e-12))
        checks[tag+'/jacobian']=bool(np.allclose(j,oldj,rtol=1e-8,atol=1e-11))
        checks[tag+'/cost']=bool(np.isclose(value,.5*oldr@oldr,rtol=1e-11))
        checks[tag+'/gradient']=bool(np.allclose(g,oldj.T@oldr,rtol=1e-8,atol=1e-10))
        direction=np.random.default_rng(2026110401).normal(size=len(xx));direction/=np.linalg.norm(direction)
        eps=1e-5
        fd=(fc.objective(xx+eps*direction,f,y,paths)[0]-fc.objective(xx-eps*direction,f,y,paths)[0])/(2*eps)
        checks[tag+'/finite_difference']=bool(np.isclose(fd,g@direction,rtol=2e-4,atol=2e-8))
        times={}
        for name,fun in [('original_r_j',ref.evaluate),('fast_r_j',fc.evaluate),('fast_cost_grad',fc.objective)]:
            tt=[]
            for _ in range(5):
                t=time.perf_counter();fun(xx,f,y,paths);tt.append(time.perf_counter()-t)
            times[name]=float(np.median(tt))
        profiler=cProfile.Profile();profiler.enable();ref.evaluate(xx,f,y,paths);profiler.disable()
        buf=io.StringIO();pstats.Stats(profiler,stream=buf).sort_stats('cumtime').print_stats(16)
        (out/(tag+'_profile.txt')).write_text(buf.getvalue(),encoding='utf-8')
        rows.append(dict(case=tag,K=len(paths),times=times))
        print(tag,times,flush=True)
    fc.write(out/'kernel.json',dict(rows=rows,checks=checks,all_pass=all(checks.values())))
    assert all(checks.values()),checks


class NumpyNoSave:
    def __getattr__(self,name):return getattr(np,name)
    def savez_compressed(self,*args,**kwargs):pass


def solve(iq,out,tag,variant):
    if variant=='original': fit,search=cal.frozen_fit,ref.search
    elif variant=='exact': fit,search=fc.fit,fc.search
    else:
        import 研究方向.多径DPD.代码.mp_dpd.solvers.structured_reference as ss
        fit,search=ss.fit,ss.search
    scope={**engine.solve_iq.__globals__,'fit':fit,'search':search,'np':NumpyNoSave(),'write':lambda *a,**k:None}
    fun=types.FunctionType(engine.solve_iq.__code__,scope)
    return fun(iq,out,tag)


def paired(out,tags,variants):
    contract=fc.load(FORMAL/'contract.json')['contract'];rows=[]
    for i,tag in enumerate(tags):
        iq=inputs(tag)
        for variant in (variants if i%2==0 else variants[::-1]):
            print('START',tag,variant,flush=True)
            t=time.perf_counter();res=solve(iq,out,tag,variant);wall=time.perf_counter()-t
            fc.write(out/(tag+'_'+variant+'.json'),res)
            cond=tag.split('_')[0]
            with np.load(contract['conditions'][cond]['bank'],allow_pickle=False) as z:truth=z['p']
            p=np.asarray(res['final']['p']);err=ref.error(p,truth)
            row=dict(case=tag,variant=variant,seconds=wall,algorithm_seconds=res['seconds'],rmse_m=float(np.sqrt(np.mean(np.asarray(err)**2))),errors_m=err,K=len(res['paths']),stop=res['stop'],J=res['final']['cost'],PG=res['final']['projected_gradient'],stationary=res['final']['stationary'],search_seconds=sum(e['search']['seconds'] for e in res['history']),fit_seconds=sum(e['fit']['seconds'] for e in res['history'] if 'fit'in e),all_checks=all(res['checks'].values()))
            rows.append(row);fc.write(out/'results.json',dict(rows=rows,complete=False))
            print(row,flush=True)
    fc.write(out/'results.json',dict(rows=rows,complete=True))


# Historical main retained only in the evidence archive.


# Historical batch entry disabled; use the public mp_dpd API.
