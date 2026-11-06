# ruff: noqa: F401
# Preserve imported globals for legacy function cloning and reexports.
"""M-station instantiation of frozen formulas; explicit audited source substitutions."""
import importlib.util
import inspect
import sys
from itertools import combinations
from pathlib import Path
import types

import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
spec=importlib.util.spec_from_file_location('station_gate_source',BASE/'acceptance_gate_20261104/solver.py')
old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
fc,ref,rg=old.fc,old.ref,old.rg


def adapt(fun,bindings=None,edits=(),defaults=None):
    source=inspect.getsource(fun)
    for before,after in edits:
        assert before in source,(fun.__name__,before)
        source=source.replace(before,after)
    env={**fun.__globals__,**(bindings or {})}
    exec(compile(source,str(HERE/('instantiated_'+fun.__name__)), 'exec'),env)
    new=env[fun.__name__]
    new.generated_source=source
    if defaults is not None:new.__defaults__=defaults
    return new


class Backend:
    def __init__(self,station_ids,reference=0):
        self.ids=list(station_ids);self.M=len(self.ids);self.reference=reference
        self.stations=ref.STATIONS[self.ids].copy()
        self.free=np.array([i for i in range(self.M) if i!=reference])
        self.G=2*(self.M-1);self.D=4+2*self.G
        self.bind=dict(M=self.M,G=self.G,D=self.D,FREE=self.free,REF=reference)
        self.ref=types.SimpleNamespace(**ref.__dict__)
        self.ref.STATIONS=self.stations
        self.ref.project=adapt(ref.project,self.bind,[('range(20,','range(D,')])
        nb={**self.bind,'ref':self.ref}
        self.core=adapt(fc.core,{**nb,'np':old.s.GuardedNumpy()},[
            ('np.ones((5,2)','np.ones((M,2)'),
            ('g[1:]=(x[4:12]+1j*x[12:20]).reshape(4,2)','g[FREE]=(x[4:4+G]+1j*x[4+G:D]).reshape(M-1,2)'),
            ('20+3*i','D+3*i')])
        self.derivatives=adapt(fc.derivatives,self.bind,[
            ('for m in range(1,5):','for ii,m in enumerate(FREE):'),
            ('4+2*(m-1)+q','4+2*ii+q'),('k+8','k+G'),('20+3*i','D+3*i')])
        eb={**nb,'core':self.core,'derivatives':self.derivatives}
        self.objective=adapt(fc.objective,eb)
        self.evaluate=adapt(fc.evaluate,eb,[('(len(f),5,len(x))','(len(f),M,len(x))')])
        self.search=adapt(ref.search,{**nb,'evaluate':self.evaluate,'np':old.s.SearchNumpy()},[
            ('len(f)*5','len(f)*M'),('reshape(len(f), 5,','reshape(len(f), M,'),
            ('*10*(n-1)','*(2*M)*(n-1)'),('range(5)','range(M)'),('(len(f), 5, 2)','(len(f), M, 2)')])
        self.base_fit=adapt(fc.fit_base,{**nb,'evaluate':self.evaluate,'objective':self.objective},[
            ('[20:]','[D:]'),('20+3*i','D+3*i'),('[22::3]','[D+2::3]')])
        self.base_fit.__globals__['fit_base']=self.base_fit
        self.finish=adapt(old.h.finish,{'ref':types.SimpleNamespace(project=self.ref.project,evaluate=self.evaluate,UNIT=ref.UNIT)})
        self.s=types.SimpleNamespace(**old.s.__dict__)
        self.s.objective,self.s.evaluate=self.objective,self.evaluate
        self.fc=types.SimpleNamespace(**fc.__dict__)
        self.fc.fit_base=self.base_fit
        self.h=types.SimpleNamespace(**old.h.__dict__)
        self.h.scope=self.base_fit.__globals__
        self.h.fit=adapt(old.h.fit,{'s':self.s,'ref':self.ref,'base_fit':self.base_fit,'finish':self.finish})
        scope={**old.__dict__,'s':self.s,'fc':self.fc,'ref':self.ref,'h':self.h}
        exec(inspect.getsource(old.Gate),scope)
        self.fit=adapt(old.fit,{**scope,'Gate':scope['Gate']})
        self.los_evaluate=lambda x,f,y,path=None:self.evaluate(x,f,y,[] if path is None else [path])
        self.los_fit=adapt(fc.engine.los_fit,{**nb,'evaluate':self.los_evaluate,'closure':self.ref.project},[
            ('np.full(16,','np.full(2*G,'),('np.full(18,','np.full(2*G+2,'),
            ('np.zeros(23)','np.zeros(D+3)'),('[20:22]','[D:D+2]'),('opt.x[22]','opt.x[D+2]')])
        pairs=list(combinations(range(self.M),2))
        self.pairs=pairs
        geom=fc.engine.geometry_fit
        quant=geom.__globals__['quantities']
        predicted=quant.__globals__['predicted']
        pred=adapt(predicted,{'STATIONS':self.stations,'PAIRS':pairs},defaults=(self.stations,))
        quantities=adapt(quant,{'predicted':pred})
        self.geometry_fit=adapt(geom,{'quantities':quantities})
        self.triples=adapt(fc.engine.triple_candidates,{**self.bind,'STATIONS':self.stations,'PAIRS':pairs},
                          [('range(5)','range(M)')],defaults=(self.stations,))
        self.extract=adapt(fc.engine.extract,{'STATIONS':self.stations,'PAIRS':pairs},defaults=(self.stations,False))
        self.rank_one=adapt(fc.engine.rank_one,{'STATIONS':self.stations,'REF':reference},[('g[0]','g[REF]')])
        self.engine=types.SimpleNamespace(**fc.engine.__dict__)
        self.engine.rank_one=self.rank_one
        self.fc.engine=self.engine
        regions=types.SimpleNamespace(frontend=self.frontend,alternate_pair=old.s.bc.regions.alternate_pair)
        self.estimate_function=adapt(old.s.estimate,{**nb,'fc':self.fc,'bc':types.SimpleNamespace(regions=regions),
            'fit':self.fit,'objective':self.objective,'fast_los':self.los_fit,'fast_search':self.search},
            [('gains[1:]','gains[FREE]'),('range(5)','range(M)')])

    def frontend(self,iq):
        assert iq.shape[0]==self.M
        extracted,_,_=self.extract(iq)
        peaks=extracted['CC_ABS']
        lags=np.array([sorted(np.asarray(peaks[f'{m}_{n}']['lag_s'][:6])*ref.C) for m,n in self.pairs])
        pool=self.triples(lags)
        if len(pool)<2:raise ValueError('Fewer than two blind geometry candidates')
        pp,cc,reward,status,iterations=self.geometry_fit(pool,lags)
        return dict(pool=pool,refined=pp,cost=cc,reward=reward,lags=lags,status=status,iterations=iterations)

    def estimate(self,iq):
        return self.estimate_function(iq,'M_GATED',True,True)
