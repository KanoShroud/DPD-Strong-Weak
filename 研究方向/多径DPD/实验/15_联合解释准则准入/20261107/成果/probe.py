"""Frozen complex closure diagnostics over the existing geometric pool."""
import importlib.util
from itertools import combinations
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'AGENTS.md').exists())
BASE=ROOT/'研究方向/多径DPD'
OLD=BASE/'实验/09_文献后机制核验/20261107'
R14=BASE/'实验/14_公开信道记忆核验/20261107/运行数据'
loader=OLD/'成果/run_gate.py'
spec=importlib.util.spec_from_file_location('gate',loader)
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)


def closure(gram):
    products=np.stack([gram[:,a,b]*gram[:,b,c]*gram[:,a,c].conj()
                      for a,b,c in combinations(range(4),3)],axis=1)
    magnitude=abs(products)
    phase=np.divide(products.real,magnitude,out=np.zeros_like(magnitude),where=magnitude>0)
    return np.mean(phase,axis=1),np.sum(products.real,axis=1),np.sum(magnitude==0,axis=1)


def scores(points,f,y,stations,c):
    tau=np.linalg.norm(np.asarray(points)[:,None,:]-stations[None],axis=2)/c
    energy=np.sum(abs(y)**2,axis=0)
    grams=[]
    for t in tau:
        aligned=y*np.exp(2j*np.pi*f[:,None]*t[None])
        g=aligned.conj().T@aligned
        grams.append(g/np.sqrt(energy[:,None]*energy[None,:]))
    return closure(np.asarray(grams))


def unit():
    rng=np.random.default_rng(2026110717)
    g=rng.normal(size=4)+1j*rng.normal(size=4)
    gram=g.conj()[:,None]*g[None,:]
    gram/=np.sqrt(np.diag(gram).real[:,None]*np.diag(gram).real[None,:])
    p,s,_=closure(gram[None]);assert abs(p[0]-1)<1e-13 and abs(s[0]-4)<1e-13
    y=rng.normal(size=(50,4))+1j*rng.normal(size=(50,4))
    a=y.conj().T@y;a/=np.sqrt(np.diag(a).real[:,None]*np.diag(a).real[None,:])
    d=np.exp(1j*rng.normal(size=4));b=d.conj()[:,None]*a*d[None,:]
    aa=closure(a[None]);bb=closure(b[None])
    assert np.allclose(aa[0],bb[0],rtol=0,atol=1e-13)
    assert np.allclose(aa[1],bb[1],rtol=0,atol=1e-13)
    # At one frequency any nonzero observation is an outer product: vacuous closure.
    single=y[:1].conj().T@y[:1]
    assert abs(closure(single[None])[0][0]-1)<1e-13
    return dict(pure_component_phase=float(p[0]),pure_component_product=float(s[0]),
                station_phase_invariance=True,single_bin_vacuous_closure=True)


def main():
    out=HERE.parent/'运行数据/closure18';out.mkdir(parents=True,exist_ok=False)
    source=list((BASE/'代码/mp_dpd').rglob('*.py'))+[Path(__file__),HERE/'执行合同.md',loader]
    identity={str(p):gate.sha(p) for p in source}
    for p in source:gate.copy_checked(p,out/'source'/p.relative_to(ROOT))
    gate.cfg.write(out/'preflight.json',dict(source_hashes=identity,checks=unit()))
    rows=[];inputs={};start=time.perf_counter()
    for old in gate.read(OLD/'运行数据/common18/rows.json'):
        tag=old['tag'];src=OLD/'运行数据/common18/inputs'/tag/'iq.npz'
        assert gate.sha(src)==old['input_sha']
        local=out/'inputs'/tag/'iq.npz';gate.copy_checked(src,local);inputs[str(local)]=gate.sha(local)
        with np.load(local,allow_pickle=False) as z:iq=z['iq']
        e=gate.api.create_engine();front=e.frontend(iq)
        f,y,n,_=e.fc.cal.spectral(iq);pool=np.asarray(front['refined'])
        phase,product,zero=scores(pool,f,y,e.stations,e.ref.C)
        # These are observable scores, not a selected pair or truth-assisted estimator.
        gate.cfg.write(out/f'{tag}_observable.json',dict(points=pool,phase=phase,product=product,
            zero_triangles=zero,geometry_cost=front['cost'],scope='NO_PAIR_SELECTION'))
        bank=Path(old['truth_evaluation_file']);assert gate.sha(bank)==old['truth_file_sha']
        dst=out/'evaluation'/bank.name
        if not dst.exists():gate.copy_checked(bank,dst)
        inputs[str(dst)]=gate.sha(dst)
        with np.load(dst,allow_pickle=False) as z:truth=z['p']
        _,ids=linear_sum_assignment(np.sum((truth[:,None]-pool[None])**2,axis=2))
        states={}
        for arm,src in [('blind',R14/'paired18'/f'{tag}_CAP_COUNTED_decision.json'),
                        ('oracle',R14/'oracle12'/f'{tag}_decision.json')]:
            if not src.exists():continue
            dest=out/'inputs'/tag/f'{arm}.json';gate.copy_checked(src,dest);inputs[str(dest)]=gate.sha(dest)
            solved=gate.read(dest);pos=np.asarray(solved['final']['p'])
            a,b,z=scores(pos,f,y,e.stations,e.ref.C)
            _,assignment=linear_sum_assignment(np.sum((truth[:,None]-pos[None])**2,axis=2))
            states[arm]=dict(p=pos,phase=a[assignment],product=b[assignment],
                errors_m=np.linalg.norm(pos[assignment]-truth,axis=1),normal=solved['normal'])
        rows.append(dict(tag=tag,condition=old['condition'],pool_size=len(pool),oracle_ids=ids,
            near_errors_m=np.linalg.norm(pool[ids]-truth,axis=1),near_phase=phase[ids],near_product=product[ids],
            phase_rank=[int(np.sum(phase>phase[k])+1) for k in ids],
            product_rank=[int(np.sum(product>product[k])+1) for k in ids],states=states,
            scope='ORACLE_RANK_DIAGNOSTIC_NOT_ALGORITHM'))
        gate.cfg.write(out/'rows.json',rows)
        print(tag,len(pool),'phase rank',rows[-1]['phase_rank'],'product rank',rows[-1]['product_rank'],flush=True)
    assert len(rows)==18
    assert all(gate.sha(Path(p))==h for p,h in {**identity,**inputs}.items())
    gate.cfg.write(out/'completion.json',dict(count=18,seconds=time.perf_counter()-start,
        source_hashes=identity,input_hashes=inputs,identity_verified=True,scope='diagnostic_no_new_localization'))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
