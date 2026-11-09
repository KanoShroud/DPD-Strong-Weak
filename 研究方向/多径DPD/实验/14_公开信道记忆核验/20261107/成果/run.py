"""Frozen 20us support study; public algorithm source remains unchanged."""
import importlib.util
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'AGENTS.md').exists())
BASE=ROOT/'研究方向/多径DPD'
OLD=BASE/'实验/09_文献后机制核验/20261107'
loader=OLD/'成果/run_gate.py'
spec=importlib.util.spec_from_file_location('gate',loader)
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
MAX_DELAY_S=20e-6


def edit_in_place(fn, replacements):
    source=fn.generated_source
    for old,new in replacements:
        assert source.count(old)==1,(fn.__name__,old,source.count(old))
        source=source.replace(old,new)
    env={**fn.__globals__,'MAX_DELAY_S':MAX_DELAY_S}
    exec(compile(source,str(HERE/f'instantiated_{fn.__name__}.py'),'exec'),env)
    fn.__globals__['MAX_DELAY_S']=MAX_DELAY_S
    fn.__code__=env[fn.__name__].__code__
    fn.generated_source=source


def engine_for(arm):
    e=gate.api.create_engine()
    if arm=='FULL':return e
    assert arm in ('CAP_CONSERVATIVE','CAP_COUNTED')
    edit_in_place(e.ref.project,[('n/FS*B','MAX_DELAY_S*B')])
    edit_in_place(e.base_fit,[('n/ref.FS*ref.B','MAX_DELAY_S*ref.B')])
    edit_in_place(e.los_fit,[('n/FS*B','MAX_DELAY_S*B')])
    changes=[('    bank = np.vstack(banks)',
        '    bank = np.vstack(banks)\n    bank = bank[bank[:,2] <= min(n-1,int(np.floor(MAX_DELAY_S*FS+1e-9)))]')]
    if arm=='CAP_COUNTED':
        changes.append(('*(2*M)*(n-1)','*(2*M)*min(n-1,int(np.floor(MAX_DELAY_S*FS+1e-9)))'))
    edit_in_place(e.search,changes)
    assert e.estimate_function.__globals__['fast_search'] is e.search
    assert e.fc.fit_base is e.base_fit and e.h.scope is e.base_fit.__globals__
    assert e.finish.__globals__['ref'].project is e.ref.project
    return e


def checks(e,iq,old):
    full=engine_for('FULL')
    f,y,n,sigma=e.fc.cal.spectral(iq)
    x=np.asarray(old['los']['x'])
    xx=np.r_[x,.2,.1,10.,-.1,.2,200.]
    paths=[(0,0),(3,1)]
    a,g=full.objective(xx,f,y,paths);b,h=e.objective(xx,f,y,paths)
    assert abs(a-b)<1e-13 and np.allclose(g,h,rtol=0,atol=1e-13)
    rng=np.random.default_rng(2026110716)
    worst=0.
    for _ in range(30):
        u=xx+rng.normal(size=len(xx))*100
        v=xx+rng.normal(size=len(xx))*100
        pu,pv=e.ref.project(u,n),e.ref.project(v,n)
        worst=max(worst,float((u-pu)@(pv-pu)))
        assert np.max(pu[e.D+2::3])<=MAX_DELAY_S*e.ref.B+1e-12
    assert worst<1e-8
    use=abs(np.fft.fftfreq(n,1/e.ref.FS))<=e.ref.B/2
    _,bb,meta=e.search(x,[],f,y,n,use,sigma,1)
    _,original,om=full.search(x,[],f,y,n,use,sigma,1)
    assert all(meta['checks'].values())
    restricted=original[original[:,2]<=1000]
    assert bb.shape==(8000,7) and np.allclose(bb,restricted,rtol=0,atol=1e-12)
    return dict(objective_difference=abs(a-b),gradient_max_difference=float(np.max(abs(g-h))),
                projection_violation=worst,candidate_count=len(bb),threshold=meta['threshold'],
                original_threshold=om['threshold'],FFT_checks=True)


def main():
    out=HERE.parent/'运行数据/paired18';out.mkdir(parents=True,exist_ok=False)
    sources=list((BASE/'代码/mp_dpd').rglob('*.py'))+[Path(__file__),HERE/'执行合同.md',loader]
    identity={str(p):gate.sha(p) for p in sources}
    for p in sources:gate.copy_checked(p,out/'source'/p.relative_to(ROOT))
    gate.cfg.write(out/'preflight.json',dict(source_hashes=identity,delay_cap_seconds=MAX_DELAY_S,threads=2))
    inputs={};rows=[];start=time.perf_counter()
    arms=['FULL','CAP_CONSERVATIVE','CAP_COUNTED']
    oldrows=gate.read(OLD/'运行数据/common18/rows.json')
    for i,previous in enumerate(oldrows):
        tag=previous['tag'];src=OLD/'运行数据/common18/inputs'/tag/'iq.npz'
        assert gate.sha(src)==previous['input_sha']
        local=out/'inputs'/tag/'iq.npz';gate.copy_checked(src,local);inputs[str(local)]=gate.sha(local)
        with np.load(local,allow_pickle=False) as z:iq=z['iq']
        oldpath=OLD/'运行数据/common18'/f'{tag}_MP_decision.json'
        gate.copy_checked(oldpath,out/'inputs'/tag/'old_decision.json')
        old=gate.read(out/'inputs'/tag/'old_decision.json')
        inputs[str(out/'inputs'/tag/'old_decision.json')]=gate.sha(oldpath)
        if i==0:
            gate.cfg.write(out/'unit_checks.json',{arm:checks(engine_for(arm),iq,old) for arm in arms[1:]})
        order=arms[i%3:]+arms[:i%3];results={}
        for arm in order:
            e=engine_for(arm)
            if i==0:
                for name,fn in [('project',e.ref.project),('base',e.base_fit),('search',e.search)]:
                    gate.cfg.write(out/f'{arm}_{name}_instantiation.json',dict(source=fn.generated_source))
            result=e.mp(iq)
            gate.cfg.write(out/f'{tag}_{arm}_decision.json',result)
            if arm!='FULL':
                assert max(result['final']['x'][e.D+2::3],default=0)<=MAX_DELAY_S*e.ref.B+1e-7
                for state in result['history']:
                    if 'fit' in state:
                        assert max(state['fit']['x'][e.D+2::3],default=0)<=MAX_DELAY_S*e.ref.B+1e-7
            results[arm]=result
            print(i+1,tag,arm,result['stop'],round(result['seconds'],3),flush=True)
        # Frontend must be unchanged across arms, even if full paths fail later.
        for arm in arms[1:]:
            assert np.allclose(results[arm]['los']['p'],results['FULL']['los']['p'],rtol=0,atol=1e-8)
        assert np.allclose(results['FULL']['final']['p'],old['final']['p'],rtol=0,atol=1e-5)
        bank=Path(previous['truth_evaluation_file']);assert gate.sha(bank)==previous['truth_file_sha']
        bc=out/'evaluation'/bank.name
        if not bc.exists():gate.copy_checked(bank,bc)
        inputs[str(bc)]=gate.sha(bc)
        with np.load(bc,allow_pickle=False) as z:truth=z['p']
        for arm in arms:
            result=results[arm];error=gate.errors(result['final']['p'],truth)
            rows.append(dict(tag=tag,condition=previous['condition'],arm=arm,errors_m=error,
                rmse=float(np.sqrt(np.mean(error**2))),normal=result['normal'],stop=result['stop'],
                seconds=result['seconds'],K=len(result['paths']),cost=result['final']['cost'],
                projected_gradient=result['final']['projected_gradient'],
                max_delay_us=max(result['final']['x'][16+2::3],default=0)/e.ref.B*1e6))
        gate.cfg.write(out/'rows.json',rows)
    assert len(rows)==54
    assert all(gate.sha(Path(p))==h for p,h in {**identity,**inputs}.items())
    gate.cfg.write(out/'completion.json',dict(count=54,seconds=time.perf_counter()-start,
        source_hashes=identity,input_hashes=inputs,identity_verified=True,all_frontends_equal=True,
        original_results_reproduced=True,scope='development_same_observation_pairing'))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
