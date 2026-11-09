"""Coherent cross-spectrum algebra and saved-IQ decomposition, evaluation only."""
from pathlib import Path
import importlib.util
import time
import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'AGENTS.md').exists())
BASE=ROOT/'研究方向/多径DPD'
P17=BASE/'实验/17_IQ分量接口核验/20261107'
spec=importlib.util.spec_from_file_location('prev',P17/'成果/probe.py')
prev=importlib.util.module_from_spec(spec);spec.loader.exec_module(prev)
gate=prev.gate


def outer(y):
    return y[..., :,None]*y[...,None,:].conj()


def identities(y,h):
    # Per-frequency QR projection, independent of production VP implementation.
    q,_=np.linalg.qr(h,mode='reduced')
    v=np.einsum('fmi,fi->fm',q,np.einsum('fmi,fm->fi',q.conj(),y))
    residual=y-v
    e=np.sum(abs(residual)**2,axis=1);t=np.sum(abs(y)**2,axis=1)
    projector=np.einsum('fmi,fni->fmn',q,q.conj())
    z=outer(y)
    trace=np.trace(z-projector@z,axis1=1,axis2=2).real
    actual=np.sum(abs(z-outer(v))**2,axis=(1,2))
    predicted=2*t*e-e**2
    return dict(trace_relative_error=float(np.linalg.norm(trace-e)/max(np.linalg.norm(t),1e-300)),
                frobenius_relative_error=float(np.linalg.norm(actual-predicted)/max(np.linalg.norm(t*t),1e-300)),
                iq_energy_loss=float(sum(e)/sum(t)))


def main():
    out=HERE.parent/'运行数据/check18'
    if out.exists():
        assert not any(out.iterdir()), 'Refuse to overwrite nonempty run'
    out.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),HERE/'执行合同.md',P17/'成果/probe.py',prev.loader] + list((BASE/'代码/mp_dpd').rglob('*.py'))
    source={str(p):gate.sha(p) for p in files}
    for p in files:gate.copy_checked(p,out/'source'/p.relative_to(ROOT))
    rng=np.random.default_rng(2026110720);units=[]
    for _ in range(100):
        y=rng.normal(size=(16,4))+1j*rng.normal(size=(16,4))
        h=rng.normal(size=(16,4,2))+1j*rng.normal(size=(16,4,2))
        units.append(identities(y,h))
    assert max(x['trace_relative_error'] for x in units)<1e-12
    assert max(x['frobenius_relative_error'] for x in units)<1e-12
    gate.cfg.write(out/'preflight.json',dict(source_hashes=source,unit_count=100,units=units))
    old_identity=gate.read(P17/'运行数据/iq18/completion.json')['input_hashes']
    inputs={};rows=[];engine=gate.api.create_engine();start=time.perf_counter()
    mask=~np.eye(4,dtype=bool)
    for old in gate.read(prev.OLD/'运行数据/common18/rows.json'):
        tag=old['tag'];layout=tag.split('_')[1]
        selected=[P17/'运行数据/iq18/inputs'/tag/'iq.npz',
                  P17/'运行数据/iq18/evaluation'/f'{layout}_00.npz',
                  P17/'运行数据/iq18/evaluation'/f'{tag}.json']
        destinations=[]
        for src in selected:
            assert gate.sha(src)==old_identity[str(src)]
            dst=out/'inputs'/tag/src.name;gate.copy_checked(src,dst)
            inputs[str(dst)]=gate.sha(dst);destinations.append(dst)
        with np.load(destinations[0],allow_pickle=False) as z:iq=z['iq']
        with np.load(destinations[1],allow_pickle=False) as z:bank=dict(z)
        meta=gate.read(destinations[2]);power=np.array(meta['power_ratios'])
        regenerated,_=gate.cfg.observation(bank,meta['condition'])
        assert gate.cfg.digest(regenerated)==gate.cfg.digest(iq)
        components=bank['los']+np.sum(bank['reflected']*np.sqrt(power.transpose(1,0,2))[...,None],axis=2)
        reconstruction=components.sum(axis=0)+bank['noise']
        reconstruction_error=float(np.linalg.norm(reconstruction-iq)/np.linalg.norm(iq))
        assert reconstruction_error<1e-14
        f,y,n,_=engine.fc.cal.spectral(iq)
        sel=abs(np.fft.fftfreq(n,1/engine.ref.FS))<=engine.ref.B/2
        sources=np.fft.fft(components,axis=-1,norm='ortho')[:,:,sel].transpose(0,2,1)
        noise=np.fft.fft(bank['noise'],axis=-1,norm='ortho')[:,sel].T
        signal=sources.sum(axis=0)
        same=outer(sources).sum(axis=0)
        inter=outer(signal)-same
        noisy=outer(signal+noise)-outer(signal)
        observed=outer(y)
        cross_identity=float(np.linalg.norm(observed-same-inter-noisy)/np.linalg.norm(observed))
        assert cross_identity<1e-14
        # Frozen LOS state, no truth-assisted position or path construction.
        state_src=prev.OLD/'运行数据/common18'/f'{tag}_MP_decision.json'
        state_dst=out/'inputs'/tag/'frozen_state.json';gate.copy_checked(state_src,state_dst);inputs[str(state_dst)]=gate.sha(state_dst)
        state=gate.read(state_dst)
        response=engine.ref.response(np.asarray(state['los']['x']),f,[])
        h=response[0] if isinstance(response,tuple) else response
        checks=identities(y,h)
        assert checks['trace_relative_error']<1e-12 and checks['frobenius_relative_error']<1e-12
        rows.append(dict(tag=tag,condition=old['condition'],iq_reconstruction_error=reconstruction_error,
            full_cross_identity_error=cross_identity,objective_checks=checks,
            inter_to_same_offdiag_norm=float(np.linalg.norm(inter[:,mask])/np.linalg.norm(same[:,mask])),
            noise_to_same_offdiag_norm=float(np.linalg.norm(noisy[:,mask])/np.linalg.norm(same[:,mask])),
            integrated_inter_to_same_offdiag=float(np.linalg.norm(inter.sum(0)[mask])/np.linalg.norm(same.sum(0)[mask])),
            scope='EVALUATION_COMPONENTS_NOT_ESTIMATOR_INPUT'))
        gate.cfg.write(out/'rows.json',rows)
        print(tag,round(rows[-1]['inter_to_same_offdiag_norm'],3),flush=True)
    assert all(gate.sha(Path(p))==h for p,h in {**inputs,**source}.items())
    summary=dict(rows=rows,seconds=time.perf_counter()-start,source_hashes=source,input_hashes=inputs,
        algebra_unit_count=100,identity_verified=True,new_localization=False)
    gate.cfg.write(out/'summary.json',summary)
    gate.cfg.write(HERE/'summary.json',summary)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
