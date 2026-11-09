"""Evaluation-only oracle conditioning control. Never a localization result."""
from pathlib import Path
import importlib.util
from itertools import product
import time
import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('iqprobe',HERE/'probe.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


def main():
    raw=HERE.parent/'运行数据/iq18'
    out=HERE.parent/'运行数据/support18';out.mkdir(parents=True,exist_ok=False)
    identity={str(x):p.gate.sha(x) for x in [Path(__file__),HERE/'probe.py',HERE/'执行合同.md']}
    inputs=p.gate.read(raw/'completion.json')['input_hashes']
    assert all(p.gate.sha(Path(x))==h for x,h in inputs.items())
    for x in identity:p.gate.copy_checked(Path(x),out/'source'/Path(x).name)
    rng=np.random.default_rng(2026110719);rows=[];start=time.perf_counter()
    engine=p.gate.api.create_engine()
    for entry in p.gate.read(raw/'rows.json'):
        tag=entry['tag'];layout=tag.split('_')[1]
        with np.load(raw/'evaluation'/f'{layout}_00.npz',allow_pickle=False) as z:bank=dict(z)
        meta=p.gate.read(raw/'evaluation'/f'{tag}.json');power=np.array(meta['power_ratios'])
        with np.load(raw/'inputs'/tag/'iq.npz',allow_pickle=False) as z:iq=z['iq']
        f,_,_,_=engine.fc.cal.spectral(iq)
        propagation=np.linalg.norm(engine.stations[:,None]-bank['p'][None],axis=2)/p.gate.cfg.C
        pairs=[]
        for m,n in p.PAIRS:
            lags=[];beta=[];labels=[];same_delay_offset=[]
            for q in range(2):
                im=[-1]+list(np.flatnonzero(power[m,q]>0));jn=[-1]+list(np.flatnonzero(power[n,q]>0))
                for l,k in product(im,jn):
                    dm=0 if l==-1 else bank['delays'][m,q,l]/p.gate.cfg.B
                    dn=0 if k==-1 else bank['delays'][n,q,k]/p.gate.cfg.B
                    am=bank['gains'][m,q]*(1 if l==-1 else np.sqrt(power[m,q,l])*bank['phases'][m,q,l])
                    an=bank['gains'][n,q]*(1 if k==-1 else np.sqrt(power[n,q,k])*bank['phases'][n,q,k])
                    lags.append(propagation[n,q]-propagation[m,q]+dn-dm)
                    beta.append(am.conjugate()*an);labels.append((q,l,k))
                    if l==k and l>=0:same_delay_offset.append(abs(dn-dm)*p.gate.cfg.B)
            beta=np.array(beta);beta/=np.linalg.norm(beta);lags=np.array(lags)
            a=np.exp(-2j*np.pi*f[:,None]*lags)
            u,s,vh=np.linalg.svd(a,full_matrices=False)
            inverse=(vh.conj().T/s)@u.conj().T
            clean=a@beta;recovered=inverse@clean
            noise_sigma=.01*np.sqrt(np.mean(abs(clean)**2))
            noise=noise_sigma/np.sqrt(2)*(rng.normal(size=(len(f),32))+1j*rng.normal(size=(len(f),32)))
            err=inverse@noise
            predicted=noise_sigma*np.sqrt(np.sum(1/s**2))/np.linalg.norm(beta)
            measured=np.sqrt(np.mean(np.sum(abs(err)**2,axis=0)))/np.linalg.norm(beta)
            direct=[i for i,x in enumerate(labels) if x[1:]==(-1,-1)]
            nearest=[]
            for i in direct:
                peers=[j for j,x in enumerate(labels) if x[0]==labels[i][0] and j!=i]
                nearest.append(None if not peers else float(np.min(abs(lags[peers]-lags[i]))*p.gate.cfg.B))
            pairs.append(dict(pair=[m,n],components=len(lags),condition=float(s[0]/s[-1]),
                noiseless_relative_coefficient_error=float(np.linalg.norm(recovered-beta)),
                direct_nearest_same_source_B=nearest,same_index_reflection_offset_B=same_delay_offset,
                one_percent_noise_predicted_relative_coefficient_rms=float(predicted),
                one_percent_noise_32draw_relative_coefficient_rms=float(measured)))
        rows.append(dict(tag=tag,condition=entry['condition'],pairs=pairs))
    assert all(p.gate.sha(Path(x))==h for x,h in {**inputs,**identity}.items())
    result=dict(rows=rows,seconds=time.perf_counter()-start,source_hashes=identity,
        input_hashes=inputs,scope='ORACLE_EXACT_SUPPORT_FLAT_KERNEL_SYNTHETIC_CROSS_SPECTRUM_NO_LOCALIZATION',
        actual_iq=False,noise_amplitude_fraction=.01,draws_per_pair=32,seed=2026110719)
    p.gate.cfg.write(out/'summary.json',result)
    for cond in ('L0','L1_P075','L2_P075'):
        pp=[x for r in rows if r['condition']==cond for x in r['pairs']]
        print(cond,'condition median/max',np.median([x['condition'] for x in pp]),max(x['condition'] for x in pp),
              'predicted relative coefficient RMS median',np.median([x['one_percent_noise_predicted_relative_coefficient_rms'] for x in pp]))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
