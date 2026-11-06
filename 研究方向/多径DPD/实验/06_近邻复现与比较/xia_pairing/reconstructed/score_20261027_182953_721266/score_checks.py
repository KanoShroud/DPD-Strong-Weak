"""Fixed-input diagnosis of Eq9 spectral division and Eq11--16 selection.

No location solves or fitted parameters. Dropping 1/5 percent of low observed
reference-power bins is a predeclared sensitivity check on SAVED particles,
not a newly evaluated localization algorithm. Truth only labels diagnostics.
"""
import csv
from collections import Counter
from datetime import datetime
import hashlib
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits
from pilot import ROOT, N, C, sha, write, observe, xia
from trace_audit import read

RUN=ROOT/'square_20261027_182013_211372'


def csv_write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def candidate_components(points,stations,y,f,masks):
    d=np.linalg.norm(points[:,None]-stations[None],axis=-1)
    delay=(d[:,1:]-d[:,:1])/C
    ratio=y[1:]/y[:1]
    all_scores=[[] for _ in masks]
    full_components=[]
    for start in range(0,len(points),64):
        phase=np.exp(2j*np.pi*delay[start:start+64,:,None]*f[None,None])
        term=phase*ratio[None]/N
        comp=term.sum(-1)
        full_components.append(comp)
        for k,mask in enumerate(masks):
            value=term[:,:,mask].sum(-1)
            # Reference self-factor is kept too; FFT normalization stays N.
            all_scores[k].append(np.prod(abs(value),axis=1)*mask.sum()/N)
    return [np.concatenate(v) for v in all_scores],np.concatenate(full_components)


def main():
    begun=time.perf_counter()
    frozen=read(ROOT/'pilot_20261027_170421_805496/preflight.json')
    for name,h in frozen['sources'].items(): assert sha(Path(name))==h
    files=list(RUN.glob('r*_G*.json'))+list(RUN.glob('input_*.npz'))+[RUN/'protocol.json',RUN/'summary.json',RUN/'result.json']
    manifest={p.name:sha(p) for p in files}
    out=ROOT/('score_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir(exist_ok=False)
    write(out/'protocol.json',dict(input=str(RUN.resolve(strict=True)),input_hashes=manifest,
        source_sha=sha(__file__),paper_sha=frozen['paper_sha'],new_solves=0,
        fractions=[0,.01,.05],selection='lowest measured |R1|^2, stable tie index, no truth',
        normalization='FFT divisor N=2000 unchanged; reference self factor retained',
        points='saved first-round 1000 particles plus two exact truths for diagnostics only',
        mechanism='linear numerator decomposition with SAME observed noisy denominator; no counterfactual clean denominator',
        noise_dominance='clean reference signal bin power < sigma2/N; known generator components used offline',
        not_claimed=['new algorithm performance','optimal bin cutoff','global peak at exact truth','author-code reproduction']))
    (out/'score_checks.py').write_bytes(Path(__file__).read_bytes())
    checks={};rows=[];sensitivity=[];filters=[];cache={}; examples=[]
    for index,file in enumerate(sorted(RUN.glob('r*_G*.json'))):
        r=read(file);i=r['realization']
        if i not in cache: cache[i]=dict(np.load(RUN/f'input_{i:02d}.npz'))
        z=cache[i];y,source,sigma=observe(z,r['snr_db'],r['sir_db'],r['paths'])
        checks[file.stem+'/input']=sha(RUN/f'input_{i:02d}.npz')==r['input_sha']
        checks[file.stem+'/observation']=hashlib.sha256(y.tobytes()).hexdigest()==r['observation_sha']
        noise=np.sqrt(sigma)*z['noise_f']; clean=source.sum(0)
        q=y[1:]/y[:1]
        checks[file.stem+'/division_identity']=np.allclose(q,y[0].conj()[None]*y[1:]/abs(y[:1])**2,rtol=1e-13,atol=1e-15)
        qparts=np.concatenate([source[:,1:]/y[None,:1],noise[None,1:]/y[None,:1]],axis=0)
        checks[file.stem+'/numerator_decomposition']=np.allclose(qparts.sum(0),q,rtol=1e-12,atol=1e-13)
        power=abs(y[0])**2; order=np.argsort(power,kind='stable')
        masks=[]
        for frac in [0,.01,.05]:
            mask=np.ones(N,bool);mask[order[:int(N*frac)]]=False;masks.append(mask)
        pts=np.asarray(r['output']['steps'][0]['points']);saved=np.asarray(r['output']['steps'][0]['scores'])
        points=np.vstack([pts,z['truth']])
        scores,comp=candidate_components(points,z['stations'],y,z['frequency'],masks)
        checks[file.stem+'/saved_scores']=np.allclose(scores[0][:-2],saved,rtol=1e-12,atol=0)
        checks[file.stem+'/truth_scores']=np.allclose(scores[0][-2:],xia.scores(z['truth'],z['stations'],y,z['frequency'],N),rtol=1e-12,atol=0)
        # Check masked expression independently at the two truth coordinates.
        for k,mask in enumerate(masks[1:],1):
            direct=xia.scores(z['truth'],z['stations'],y[:,mask],z['frequency'][mask],N)
            checks[file.stem+f'/mask{k}']=np.allclose(scores[k][-2:],direct,rtol=1e-12,atol=0)
        winner=int(saved.argmax()); olderr=np.linalg.norm(pts[winner]-z['truth'],axis=1)
        ratio_energy=np.sum(abs(q)**2,axis=0)
        signal_power=abs(clean[0])**2; noisefloor=sigma/N
        low=~masks[2]
        denom=abs(source[0,0])**2+abs(source[1,0])**2
        cancel=np.divide(signal_power,denom,out=np.ones_like(denom),where=denom>0)
        row=dict(case=file.stem,realization=i,G=r['paths'],SNR=r['snr_db'],SIR=r['sir_db'],
            reference_min_over_median=float(power.min()/np.median(power)),
            low1_ratio_energy_share=float(ratio_energy[~masks[1]].sum()/ratio_energy.sum()),
            low5_ratio_energy_share=float(ratio_energy[low].sum()/ratio_energy.sum()),
            all_noise_dominated_fraction=float(np.mean(signal_power<noisefloor)),
            low5_noise_dominated_fraction=float(np.mean(signal_power[low]<noisefloor)),
            low5_strong_cancellation_fraction=float(np.mean(cancel[low]<.1)),
            low5_reference_clean_median_over_noise=float(np.median(signal_power[low]/noisefloor)),
            original_first_A_m=float(olderr[0]),original_first_nearest_m=float(olderr.min()),
            A_rank=int(np.sum(saved>scores[0][-2]))+1,B_rank=int(np.sum(saved>scores[0][-1]))+1)
        for k,(frac,mask) in enumerate(zip([0,.01,.05],masks)):
            v=scores[k][:-2];w=int(v.argmax());err=np.linalg.norm(pts[w]-z['truth'],axis=1)
            sensitivity.append(dict(case=file.stem,G=r['paths'],SNR=r['snr_db'],SIR=r['sir_db'],removed=frac,
                winner_changed=w!=winner,first_A_m=float(err[0]),first_B_m=float(err[1]),first_nearest_m=float(err.min()),
                A_rank=int(np.sum(v>scores[k][-2]))+1,B_rank=int(np.sum(v>scores[k][-1]))+1,
                A_score=float(scores[k][-2]),B_score=float(scores[k][-1]),winner_score=float(v[w]),
                old_winner_score=float(v[winner])))
        # Station-resolved phasor contributions at A, B and original winner.
        diagnostic_points=np.vstack([z['truth'],pts[winner]])
        d=np.linalg.norm(diagnostic_points[:,None]-z['stations'][None],axis=-1)
        phase=np.exp(2j*np.pi*((d[:,1:]-d[:,:1])/C)[:,:,None]*z['frequency'][None,None])
        terms=phase*q[None]/N
        total=terms.sum(-1); lowcomp=terms[:,:,low].sum(-1)
        decomposition=np.einsum('pmf,smf->psm',phase,qparts)/N
        checks[file.stem+'/phasor_decomposition']=np.allclose(decomposition.sum(1),total,rtol=1e-12,atol=1e-14)
        row['winner_low5_phasor_relative_norm']=float(np.linalg.norm(lowcomp[-1])/np.linalg.norm(total[-1]))
        row['winner_noise_numerator_relative_norm']=float(np.linalg.norm(decomposition[-1,2])/np.linalg.norm(total[-1]))
        rows.append(row)
        np.savez_compressed(out/(file.stem+'_spectral.npz'),frequency=z['frequency'],reference_power=power,
            clean_reference_power=signal_power,noise_expected_bin_power=noisefloor,ratio_energy=ratio_energy,
            cancellation_ratio=cancel,diagnostic_points=diagnostic_points,full_components=total,
            low5_components=lowcomp,numerator_components=decomposition,particle_scores=np.array(scores),
            low_power_order=order)
        for step in r['output']['steps']:
            v=np.array(step['scores']);p=np.array(step['points']);dec=step['output'];filt=xia.threshold(v);keep=filt['keep']
            checks[file.stem+f'/decode{step["round"]}']=xia.decode(p,v,2,400)==dec
            filtrow=dict(case=file.stem,round=step['round'],status=dec['status'],mode=filt['modal_bin'],
                min_over_max=float(v.min()/v.max()),threshold_over_max=float(filt['threshold']/v.max()),
                keep=int(keep.sum()),parent1_keep=int(keep[::2].sum()) if step['round']>1 else None,
                parent2_keep=int(keep[1::2].sum()) if step['round']>1 else None,
                parent1_min=float(v[::2].min()) if step['round']>1 else None,
                parent1_max=float(v[::2].max()) if step['round']>1 else None,
                parent2_min=float(v[1::2].min()) if step['round']>1 else None,
                parent2_max=float(v[1::2].max()) if step['round']>1 else None,
                threshold=float(filt['threshold']),max=float(v.max()),
                cluster_radius=dec.get('trace',[[None]])[-1][0])
            filters.append(filtrow)
        if index==0: examples.append(dict(case=file.stem,points=diagnostic_points,total=total,low=lowcomp))
        assert all(checks.values()),[k for k,v in checks.items() if not v]
        print(f'原评分分解 {index+1}/84 | {file.stem}',flush=True)
    for name,h in manifest.items():checks['input_end/'+name]=sha(RUN/name)==h
    for name,h in frozen['sources'].items(): checks['code_end/'+name]=sha(Path(name))==h
    assert all(checks.values())
    csv_write(out/'spectral.csv',rows);csv_write(out/'sensitivity.csv',sensitivity);csv_write(out/'filter.csv',filters)
    summary=dict(n=84,independent_realizations=12,new_solves=0,
        spectral={k:dict(median=float(np.median([r[k] for r in rows])),minimum=float(min(r[k] for r in rows)),maximum=float(max(r[k] for r in rows))) for k in ['low1_ratio_energy_share','low5_ratio_energy_share','all_noise_dominated_fraction','low5_noise_dominated_fraction','low5_strong_cancellation_fraction','winner_low5_phasor_relative_norm','winner_noise_numerator_relative_norm']},
        sensitivity=[],rounds={},seconds=time.perf_counter()-begun,checks=len(checks))
    for frac in [0,.01,.05]:
        sub=[r for r in sensitivity if r['removed']==frac]
        summary['sensitivity'].append(dict(removed=frac,winner_changed=sum(r['winner_changed'] for r in sub),
            first_A_RMSE=float(np.sqrt(np.mean([r['first_A_m']**2 for r in sub]))),
            nearest100=sum(r['first_nearest_m']<=100 for r in sub),A_top10=sum(r['A_rank']<=10 for r in sub),B_top10=sum(r['B_rank']<=10 for r in sub),
            recovered_nearest100=sum(r['first_nearest_m']<=100 and old['original_first_nearest_m']>100 for r,old in zip(sub,rows)),
            damaged_nearest100=sum(r['first_nearest_m']>100 and old['original_first_nearest_m']<=100 for r,old in zip(sub,rows))))
    for j in [1,2,3]:
        sub=[r for r in filters if r['round']==j]
        summary['rounds'][str(j)]=dict(n=len(sub),status=dict(Counter(r['status'] for r in sub)),
            modal_bins=dict(Counter(r['mode'] for r in sub)),
            min_max_median=float(np.median([r['min_over_max'] for r in sub])) if sub else None,
            threshold_above_max=sum(r['threshold_over_max']>1 for r in sub),
            second_parent_all_deleted=sum(r['parent2_keep']==0 and r['parent1_keep']>0 for r in sub if r['parent2_keep'] is not None))
    write(out/'summary.json',summary)
    write(out/'audit.json',dict(passed=True,count=len(checks),checks={k:bool(v) for k,v in checks.items()}))
    print(summary);print('REPORT',out,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2): main()
