"""Observed-IQ cross-spectrum component interface, not a localization solver."""
import argparse
import importlib.util
from itertools import combinations, product
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'AGENTS.md').exists())
BASE = ROOT/'研究方向/多径DPD'
OLD = BASE/'实验/09_文献后机制核验/20261107'
loader = OLD/'成果/run_gate.py'
spec = importlib.util.spec_from_file_location('gate', loader)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
PAIRS = list(combinations(range(4), 2))


def fit_components(z, kernel, f, fs, n, bound, capacity=64):
    """No truth/condition argument. Joint complex LS on incrementally found lags."""
    nf = len(f)
    length = 8*n
    bins = np.rint(f/fs*n).astype(int) % length
    grid = np.fft.fftfreq(length, d=fs/n)
    allowed = abs(grid) <= bound
    norm = np.linalg.norm(z)
    if norm == 0:
        return dict(delays=[],coefficients=[],history=[],stop='zero_input')
    target = z/norm
    residual = target.copy()
    ds, atoms, history = [], [], []
    score = 2*nf*np.log(np.vdot(residual,residual).real/(2*nf))
    beta = np.empty(0,complex)
    stop = 'capacity'
    for k in range(1,capacity+1):
        padded = np.zeros(length,complex)
        padded[bins] = kernel*residual
        correlation = abs(np.fft.ifft(padded)*length)**2
        correlation[~allowed] = -1
        for d in ds:
            correlation[abs(grid-d) < .5/fs/8] = -1
        index = int(np.argmax(correlation))
        d = float(grid[index])
        atom = kernel*np.exp(-2j*np.pi*f*d)
        matrix = np.column_stack(atoms+[atom])
        trial,_,rank,_ = np.linalg.lstsq(matrix,target,rcond=None)
        rr = target-matrix@trial
        rss = float(np.vdot(rr,rr).real)
        new_score = 2*nf*np.log(max(rss,np.finfo(float).tiny)/(2*nf))+3*k*np.log(2*nf)
        accepted = new_score < score and rank == k
        history.append(dict(k=k,delay=d,rss=rss,criterion=new_score,accepted=accepted))
        if not accepted:
            stop = 'criterion_no_decrease'
            break
        ds.append(d);atoms.append(atom);beta=trial;residual=rr;score=new_score
    return dict(delays=ds,coefficients=np.column_stack(((beta*norm).real,(beta*norm).imag)),
                history=history,stop=stop)


def observed(iq,engine):
    f,y,n,noise=engine.fc.cal.spectral(iq)
    order=np.argsort(f);f=f[order];y=y[order]
    width=int(np.ceil(np.sqrt(len(f)))) | 1
    envelope=uniform_filter1d(np.mean(abs(y)**2,axis=1)-noise,size=width,mode='nearest')
    envelope=np.maximum(envelope,0)
    if not np.any(envelope):
        return f,{},dict(status='nonpositive_envelope',width=width)
    envelope/=np.mean(envelope)
    kernels={'FLAT':np.ones(len(f)),'OBSERVED_COMMON':envelope}
    answers={}
    for name,kernel in kernels.items():
        rows=[]
        for m,j in PAIRS:
            cross=y[:,m].conj()*y[:,j]
            bound=np.linalg.norm(engine.stations[m]-engine.stations[j])/engine.ref.C+20e-6
            row=fit_components(cross,kernel,f,engine.ref.FS,n,bound)
            row.update(pair=[m,j]);rows.append(row)
        answers[name]=rows
    return f,answers,dict(status='output',width=width,noise=noise,kernels=kernels,y=y)


def evaluate(f,answers,context,bank,meta,stations):
    # All labels are consumed only after blind outputs have been saved.
    power=np.asarray(meta['power_ratios'])
    arrival=np.linalg.norm(stations[:,None]-bank['p'][None],axis=2)/gate.cfg.C
    gain=bank['gains']
    lagsets=[];coeffsets=[];labelsets=[]
    for m,j in PAIRS:
        lags=[];coef=[];labels=[]
        for q in range(2):
            ixm=[-1]+list(np.flatnonzero(power[m,q]>0))
            ixj=[-1]+list(np.flatnonzero(power[j,q]>0))
            for l,k in product(ixm,ixj):
                dm=0 if l==-1 else bank['delays'][m,q,l]/gate.cfg.B
                dj=0 if k==-1 else bank['delays'][j,q,k]/gate.cfg.B
                am=gain[m,q]*(1 if l==-1 else np.sqrt(power[m,q,l])*bank['phases'][m,q,l])
                aj=gain[j,q]*(1 if k==-1 else np.sqrt(power[j,q,k])*bank['phases'][j,q,k])
                lags.append(arrival[j,q]-arrival[m,q]+dj-dm)
                coef.append(am.conjugate()*aj);labels.append((q,l,k))
        lagsets.append(np.array(lags));coeffsets.append(np.array(coef));labelsets.append(labels)
    report={}
    for name,rows in answers.items():
        details=[];direct_coeff={};oracle_coeff={}
        for pairid,(row,truth,labels) in enumerate(zip(rows,lagsets,labelsets)):
            d=np.array(row['delays']);cc=np.asarray(row['coefficients']).reshape(-1,2)
            coef=cc[:,0]+1j*cc[:,1]
            errors=np.full(len(truth),np.inf);matched=np.full(len(truth),-1,int)
            if len(d):
                ii,jj=linear_sum_assignment(abs(truth[:,None]-d[None]))
                errors[ii]=abs(truth[ii]-d[jj])*gate.cfg.B;matched[ii]=jj
            direct=[i for i,x in enumerate(labels) if x[1:]==(-1,-1)]
            dc=[]
            for i in direct:
                dc.append(None if matched[i]<0 else complex(coef[matched[i]]))
            direct_coeff[PAIRS[pairid]]=dc
            # ORACLE support LS: isolate missed-support vs coefficient-model mismatch.
            kernel=context['kernels'][name]
            matrix=kernel[:,None]*np.exp(-2j*np.pi*f[:,None]*truth[None])
            m,j=PAIRS[pairid];target=context['y'][:,m].conj()*context['y'][:,j]
            beta,_,rank,s=np.linalg.lstsq(matrix,target,rcond=None)
            oracle_coeff[PAIRS[pairid]]=[complex(beta[i]) for i in direct]
            details.append(dict(pair=[m,j],selected=len(d),true_count=len(truth),
                matched_count=int(sum(errors<=.25)),direct_errors_B=[None if not np.isfinite(errors[i]) else float(errors[i]) for i in direct],
                direct_hits=int(sum(errors[direct]<=.25)),oracle_rank=int(rank),
                oracle_condition=float(s[0]/s[-1]),oracle_relative_residual=float(np.linalg.norm(target-matrix@beta)/np.linalg.norm(target))))
        def phases(coeff):
            result=[]
            for q in range(2):
                for a,b,c in combinations(range(4),3):
                    vals=[coeff[a,b][q],coeff[b,c][q],coeff[a,c][q]]
                    result.append(None if any(z is None or abs(z)==0 for z in vals) else float(abs(np.angle(vals[0]*vals[1]*vals[2].conjugate()))))
            return result
        report[name]=dict(pairs=details,direct_hits=sum(v['direct_hits'] for v in details),direct_total=12,
            matched=sum(v['matched_count'] for v in details),total=sum(v['true_count'] for v in details),
            phase_error_rad=phases(direct_coeff),oracle_phase_error_rad=phases(oracle_coeff))
    return report


def unit():
    n=256;fs=50e6;f=np.fft.fftfreq(n,1/fs);f=f[abs(f)<=12.5e6/2]
    kernel=1+.2*np.cos(f/12.5e6*2*np.pi)
    d=np.array([16.,-24.])/fs
    beta=np.array([1+.4j,-.3+.6j]);z=(kernel[:,None]*np.exp(-2j*np.pi*f[:,None]*d))@beta
    out=fit_components(z,kernel,f,fs,n,2e-6,capacity=2)
    ids=np.argsort(out['delays']);actual=np.array(out['delays'])[ids]
    coeff=np.array(out['coefficients']);coeff=(coeff[:,0]+1j*coeff[:,1])[ids]
    assert np.allclose(actual,np.sort(d),atol=1e-15,rtol=0)
    assert np.allclose(coeff,beta[np.argsort(d)],atol=1e-10,rtol=0)
    return dict(sign_and_complex_gain=True,scope='exact_grid_kernel_unit_not_IQ')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=18);parser.add_argument('--name',required=True)
    args=parser.parse_args();assert 0<args.limit<=18
    out=HERE.parent/'运行数据'/args.name;out.mkdir(parents=True,exist_ok=False)
    files=list((BASE/'代码/mp_dpd').rglob('*.py'))+[Path(__file__),HERE/'执行合同.md',loader]
    identity={str(p):gate.sha(p) for p in files}
    for p in files:gate.copy_checked(p,out/'source'/p.relative_to(ROOT))
    gate.cfg.write(out/'preflight.json',dict(checks=unit(),source_hashes=identity))
    start=time.perf_counter();results=[];inputs={}
    for old in gate.read(OLD/'运行数据/common18/rows.json')[:args.limit]:
        tag=old['tag'];src=OLD/'运行数据/common18/inputs'/tag/'iq.npz'
        assert gate.sha(src)==old['input_sha']
        dst=out/'inputs'/tag/'iq.npz';gate.copy_checked(src,dst);inputs[str(dst)]=gate.sha(dst)
        with np.load(dst,allow_pickle=False) as z:iq=z['iq']
        engine=gate.api.create_engine();tick=time.perf_counter()
        f,answers,context=observed(iq,engine);seconds=time.perf_counter()-tick
        gate.cfg.write(out/f'{tag}_blind.json',dict(answers=answers,seconds=seconds,status=context['status'],smoothing_bins=context['width']))
        # Frozen evaluation references, read only after output.
        bank_path=Path(old['truth_evaluation_file']);assert gate.sha(bank_path)==old['truth_file_sha']
        bank_dst=out/'evaluation'/bank_path.name
        if not bank_dst.exists():gate.copy_checked(bank_path,bank_dst)
        inputs[str(bank_dst)]=gate.sha(bank_dst)
        meta_src=gate.OLD_C/'cases'/tag/'input.json';meta_dst=out/'evaluation'/f'{tag}.json'
        gate.copy_checked(meta_src,meta_dst);inputs[str(meta_dst)]=gate.sha(meta_dst)
        with np.load(bank_dst,allow_pickle=False) as z:bank=dict(z)
        report=evaluate(f,answers,context,bank,gate.read(meta_dst),engine.stations)
        row=dict(tag=tag,condition=old['condition'],seconds=seconds,report=report)
        results.append(row);gate.cfg.write(out/'rows.json',results)
        print(tag,round(seconds,3),{a:r['direct_hits'] for a,r in report.items()},flush=True)
    assert all(gate.sha(Path(p))==h for p,h in {**identity,**inputs}.items())
    gate.cfg.write(out/'completion.json',dict(count=len(results),seconds=time.perf_counter()-start,input_hashes=inputs,source_hashes=identity,identity_verified=True))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
