"""Saved-state statistical diagnostics; no estimator changes or truth inputs."""
import importlib.util
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'AGENTS.md').exists())
BASE = ROOT/'研究方向/多径DPD'
OLD = BASE/'实验/09_文献后机制核验/20261107'
spec = importlib.util.spec_from_file_location('old_gate', OLD/'成果/run_gate.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def stats(s):
    s = s-s.mean(axis=0)
    power = np.mean(abs(s)**2, axis=0)
    pseudo = np.mean(s*s, axis=0)
    kurt = np.mean(abs(s)**4, axis=0)/power**2-2-abs(pseudo)**2/power**2
    a,b = s.T
    cov, noncov = np.mean(a*b.conj()), np.mean(a*b)
    mixed = (np.mean(abs(a)**2*abs(b)**2)-power.prod()-abs(cov)**2-abs(noncov)**2)/power.prod()
    return dict(noncircularity=abs(pseudo)/power, naive_kurtosis=kurt,
                naive_mixed_cumulant=float(mixed), zero_lag_correlation=float(abs(cov)/np.sqrt(power.prod())))


def match(s, truth):
    s = s-s.mean(axis=0)
    truth = truth-truth.mean(axis=0)
    corr = np.empty((2,2))
    lag = np.empty((2,2),int)
    for i in range(2):
        for j in range(2):
            curve = abs(np.fft.ifft(np.fft.fft(s[:,i])*np.fft.fft(truth[:,j]).conj()))
            k = int(curve.argmax())
            corr[i,j] = curve[k]/(np.linalg.norm(s[:,i])*np.linalg.norm(truth[:,j]))
            lag[i,j] = k
    ii,jj = linear_sum_assignment(-corr**2)
    return dict(correlations=corr, assigned=jj, matched_correlation=corr[ii,jj],lags=lag[ii,jj],
                scope='oracle_waveform_diagnostic_integer_circular_shift_not_channel_recovery')


def main():
    out = HERE.parent/'运行数据/saved30'
    out.mkdir(parents=True, exist_ok=False)
    source = list((BASE/'代码/mp_dpd').rglob('*.py'))+[Path(__file__),HERE/'执行合同.md',OLD/'成果/run_gate.py']
    identity = {str(p): gate.sha(p) for p in source}
    for p in source:
        gate.copy_checked(p,out/'source'/p.relative_to(ROOT))
    gate.cfg.write(out/'preflight.json',dict(source_hashes=identity,kind='saved_state_diagnostic'))
    rows=[]; inputs={}; start=time.perf_counter()
    for old in gate.read(OLD/'运行数据/common18/rows.json'):
        tag=old['tag']
        src=OLD/'运行数据/common18/inputs'/tag/'iq.npz'
        assert gate.sha(src)==old['input_sha']
        local=out/'inputs'/tag/'iq.npz'
        gate.copy_checked(src,local); inputs[str(local)]=gate.sha(local)
        with np.load(local,allow_pickle=False) as z: iq=z['iq']
        engine=gate.api.create_engine()
        f,y,n,_=engine.fc.cal.spectral(iq)
        mask=abs(np.fft.fftfreq(n,1/engine.ref.FS))<=engine.ref.B/2
        zz=np.fft.fft(iq,axis=1,norm='ortho');zz[:,~mask]=0
        band=np.fft.ifft(zz,axis=1,norm='ortho')
        power=abs(band)**2
        power=power/power.mean(axis=1,keepdims=True)-1
        spectrum=np.sum(abs(np.fft.fft(power,axis=1))**2,axis=0)
        freq=np.fft.fftfreq(n,1/engine.ref.FS)
        ids=np.flatnonzero((freq>0)&(freq<=engine.ref.B))
        peak=int(ids[np.argmax(spectrum[ids])])
        blind_frequency=float(freq[peak])
        states=[('blind',OLD/'运行数据/common18'/f'{tag}_MP_decision.json')]
        if old['condition']!='L0':states.append(('oracle_pair_control',OLD/'运行数据/cold12'/f'{tag}_decision.json'))
        outputs=[]
        for arm,p in states:
            snapshot=out/'inputs'/tag/f'{arm}.json'
            gate.copy_checked(p,snapshot); inputs[str(snapshot)]=gate.sha(snapshot)
            solved=gate.read(snapshot)
            core=engine.core(np.asarray(solved['final']['x']),f,y,solved['paths'])
            sf=core[2]
            padded=np.zeros((n,2),complex);padded[mask]=sf
            s=np.fft.ifft(padded,axis=0,norm='ortho')
            outputs.append((arm,solved,s,stats(s)))
        # Save observation-only diagnostics before reading truth.
        gate.cfg.write(out/f'{tag}_observable.json',dict(energy_cycle_peak_hz=blind_frequency,
            states=[dict(arm=a,stats=t) for a,_,_,t in outputs],scope='descriptive_not_selection'))
        bank=Path(old['truth_evaluation_file'])
        assert gate.sha(bank)==old['truth_file_sha']
        bankcopy=out/'evaluation'/bank.name
        if not bankcopy.exists():gate.copy_checked(bank,bankcopy)
        inputs[str(bankcopy)]=gate.sha(bankcopy)
        with np.load(bankcopy,allow_pickle=False) as z:
            truth_position=z['p']; true_signal=z['los'][:,0,:].T
        ft=np.fft.fft(true_signal,axis=0,norm='ortho');ft[~mask]=0
        true_signal=np.fft.ifft(ft,axis=0,norm='ortho')
        for arm,solved,s,st in outputs:
            err=gate.errors(solved['final']['p'],truth_position)
            rows.append(dict(tag=tag,condition=old['condition'],arm=arm,stats=st,
                energy_cycle_peak_hz=blind_frequency,
                position_rmse=float(np.sqrt(np.mean(err**2))),waveform=match(s,true_signal),
                truth_stats=stats(true_signal),normal=solved['normal'],
                cost=solved['final']['cost'],K=len(solved['paths'])))
        gate.cfg.write(out/'rows.json',rows)
        print(tag,blind_frequency,flush=True)
    assert len(rows)==30
    assert all(gate.sha(Path(p))==h for p,h in {**identity,**inputs}.items())
    gate.cfg.write(out/'completion.json',dict(count=len(rows),seconds=time.perf_counter()-start,
        source_and_inputs_verified=True,source_hashes=identity,input_hashes=inputs,
        scope='saved_state_diagnostic_no_new_localization'))


if __name__=='__main__':main()
