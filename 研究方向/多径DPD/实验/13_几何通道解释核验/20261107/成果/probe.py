"""Diagnose Fourier delay aliases in all saved states, without localization."""
import importlib.util
from pathlib import Path
import time

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'AGENTS.md').exists())
BASE=ROOT/'研究方向/多径DPD'
PREV=BASE/'实验/12_独立性统计准入/20261107/运行数据/saved30'
loader=BASE/'实验/09_文献后机制核验/20261107/成果/run_gate.py'
spec=importlib.util.spec_from_file_location('gate',loader)
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)


def main():
    out=HERE.parent/'运行数据/alias30';out.mkdir(parents=True,exist_ok=False)
    inputs={}
    def copy(p,dst):
        digest=gate.copy_checked(p,dst);inputs[str(dst)]=digest
    sources=list((BASE/'代码/mp_dpd').rglob('*.py'))+[Path(__file__),HERE/'执行合同.md',loader]
    identity={str(p):gate.sha(p) for p in sources}
    for p in sources:copy(p,out/'source'/p.relative_to(ROOT))
    done=gate.read(PREV/'completion.json')
    assert all(gate.sha(Path(p))==h for p,h in done['input_hashes'].items())
    copy(PREV/'rows.json',out/'previous_rows.json')
    gate.cfg.write(out/'preflight.json',dict(source_hashes=identity,input_hashes=inputs))
    rows=[];start=time.perf_counter()
    for old in gate.read(out/'previous_rows.json'):
        tag,arm=old['tag'],old['arm']
        dst=out/'inputs'/tag
        if not (dst/'iq.npz').exists():copy(PREV/'inputs'/tag/'iq.npz',dst/'iq.npz')
        copy(PREV/'inputs'/tag/f'{arm}.json',dst/f'{arm}.json')
        solved=gate.read(dst/f'{arm}.json')
        with np.load(dst/'iq.npz',allow_pickle=False) as z:iq=z['iq']
        engine=gate.api.create_engine();f,y,n,_=engine.fc.cal.spectral(iq)
        x=np.asarray(solved['final']['x']);paths=solved['paths']
        assert engine.D==16 and len(x)==engine.D+3*len(paths)
        T=n/engine.ref.FS;period=T*engine.ref.B
        delay=x[engine.D+2::3];amplitude=abs(x[engine.D::3]+1j*x[engine.D+1::3])
        wrapped=delay>period/2
        data=engine.core(x,f,y,paths)
        xx=x.copy();xx[engine.D+2::3]-=wrapped*period
        alias=engine.core(xx,f,y,paths)
        relative=float(np.linalg.norm(alias[0]-data[0])/np.linalg.norm(data[0]))
        assert relative<1e-10
        keep=np.flatnonzero(~wrapped)
        xp=np.r_[x[:engine.D],x[engine.D:].reshape(-1,3)[keep].ravel()]
        pp=[paths[k] for k in keep]
        removed=engine.core(xp,f,y,pp)
        # Same y normalization; this is a fixed-channel deletion, not optimization.
        cost=float(np.linalg.norm(data[3])**2/np.linalg.norm(y)**2)
        deletecost=float(np.linalg.norm(removed[3])**2/np.linalg.norm(y)**2)
        pathrows=[dict(station=m,source=q,delay_us=float(delay[k]/engine.ref.B*1e6),
                      signed_principal_delay_us=float((delay[k]-period*wrapped[k])/engine.ref.B*1e6),
                      relative_amplitude=float(amplitude[k]),negative_alias=bool(wrapped[k]))
                  for k,(m,q) in enumerate(paths)]
        row=dict(tag=tag,arm=arm,position_rmse=old['position_rmse'],normal=solved['normal'],
                 K=len(paths),negative_alias_count=int(wrapped.sum()),
                 max_negative_alias_amplitude=float(amplitude[wrapped].max()) if wrapped.any() else 0.,
                 normalized_rss=cost,deletion_normalized_rss=deletecost,
                 channel_alias_relative_error=relative,record_us=T*1e6,paths=pathrows,
                 matched_correlation=old['waveform']['matched_correlation'])
        rows.append(row);gate.cfg.write(out/'rows.json',rows)
        print(tag,arm,'RMSE',round(row['position_rmse'],3),'alias',row['negative_alias_count'],
              'amp',round(row['max_negative_alias_amplitude'],3),'rss',round(cost,4),round(deletecost,4),flush=True)
    assert len(rows)==30
    assert all(gate.sha(Path(p))==h for p,h in {**identity,**inputs}.items())
    gate.cfg.write(out/'completion.json',dict(count=len(rows),seconds=time.perf_counter()-start,
        source_hashes=identity,input_hashes=inputs,identity_verified=True,scope='saved_state_no_new_localization'))


if __name__=='__main__':main()
