"""Saved geometry -> conventional LOS/MP DPD; no truth accepted by solve."""
from pathlib import Path
from datetime import datetime
import json
import time
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

from geometry_association_gate import ROOT, B, sha, write, signature
from blind_initialization import rank_one, los_vector, local_los, seed_paths, residual, feasible
from boundary_update import run, state

IQ=ROOT/'finite_iq_clues_20261029_204430_528859'
INIT=ROOT/'robust_geometry_20261029_210612_452075'
PROTOCOL=ROOT/'盲后端短测协议.md'


def solve(p,f,y):
    before=signature(p,f,y)
    start=time.perf_counter()
    gains=np.column_stack([rank_one(y,f,position)[0] for position in p])
    v0=los_vector(gains)
    pp,vv,meta=local_los(p,v0,f,y)
    los_time=time.perf_counter()-start
    seed_start=time.perf_counter()
    seeded,seed_meta=seed_paths(pp,vv,f,y)
    seed_time=time.perf_counter()-seed_start
    rr,_=residual(seeded,pp,f,y)
    fit=run(pp,seeded,f,y)
    assert before==signature(p,f,y)
    return dict(initial=p,initial_v=v0,los_p=pp,los_v=vv,los=meta,los_seconds=los_time,
          seeded_v=seeded,seed=seed_meta,seed_seconds=seed_time,seed_cost=.5*float(rr@rr),
          mp=fit,seconds=time.perf_counter()-start,input_signature=before)


def errors(p,truth):
    d=np.linalg.norm(np.asarray(p)[:,None]-truth[None],axis=2)
    i,j=linear_sum_assignment(d*d)
    e=np.zeros(2)
    e[j]=d[i,j]
    return e


def main():
    files=[Path(__file__),PROTOCOL]+[ROOT/name for name in [
         'geometry_association_gate.py','blind_initialization.py','boundary_update.py',
         'structural_probe.py','position_reduction.py','solver_reliability.py']]
    for l in [0,1]:
        for r in [0,5]:
            files.append(IQ/f'L{l}_R{r}_components.npz')
            for path in ['LOS','MP']:
                files.append(INIT/f'L{l}_R{r}_{path}_AB_snr6_CC_ABS.json')
    hashes={str(p.resolve(strict=True)):sha(p) for p in files}
    assert all(Path(p).is_relative_to(ROOT) for p in hashes)
    frozen=json.loads((INIT/'protocol.json').read_text(encoding='utf-8'))['hashes']
    for p,h in hashes.items():
        if p in frozen:
            assert h==frozen[p]
    out=ROOT/('blind_backend_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    (out/'source').mkdir()
    for p in files[:8]:
        (out/'source'/p.name).write_bytes(p.read_bytes())
    write(out/'protocol.json',dict(hashes=hashes,truth_in_solver=False,oracle=False,
             samples=8,max_mp_iterations=60,window='unchanged rectangular',source_count=2))
    freq=np.fft.fftfreq(8192,1/50e6)
    mask=abs(freq)<=B/2
    f=freq[mask]
    checks={}
    rows=[]
    begin=time.perf_counter()
    print('OUTPUT',out,flush=True)
    for l in [0,1]:
        for rep in [0,5]:
            data=dict(np.load(IQ/f'L{l}_R{rep}_components.npz'))
            for path in ['LOS','MP']:
                tag=f'L{l}_R{rep}_{path}'
                original=json.loads((INIT/(tag+'_AB_snr6_CC_ABS.json')).read_text(encoding='utf-8'))
                p=np.array(original['methods']['GEO_ALL']['p'])
                iq=(data['los']+(data['reflected'] if path=='MP' else 0)).sum(axis=0)+data['noise']
                checks[tag+'/iq_identity']=signature(iq)==original['iq_sha']
                y=np.fft.fft(iq,axis=1,norm='ortho')[:,mask].T
                result=solve(p,f,y)
                for name,positions in [('geometry',p),('los',result['los_p']),('mp',result['mp']['p'])]:
                    result[name+'_errors_m']=errors(positions,data['truth'])
                checks[tag+'/seed_descent']=result['seed_cost']<=result['los']['cost']+1e-10
                checks[tag+'/mp_descent']=result['mp']['cost']<=result['seed_cost']+1e-10
                checks[tag+'/feasible']=feasible(result['mp']['p'],result['mp']['v'])
                history=result['mp']['history']
                checks[tag+'/trajectory_descent']=all(b['cost']<=a['cost']+1e-10 for a,b in zip(history,history[1:]))
                if not rows:
                    from structural_probe import C
                    x=np.r_[result['los_p'].ravel()/(C/B),result['seeded_v']]
                    rr,j,_=state(x,f,y)
                    ids=[0,3,4,12,20,40,60,79]
                    eps=1e-6
                    numeric=np.column_stack([(state(x+np.eye(80)[i]*eps,f,y)[0]-state(x-np.eye(80)[i]*eps,f,y)[0])/(2*eps) for i in ids])
                    checks['new_axis_representative_jacobian']=np.linalg.norm(j[:,ids]-numeric)/np.linalg.norm(numeric)<1e-6
                result.update(tag=tag,oracle=False)
                rows.append(result)
                write(out/(tag+'.json'),result)
                np.savez_compressed(out/(tag+'.npz'),f=f,y=y,initial=p,truth=data['truth'])
                print(tag,'geom',np.round(result['geometry_errors_m'],3),'LOS',np.round(result['los_errors_m'],3),
                      'MP',np.round(result['mp_errors_m'],3),result['mp']['reason'],round(result['seconds'],2),flush=True)
                assert all(checks.values()),checks
    checks.update({p:sha(p)==h for p,h in hashes.items()})
    assert all(checks.values())
    write(out/'result.json',dict(rows=rows,checks=checks,seconds=time.perf_counter()-begin,oracle=False))
    print('COMPLETE',out,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2),warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='divide by zero encountered in log10',category=RuntimeWarning)
        main()
