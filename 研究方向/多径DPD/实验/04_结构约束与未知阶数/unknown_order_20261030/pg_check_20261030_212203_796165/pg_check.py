"""Replay affected observations from blind starts, reuse only unchanged branches."""
from datetime import datetime
import json
from pathlib import Path
import time
import hashlib

import numpy as np
from threadpoolctl import threadpool_limits

import reference as ref
from reference_pg import fit


def main():
    root=ref.HERE
    old=Path(json.loads((root/'full_latest.json').read_text(encoding='utf-8'))['path'])
    full=json.loads((old/'result.json').read_text(encoding='utf-8'))
    known=json.loads((old/'preflight.json').read_text(encoding='utf-8'))['hashes']
    assert all(ref.sha(p)==h for p,h in known.items())
    selected=[r for r in full['rows'] if any(not e['fit']['stationary'] and e['fit']['iterations']<300 for e in r['history'] if 'fit' in e)]
    lookup={tag:(saved,inp,mode) for tag,saved,inp,mode in ref.cases(False)}
    files=[Path(__file__),root/'reference.py',root/'reference_pg.py',root/'投影收尾批准协议.md',old/'result.json']
    files += [p for r in selected for p in lookup[r['tag']][:2]]
    hashes={str(p.resolve(strict=True)):ref.sha(p) for p in files}
    out=root/('pg_check_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    for name in ['pg_check.py','reference_pg.py']:
        (out/name).write_bytes((root/name).read_bytes())
    ref.write(out/'preflight.json',dict(hashes=hashes,affected=[r['tag'] for r in selected],unchanged_reused=48-len(selected)))
    checks={}
    rows=[]
    begin=time.perf_counter()
    print('输出',out,'受影响',len(selected),flush=True)
    for original in selected:
        tag=original['tag']
        savedp,inp,mode=lookup[tag]
        saved=json.loads(savedp.read_text(encoding='utf-8'))
        with np.load(inp,allow_pickle=False) as data:
            iq=(data['los']+(data['reflected'] if mode=='MP' else 0)).sum(axis=0)+data['noise']
        checks[tag+'/iq_identity']=hashlib.sha256(np.ascontiguousarray(iq).tobytes()).hexdigest()==saved['iq_sha']
        n=iq.shape[1];f0=np.fft.fftfreq(n,1/ref.FS);use=abs(f0)<=ref.B/2
        ft=np.fft.fft(iq,axis=1,norm='ortho');f,y=f0[use],ft[:,use].T
        sigma2=float(np.mean(abs(ft[:,~use])**2))
        x0=np.r_[np.array(saved['initial']).ravel()/ref.UNIT,np.array(saved['initial_v'])[:16]]
        x,los=ref.los_fit(x0,f,y,n)
        paths,history=[],[]
        state=los;stop='capacity';finish_seen=False
        for step in range(1,65):
            pick,bank,meta=ref.search(x,paths,f,y,n,use,sigma2,step)
            checks.update({tag+f'/{step}/'+k:v for k,v in meta['checks'].items()})
            np.savez_compressed(out/f'{tag}_{step}_candidates.npz',bank=bank)
            ev=dict(step=step,pick=pick,search=meta);history.append(ev)
            if pick is None:
                stop='no_detected_increment';break
            pp=paths+[(pick['m'],pick['q'])]
            xx,new=fit(np.r_[x,pick['rho'],pick['delay']],f,y,pp,n)
            ev['fit']=new
            checks[tag+f'/{step}/budget']=new['iterations']<=300
            checks[tag+f'/{step}/feasible']=new['closure_error']<1e-7
            checks[tag+f'/{step}/descent']=new['cost']<=state['cost']+1e-10
            if not finish_seen and 'pg_finish' not in new:
                prior=original['history'][step-1]['fit']
                checks[tag+f'/{step}/prefix_equivalence']=np.allclose(new['x'],prior['x'],atol=1e-9,rtol=0)
            finish_seen |= 'pg_finish' in new
            x,paths,state=xx,pp,new
            print(tag,step,'梯度',format(new['projected_gradient'],'.2g'),'收尾','pg_finish' in new,flush=True)
            if not state['stationary']:
                stop='numerical_nonstationary';break
            if not all(checks.values()):
                stop='correctness_failure';break
        with np.load(inp,allow_pickle=False) as data:
            truth=data['p']
        row=dict(tag=tag,stop=stop,los=los,final=state,paths=paths,history=history,
                 los_errors_m=ref.error(los['p'],truth),errors_m=ref.error(state['p'],truth))
        rows.append(row);ref.write(out/(tag+'.json'),row)
    checks['identity_unchanged']=all(ref.sha(p)==h for p,h in hashes.items())
    complete=all(r['stop']=='no_detected_increment' for r in rows)
    ref.write(out/'result.json',dict(rows=rows,checks=checks,all_checks_pass=all(checks.values()),
                                   all_order_stopped=complete,unchanged_reused=48-len(selected),seconds=time.perf_counter()-begin))
    ref.write(root/'pg_check_latest.json',dict(path=str(out)))
    print('完成',all(checks.values()),complete,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
