"""Conditional fresh-IQ order transfer; physical truth never enters solve_iq."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

from reference import HERE, BASE, B, C, FS, UNIT, STATIONS, search, los_fit, sha, write, error
from reference_pg import fit
sys.path.insert(0,str(BASE))
from finite_iq_clues import synthesize, extract, RS, SPAN  # noqa: E402
from geometry_association_gate import PAIRS  # noqa: E402
from robust_geometry import triple_candidates  # noqa: E402
from bounded_consensus import fit as geometry_fit, joint_select  # noqa: E402
from blind_initialization import rank_one  # noqa: E402

N=8192


def generate_bank(p, li, unseen):
    rng=np.random.default_rng((2026103023 if unseen else 2026103022)+10000*li)
    first=-int(np.ceil((np.sqrt(2)*6000/C+40/B)*RS))-2*SPAN-4
    last=int(np.ceil(N/FS*RS))+2*SPAN+4
    bits=rng.choice([-1.,1.],(2,last-first+1))
    offsets=rng.uniform(0,1/RS,2)
    distance=np.linalg.norm(STATIONS[:,None]-p[None],axis=2)
    gains=1000/distance*np.exp(1j*rng.uniform(-np.pi,np.pi,(5,2)))
    delays=np.array([3.,7.,11.,15.])[None,None]+rng.uniform(0,.5,(5,2,4))
    phases=np.exp(1j*rng.uniform(-np.pi,np.pi,(5,2,4)))
    weights=rng.uniform(.8,1.2,(5,2,4))
    single=rng.integers(0,4,(5,2))
    t=np.arange(N)/FS
    los=np.empty((2,5,N),complex)
    bank=np.empty((2,5,4,N),complex)
    for q in range(2):
        for m in range(5):
            tau=distance[m,q]/C
            los[q,m]=gains[m,q]*synthesize(bits[q],first,offsets[q],t-tau)
            for ell in range(4):
                bank[q,m,ell]=gains[m,q]*phases[m,q,ell]*synthesize(bits[q],first,offsets[q],t-tau-delays[m,q,ell]/B)
    variance=np.mean(abs(los[1])**2)/(10**.6*(B/FS))
    noise=np.sqrt(variance/2)*(rng.normal(size=(5,N))+1j*rng.normal(size=(5,N)))
    los_snr_db=10*np.log10(np.mean(abs(los)**2,axis=2)/(variance*(B/FS)))
    return dict(los=los,bank=bank,noise=noise,bits=bits,first=first,offsets=offsets,los_snr_db=los_snr_db,
                gains=gains,delays=delays,phases=phases,weights=weights,single=single,variance=variance,p=p)


def observation(bank, condition):
    counts=np.full((5,2),int(condition[1:]) if condition.startswith('L') else 4)
    if condition=='MIX':
        counts=np.resize([0,1,2,4],10).reshape(5,2)
    ratio=1.2 if condition=='HIGH' else 10**(-.6)
    reflected=np.zeros_like(bank['los'])
    coefficients=np.zeros((5,2,4),complex)
    for m in range(5):
        for q in range(2):
            k=counts[m,q]
            ids={0:[],1:[int(bank['single'][m,q])],2:[0,3],3:[0,1,3],4:[0,1,2,3]}[k]
            if ids:
                amp=np.sqrt(ratio*bank['weights'][m,q,ids]/sum(bank['weights'][m,q,ids]))
                for ii,ell in enumerate(ids):
                    reflected[q,m]+=amp[ii]*bank['bank'][q,m,ell]
                    coefficients[m,q,ell]=amp[ii]*bank['phases'][m,q,ell]
    assert np.max(abs(coefficients))<1
    total=np.sum(abs(coefficients)**2,axis=2)
    assert np.allclose(total[counts>0],ratio) and np.all(total[counts==0]==0)
    iq=(bank['los']+reflected).sum(axis=0)+bank['noise']
    return iq,dict(counts=counts,relative_coefficients=coefficients,total_reflection_power_ratio=total)


def solve_iq(iq, out, label):
    """label is used only for filenames; no physical condition accepted."""
    start=time.perf_counter()
    extracted,_,_=extract(iq)
    peaks=extracted['CC_ABS']
    lags=np.array([sorted(np.array(peaks[f'{m}_{nn}']['lag_s'][:6])*C) for m,nn in PAIRS])
    pool=triple_candidates(lags)
    pp,_,reward,_,_=geometry_fit(pool,lags)
    ids,_=joint_select(reward)
    initial=pp[ids]
    f0=np.fft.fftfreq(N,1/FS)
    use=abs(f0)<=B/2
    ft=np.fft.fft(iq,axis=1,norm='ortho')
    f,y=f0[use],ft[:,use].T
    sigma2=float(np.mean(abs(ft[:,~use])**2))
    gains=np.column_stack([rank_one(y,f,p)[0] for p in initial])
    x0=np.r_[initial.ravel()/UNIT,gains[1:].real.ravel(),gains[1:].imag.ravel()]
    frontend=time.perf_counter()-start
    x,los=los_fit(x0,f,y,N)
    paths,history=[],[]
    state=los
    checks={}
    stop='capacity'
    for step in range(1,65):
        pick,bank,meta=search(x,paths,f,y,N,use,sigma2,step)
        checks.update({f'{step}/'+k:v for k,v in meta['checks'].items()})
        np.savez_compressed(out/f'{label}_step{step}_candidates.npz',bank=bank)
        ev=dict(step=step,pick=pick,search=meta)
        history.append(ev)
        if pick is None:
            stop='no_detected_increment'
            break
        trial=np.r_[x,pick['rho'],pick['delay']]
        nextpaths=paths+[(pick['m'],pick['q'])]
        trial,new=fit(trial,f,y,nextpaths,N)
        ev['fit']=new
        ev['decrease']=new['cost']<=state['cost']+1e-10
        checks[f'{step}/finite']=bool(np.isfinite(trial).all())
        checks[f'{step}/closure']=new['closure_error']<1e-7
        checks[f'{step}/budget']=new['iterations']<=300
        if not ev['decrease']:
            stop='numerical_non_decrease'
            break
        x,paths,state=trial,nextpaths,new
        write(out/f'{label}_progress.json',dict(paths=paths,state=state,history=history))
        if not state['stationary']:
            stop='numerical_nonstationary'
            break
        if not all(checks.values()):
            stop='correctness_failure'
            break
    return dict(initial=initial,los=los,final=state,paths=paths,history=history,stop=stop,
                frontend_seconds=frontend,seconds=time.perf_counter()-start,checks=checks,
                delay_alias=sum(v>N/FS*B/2 for v in x[22::3]),
                power_boundary=sum(x[k:k+2]@x[k:k+2]>=1-1e-10 for k in range(20,len(x),3)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--unseen',action='store_true')
    args=parser.parse_args()
    admission=json.loads((HERE/'跨阶数放行.json').read_text(encoding='utf-8'))
    assert admission['approved_to_run'] and sha(HERE/'reference.py')==admission['reference_sha']
    assert sha(HERE/'reference_pg.py')==admission['pg_sha']
    if args.unseen:
        pilot=Path(json.loads((HERE/'cross_latest.json').read_text(encoding='utf-8'))['path'])
        d=json.loads((pilot/'result.json').read_text(encoding='utf-8'))
        assert d['all_checks_pass'] and all(r.get('stop')=='no_detected_increment' for r in d['rows'])
        assert json.loads((HERE/'未见阶数放行.json').read_text(encoding='utf-8'))['approved_to_run']
    geometries=[BASE/'independent_backend_20261029_222111_026721/L20_R200_input.npz',
                BASE/'consensus_validation_20261029_223520_495993/L40_R300_input.npz']
    files=[Path(__file__),HERE/'reference.py',HERE/'reference_pg.py',HERE/'投影收尾批准协议.md',HERE/'跨阶数条件协议.md',HERE/'跨阶数放行.json',
           BASE/'risk_step_20261030/pilot.py']+geometries
    files += [BASE/n for n in ['finite_iq_clues.py','geometry_association_gate.py','robust_geometry.py',
                               'bounded_consensus.py','blind_initialization.py','structural_probe.py']]
    for mod in tuple(sys.modules.values()):
        name=getattr(mod,'__file__',None)
        if name:
            p=Path(name).resolve()
            if p.is_relative_to(BASE) and p.suffix=='.py':
                files.append(p)
    hashes={str(p.resolve(strict=True)):sha(p) for p in files}
    assert all(Path(p).is_relative_to(BASE) for p in hashes)
    oldrun=Path(json.loads((HERE/'full_latest.json').read_text(encoding='utf-8'))['path'])
    known=json.loads((oldrun/'preflight.json').read_text(encoding='utf-8'))['hashes']
    assert all(h==known[p] for p,h in hashes.items() if p in known)
    out=HERE/(('unseen_' if args.unseen else 'cross_')+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    (out/'cross_order.py').write_bytes(Path(__file__).read_bytes())
    for name in hashes:
        source=Path(name)
        dest=out/'source'/source.relative_to(BASE)
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(source.read_bytes())
    write(out/'preflight.json',dict(hashes=hashes,seed_base=2026103023 if args.unseen else 2026103022,
                                  unknown_count_interface=True,unseen=args.unseen))
    begin=time.perf_counter()
    rows,checks=[],{}
    conditions=['L3'] if args.unseen else ['L0','L1','L2','L4','MIX','HIGH']
    print('输出',out,flush=True)
    for li,gpath in zip([20,40],geometries):
        with np.load(gpath,allow_pickle=False) as old:
            p=old['p'].copy()
        bank=generate_bank(p,li,args.unseen)
        np.savez_compressed(out/f'L{li}_generation.npz',**bank)
        checks[f'L{li}/gap']=bool(np.min(np.diff(bank['delays'],axis=2))>=3.5)
        for condition in conditions:
            label=f'L{li}_{condition}'
            iq,physical=observation(bank,condition)
            identity=hashlib.sha256(np.ascontiguousarray(iq).tobytes()).hexdigest()
            np.savez_compressed(out/f'{label}_input.npz',iq=iq,**physical)
            try:
                row=solve_iq(iq,out,label)
            except (ValueError,IndexError) as exc:
                row=dict(stop='frontend_or_numerical_exception',exception=repr(exc),checks={})
            checks.update({label+'/'+k:v for k,v in row['checks'].items()})
            checks[label+'/input_unchanged']=identity==hashlib.sha256(np.ascontiguousarray(iq).tobytes()).hexdigest()
            # Truth first used for evaluation after the complete blind estimator.
            if 'final' in row:
                row['geometry_errors_m']=error(row['initial'],p)
                row['los_errors_m']=error(row['los']['p'],p)
                row['errors_m']=error(row['final']['p'],p)
                for ev in row['history']:
                    if 'fit' in ev:
                        ev['errors_m']=error(ev['fit']['p'],p)
            row.update(tag=label,iq_sha=identity,condition=condition,layout=li)
            rows.append(row)
            write(out/f'{label}.json',row)
            print(label,row['stop'],'误差',row.get('errors_m'),'分量',len(row.get('paths',[])),flush=True)
    checks['identity_unchanged']=all(sha(p)==h for p,h in hashes.items())
    write(out/'result.json',dict(rows=rows,checks=checks,all_checks_pass=all(checks.values()),seconds=time.perf_counter()-begin))
    write(HERE/('unseen_latest.json' if args.unseen else 'cross_latest.json'),dict(path=str(out)))
    print('完成',len(rows),'正确性',all(checks.values()),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
