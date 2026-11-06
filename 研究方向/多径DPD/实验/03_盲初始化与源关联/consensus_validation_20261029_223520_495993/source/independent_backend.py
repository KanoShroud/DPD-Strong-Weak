"""Frozen geometry/LOS/MP pipeline on predeclared fresh geometries and seeds."""
from datetime import datetime
import json
import time
import warnings

import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from geometry_association_gate import ROOT, STATIONS, PAIRS, C, B, sha, write, signature
from finite_iq_clues import generate, extract
from robust_geometry import triple_candidates, refine, select
from blind_backend_pilot import solve, errors

PROTOCOL=ROOT/'独立盲后端复核协议.md'


def layouts():
    rng=np.random.default_rng(20261029220)
    found=[]
    while len(found)<4:
        p=rng.uniform(-2600,2600,(2,2))
        if cdist(p,STATIONS).min()>=500 and np.linalg.norm(p[0]-p[1])>=600:
            found.append(p)
    return found


def case(p,li,rep):
    rng=np.random.default_rng(20261029221+10000*li+rep)
    d=np.zeros((5,2,3))
    d[:,:,1]=rng.uniform(3,5,(5,2))
    d[:,:,2]=rng.uniform(8,10,(5,2))
    distances=np.linalg.norm(STATIONS[:,None]-p[None],axis=2)
    g=np.empty((5,2,3),complex)
    g[:,:,0]=1000/distances*np.exp(1j*rng.uniform(-np.pi,np.pi,(5,2)))
    split=rng.uniform(.3,.7,(5,2))
    g[:,:,1]=abs(g[:,:,0])*np.sqrt(10**(-.6)*split)*np.exp(1j*rng.uniform(-np.pi,np.pi,(5,2)))
    g[:,:,2]=abs(g[:,:,0])*np.sqrt(10**(-.6)*(1-split))*np.exp(1j*rng.uniform(-np.pi,np.pi,(5,2)))
    return dict(p=p,g=g,d=d)


def main():
    names=['independent_backend.py','blind_backend_pilot.py','finite_iq_clues.py','robust_geometry.py',
           'geometry_association_gate.py','blind_initialization.py','boundary_update.py',
           'structural_probe.py','position_reduction.py','solver_reliability.py']
    files=[ROOT/n for n in names]+[PROTOCOL]
    hashes={str(p.resolve(strict=True)):sha(p) for p in files}
    frozen=json.loads((ROOT/'blind_backend_20261029_221536_252934/protocol.json').read_text(encoding='utf-8'))['hashes']
    for p,h in hashes.items():
        if p in frozen:
            assert h==frozen[p]
    points=layouts()
    out=ROOT/('independent_backend_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    (out/'source').mkdir()
    for file in files:
        (out/'source'/file.name).write_bytes(file.read_bytes())
    write(out/'protocol.json',dict(hashes=hashes,layouts=points,layout_ids=list(range(20,24)),
          repetitions=[200,201,202],layout_seed=20261029220,channel_seed=20261029221,oracle=False,
          unchanged_backend=True,planned=24))
    checks={}
    rows=[]
    begin=time.perf_counter()
    f=np.fft.fftfreq(8192,1/50e6)
    mask=abs(f)<=B/2
    print('OUTPUT',out,flush=True)
    for li,p in enumerate(points,20):
        for rep in [200,201,202]:
            physical=case(p,li,rep)
            data=generate(physical,li,rep)
            base=f'L{li}_R{rep}'
            np.savez_compressed(out/(base+'_input.npz'),**data,**physical)
            checks[base+'/snr']=bool(abs(data['nominal_average_snr_db'][1]-6)<1e-10)
            ratio=np.sum(abs(physical['g'][:,:,1:])**2,axis=2)/abs(physical['g'][:,:,0])**2
            checks[base+'/power']=bool(np.allclose(ratio,10**(-.6)))
            for path in ['LOS','MP']:
                tag=base+'_'+path
                iq=(data['los']+(data['reflected'] if path=='MP' else 0)).sum(axis=0)+data['noise']
                t=time.perf_counter()
                observed,_,extraction_seconds=extract(iq)
                lags=np.array([sorted(np.array(observed['CC_ABS'][f'{m}_{n}']['lag_s'][:6])*C) for m,n in PAIRS])
                pool=triple_candidates(lags)
                fitted,scores,reasons,iterations=refine(pool,lags,10)
                ids=select(fitted,scores,2)
                initial=fitted[ids]
                frontend_seconds=time.perf_counter()-t
                if len(initial)!=2:
                    raise RuntimeError('Incomplete geometry output: preserve this case, do not substitute truth')
                y=np.fft.fft(iq,axis=1,norm='ortho')[:,mask].T
                result=solve(initial,f[mask],y)
                for name,positions in [('geometry',initial),('los',result['los_p']),('mp',result['mp']['p'])]:
                    result[name+'_errors_m']=errors(positions,p)
                result.update(tag=tag,layout=li,rep=rep,path=path,iq_sha=signature(iq),oracle=False,
                    extraction_seconds=extraction_seconds,frontend_seconds=frontend_seconds,
                    whole_seconds=frontend_seconds+result['seconds'],peaks=observed['CC_ABS'])
                checks[tag+'/seed_descent']=result['seed_cost']<=result['los']['cost']+1e-10
                checks[tag+'/mp_descent']=result['mp']['cost']<=result['seed_cost']+1e-10
                checks[tag+'/feasible']=bool(result['mp']['feasible'])
                rows.append(result)
                write(out/(tag+'.json'),result)
                np.savez_compressed(out/(tag+'_geometry.npz'),initial_candidates=pool,p=fitted,
                         scores=scores,chosen=ids,reason=reasons,iterations=iterations)
                assert all(checks.values()),checks
                print(len(rows),tag,'GEO',np.round(result['geometry_errors_m'],2),
                   'LOS',np.round(result['los_errors_m'],2),'MP',np.round(result['mp_errors_m'],2),flush=True)
    checks.update({p:sha(p)==h for p,h in hashes.items()})
    assert all(checks.values())
    write(out/'result.json',dict(rows=rows,checks=checks,seconds=time.perf_counter()-begin,oracle=False))
    print('COMPLETE',out,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2),warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='divide by zero encountered in log10',category=RuntimeWarning)
        main()
