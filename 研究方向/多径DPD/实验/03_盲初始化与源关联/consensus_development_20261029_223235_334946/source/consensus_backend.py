"""Frozen conventional robust start with paired blind LOS/multipath backends."""
from datetime import datetime
import argparse
import json
import time
import warnings

import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from geometry_association_gate import ROOT, STATIONS, PAIRS, C, B, sha, write, signature
from finite_iq_clues import generate, extract
from robust_geometry import triple_candidates, refine, select, evaluate
from bounded_consensus import fit, joint_select, selftest
from blind_backend_pilot import solve, errors
from independent_backend import case

SOURCE=ROOT/'independent_backend_20261029_222111_026721'


def new_layouts():
    rng=np.random.default_rng(20261029222)
    found=[]
    while len(found)<4:
        p=rng.uniform(-2600,2600,(2,2))
        if cdist(p,STATIONS).min()>=500 and np.linalg.norm(p[0]-p[1])>=600:
            found.append(p)
    return found


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['development','validation'],required=True)
    args=parser.parse_args()
    files=[ROOT/n for n in ['consensus_backend.py','bounded_consensus.py','有界起点与多径后端配对协议.md',
      '有界共识诊断协议.md','blind_backend_pilot.py','independent_backend.py','finite_iq_clues.py',
      'robust_geometry.py','geometry_association_gate.py','blind_initialization.py','boundary_update.py',
      'structural_probe.py','position_reduction.py','solver_reliability.py']]
    if args.phase=='development':
        files+=[SOURCE/'result.json',SOURCE/'protocol.json']+sorted(SOURCE.glob('*_input.npz'))
    hashes={str(p.resolve(strict=True)):sha(p) for p in files}
    prior=json.loads((SOURCE/'protocol.json').read_text(encoding='utf-8'))['hashes']
    assert all(h==prior[p] for p,h in hashes.items() if p in prior)
    out=ROOT/('consensus_'+args.phase+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    (out/'source').mkdir()
    for p in files[:14]:
        (out/'source'/p.name).write_bytes(p.read_bytes())
    layouts=new_layouts()
    write(out/'protocol.json',dict(hashes=hashes,phase=args.phase,layouts=layouts if args.phase=='validation' else None,
       random_geometry_seed=20261029222,planned=24,truth_in_solver=False))
    source=json.loads((SOURCE/'result.json').read_text(encoding='utf-8'))
    old={r['tag']:r for r in source['rows']}
    checks=selftest()
    rows=[]
    start=time.perf_counter()
    f=np.fft.fftfreq(8192,1/50e6)
    mask=abs(f)<=B/2
    print('OUTPUT',out,flush=True)
    layout_ids=range(20,24) if args.phase=='development' else range(40,44)
    reps=range(200,203) if args.phase=='development' else range(300,303)
    for li in layout_ids:
        for rep in reps:
            base=f'L{li}_R{rep}'
            if args.phase=='development':
                data=dict(np.load(SOURCE/(base+'_input.npz')))
            else:
                physical=case(layouts[li-40],li,rep)
                data=generate(physical,li,rep)
                data.update(physical)
                np.savez_compressed(out/(base+'_input.npz'),**data)
            for path in ['LOS','MP']:
                tag=base+'_'+path
                iq=(data['los']+(data['reflected'] if path=='MP' else 0)).sum(axis=0)+data['noise']
                begin=time.perf_counter()
                if args.phase=='development':
                    peaks=old[tag]['peaks']
                    extraction_seconds=old[tag]['extraction_seconds']
                    checks[tag+'/iq_identity']=signature(iq)==old[tag]['iq_sha']
                else:
                    extracted,_,extraction_seconds=extract(iq)
                    peaks=extracted['CC_ABS']
                lags=np.array([sorted(np.array(peaks[f'{m}_{n}']['lag_s'][:6])*C) for m,n in PAIRS])
                pool=triple_candidates(lags)
                pp,cost,reward,status,nit=fit(pool,lags)
                ids,value=joint_select(reward)
                initial=pp[ids]
                frontend_seconds=time.perf_counter()-begin
                if args.phase=='development':
                    frontend_seconds+=extraction_seconds
                # Additional diagnostic estimators are not included in candidate algorithm time.
                front={}
                for keep in [10,8]:
                    pf,cf,_,_=refine(pool,lags,keep)
                    front[str(keep)]=evaluate(pf[select(pf,cf,2)],data['p'])
                front['bounded_nms']=evaluate(pp[select(pp,cost,2)],data['p'])
                y=np.fft.fft(iq,axis=1,norm='ortho')[:,mask].T
                record=solve(initial,f[mask],y)
                for name,pos in [('geometry',initial),('los',record['los_p']),('mp',record['mp']['p'])]:
                    record[name+'_errors_m']=errors(pos,data['p'])
                record.update(tag=tag,path=path,layout=li,rep=rep,iq_sha=signature(iq),peaks=peaks,
                    front_diagnostics=front,frontend_seconds=frontend_seconds,
                    whole_seconds=frontend_seconds+record['seconds'],joint_reward=value,oracle=False)
                checks[tag+'/feasible']=bool(record['mp']['feasible'])
                checks[tag+'/descent']=record['mp']['cost']<=record['seed_cost']+1e-10<=record['los']['cost']+2e-10
                rows.append(record)
                write(out/(tag+'.json'),record)
                np.savez_compressed(out/(tag+'_geometry.npz'),initial_candidates=pool,p=pp,cost=cost,
                    reward=reward,chosen=ids,status=status,iterations=nit)
                assert all(checks.values()),checks
                print(len(rows),tag,'GEO',np.round(record['geometry_errors_m'],2),
                    'LOS',np.round(record['los_errors_m'],2),'MP',np.round(record['mp_errors_m'],2),flush=True)
    checks.update({p:sha(p)==h for p,h in hashes.items()})
    assert all(checks.values())
    write(out/'result.json',dict(rows=rows,checks=checks,seconds=time.perf_counter()-start,phase=args.phase))
    print('COMPLETE',out,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2),warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='divide by zero encountered in log10',category=RuntimeWarning)
        main()
