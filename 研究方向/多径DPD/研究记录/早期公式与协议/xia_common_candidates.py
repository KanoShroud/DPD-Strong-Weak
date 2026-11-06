"""Xia score/decoder with a public common candidate pool, explicitly adapted."""
from datetime import datetime
import argparse
import json
import sys
import time

import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from geometry_association_gate import ROOT, STATIONS, B, sha, write, signature
from robust_geometry import evaluate

XIA=ROOT.parent/'xia_pairing'
sys.path.insert(0,str(XIA/'recheck'))
import xia_checked as xia  # noqa: E402


def solve(pool,y,f,seed):
    before=signature(pool,y,f)
    start=time.perf_counter();rng=np.random.default_rng(seed)
    points=pool.copy();steps=[];retained=None
    for iteration in range(3):
        score=xia.scores(points,STATIONS,y,f,8192)
        decoded=xia.decode(points,score,2,400.)
        steps.append(dict(iteration=iteration,points=points,scores=score,decoded=decoded))
        if decoded['positions'] is None:
            break
        retained=np.array(decoded['positions'])
        if iteration<2:
            points=retained[np.arange(1000)%2]+rng.normal(size=(1000,2))*10.
    assert signature(pool,y,f)==before
    return dict(steps=steps,terminal=decoded['positions'],retained=retained,seconds=time.perf_counter()-start)


def ev(p,truth):
    return evaluate(np.array(p).reshape(-1,2) if p is not None else np.empty((0,2)),truth)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['development','validation'],required=True)
    args=parser.parse_args()
    if args.phase=='development':
        source=ROOT/'consensus_development_20261029_223235_334946'
        inputs=ROOT/'independent_backend_20261029_222111_026721'
    else:
        source=inputs=ROOT/'consensus_validation_20261029_223520_495993'
    files=[ROOT/'xia_common_candidates.py',ROOT/'Xia公共候选对照协议.md',XIA/'recheck/xia_checked.py',
       XIA/'xia_reference.py',source/'result.json',source/'protocol.json']
    files+=sorted(source.glob('*_geometry.npz'))+sorted(inputs.glob('*_input.npz'))
    hashes={str(p.resolve(strict=True)):sha(p) for p in files}
    out=ROOT/('xia_common_'+args.phase+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False);(out/'source').mkdir()
    for p in files[:4]:
        (out/'source'/p.name).write_bytes(p.read_bytes())
    write(out/'protocol.json',dict(hashes=hashes,phase=args.phase,base_seed=20261029223,
       initial='same raw geometric candidates, no truth',resampling_particles=1000,std_m=10.,delta_h_m=400.,rounds=3))
    f=np.fft.fftfreq(8192,1/50e6);mask=abs(f)<=B/2
    records=json.loads((source/'result.json').read_text(encoding='utf-8'))['rows']
    rows=[];checks={};begin=time.perf_counter()
    print('OUTPUT',out,flush=True)
    for row in records:
        tag=row['tag'];base=tag.rsplit('_',1)[0]
        data=dict(np.load(inputs/(base+'_input.npz')))
        iq=(data['los']+(data['reflected'] if row['path']=='MP' else 0)).sum(axis=0)+data['noise']
        pool=np.load(source/(tag+'_geometry.npz'))['initial_candidates']
        y=np.fft.fft(iq,axis=1,norm='ortho')[:,mask]
        checks[tag+'/IQ']=signature(iq)==row['iq_sha']
        record=solve(pool,y,f[mask],20261029223+10000*row['layout']+row['rep'])
        record.update(tag=tag,path=row['path'],layout=row['layout'],rep=row['rep'],iq_sha=signature(iq),
           terminal_evaluation=ev(record['terminal'],data['p']),retained_evaluation=ev(record['retained'],data['p']),
           initial_coverage_m=cdist(pool,data['p']).min(axis=0))
        checks[tag+'/finite']=all(np.isfinite(s['scores']).all() for s in record['steps'])
        if not rows:
            from xia_reference import scalar_score
            loop=scalar_score(pool[0],STATIONS,y[:,:31],f[mask][:31],8192)
            batch=xia.scores(pool[:1],STATIONS,y[:,:31],f[mask][:31],8192)[0]
            checks['score_scalar']=bool(np.isclose(loop,batch,rtol=1e-10,atol=0))
        write(out/(tag+'.json'),record);rows.append(record)
        assert all(checks.values())
        print(len(rows),tag,record['steps'][-1]['decoded']['status'],'retained',record['retained_evaluation']['errors_m'],flush=True)
    checks.update({p:sha(p)==h for p,h in hashes.items()})
    assert all(checks.values())
    write(out/'result.json',dict(rows=rows,checks=checks,seconds=time.perf_counter()-begin))
    print('COMPLETE',out,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
