"""Explicit oracle block-initialization attribution and saved-state replay."""
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from joint_model import Model
from output_json import dumps

ROOT=Path(__file__).resolve().parent
JOINT=ROOT/'joint_20261025_204840_719110'
FRONT=ROOT/'frontend_20261025_204005_991746'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    begin=time.perf_counter()
    joint=json.loads((JOINT/'result.json').read_text(encoding='utf-8'))
    if not joint['all_checks_pass'] or not all(sha(Path(p))==h for p,h in joint['identity'].items()):
        raise RuntimeError('Frozen joint identity failed')
    front=json.loads((FRONT/'result.json').read_text(encoding='utf-8'))
    geometry=next(Path(p) for p in front['identity'] if 'finite_scene' in p and p.endswith('result.json'))
    data=json.loads(geometry.read_text(encoding='utf-8'))
    public=json.loads((FRONT/'public.json').read_text(encoding='utf-8'))
    with np.load(FRONT/'observations.npz') as d:
        y=d['mixture']
    out=ROOT/('attribution_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    state_paths=[JOINT/f"state_{r['pair_id']}_{r['seed']}.npz" for r in joint['runs'] if r['status']=='PROTOTYPE_BUDGET']
    files=[Path(__file__),ROOT/'joint_model.py',ROOT/'output_json.py',ROOT/'归因检查合同.md',JOINT/'result.json',geometry,FRONT/'observations.npz',FRONT/'public.json',*state_paths]
    identity={str(p.resolve(strict=True)):sha(p) for p in files}
    (out/'preflight.json').write_text(dumps({'sha256':identity,'threads':2},indent=2),encoding='utf-8')
    for p in files[:4]:
        (out/p.name).write_bytes(p.read_bytes())
    model=Model(y,public)
    x=list(np.array(data['positions_m'])[[9,12]].ravel()/1000);extents=[]
    for panel in data['panels']:
        n=np.array(panel['normal']);center=np.array(panel['center_m'])
        x.extend([np.arctan2(n[1],n[0]),center@n/1000])
        extents.append([center@np.cross([0.,0.,1.],n)/1000,panel['width_m']/1000,panel['height_m']/1000])
    x=np.array(x);extents=np.array(extents)
    paths,_=model.paths(x,extents)
    correct_gains=np.zeros((2,5,3),complex)
    for q,m,label,length,_ in paths:
        correct_gains[q,m,label+1]=(1 if label==-1 else -.5)/length
    results=[]
    for name in ['ORACLE_GEOMETRY','ORACLE_GEOMETRY_CHANNEL_INIT']:
        coeff=model.initial_coeff.copy()
        if name=='ORACLE_GEOMETRY':
            loss,gains,_,_=model.gain_fit(x,extents,coeff)
        else:
            gains=correct_gains.copy()
            coeff,rank=model.wave_fit(x,extents,gains)
            loss=float(np.linalg.norm(model.predict(x,extents,coeff,gains)-model.y)**2)
        trace=[{'iteration':0,'loss':loss,'relative_residual':float(np.sqrt(loss))}]
        for iteration in range(3):
            new,rank=model.wave_fit(x,extents,gains)
            value=float(np.linalg.norm(model.predict(x,extents,new,gains)-model.y)**2)
            if value<=loss+1e-10:
                coeff=new;loss=value
            fit=model.gain_fit(x,extents,coeff)
            if fit[0]<=loss+1e-10:
                loss,gains,_,_=fit
            trace.append({'iteration':iteration+1,'loss':loss,'relative_residual':float(np.sqrt(loss)),'wave_rank':rank})
        results.append({'mode':name,'trace':trace})
        print(dumps(results[-1]),flush=True)
    replay=[]
    for r in joint['runs']:
        if r['status']!='PROTOTYPE_BUDGET':
            continue
        xx=np.array(r['geometry_parameters']);ext=np.array(r['extents_normalized'])
        with np.load(JOINT/f"state_{r['pair_id']}_{r['seed']}.npz") as state:
            coeff,gains,saved=[state[k] for k in ['coefficients','gains','prediction']]
        paths,_=model.paths(xx,ext);pred=np.zeros_like(y)
        for q,m,label,_,_ in paths:
            p=xx[3*q:3*q+3]*1000
            if label>=0:
                theta,b=xx[6+2*label:8+2*label];n=np.array([np.cos(theta),np.sin(theta),0.])
                p=p-2*(p@n-b*1000)*n
            length=np.linalg.norm(p-model.stations[m])
            pred[m]+=gains[q,m,label+1]*(coeff[q]*np.exp(-2j*np.pi*model.freq*length/299792458.))@model.V*model.scale
        err=float(np.linalg.norm(pred-saved)/max(np.linalg.norm(saved),1e-30))
        replay.append({'pair':r['pair_id'],'seed':r['seed'],'relative_prediction_error':err})
    checks={'blind_prediction_replay':all(r['relative_prediction_error']<1e-10 for r in replay),
            'oracle_channel_positive_control':results[1]['trace'][-1]['relative_residual']<1e-6,
            'inputs_unchanged':all(sha(Path(p))==h for p,h in identity.items())}
    final={'identity':identity,'oracle_results':results,'blind_replay':replay,'checks':checks,
           'all_checks_pass':all(checks.values()),'seconds':time.perf_counter()-begin}
    (out/'result.json').write_text(dumps(final,indent=2),encoding='utf-8')
    print(dumps({'out':str(out),'checks':checks,'replay':replay,'seconds':final['seconds']}))
    if not final['all_checks_pass']:
        raise RuntimeError('Oracle or independent replay check failed')


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
