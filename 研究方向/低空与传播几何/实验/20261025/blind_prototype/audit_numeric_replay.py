"""Explain, do not erase, the failed strict image-length replay check."""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from joint_model import Model
from output_json import dumps

ROOT=Path(__file__).resolve().parent
JOINT=ROOT/'joint_20261025_204840_719110'
ATTR=ROOT/'attribution_20261025_205117_892831/result.json'
FRONT=ROOT/'frontend_20261025_204005_991746'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    prior=json.loads(ATTR.read_text(encoding='utf-8'))
    if not all(sha(Path(p))==h for p,h in prior['identity'].items()):
        raise RuntimeError('Attribution input identity failed')
    data=json.loads((JOINT/'result.json').read_text(encoding='utf-8'))
    public=json.loads((FRONT/'public.json').read_text(encoding='utf-8'))
    with np.load(FRONT/'observations.npz') as obs:
        y=obs['mixture']
    out=ROOT/('numeric_replay_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir(exist_ok=False)
    files=[Path(__file__),ROOT/'joint_model.py',ROOT/'output_json.py',ATTR,JOINT/'result.json',FRONT/'public.json',FRONT/'observations.npz']
    files += [JOINT/f"state_{r['pair_id']}_{r['seed']}.npz" for r in data['runs']]
    identity={str(p.resolve(strict=True)):sha(p) for p in files}
    (out/'preflight.json').write_text(dumps({'sha256':identity},indent=2),encoding='utf-8')
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    model=Model(y,public);results=[]
    for r in data['runs']:
        x=np.array(r['geometry_parameters']);ext=np.array(r['extents_normalized'])
        with np.load(JOINT/f"state_{r['pair_id']}_{r['seed']}.npz") as state:
            coeff,gains,saved=[state[k] for k in ['coefficients','gains','prediction']]
        paths,_=model.paths(x,ext)
        same=np.zeros_like(y);other=np.zeros_like(y);bounds=np.zeros(5);deltas=[]
        for q,m,label,length,_ in paths:
            p=x[3*q:3*q+3]*1000
            if label>=0:
                theta,b=x[6+2*label:8+2*label];n=np.array([np.cos(theta),np.sin(theta),0.])
                p=p-2*(p@n-b*1000)*n
            mirror_length=np.linalg.norm(p-model.stations[m]);delta=mirror_length-length;deltas.append(abs(delta))
            g=gains[q,m,label+1]
            same[m]+=g*(coeff[q]*np.exp(-2j*np.pi*model.freq*length/299792458.))@model.V*model.scale
            other[m]+=g*(coeff[q]*np.exp(-2j*np.pi*model.freq*mirror_length/299792458.))@model.V*model.scale
            bounds[m]+=abs(g)*np.sum(abs(coeff[q])*abs(2*np.pi*model.freq/299792458.*delta))*model.scale
        denominator=np.linalg.norm(saved)
        bound=float(np.sqrt(y.shape[1])*np.linalg.norm(bounds)/denominator)
        same_error=float(np.linalg.norm(same-saved)/denominator)
        other_error=float(np.linalg.norm(other-saved)/denominator)
        results.append({'pair':r['pair_id'],'seed':r['seed'],'max_length_delta_m':max(deltas),
                        'same_length_replay_relative_error':same_error,'image_length_replay_relative_error':other_error,
                        'first_order_phase_perturbation_upper_bound':bound,
                        'coefficient_norm':float(np.linalg.norm(coeff)),
                        'scale_invariant_coefficient_gain_products':[float(np.linalg.norm(coeff[q])*np.linalg.norm(gains[q])) for q in range(2)]})
    checks={'same_length_replay_1e10':all(r['same_length_replay_relative_error']<1e-10 for r in results),
            'image_difference_within_length_bound':all(r['image_length_replay_relative_error']<=r['first_order_phase_perturbation_upper_bound']+2*r['same_length_replay_relative_error']+1e-12 for r in results),
            'inputs_unchanged':all(sha(Path(p))==h for p,h in identity.items())}
    final={'identity':identity,'results':results,'checks':checks,'all_checks_pass':all(checks.values()),
           'original_strict_image_replay_status':'FAILED_RETAINED_NOT_RELAXED',
           'scope':'Roundoff attribution only; no changed estimator, output, stopping rule or threshold.'}
    (out/'result.json').write_text(dumps(final,indent=2),encoding='utf-8');print(dumps({'out':str(out),**final}))
    if not final['all_checks_pass']:
        raise RuntimeError('Roundoff attribution did not explain mismatch')


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
