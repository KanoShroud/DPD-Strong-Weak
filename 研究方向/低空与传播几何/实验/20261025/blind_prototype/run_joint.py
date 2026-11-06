"""Bounded four-start shared-environment prototype; evaluation is post hoc."""
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

from joint_model import MODULE, Model, initial_pairs
from output_json import dumps

ROOT=Path(__file__).resolve().parent
FRONT=ROOT/'frontend_20261025_204005_991746'
CAPACITY=ROOT/'representation_20261025_204157_466399/result.json'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    start=time.perf_counter()
    prior=json.loads(CAPACITY.read_text(encoding='utf-8'))
    if not prior['all_checks_pass'] or not prior['capacity_gate_pass'] or not all(sha(Path(p))==h for p,h in prior['identity'].items()):
        raise RuntimeError('Capacity identity failed')
    front=json.loads((FRONT/'result.json').read_text(encoding='utf-8'))
    if not front['all_checks_pass'] or not all(sha(FRONT/p)==h for p,h in front['public_input_sha256'].items()):
        raise RuntimeError('Public observations failed')
    public=json.loads((FRONT/'public.json').read_text(encoding='utf-8'))
    blind=json.loads((FRONT/'blind_output.json').read_text(encoding='utf-8'))
    with np.load(FRONT/'observations.npz') as d:
        y=d['mixture']
    out=ROOT/('joint_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    files=[Path(__file__),ROOT/'joint_model.py',ROOT/'output_json.py',ROOT/'联合拟合短测合同.md',MODULE,CAPACITY,
           FRONT/'result.json',FRONT/'public.json',FRONT/'blind_output.json',FRONT/'observations.npz']
    identity={str(p.resolve(strict=True)):sha(p) for p in files}
    (out/'preflight.json').write_text(dumps({'sha256':identity,'numpy':np.__version__,'threads':2},indent=2),encoding='utf-8')
    for p in files[:5]:
        (out/p.name).write_bytes(p.read_bytes())
    pairs=initial_pairs(blind,public)
    (out/'initial_pairs.json').write_text(dumps(pairs,indent=2),encoding='utf-8')
    model=Model(y,public)
    runs=[];stop=False
    for pair_id,pair in enumerate(pairs):
        for seed in [811,812]:
            result=model.run(pair['positions_m'],seed)
            if 'coefficients' in result:
                np.savez_compressed(out/f'state_{pair_id}_{seed}.npz',**{k:result.pop(k) for k in ['coefficients','gains','prediction']})
            result['pair_id']=pair_id
            (out/f'run_{pair_id}_{seed}.json').write_text(dumps(result,indent=2),encoding='utf-8')
            runs.append(result)
            print(dumps({k:v for k,v in result.items() if k not in ['initial','trace','geometry_parameters','extents_normalized']}),flush=True)
            if result.get('seconds',0)>600 or result['status']=='GRADIENT_CHECK_FAILED':
                stop=True;break
        if stop:
            break
    admissible=[(i,r) for i,r in enumerate(runs) if r['status']=='PROTOTYPE_BUDGET' and r['monotone']]
    selected=min(admissible,key=lambda item:item[1]['loss'])[0] if admissible else None
    # No truth has been loaded by the estimator; all runs saved before this evaluation.
    geometry=next(Path(p) for p in prior['identity'] if 'finite_scene' in p and p.endswith('result.json'))
    data=json.loads(geometry.read_text(encoding='utf-8'));truth=np.array(data['positions_m'])[[9,12]]
    evaluation=[]
    for i,r in admissible:
        positions=np.array(r['positions_m']);distance=np.linalg.norm(positions[:,None,:]-truth[None,:,:],axis=2)
        a,b=linear_sum_assignment(distance**2)
        errors=np.empty(2);errors[b]=distance[a,b]
        evaluation.append({'run_index':i,'errors_by_true_source_m':errors.tolist(),'RMSE_m':float(np.sqrt(np.mean(errors**2)))})
    checks={'input_unchanged':all(sha(Path(p))==h for p,h in identity.items()),
            'finite_gradients':all(r.get('gradient_error',0)<1e-5 for r in runs),
            'accepted_steps_monotone':all(r.get('monotone',True) for r in runs),
            'loss_replay':all(r.get('loss_replay_error',0)<1e-9 for r in runs)}
    summary={'identity':identity,'scope':'BUDGETED_BLOCK_LS_PROTOTYPE_NOT_CONVERGED_BASELINE',
             'runs':runs,'selected_by_data_loss':selected,'posthoc_evaluation':evaluation,
             'checks':checks,'all_checks_pass':all(checks.values()),'stopped_early':stop,'seconds':time.perf_counter()-start}
    (out/'result.json').write_text(dumps(summary,indent=2),encoding='utf-8')
    print(dumps({'out':str(out),'selected':selected,'evaluation':evaluation,'checks':checks,'seconds':summary['seconds']}),flush=True)
    if not summary['all_checks_pass']:
        raise RuntimeError('Prototype numerical check failed')


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
