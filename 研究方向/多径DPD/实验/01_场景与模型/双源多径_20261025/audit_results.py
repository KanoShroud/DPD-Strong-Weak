"""Read-only original-coordinate replay and paired summary of saved estimates."""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from experiment import ROOT, preprocess, write, sha
from model import Model, evaluate


def main():
    parser=argparse.ArgumentParser();parser.add_argument('folders',nargs='+');args=parser.parse_args()
    out=ROOT/('audit_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir(exist_ok=False)
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    rows=[];source_checks=[]
    for folder in args.folders:
        base=Path(folder).resolve(strict=True)
        result=json.loads((base/'result.json').read_text(encoding='utf-8'))
        source_checks.append(all(sha(Path(p))==h for p,h in result['identity'].items()))
        for file in sorted(base.glob('[01]_*.json')):
            r=json.loads(file.read_text(encoding='utf-8'))
            label=r['condition'];ids=[0,1] if label.startswith('dual') else ([0] if label.startswith('A') else [1])
            infile=base/f'input_{r["layout"]}_{r["seed"]}.npz'
            with np.load(infile) as data:
                if label.endswith('power_only'):
                    components=data['power_only']
                elif label.endswith('_mp') or label.endswith('_ground'):
                    components=data['los']+data['echo']
                else:
                    components=data['los']
                iq=components[ids].sum(axis=0)+data['noise'];truth=data['truth'][ids]
                power_mismatch=None
                if r['stage']=='ground':
                    real_power=[];control_power=[]
                    for q in range(2):
                        yy,_=preprocess(data['los'][q]+data['echo'][q],True)
                        zz,_=preprocess(data['power_only'][q],True)
                        real_power.append(np.sum(abs(yy)**2,axis=-1));control_power.append(np.sum(abs(zz)**2,axis=-1))
                    power_mismatch=float(np.max(abs(10*np.log10(np.array(real_power)/np.array(control_power)))))
            y,f=preprocess(iq,r['stage'] in ['nonperiodic','ground'])
            model=Model(y,f);selected=r['output']['selected'];x=np.array(selected['x'])
            a,_,_,s,res=model.components(x,len(ids))
            coef=np.stack([np.linalg.lstsq(aa,yy,rcond=None)[0] for aa,yy in zip(a,model.y)])
            independent=model.y-np.einsum('fmk,fk->fm',a,coef)
            replay=float(np.linalg.norm(res-independent)/np.linalg.norm(model.y))
            value=float(np.vdot(independent,independent).real/model.energy)
            ev=evaluate(selected['positions'],truth)
            rows.append({'file':str(file),'file_sha256':sha(file),'input_sha_ok':sha(infile)==r['input_sha256'],
                'stage':r['stage'],'layout':r['layout'],'seed':r['seed'],'condition':label,
                'residual_replay_relative':replay,'loss_replay_difference':abs(value-selected['loss']),
                'evaluation_difference':abs(ev['rmse_m']-r['evaluation']['rmse_m']),
                'finite':bool(np.isfinite(x).all() and np.isfinite(value)),
                'postprocessing_power_control_mismatch_max_db':power_mismatch,
                **ev,'seconds':r['output']['seconds'],'gradient_inf':selected['gradient_inf'],
                'stop':selected['message'],'evaluations':selected['iterations'],
                'budget_exhausted':'maximum number' in selected['message'].lower(),
                'position_boundary':bool(np.any(np.isclose(np.array(selected['positions'])[:,2],20,atol=.01)) or np.any(np.isclose(np.array(selected['positions'])[:,2],150,atol=.01)))})
    summaries=[]
    for stage in sorted(set(r['stage'] for r in rows)):
        for condition in sorted(set(r['condition'] for r in rows if r['stage']==stage)):
            selected=[r for r in rows if r['stage']==stage and r['condition']==condition]
            e=np.array([r['errors_m'] for r in selected]);v=np.array([r['vertical_m'] for r in selected]);h=np.array([r['horizontal_m'] for r in selected])
            summaries.append({'stage':stage,'condition':condition,'count':len(selected),'targets_per_case':e.shape[1],
                'rmse_m':float(np.sqrt(np.mean(e**2))),'vertical_rmse_m':float(np.sqrt(np.mean(v**2))),
                'horizontal_rmse_m':float(np.sqrt(np.mean(h**2))),
                'all_success':sum(r['all_within_20m'] for r in selected),
                'mean_seconds':float(np.mean([r['seconds'] for r in selected])),
                'budget_exhausted':sum(r['budget_exhausted'] for r in selected)})
    checks={'sources_unchanged':all(source_checks),'inputs_unchanged':all(r['input_sha_ok'] for r in rows),
            'all_finite':all(r['finite'] for r in rows),'independent_ls_replay':all(r['residual_replay_relative']<1e-10 and r['loss_replay_difference']<1e-10 for r in rows),
            'evaluation_replay':all(r['evaluation_difference']<1e-10 for r in rows)}
    write(out/'result.json',{'checks':checks,'all_checks_pass':all(checks.values()),'rows':rows,'summary':summaries})
    print(json.dumps({'out':str(out),'checks':checks,'summary':summaries},ensure_ascii=False),flush=True)
    if not all(checks.values()):
        raise RuntimeError('Audit failed; preserved original status')


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
