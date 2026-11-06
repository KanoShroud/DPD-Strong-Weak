"""Explicit algorithm modification: preserve two resampling ancestries.

Original score/initial decode unchanged. Later rounds select the highest
original score separately within each resampling group; no global threshold,
reclustering, incumbent injection, exclusion radius, or truth in solver.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import time
from pathlib import Path
import csv

import numpy as np
from threadpoolctl import threadpool_limits
from pilot import ROOT, N, sha, write, observe, evaluate, select_output, xia
from square_check import first_metrics
from trace_audit import read, stats

SQUARE=ROOT/'square_20261027_182013_211372'
ORIGINAL=ROOT/'pilot_20261027_170421_805496'


def group_select(points,values,owner):
    assert np.isfinite(values).all() and np.isfinite(points).all()
    selected=[]
    for group in [0,1]:
        indices=np.flatnonzero(owner==group)
        assert len(indices)>0
        selected.append(int(indices[np.argmax(values[indices])]))
    return dict(status='GROUPWISE_COMPLETE',positions=points[selected].tolist(),
                selected_indices=selected,group_sizes=[int(np.sum(owner==g)) for g in [0,1]])


def repaired_solve(y,frequency,stations,seed):
    started=time.perf_counter();rng=np.random.default_rng(seed)
    points=rng.uniform(-3000,3000,(1000,2));owner=np.arange(1000)%2;steps=[]
    for j in range(3):
        values=xia.scores(points,stations,y,frequency,N)
        output=xia.decode(points,values,2,400.) if j==0 else group_select(points,values,owner)
        steps.append(dict(round=j+1,points=points.tolist(),scores=values.tolist(),output=output))
        if output['positions'] is None: break
        if j<2:
            centers=np.array(output['positions'])
            points=centers[owner]+rng.normal(size=(1000,2))*10.
    return dict(version='XIA_FLOW_REPAIR_GROUPWISE_V1',positions=output['positions'],
                status=output['status'],steps=steps,seconds=time.perf_counter()-started,
                score_positions=1000*len(steps))


def tests():
    p=np.array([[0.,0.],[10.,0.],[0.,1.],[10.,1.]])
    v=np.array([2.,1.,3.,1.5]);owner=np.array([0,1,0,1])
    result=group_select(p,v,owner)
    swapped=group_select(p,v,1-owner)
    return dict(select_original_score_per_group=result['selected_indices']==[2,3],
                preserve_group_under_power_difference=np.allclose(group_select(p,np.array([100.,.01,200.,.02]),owner)['positions'],p[[2,3]]),
                swap_group_not_truth=swapped['positions']==result['positions'][::-1],
                deterministic_tie=group_select(p,np.ones(4),owner)['selected_indices']==[0,1],
                no_position_merge_rule=group_select(np.zeros_like(p),v,owner)['positions']==[[0.,0.],[0.,0.]])


def main(limit):
    checks=tests();assert all(checks.values())
    frozen=read(ORIGINAL/'preflight.json')
    for p,h in frozen['sources'].items(): assert sha(Path(p))==h
    files=list(SQUARE.glob('r*_G*.json'))+list(SQUARE.glob('input_*.npz'))+[SQUARE/'protocol.json']
    identities={p.name:sha(p) for p in files}
    out=ROOT/('repair_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir(exist_ok=False)
    protocol=dict(input=str(SQUARE.resolve(strict=True)),input_hashes=identities,solver_hashes=frozen['sources'],
        repair_source_sha=sha(__file__),scope='EXPLICIT_FLOW_MODIFICATION_NOT_PAPER_REPRODUCTION',
        original_score_unchanged=True,initial_global_decode_unchanged=True,
        later_update='argmax original score independently within two ancestry groups; no histogram or radial re-clustering',
        particles=1000,per_group=500,std_m=10,rounds=3,no_elitism=True,no_collision_guard=True,
        no_truth_solver=True,limit=limit,threads=2,
        compare='terminal availability separately; paired errors on common last-valid-complete subset; fixed A first output on all available',
        timing='two versions in alternating order, same process/threadpool; increased work counted, no promised speedup')
    write(out/'protocol.json',protocol);(out/'flow_repair.py').write_bytes(Path(__file__).read_bytes())
    records=sorted(SQUARE.glob('r*_G*.json'))
    if limit is not None: records=records[:limit]
    started=time.perf_counter();rows=[]
    for index,file in enumerate(records):
        saved=read(file);i=saved['realization'];old=read(ORIGINAL/file.name)
        assert sha(ORIGINAL/file.name)==saved['original_record_sha']
        z=dict(np.load(SQUARE/f'input_{i:02d}.npz'))
        y,_,_=observe(z,saved['snr_db'],saved['sir_db'],saved['paths'])
        checks[file.stem+'/input']=sha(SQUARE/f'input_{i:02d}.npz')==saved['input_sha']
        checks[file.stem+'/observation']=hashlib.sha256(y.tobytes()).hexdigest()==saved['observation_sha']
        seed=old['particle_seed']
        outputs={}
        order=['original','repair'] if index%2==0 else ['repair','original']
        for name in order:
            if name=='original': outputs[name]=xia.solve(y,z['frequency'],z['stations'],2,N,3000,seed=seed)
            else: outputs[name]=repaired_solve(y,z['frequency'],z['stations'],seed)
        ref,new=outputs['original'],outputs['repair']
        checks[file.stem+'/reference_replay']=ref['steps']==saved['output']['steps']
        checks[file.stem+'/identical_initial_step']=ref['steps'][0]==new['steps'][0]
        if len(ref['steps'])>1:
            checks[file.stem+'/identical_round2_points']=ref['steps'][1]['points']==new['steps'][1]['points']
            checks[file.stem+'/identical_round2_scores']=ref['steps'][1]['scores']==new['steps'][1]['scores']
        if ref['steps'][0]['output']['positions'] is None:
            checks[file.stem+'/initial_failure_unchanged']=ref['positions']==new['positions'] and len(new['steps'])==1
        for step in new['steps'][1:]:
            pts=np.asarray(step['points']);v=np.asarray(step['scores'])
            expected=np.array([pts[g::2][v[g::2].argmax()] for g in [0,1]])
            checks[file.stem+f'/group_independent{step["round"]}']=np.array_equal(expected,np.array(step['output']['positions']))
        selected,round_id,_=select_output(ref)
        row=dict(case=file.stem,realization=i,G=saved['paths'],SNR=saved['snr_db'],SIR=saved['sir_db'],order=order,
            original_terminal=evaluate(ref['positions'],z['truth']),original_retained=evaluate(selected,z['truth']),
            original_selected_round=round_id,repair_terminal=evaluate(new['positions'],z['truth']),
            original_first=first_metrics(ref,z['truth']),repair_first=first_metrics(new,z['truth']),
            original_seconds=ref['seconds'],repair_seconds=new['seconds'],
            original_score_positions=1000*len(ref['steps']),repair_score_positions=new['score_positions'],
            original_status=ref['status'],repair_status=new['status'],
            repair_final_separation_m=float(np.linalg.norm(np.subtract(*new['positions']))) if new['positions'] is not None else None)
        checks[file.stem+'/paired_available_set']=row['original_retained']['complete']==row['repair_terminal']['complete']
        assert all(checks.values()),[k for k,v in checks.items() if not v]
        write(out/file.name,dict(**row,outputs=outputs,input_sha=saved['input_sha'],observation_sha=saved['observation_sha']))
        rows.append(row)
        print(f'流程修正配对 {index+1}/{len(records)} | {file.stem} | {new["status"]}',flush=True)
    for name,h in identities.items():checks['unchanged/'+name]=sha(SQUARE/name)==h
    for p,h in frozen['sources'].items():checks['solver/'+p]=sha(Path(p))==h
    assert all(checks.values())
    write(out/'result.json',dict(rows=rows,seconds=time.perf_counter()-started,checks={k:bool(v) for k,v in checks.items()}))
    write(out/'audit.json',dict(passed=True,count=len(checks),checks={k:bool(v) for k,v in checks.items()}))
    summarize(out,rows,time.perf_counter()-started,len(checks),limit)
    print('REPORT',out,flush=True)


def summarize(out,rows,seconds,check_count,limit):
    common=[r for r in rows if r['original_retained']['complete'] and r['repair_terminal']['complete']]
    groups=[]
    for g,snr,sir in sorted(set((r['G'],r['SNR'],r['SIR']) for r in rows)):
        sub=[r for r in rows if (r['G'],r['SNR'],r['SIR'])==(g,snr,sir)]
        matched=[r for r in sub if r['original_retained']['complete'] and r['repair_terminal']['complete']]
        row=dict(G=g,SNR=snr,SIR=sir,n=len(sub),common_n=len(matched))
        for v,key in [('original','original_retained'),('repair','repair_terminal')]:
            errors=np.array([r[key]['error_m'] for r in matched])
            row[v+'_joint_RMSE_m']=float(np.sqrt(np.mean(errors**2))) if len(errors) else None
            row[v+'_A_RMSE_m']=float(np.sqrt(np.mean(errors[:,0]**2))) if len(errors) else None
            row[v+'_B_RMSE_m']=float(np.sqrt(np.mean(errors[:,1]**2))) if len(errors) else None
            row[v+'_both20']=sum(r[key]['both20'] for r in matched)
        groups.append(row)
    overall={}
    for v,key in [('original','original_retained'),('repair','repair_terminal')]:
        errors=np.array([r[key]['error_m'] for r in common])
        overall[v]=dict(terminal_complete=sum(r['original_terminal' if v=='original' else 'repair_terminal']['complete'] for r in rows),
            comparable_complete=len(errors),joint_RMSE_m=float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
            A_RMSE_m=float(np.sqrt(np.mean(errors[:,0]**2))) if len(errors) else None,
            B_RMSE_m=float(np.sqrt(np.mean(errors[:,1]**2))) if len(errors) else None,
            both20=sum(r[key]['both20'] for r in common),
            last_first_A=stats([r[v+'_first']['last']['A'] for r in rows]),
            mean_seconds=float(np.mean([r[v+'_seconds'] for r in rows])),
            score_positions=sum(r[v+'_score_positions'] for r in rows),
            statuses=dict(Counter(r[v+'_status'] for r in rows)))
    delta=[r['repair_terminal']['rmse_m']**2-r['original_retained']['rmse_m']**2 for r in common]
    summary=dict(n=len(rows),full_run=limit is None,independent_realizations=len(set(r['realization'] for r in rows)),
        overall=overall,groups=groups,checks=check_count,seconds=seconds,
        common_error_improved=sum(d< -1e-8 for d in delta),common_error_worsened=sum(d>1e-8 for d in delta),
        common_error_equal=sum(abs(d)<=1e-8 for d in delta),
        final_pair_separation_min_m=min((r['repair_final_separation_m'] for r in common),default=None))
    write(out/'summary.json',summary)
    with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(groups[0]));w.writeheader();w.writerows(groups)
    print(summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int);args=parser.parse_args()
    with threadpool_limits(limits=2):main(args.limit)
