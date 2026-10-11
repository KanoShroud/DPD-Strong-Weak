"""G9-R3: bounded small comparison; no neural training or test access."""
import argparse
from functools import partial
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from .. import solver_v2
from ..runner import last_positions
from ..runtime import BASE, Guard, identity, preflight, read, verify, write
from ..scenes import make
from ..solver import local
from .bounded_search import BoundScore, update
from .coverage_search import BatchedScore
from .failure_attribution import load_inputs, regenerate, full_score
from .reevaluate_baseline import evaluate, summary

METHODS=('D','shift10','S','E')


def tests(iq, predicted, guard):
    calc=BoundScore(iq,guard,128 if iq.shape[-1]==4096 else 64)
    rng=np.random.default_rng(2026101103)
    records=[]
    points=np.array([[500.,0.], [499.,0.], [0.,0.], [900.,900.], [-975.,-975.]])
    for k in (0,1,2):
        calc.set_sources(predicted[:k])
        for width in (0.,1.,5.,25.):
            values,bounds=calc.bound(points,width)
            ordinary=calc.evaluate(points)
            error=float(np.max(np.abs(values-ordinary)))
            assert error<1e-12
            samples=points[:,None]+rng.uniform(-width,width,(5,16,2))
            actual=calc.evaluate(samples.reshape(-1,2)).reshape(5,16).max(-1)
            excess=float(np.max(actual-bounds))
            assert excess<1e-10,(k,width,excess)
            records.append(dict(projected_sources=k,width=width,score_error=error,upper_excess=excess))
    # Boundary semantics are separate from numerical geometry.
    assert evaluate([[0,0]],[[20,0]])['success']['20']==[True]
    return records


def develop(out, records, contract, guard, smoke):
    selected=read(BASE/'r1_failure_20261011_01/contract.json')['selected']
    items=[r for r in records if r['tag'] in selected]
    if smoke:
        items=[next(r for r in items if r['row']['N']==n) for n in (4096,16384)]
    rows=[]
    for i,item in enumerate(items):
        iq,truth,predicted,_=regenerate(item,contract)
        if smoke:
            write(out/f'numerical_N{item["row"]["N"]}.json',tests(iq,predicted,guard))
            continue
        row=dict(tag=item['tag'],N=item['row']['N'],selection=selected[item['tag']],results={})
        # Rotate evaluation order; identical fixed first-source estimate for all arms.
        for method in METHODS[i%4:]+METHODS[:i%4]:
            calc=BoundScore(iq,guard,128 if item['row']['N']==4096 else 64)
            result=update(calc,predicted[0],predicted[1],method,guard)
            result['error_m']=float(np.linalg.norm(np.asarray(result['position'])-truth[1]))
            assert result['score']>=result['initial_score']-1e-12
            row['results'][method]=result
            del calc
        rows.append(row)
        write(out/'rows'/f'{item["tag"]}.json',row)
        if len(rows)%4==0:print(f'开发比较 {len(rows)}/24',flush=True)
    if smoke:
        write(out/'report.json',dict(status='SMOKE_PASS',performance_evidence=False))
        return
    stats={}
    for method in METHODS:
        r=[x['results'][method] for x in rows]
        stats[method]=dict(cases=len(r),success20=sum(x['error_m']<=20 for x in r),
            failure_recovered20=sum(x['results'][method]['error_m']<=20 for x in rows if x['selection']=='failure'),
            controls20=sum(x['results'][method]['error_m']<=20 for x in rows if x['selection']=='success_control'),
            seconds_mean=float(np.mean([x['seconds'] for x in r])),
            grid_points_mean=float(np.mean([x['grid_points'] for x in r])),
            pruned_cells_mean=float(np.mean([x['pruned_cells'] for x in r])))
    e,d=stats['E'],stats['D']
    eligible=(e['failure_recovered20']>=d['failure_recovered20']-1 and e['controls20']==8
              and e['seconds_mean']<=.6*d['seconds_mean']
              and e['success20']>=max(stats[m]['success20'] for m in ('shift10','S')))
    write(out/'report.json',dict(status='CANDIDATE_TO_FRESH' if eligible else 'STOP_BOUND_RULE',
        eligible=eligible,summary=stats,scope='fixed selected diagnostics, not population success'))


def complete_update(iq,predicted,method,guard):
    calc=BoundScore(iq,guard,128 if iq.shape[-1]==4096 else 64)
    positions=np.asarray(predicted).copy()
    for k in range(2):
        old=full_score(calc,positions)
        result=update(calc,positions[1-k],positions[k],method,guard)
        proposal=positions.copy();proposal[k]=result['position']
        if full_score(calc,proposal)>=old:positions=proposal
    for _ in range(2):
        for k in range(2):
            other=positions[1-k:2-k]
            calc.set_sources(other)
            candidate=local(calc,positions[k],other)
            if calc.evaluate([candidate])[0]>=calc.evaluate([positions[k]])[0]:positions[k]=candidate
    return positions


def fresh(out,guard):
    dev=read(BASE/'r3_bound_dev_20261011_01/report.json')
    assert dev['eligible'],'Development did not authorize fresh stage'
    rows=[]
    for group in range(106,110):
        for overlap in (.5,1.):
            scene=make('check',group,0,overlap,3,12)
            for n in (4096,16384):
                iq=scene['signal'][:,:,:n].sum(0)+scene['noise'][:,:n]
                original=solver_v2.Score
                solver_v2.Score=partial(BatchedScore,chunk=128 if n==4096 else 64)
                try:base=solver_v2.run(iq,'B',60.,guard=guard)
                finally:solver_v2.Score=original
                assert base['status']=='FINISHED'
                initial=last_positions(base,60.)
                row=dict(group=group,overlap=overlap,N=n,results={})
                row['results']['B']=dict(metrics=evaluate(scene['positions'],initial),seconds=base['seconds'])
                for method in ('D','S','E'):
                    torch.cuda.synchronize();start=time.perf_counter()
                    positions=complete_update(iq,initial,method,guard)
                    torch.cuda.synchronize();elapsed=time.perf_counter()-start
                    row['results'][method]=dict(positions=positions.tolist(),metrics=evaluate(scene['positions'],positions),seconds=base['seconds']+elapsed)
                rows.append(row)
                write(out/'rows'/f'{group}_{overlap}_{n}.json',row)
        print(f'新布局完成 {group-105}/4',flush=True)
    write(out/'report.json',dict(status='FRESH_COMPLETE',summary={m:summary([r['results'][m] for r in rows]) for m in ('B','D','S','E')}))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('smoke','develop','fresh'),required=True)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.zeros(1,device='cuda').sum().item()
    report,records,inputs=load_inputs()
    previous=read(BASE/'r1_failure_20261011_01/ledger.json')['cumulative_seconds']
    previous+=sum(read(p)['seconds_this_run'] for p in BASE.glob('r2_coverage_*/ledger.json'))
    spent=sum(read(p)['seconds_this_run'] for p in BASE.glob('r3_bound_*/ledger.json'))
    a,b=Guard(45*60,spent),Guard(8*3600,previous+spent)
    def guard():a();b()
    out=preflight(BASE/args.run_id)
    names=('bounded_search.py','bound_experiment.py','coverage_search.py','failure_attribution.py','reevaluate_baseline.py')
    sources=report['contract']['sources']+[identity(Path(__file__).with_name(x)) for x in names]
    inputs.append(identity(BASE/'r1_failure_20261011_01/contract.json'))
    if args.stage=='fresh':inputs.append(identity(BASE/'r3_bound_dev_20261011_01/report.json'))
    verify(inputs);verify(sources)
    write(out/'contract.json',dict(stage=args.stage,sources=sources,inputs=inputs,prior_seconds=previous+spent,
        r3_limit_seconds=2700,gate_limit_seconds=28800,protocol='EVAL-20261011-v2',methods=list(METHODS),
        grid_limit=16384,min_halfwidth_m=1.5625,test_access=False))
    start=time.perf_counter()
    try:
        if args.stage=='fresh':fresh(out,guard)
        else:develop(out,records,report['contract'],guard,args.stage=='smoke')
        verify(sources);verify(inputs)
        write(out/'final_audit_report.json',dict(status='PASS',sources_unchanged=True,inputs_unchanged=True,
            report=identity(out/'report.json'),test_access=False))
    except Exception:
        write(out/'failure.json',dict(status='FAILED',traceback=traceback.format_exc()))
        raise
    finally:
        elapsed=time.perf_counter()-start
        write(out/'ledger.json',dict(seconds_this_run=elapsed,cumulative_seconds=previous+spent+elapsed))
    print('阶段完成：'+str(out/'report.json'),flush=True)


if __name__=='__main__':main()
