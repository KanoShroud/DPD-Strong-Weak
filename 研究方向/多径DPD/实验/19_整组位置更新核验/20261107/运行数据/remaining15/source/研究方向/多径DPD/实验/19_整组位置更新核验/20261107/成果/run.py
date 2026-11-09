"""Same-objective joint position-block proposal using complete multipath columns."""
import argparse
import importlib.util
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'AGENTS.md').exists())
BASE=ROOT/'研究方向/多径DPD'
R14=BASE/'实验/14_公开信道记忆核验/20261107'
spec=importlib.util.spec_from_file_location('r14',R14/'成果/run.py')
r14=importlib.util.module_from_spec(spec);spec.loader.exec_module(r14)
gate=r14.gate


def exact_loss(h,y):
    scale=np.max(abs(h),axis=(1,2))
    u,s,_=np.linalg.svd(h/scale[:,None,None],full_matrices=False)
    basis=u*(s>1e-12*s[:,:1])[:,None,:]
    r=y-(basis@(basis.conj().transpose(0,2,1)@y[:,:,None]))[:,:,0]
    return float(np.sum(abs(r)**2)/np.sum(abs(y)**2))


def score_pairs(h1,h2,y,block=64):
    """h arrays: F x candidates x stations. Constant-rank algebra + SVD fallback."""
    count=len(h1[0]);scores=np.zeros((count,count));fallback=0
    for start in range(0,len(y),block):
        a=h1[start:start+block];b=h2[start:start+block];v=y[start:start+block]
        na=np.linalg.norm(a,axis=2);nb=np.linalg.norm(b,axis=2)
        assert np.all(na>0) and np.all(nb>0)
        aa=a/na[:,:,None];bb=b/nb[:,:,None]
        cross=aa.conj()@bb.transpose(0,2,1)
        ay=np.einsum('fcm,fm->fc',aa.conj(),v)
        by=np.einsum('fcm,fm->fc',bb.conj(),v)
        denominator=1-abs(cross)**2
        safe=denominator>1e-6
        second=np.divide(abs(by[:,None,:]-cross.conj()*ay[:,:,None])**2,denominator,
                         out=np.zeros_like(denominator),where=safe)
        residual=np.sum(abs(v)**2,axis=1)[:,None,None]-abs(ay[:,:,None])**2-second
        for fi,i,j in np.argwhere(~safe):
            h=np.column_stack((a[fi,i],b[fi,j]))[None]
            residual[fi,i,j]=exact_loss(h,v[fi:fi+1])*np.sum(abs(v[fi])**2)
            fallback+=1
        assert np.min(residual)>-1e-10*max(float(np.max(np.sum(abs(v)**2,axis=1))),1e-300)
        scores+=np.sum(residual,axis=0)
    return scores/np.sum(abs(y)**2),fallback


def unit():
    rng=np.random.default_rng(2026110721)
    a=rng.normal(size=(25,6,4))+1j*rng.normal(size=(25,6,4))
    b=rng.normal(size=a.shape)+1j*rng.normal(size=a.shape)
    b[:,0]=a[:,0]*(1+1j)  # exact rank one must use original SVD cutoff
    y=rng.normal(size=(25,4))+1j*rng.normal(size=(25,4))
    scores,n=score_pairs(a,b,y,block=7)
    worst=0.
    for i in range(6):
        for j in range(6):
            value=exact_loss(np.stack((a[:,i],b[:,j]),axis=2),y)
            worst=max(worst,abs(value-scores[i,j]))
    assert worst<1e-12 and n>=25
    return dict(score_max_absolute_error=worst,rank_fallback=n,pairs=36)


def propose(e,x,paths,iq):
    f,y,n,_=e.fc.cal.spectral(iq)
    start=time.perf_counter();front=e.frontend(iq)
    pool=np.vstack((np.asarray(front['refined']),x[:4].reshape(2,2)*e.ref.UNIT))
    assert np.all(abs(pool)<=3000+1e-7)
    distance=np.linalg.norm(pool[:,None]-e.stations[None],axis=2)
    assert np.all(distance>1e-8)
    h=e.core(x,f,y,paths)[0]
    current=x[:4].reshape(2,2)*e.ref.UNIT
    tau=np.linalg.norm(current[:,None]-e.stations[None],axis=2)/e.ref.C
    shape=h*np.exp(2j*np.pi*f[:,None,None]*tau.T[None])
    phase=np.exp(-2j*np.pi*f[:,None,None]*distance[None]/e.ref.C)
    a=phase*shape[:,None,:,0];b=phase*shape[:,None,:,1]
    scores,fallback=score_pairs(a,b,y)
    parent=len(pool)-2,len(pool)-1
    assert abs(scores[parent]-e.objective(x,f,y,paths)[0])<1e-9
    winner=np.unravel_index(np.argmin(scores),scores.shape)
    rng=np.random.default_rng(2026110721)
    audit=[parent,winner]+[(int(i),int(j)) for i,j in rng.integers(0,len(pool),(12,2))]
    worst=0.
    for i,j in audit:
        trial=x.copy();trial[:4]=pool[[i,j]].ravel()/e.ref.UNIT
        cost=e.objective(trial,f,y,paths)[0]
        worst=max(worst,abs(cost-scores[i,j]))
    assert worst<1e-9
    proposal=x.copy();proposal[:4]=pool[list(winner)].ravel()/e.ref.UNIT
    assert np.linalg.norm(proposal-e.ref.project(proposal,n))<1e-7
    return proposal,dict(pool=pool,scores=scores,parent_pair=parent,winner=winner,
        parent_cost=scores[parent],winner_cost=scores[winner],audit_error=worst,
        svd_fallbacks=fallback,seconds=time.perf_counter()-start)


def refine(e,x,f,y,paths,n):
    start=time.perf_counter()
    z,state=e.base_fit(x,f,y,paths,n)
    # Use the actual four-station evaluator, not the historical five-station module.
    feasible=e.ref.project(z,n)
    cost,grad=e.objective(feasible,f,y,paths)
    closure=float(np.linalg.norm(feasible-e.ref.project(feasible,n)))
    pg=float(np.max(abs(feasible-e.ref.project(feasible-grad,n))))
    before=e.objective(x,f,y,paths)[0]
    accepted=np.isfinite(cost) and closure<1e-7 and cost<=before+1e-10
    if not accepted:feasible=x.copy();cost,grad=e.objective(feasible,f,y,paths);pg=float(np.max(abs(feasible-e.ref.project(feasible-grad,n))))
    return dict(x=feasible,p=feasible[:4].reshape(2,2)*e.ref.UNIT,cost=cost,
        projected_gradient=pg,normal=bool(accepted and pg<=1e-7),accepted=bool(accepted),
        fit=state,seconds=time.perf_counter()-start)


def main():
    p=argparse.ArgumentParser();p.add_argument('--start',type=int,default=0);p.add_argument('--limit',type=int,default=3);p.add_argument('--name',required=True);args=p.parse_args()
    assert args.start>=0 and args.limit>0 and args.start+args.limit<=18
    out=HERE.parent/'运行数据'/args.name;out.mkdir(parents=True,exist_ok=False)
    files=list((BASE/'代码/mp_dpd').rglob('*.py'))+[Path(__file__),HERE/'执行合同.md',R14/'成果/run.py',r14.loader]
    identity={str(p):gate.sha(p) for p in files}
    for p in files:gate.copy_checked(p,out/'source'/p.relative_to(ROOT))
    gate.cfg.write(out/'preflight.json',dict(source_hashes=identity,checks=unit()))
    done=gate.read(R14/'运行数据/paired18/completion.json')
    assert done['identity_verified']
    inputs={};rows=[];start=time.perf_counter()
    old=gate.read(r14.OLD/'运行数据/common18/rows.json')
    for index in range(args.start,args.start+args.limit):
        record=old[index];tag=record['tag'];src=r14.OLD/'运行数据/common18/inputs'/tag/'iq.npz'
        assert gate.sha(src)==record['input_sha']
        dst=out/'inputs'/tag/'iq.npz';gate.copy_checked(src,dst);inputs[str(dst)]=gate.sha(dst)
        state_src=R14/'运行数据/paired18'/f'{tag}_CAP_COUNTED_decision.json'
        state_dst=out/'inputs'/tag/'parent.json';gate.copy_checked(state_src,state_dst);inputs[str(state_dst)]=gate.sha(state_dst)
        parent=gate.read(state_dst);x=np.asarray(parent['final']['x']);paths=parent['paths']
        with np.load(dst,allow_pickle=False) as z:iq=z['iq']
        e=r14.engine_for('CAP_COUNTED');f,y,n,_=e.fc.cal.spectral(iq)
        assert abs(e.objective(x,f,y,paths)[0]-parent['final']['cost'])<1e-9
        proposal,scan=propose(e,x,paths,iq)
        np.savez_compressed(out/f'{tag}_scan.npz',pool=scan.pop('pool'),scores=scan.pop('scores'))
        answers={}
        for arm in (['CONTINUE','JOINT_POSITION'] if index%2==0 else ['JOINT_POSITION','CONTINUE']):
            answers[arm]=refine(e,x if arm=='CONTINUE' else proposal,f,y,paths,n)
        gate.cfg.write(out/f'{tag}_decision.json',dict(parent=parent['final'],K=len(paths),scan=scan,answers=answers))
        bank=Path(record['truth_evaluation_file']);assert gate.sha(bank)==record['truth_file_sha']
        ev=out/'evaluation'/bank.name
        if not ev.exists():gate.copy_checked(bank,ev)
        inputs[str(ev)]=gate.sha(ev)
        with np.load(ev,allow_pickle=False) as z:truth=z['p']
        measures={}
        for arm,state in [('PARENT',parent['final']),*answers.items()]:
            err=gate.errors(state['p'],truth)
            measures[arm]=dict(errors_m=err,cost=state['cost'],normal=state.get('normal',parent['normal']),
                seconds=state.get('seconds'),projected_gradient=state.get('projected_gradient'))
        rows.append(dict(tag=tag,condition=record['condition'],K=len(paths),scan=scan,arms=measures))
        gate.cfg.write(out/'rows.json',rows)
        print(tag,'scan',round(scan['seconds'],3),'RMSE',{a:round(float(np.sqrt(np.mean(np.array(v['errors_m'])**2))),3) for a,v in measures.items()},flush=True)
    assert all(gate.sha(Path(p))==h for p,h in {**identity,**inputs}.items())
    gate.cfg.write(out/'completion.json',dict(count=len(rows),seconds=time.perf_counter()-start,source_hashes=identity,input_hashes=inputs,identity_verified=True))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
