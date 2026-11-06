"""Fixed-design focused-risk tests; RF tangent diagnostics, never a selector."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request
import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path.insert(0,str(BASE/'unknown_order_20261030'))
import reference as ref  # noqa: E402


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def counterexample():
    sigma=.1
    a=np.array([[1.],[0.]])
    mu=np.ones(2)
    ls=np.array([[1.,0.]])
    added=np.array([[1.,-1.]])
    rows=[]
    for p in (0.,1.):
        omitted=mu-a[:,0]*p
        risks=[float(np.linalg.norm(l@mu-p)**2+sigma**2*np.sum(l*l)) for l in (ls,added)]
        rows.append(dict(p=p,omitted=omitted,mean=a[:,0]*p+omitted,risks=risks,gain=risks[0]-risks[1]))
    return rows


def linear(rng):
    q,_=np.linalg.qr(rng.normal(size=(12,12)))
    a=q[:,:4];h=rng.normal(size=(4,4))*.3
    sigma=.2;rows=[];checks={}
    for spacing in (.2,1.):
        c=a@h+spacing*q[:,4:8];g=np.column_stack((a,c));t=np.linalg.pinv(g)[:4]
        checks[f'{spacing}/focus_identity']=bool(np.allclose(t@g,np.column_stack((np.eye(4),np.zeros((4,4)))),atol=1e-12))
        models=[np.linalg.pinv(g[:,:4+k])[:4] for k in (0,2,4)]
        for amp in (0.,.25,1.):
            theta=np.r_[[.2,-.1,.3,-.4],amp*np.array([1.,-.8,.6,-.4])]
            mu=g@theta;y=mu[:,None]+sigma*rng.normal(size=(12,20000))
            scores=[];losses=[];stats=[]
            for k,l in zip((0,2,4),models):
                d=l-t
                score=np.sum((d@y)**2,axis=0)+2*sigma**2*np.sum(l*t)-sigma**2*np.sum(t*t)
                loss=np.sum((l@y-theta[:4,None])**2,axis=0)
                risk=float(np.sum((l@mu-theta[:4])**2)+sigma**2*np.sum(l*l))
                expectation=float(np.sum((d@mu)**2)+sigma**2*np.sum(d*d)+2*sigma**2*np.sum(l*t)-sigma**2*np.sum(t*t))
                error=score-loss;se=float(error.std(ddof=1)/np.sqrt(len(error)))
                key=f'{spacing}/{amp}/{k}'
                checks[key+'/analytic']=abs(risk-expectation)<1e-10
                checks[key+'/mc']=abs(float(error.mean()))<=6*se+1e-10
                stats.append(dict(nuisance_columns=k,risk=risk,analytic_score=expectation,mc_score=float(score.mean()),mc_loss=float(loss.mean()),paired_error=float(error.mean()),paired_se=se))
                scores.append(score);losses.append(loss)
            scores=np.array(scores);losses=np.array(losses);choice=np.argmin(scores,axis=0);ids=np.arange(y.shape[1])
            rows.append(dict(spacing=spacing,amplitude=amp,models=stats,
                selected_score=float(scores[choice,ids].mean()),selected_loss=float(losses[choice,ids].mean()),
                note='Fixed-model unbiasedness does not imply selected-score unbiasedness.'))
    return rows,checks


def sources():
    old=BASE/'unknown_order_20261030'
    cross=Path(read(old/'cross_latest.json')['path']);unseen=Path(read(old/'feasible_unseen_latest.json')['path'])
    for root in (cross,unseen):
        pre=read(root/'preflight.json')['hashes']
        assert all(ref.sha(p)==h for p,h in pre.items())
    return [(root,tag) for root,tags in [(cross,['L20_L0','L40_L0']),(unseen,['L20_L3','L40_L3'])] for tag in tags]


def rf(root,tag):
    row=read(root/(tag+'.json'))
    with np.load(root/(tag+'_input.npz'),allow_pickle=False) as inp:iq=inp['iq']
    assert hashlib.sha256(np.ascontiguousarray(iq).tobytes()).hexdigest()==row['iq_sha']
    f0=np.fft.fftfreq(iq.shape[1],1/ref.FS);use=abs(f0)<=ref.B/2
    ft=np.fft.fft(iq,axis=1,norm='ortho');y=ft[:,use].T;f=f0[use]
    sigma2=float(np.mean(abs(ft[:,~use])**2))
    x=np.array(row['final']['x']);paths=row['paths']
    _,j,_=ref.evaluate(x,f,y,paths,True)
    a=j[:,:4];c=j[:,4:]
    u,s,_=np.linalg.svd(c,full_matrices=False)
    rank=int(np.sum(s>s[0]*1e-10));u=u[:,:rank]
    ae=a-u@(u.T@a);sv=np.linalg.svd(ae,compute_uv=False)
    position_rank=int(np.sum(sv>np.linalg.svd(a,compute_uv=False)[0]*1e-10))
    result=dict(tag=tag,observed_support=len(paths),nuisance_columns=c.shape[1],nuisance_rank=rank,
        position_singular_values=sv,position_rank=position_rank,
        min_information_ratio=float(sv[-1]/np.linalg.svd(a,compute_uv=False)[0]),sigma2_est=sigma2,
        all_gains_interior=bool(all(np.linalg.norm(x[k:k+2])<1-1e-8 for k in range(20,len(x),3))),
        note='Estimated selected-support mean tangent only. Not full nonzero-residual estimator risk or global identifiability.')
    if position_rank==4:
        t=np.linalg.pinv(ae,rcond=1e-10)
        result['focus_identity_error']=float(np.linalg.norm(t@a-np.eye(4)))
        result['nuisance_annihilation_error']=float(np.linalg.norm(t@c)/max(np.linalg.norm(t)*np.linalg.norm(c),1e-300))
        result['conditional_rms_noise_scale_m']=float(np.sqrt(sigma2*np.sum(t*t)/2)*ref.UNIT)
    return result


def main():
    inputs=sources()
    files=[Path(__file__),HERE/'执行协议.md',BASE/'unknown_order_20261030/reference.py',BASE/'risk_step_20261030/pilot.py']
    files += [root/(tag+suffix) for root,tag in inputs for suffix in ('.json','_input.npz')]
    hashes={str(p.resolve(strict=True)):ref.sha(p) for p in files}
    assert all(Path(p).is_relative_to(BASE) for p in hashes)
    out=HERE/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir(exist_ok=False)
    for name in hashes:
        p=Path(name);dest=out/'source'/p.relative_to(BASE);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    ref.write(out/'preflight.json',dict(hashes=hashes,seed=2026103032,synthetic_repeats=20000))
    begin=time.perf_counter();rng=np.random.default_rng(2026103032)
    counter=counterexample();linear_rows,checks=linear(rng)
    checks['identical_distribution_opposite_risk']=bool(np.array_equal(counter[0]['mean'],counter[1]['mean']) and counter[0]['gain']>0>counter[1]['gain'])
    rf_rows=[]
    if all(checks.values()):
        for root,tag in inputs:
            r=rf(root,tag);rf_rows.append(r)
            checks[tag+'/finite']=bool(np.isfinite(r['position_singular_values']).all())
            if r['position_rank']==4:
                checks[tag+'/focus_identity']=r['focus_identity_error']<1e-6
                checks[tag+'/nuisance_identity']=r['nuisance_annihilation_error']<1e-7
            print(tag,json.dumps(r,ensure_ascii=False,default=lambda a:a.tolist()),flush=True)
    queries=[]
    for doi in ('10.1109/TSP.2008.2008212','10.3389/fevo.2019.00415'):
        url='https://api.crossref.org/works/'+doi.replace('/','%2F')
        try:
            with urllib.request.urlopen(url,timeout=20) as response:raw=json.load(response)
            name=doi.split('/')[-1]+'.json';ref.write(out/name,raw)
            queries.append(dict(url=url,raw_file=name,title=raw['message']['title'],status='ok'))
        except Exception as exc:queries.append(dict(url=url,status='failed',error=repr(exc)))
    checks['identities']=all(ref.sha(p)==h for p,h in hashes.items())
    ref.write(out/'result.json',dict(counterexample=counter,linear=linear_rows,rf=rf_rows,queries=queries,
        checks=checks,all_checks_pass=all(checks.values()),seconds=time.perf_counter()-begin,
        nonlinear_risk_calibrated=False,new_localization_runs=0))
    ref.write(HERE/'latest.json',dict(path=str(out)))
    print('完成',len(checks),all(checks.values()),'输出',out,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
