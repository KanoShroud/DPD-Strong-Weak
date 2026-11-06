"""Offline native-score acceptance; no truth is used to choose endpoints."""
import numpy as np
from threadpoolctl import threadpool_limits
from route_screen import ROOT, INPUT, OLD, Scorer, seeds_from_points
from pilot import observe, evaluate, write, sha
from trace_audit import read

OUT=ROOT/'numerical_20261027_191234_666421'


def main():
    rows=[];checks={}
    for p in sorted(OUT.glob('r*_G*_*.json')):
        row=read(p);kind=row['version'].split('_')[0];name=row['case'];rec=read(INPUT/(name+'.json'))
        z=dict(np.load(INPUT/f"input_{rec['realization']:02d}.npz"))
        y,_,_=observe(z,rec['snr_db'],rec['sir_db'],rec['paths']);s=Scorer(y,z['frequency'],z['stations'],kind)
        points=np.random.default_rng(read(OLD/(name+'.json'))['particle_seed']).uniform(-3000,3000,(1000,2))
        scores=s.matrix(points);indices=seeds_from_points(points,scores,6);accepted=[];rejected=0
        for i,r in zip(indices,row['result']['refinements']):
            initial=-2*np.log(scores[i]);end=float(-2*np.log(s.matrix([r['position']])[0]))
            checks[name+'/'+kind+'/'+str(i)]=np.isclose(end,r['loss'],rtol=1e-10,atol=1e-10)
            if end>initial:
                rejected+=1;accepted.append(dict(position=points[i].tolist(),loss=float(initial)))
            else:accepted.append(dict(position=r['position'],loss=end))
        selected=[]
        for r in sorted(accepted,key=lambda x:x['loss']):
            if all(np.linalg.norm(np.array(r['position'])-q)>=400 for q in selected):selected.append(r['position'])
            if len(selected)==2:break
        pred=selected if len(selected)==2 else None
        ev=evaluate(pred,z['truth'])
        if pred is not None:
            a=np.linalg.norm(np.array(pred)-z['truth'],axis=1)
            b=np.linalg.norm(np.array(pred)[::-1]-z['truth'],axis=1)
            err=a if np.sum(a*a)<=np.sum(b*b) else b
            checks[name+'/'+kind+'/matching']=np.allclose(err,ev['error_m'])
        rows.append(dict(case=name,version=kind+'_accepted',SNR=row['SNR'],evaluation=ev,
                         positions=pred,rejected_worse_endpoints=rejected,source_sha=sha(p)))
    summaries=[]
    for v in ['ratio_accepted','ccf_accepted']:
        sub=[r for r in rows if r['version']==v];good=[r for r in sub if r['evaluation']['complete']]
        summaries.append(dict(version=v,n=len(sub),complete=len(good),both20=sum(r['evaluation']['both20'] for r in sub),
            conditional_RMSE=float(np.sqrt(np.mean([np.array(r['evaluation']['error_m'])**2 for r in good]))),
            rejected_worse_endpoints=sum(r['rejected_worse_endpoints'] for r in sub)))
    assert all(checks.values())
    write(OUT/'acceptance_audit.json',dict(scope='OFFLINE_OUTPUT_ACCEPTANCE_NOT_NEW_OPTIMIZER',
        caution='Powell success flag alone does not establish useful refinement. Retain initial seed whenever native objective worsens; no truth used. Not evidence that search is exhaustive.',
        source_sha=sha(__file__),rows=rows,summary=summaries,checks=checks,passed=True))
    print(summaries)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
