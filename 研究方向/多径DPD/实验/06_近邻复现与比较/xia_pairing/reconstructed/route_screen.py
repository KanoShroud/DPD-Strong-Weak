"""Bounded 2x2 Xia-route screen: two scores x two search backends.

Standard CCF and bounded continuous peak refinement are controls, not claimed
novelties. Reuse 28 fixed conditions (IDs 0,3,6,9), no truth in solver.
"""
import csv
from datetime import datetime
from pathlib import Path
import time
import hashlib

import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
from pilot import ROOT,N,C,sha,write,observe,evaluate,xia
from trace_audit import read
from flow_repair import group_select

INPUT=ROOT/'square_20261027_182013_211372'
OLD=ROOT/'pilot_20261027_170421_805496'
IDS=[0,3,6,9]


class Scorer:
    def __init__(self,y,f,stations,kind):
        self.f=f;self.stations=stations;self.calls=0;self.points=0
        if kind=='ratio': self.q=y[1:]/y[:1]
        else:
            power=np.mean(abs(y)**2,axis=1)
            self.q=y[0].conj()[None]*y[1:]/np.sqrt(power[0]*power[1:,None])
    def matrix(self,p):
        p=np.atleast_2d(p);values=[]
        for j in range(0,len(p),64):
            d=np.linalg.norm(p[j:j+64,None]-self.stations[None],axis=-1)
            delay=(d[:,1:]-d[:,:1])/C
            z=(self.q[None]*np.exp(2j*np.pi*delay[:,:,None]*self.f)).mean(-1)
            values.extend(np.prod(abs(z),axis=1))
        self.calls+=1;self.points+=len(p)
        return np.array(values)
    def objective(self,km):
        p=np.asarray(km)*1000
        vec=p-self.stations;d=np.linalg.norm(vec,axis=1)
        if np.any(d<1e-10): raise ValueError('STATION_COORDINATE_NONDIFFERENTIABLE')
        unit=vec/d[:,None]
        delay=(d[1:]-d[0])/C
        dt=(unit[1:]-unit[0])*1000/C
        term=self.q*np.exp(2j*np.pi*delay[:,None]*self.f)
        z=term.mean(1)
        dz=(term*(2j*np.pi*self.f)).mean(1)[:,None]*dt
        power=abs(z)**2
        # Machine underflow guard only; it is not a tuned regularizer.
        value=-np.log(power+np.finfo(float).tiny).sum()
        grad=-2*np.sum(np.real(z.conj()[:,None]*dz)/(power[:,None]+np.finfo(float).tiny),axis=0)
        self.calls+=1;self.points+=1
        return float(value),grad


def seeds_from_points(p,v,count):
    chosen=[]
    for i in np.argsort(-v,kind='stable'):
        if all(np.linalg.norm(p[i]-p[j])>=400 for j in chosen):chosen.append(int(i))
        if len(chosen)==count:break
    return chosen


def solve(y,f,stations,kind,backend,seed):
    start=time.perf_counter();scorer=Scorer(y,f,stations,kind)
    rng=np.random.default_rng(seed);p=rng.uniform(-3000,3000,(1000,2));v=scorer.matrix(p)
    initial_hash=hashlib.sha256(p.tobytes()).hexdigest()
    detail={}
    if backend=='groups':
        output=xia.decode(p,v,2,400.)
        detail['first']=output
        if output['positions'] is not None:
            owner=np.arange(1000)%2
            for _ in range(2):
                p=np.array(output['positions'])[owner]+rng.normal(size=(1000,2))*10
                v=scorer.matrix(p);output=group_select(p,v,owner)
        pred=output['positions'];status=output['status']
    else:
        indices=seeds_from_points(p,v,6);results=[]
        for i in indices:
            try:
                opt=minimize(scorer.objective,p[i]/1000,jac=True,method='L-BFGS-B',bounds=[(-3.,3.)]*2,
                    options=dict(maxiter=80,gtol=1e-5,ftol=1e-10,maxls=30))
                results.append(dict(position=(opt.x*1000).tolist(),loss=float(opt.fun),success=bool(opt.success),
                                    iterations=opt.nit,status=str(opt.message)))
            except ValueError as exc:
                results.append(dict(position=p[i].tolist(),loss=None,success=False,iterations=0,status=str(exc)))
        available=[r for r in results if r['loss'] is not None and np.isfinite(r['loss'])]
        available.sort(key=lambda r:r['loss']);selected=[]
        for item in available:
            if all(np.linalg.norm(np.array(item['position'])-q)>=400 for q in selected):selected.append(item['position'])
            if len(selected)==2:break
        pred=selected if len(selected)==2 else None
        status='TWO_REFINED_PEAKS' if pred is not None else 'LESS_THAN_TWO_DISTINCT_REFINED_PEAKS'
        detail=dict(refinements=results,seeds=indices,nonconverged=sum(not r['success'] for r in results))
    return dict(positions=pred,status=status,seconds=time.perf_counter()-start,calls=scorer.calls,
                points=scorer.points,initial_hash=initial_hash,detail=detail)


def main():
    start=time.perf_counter();out=ROOT/('screen_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir(exist_ok=False)
    files=[p for p in INPUT.glob('r*_G*.json') if read(p)['realization'] in IDS]
    files=sorted(files);assert len(files)==28
    hashes={str(p):sha(p) for p in files+[INPUT/f'input_{i:02d}.npz' for i in IDS]}
    code=[Path(__file__),ROOT/'pilot.py',ROOT/'flow_repair.py',ROOT.parent/'recheck/xia_checked.py',ROOT.parent/'xia_reference.py']
    code_hash={str(p):sha(p) for p in code}
    protocol=dict(scope='BOUNDED_ROUTE_SCREEN_NOT_PAPER_REPRODUCTION',cases=28,ids=IDS,
        input_hashes=hashes,code_hashes=code_hash,
        variants=['ratio_groups','ccf_groups','ratio_peaks','ccf_peaks'],
        scoring='ratio = original Eq9; ccf = unwhitened cross-correlation normalized only by per-station total powers',
        sources='Xia 2020 Eq8-9 static/no-Doppler correlation inspiration; not its exponential kernel or tracking reproduction',
        groups='published first decode plus previously approved explicit groupwise flow correction',
        peaks='six separated initial seeds from same 1000 points, 400m exclusion inherited from original delta_h, L-BFGS-B 80 max, gtol1e-5,ftol1e-10, continuous peaks',
        no_tuning=True,no_truth_solver=True,no_new_IQ=True,
        continuation_rule='No further Xia module tuning this turn. Only consider fresh confirmation if 5/20dB cases have >=90% complete outputs, joint RMSE <=100m and both20 rate >=50%; these are loose route-screen criteria, not publication thresholds. Standard controls alone never establish novelty.',
        failure_policy='Missing outputs explicit; no invented errors or treating available subset RMSE as population RMSE')
    write(out/'protocol.json',protocol);(out/'route_screen.py').write_bytes(Path(__file__).read_bytes())
    checks={};rows=[]
    for count,p in enumerate(files):
        rec=read(p);i=rec['realization'];z=dict(np.load(INPUT/f'input_{i:02d}.npz'));old=read(OLD/p.name)
        y,_,_=observe(z,rec['snr_db'],rec['sir_db'],rec['paths'])
        checks[p.stem+'/input']=hashlib.sha256(y.tobytes()).hexdigest()==rec['observation_sha']
        testp=np.array([[123.,456.],[-1420.,950.]])
        ratio=Scorer(y,z['frequency'],z['stations'],'ratio')
        checks[p.stem+'/ratio_equivalence']=np.allclose(ratio.matrix(testp),xia.scores(testp,z['stations'],y,z['frequency'],N),rtol=1e-12,atol=0)
        for kind in ['ratio','ccf']:
            s=Scorer(y,z['frequency'],z['stations'],kind);x=np.array([.123,.456]);value,grad=s.objective(x);eps=1e-6
            numeric=np.array([(s.objective(x+np.eye(2)[j]*eps)[0]-s.objective(x-np.eye(2)[j]*eps)[0])/(2*eps) for j in [0,1]])
            checks[p.stem+'/'+kind+'_gradient']=np.allclose(grad,numeric,rtol=1e-4,atol=1e-5)
            checks[p.stem+'/'+kind+'_log_equivalence']=np.allclose(np.exp(-value/2),s.matrix(x[None]*1000)[0],rtol=1e-12,atol=0)
        assert all(checks.values()),[k for k,v in checks.items() if not v]
        variants=[('ratio','groups'),('ccf','groups'),('ratio','peaks'),('ccf','peaks')]
        variants=variants[count%4:]+variants[:count%4]
        for kind,backend in variants:
            result=solve(y,z['frequency'],z['stations'],kind,backend,old['particle_seed'])
            saved_initial=np.array(rec['output']['steps'][0]['points'])
            checks[p.stem+'/'+kind+'_'+backend+'_same_initial']=result['initial_hash']==hashlib.sha256(saved_initial.tobytes()).hexdigest()
            if kind=='ratio' and backend=='groups':
                prior=read(ROOT/'repair_20261027_184715_930592'/p.name)
                checks[p.stem+'/frozen_repair_reproduction']=result['positions']==prior['outputs']['repair']['positions']
            row=dict(case=p.stem,realization=i,G=rec['paths'],SNR=rec['snr_db'],SIR=rec['sir_db'],version=kind+'_'+backend,
                     evaluation=evaluate(result['positions'],z['truth']),result=result)
            rows.append(row);write(out/(p.stem+'_'+kind+'_'+backend+'.json'),row)
        assert all(checks.values()),[k for k,v in checks.items() if not v]
        print(f'有界路线筛查 {count+1}/28',flush=True)
    for p,h in hashes.items():checks['input_end/'+p]=sha(Path(p))==h
    for p,h in code_hash.items():checks['code_end/'+p]=sha(Path(p))==h
    assert all(checks.values())
    summaries=[]
    for scope in ['all','snr5_20']:
        for version in ['ratio_groups','ccf_groups','ratio_peaks','ccf_peaks']:
            sub=[r for r in rows if r['version']==version and (scope=='all' or r['SNR']>=5)]
            good=[r for r in sub if r['evaluation']['complete']]
            errors=np.array([r['evaluation']['error_m'] for r in good])
            rmse=float(np.sqrt(np.mean(errors**2))) if len(errors) else None
            summaries.append(dict(scope=scope,version=version,n=len(sub),complete=len(good),both20=sum(r['evaluation']['both20'] for r in sub),
                conditional_RMSE=rmse,mean_seconds=float(np.mean([r['result']['seconds'] for r in sub])),
                screen_pass=bool(scope=='snr5_20' and len(good)>=.9*len(sub) and rmse is not None and rmse<=100 and sum(r['evaluation']['both20'] for r in sub)>=.5*len(sub))))
    write(out/'result.json',dict(rows=rows,summary=summaries,seconds=time.perf_counter()-start))
    write(out/'audit.json',dict(passed=True,count=len(checks),checks={k:bool(v) for k,v in checks.items()}))
    with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    print(summaries);print('REPORT',out,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
