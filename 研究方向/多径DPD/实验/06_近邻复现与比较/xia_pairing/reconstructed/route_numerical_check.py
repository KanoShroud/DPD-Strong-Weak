"""Bounded derivative-free check after exact-station gradient aborts.

Same frozen observations, seeds, score, exclusion and box. This is an explicit
optimizer change, not a silent repair or a reproduction of Xia's algorithm.
"""
from datetime import datetime
from pathlib import Path
import hashlib
import time
import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
from route_screen import ROOT, INPUT, OLD, IDS, Scorer, seeds_from_points
from pilot import observe, evaluate, sha, write
from trace_audit import read


def main():
    start=time.perf_counter()
    out=ROOT/('numerical_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    files=sorted(p for p in INPUT.glob('r*_G*.json') if read(p)['realization'] in IDS)
    sources=files+[INPUT/f'input_{i:02d}.npz' for i in IDS]+[Path(__file__),ROOT/'route_screen.py',ROOT/'pilot.py']
    hashes={str(p):sha(p) for p in sources}
    write(out/'protocol.json',dict(reason='Exact station coordinate makes norm gradient undefined; previous continuous screen is not an adequate converged baseline',
        cases=28,variants=['ratio_powell','ccf_powell'],hashes=hashes,
        optimizer=dict(method='Powell',maxiter=80,maxfev=10000,xtol=1e-7,ftol=1e-10),
        fixed='Same IQ, same 1000 points, six seeds and 400m exclusion, same spatial score and box; no truth in solver',
        stop='One numerical follow-up only; no tuning after outcomes. No novelty claim for conventional peak extraction.'))
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    rows=[];checks={}
    for j,p in enumerate(files):
        rec=read(p);z=dict(np.load(INPUT/f"input_{rec['realization']:02d}.npz"))
        y,_,_=observe(z,rec['snr_db'],rec['sir_db'],rec['paths'])
        checks[p.stem+'/IQ']=hashlib.sha256(y.tobytes()).hexdigest()==rec['observation_sha']
        points=np.random.default_rng(read(OLD/p.name)['particle_seed']).uniform(-3000,3000,(1000,2))
        checks[p.stem+'/initial']=np.array_equal(points,rec['output']['steps'][0]['points'])
        for kind in (['ratio','ccf'] if j%2==0 else ['ccf','ratio']):
            t=time.perf_counter();s=Scorer(y,z['frequency'],z['stations'],kind)
            values=s.matrix(points);indices=seeds_from_points(points,values,6)
            def objective(x):
                return float(-2*np.log(max(s.matrix(np.asarray(x)[None]*1000)[0],np.finfo(float).tiny)))
            # Cost remains finite even when a trial touches a station coordinate.
            checks[p.stem+'/'+kind+'/stations']=all(np.isfinite(objective(a/1000)) for a in z['stations'])
            refinements=[]
            for i in indices:
                opt=minimize(objective,points[i]/1000,method='Powell',bounds=[(-3.,3.)]*2,
                             options=dict(maxiter=80,maxfev=10000,xtol=1e-7,ftol=1e-10))
                refinements.append(dict(position=(opt.x*1000).tolist(),loss=float(opt.fun),success=bool(opt.success),
                    iterations=int(opt.nit),status=str(opt.message)))
            selected=[]
            for r in sorted(refinements,key=lambda r:r['loss']):
                if np.isfinite(r['loss']) and all(np.linalg.norm(np.array(r['position'])-q)>=400 for q in selected):
                    selected.append(r['position'])
                if len(selected)==2:break
            pred=selected if len(selected)==2 else None
            row=dict(case=p.stem,version=kind+'_powell',SNR=rec['snr_db'],evaluation=evaluate(pred,z['truth']),
                result=dict(positions=pred,seconds=time.perf_counter()-t,refinements=refinements,points=s.points,calls=s.calls))
            write(out/(p.stem+'_'+kind+'.json'),row);rows.append(row)
        assert all(checks.values())
        print(f'无梯度核验 {j+1}/28',flush=True)
    checks.update({'hash/'+p:sha(Path(p))==h for p,h in hashes.items()})
    assert all(checks.values())
    summary=[]
    for version in ['ratio_powell','ccf_powell']:
        for scope in ['all','snr5_20']:
            sub=[r for r in rows if r['version']==version and (scope=='all' or r['SNR']>=5)]
            good=[r for r in sub if r['evaluation']['complete']]
            summary.append(dict(version=version,scope=scope,n=len(sub),complete=len(good),
                both20=sum(r['evaluation']['both20'] for r in sub),
                conditional_RMSE=float(np.sqrt(np.mean([np.array(r['evaluation']['error_m'])**2 for r in good]))) if good else None,
                nonconverged=sum(not x['success'] for r in sub for x in r['result']['refinements']),
                mean_seconds=float(np.mean([r['result']['seconds'] for r in sub]))))
    write(out/'result.json',dict(rows=rows,summary=summary,seconds=time.perf_counter()-start))
    write(out/'audit.json',dict(passed=all(checks.values()),checks=checks))
    print(summary);print('REPORT',out,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
