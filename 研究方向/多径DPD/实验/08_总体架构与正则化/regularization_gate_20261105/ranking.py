"""Fixed saved-interpretation ranking; explicitly oracle-assisted diagnosis."""
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from discrepancy import HERE, BASE, INPUT, cfg, ad, stable
from conditional import calibrate


def main():
    out=HERE/'ranking_v1';out.mkdir(exist_ok=False)
    source=BASE/'source_path_gate_20261105/run_v1'
    stable.inside.rr.check_hashes(cfg.read(source/'identity.json'))
    stable.inside.rr.check_hashes(cfg.read(HERE/'discrepancy_v1/identity.json'))
    hashes=ad.source_hashes()
    for p in [Path(__file__).resolve(),HERE/'ranking_protocol.md',HERE/'conditional.py']:
        hashes[str(p)]=cfg.sha(p)
    cases=[d for d in cfg.read(HERE/'discrepancy_v1/blind_decisions.json') if d['condition'] in ['L2_P075','L3_P150']]
    for d in cases:
        for arm in ['GEOMETRY_FIRST','DIAGNOSTIC_NEAREST']:
            p=source/(d['tag']+'_'+arm+'.json');hashes[str(p)]=cfg.sha(p)
        p=INPUT/'cases'/d['tag']/'iq.npz';hashes[str(p)]=cfg.sha(p)
    cfg.write(out/'identity.json',hashes)
    rows=[]
    for d in cases:
        tag=d['tag'];print('START',tag,flush=True)
        with np.load(INPUT/'cases'/tag/'iq.npz',allow_pickle=False) as z: iq=z['iq']
        f,y,n,_=ad.b.fc.cal.spectral(iq)
        row=dict(tag=tag,radius2=d['radius2'],diagnostic_only=True,arms={})
        for arm in ['GEOMETRY_FIRST','DIAGNOSTIC_NEAREST']:
            saved=cfg.read(source/(tag+'_'+arm+'.json'))
            assert saved['iq_sha']==cfg.digest(iq)
            r=saved['result'];x=np.asarray(r['final']['x']);paths=r['paths'];e=stable.Engine()
            assert np.max(abs(x-e.ref.project(x,n)))<1e-7
            h,_,s0,res,*_=e.core(x,f,y,paths)
            g=np.ones((4,2),complex);g[1:]=(x[4:10]+1j*x[10:16]).reshape(3,2)
            c=np.zeros(2)
            for k,(m,q) in enumerate(paths): c[q]+=abs(g[m,q])*np.linalg.norm(x[16+3*k:18+3*k])
            start=time.perf_counter();s,info=calibrate(h,y,c,d['radius2'])
            row['arms'][arm]=dict(conditional=info,seconds=time.perf_counter()-start,p=r['final']['p'],
                old_rss=float(np.linalg.norm(res)**2),old_penalty=float(c@np.linalg.norm(s0,axis=0)),
                source_norms=np.linalg.norm(s,axis=0),K=len(paths),old_normal=r['normal'],old_stop=r['stop'])
        rows.append(row);cfg.write(out/(tag+'_scores.json'),row);cfg.write(out/'scores.json',rows)
    # Evaluation is deliberately after saving all scores.
    evaluated=[]
    for row in rows:
        layout=row['tag'].split('_')[1]
        with np.load(INPUT/'banks'/(layout+'_00.npz'),allow_pickle=False) as z: truth=z['p']
        for arm in row['arms'].values(): arm['evaluation']=stable.inside.rr.evaluation(np.asarray(arm['p']),truth)
        a=row['arms']['GEOMETRY_FIRST'];b=row['arms']['DIAGNOSTIC_NEAREST']
        ca,cb=a['conditional'],b['conditional']
        if ca['passed'] and cb['passed']:
            v1,v2=ca['value'],cb['value']
            relative=(v1-v2)/max(abs(v1),abs(v2),np.finfo(float).tiny)
            row['ranking']='tie' if abs(relative)<=1e-5 else ('GEOMETRY_FIRST' if relative<0 else 'DIAGNOSTIC_NEAREST')
            row['comparison']='both_feasible';row['relative_value_gap']=relative
        elif ca['passed'] or cb['passed']:
            row['ranking']='GEOMETRY_FIRST' if ca['passed'] else 'DIAGNOSTIC_NEAREST'
            row['comparison']='one_feasible'
        else:
            row['ranking']=None;row['comparison']='neither_certified'
        evaluated.append(row)
    cfg.write(out/'evaluation.json',evaluated)
    stable.inside.rr.check_hashes(hashes)
    cfg.write(out/'summary.json',dict(completed=len(rows),diagnostic_only=True,new_positions=0,
        comparison_counts={k:sum(r['comparison']==k for r in evaluated) for k in sorted({r['comparison'] for r in evaluated})},
        identity_unchanged=True))
    print('DONE',len(rows),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2): main()
