"""Oracle component attribution ONLY; fixed observation-derived projectors."""
from pathlib import Path
from datetime import datetime
import argparse
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
from structural_probe import ROOT, STATIONS, C, response, sha, write
from alternative_fit import SOURCE
from blind_initialization import GRID, rank_one, peaks, score_positions, signature

OLD = ROOT/'blind_init_20261028_200514_640818'
NAMES = ['actual', 'target_only', 'projected_target', 'drop_first_reflection',
         'projected_target_LOS', 'restore_target']


def projector(data, f, p, g):
    phase = np.exp(-2j*np.pi*f[:, None]*np.linalg.norm(p[None]-STATIONS, axis=1)[None]/C)
    h = phase*g[None]
    return h*np.sum(h.conj()*data, axis=1)[:, None]/np.sum(abs(h)**2, axis=1)[:, None]


def multi_score(data, f, points):
    result = np.empty((len(data), len(points)))
    for start in range(0, len(points), 128):
        pts = points[start:start+128]
        dist = np.linalg.norm(pts[:, None]-STATIONS[None], axis=2)
        phase = np.exp(2j*np.pi*f[None, :, None]*dist[:, None]/C)
        z = phase[:, None]*data[None]
        cov = np.einsum('bvf m,bvf n->bvmn', z, z.conj(), optimize=True)
        result[:, start:start+len(pts)] = np.linalg.eigvalsh(cov)[:, :, -1].T.real
    return result


def score_breakdown(components, f, p):
    phase = np.exp(2j*np.pi*f[:, None]*np.linalg.norm(p[None]-STATIONS, axis=1)[None]/C)
    z = components*phase[None]
    total = z.sum(axis=0)
    eig, vectors = np.linalg.eigh(total.T@total.conj())
    w = vectors[:, -1]
    projected = np.einsum('cfm,m->cf', z, w.conj())
    gram = (projected.conj()@projected.T).real
    return dict(score=float(eig[-1]), diagonal=np.diag(gram),
                cross=2*np.triu(gram, 1), reconstructed=float(gram.sum()))


def selftest():
    rng = np.random.default_rng(20261028031)
    f = np.linspace(-6e6, 6e6, 25)
    data = rng.normal(size=(4,25,5))+1j*rng.normal(size=(4,25,5))
    p = np.array([123., 789.])
    g, _, fit = rank_one(data.sum(axis=0), f, p)
    applied = np.array([projector(c, f, p, g) for c in data])
    b = score_breakdown(data, f, p)
    s = multi_score(data, f, np.array([p, p+[20.,20.]]))
    checks = dict(linearity=np.allclose(applied.sum(axis=0), fit, atol=1e-11),
        idempotence=np.allclose(projector(fit, f, p, g), fit, atol=1e-11),
        energy=all(np.isclose(np.linalg.norm(c)**2, np.linalg.norm(pc)**2+np.linalg.norm(c-pc)**2) for c,pc in zip(data,applied)),
        score_cross=np.isclose(b['score'], b['reconstructed'], rtol=1e-12),
        batched=np.allclose(s,np.array([score_positions(v,f,np.array([p,p+[20.,20.]])) for v in data]),rtol=1e-12))
    assert all(checks.values()), checks
    return checks


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--test-only',action='store_true')
    args=parser.parse_args()
    checks=selftest()
    if args.test_only:
        print(checks)
        return
    start=time.perf_counter()
    old=json.loads((OLD/'result.json').read_text(encoding='utf-8'))
    protocol=json.loads((OLD/'protocol.json').read_text(encoding='utf-8'))
    assert all(sha(p)==h for p,h in protocol['hashes'].items())
    frozen=SOURCE/'inputs.npz'
    assert sha(frozen)==json.loads((SOURCE/'result.json').read_text(encoding='utf-8'))['input_sha']
    files=[Path(__file__),OLD/'result.json',frozen,ROOT/'blind_initialization.py',ROOT/'structural_probe.py',
           *[OLD/(r['case']+'_input.npz') for r in old['rows']]]
    hashes={str(p):sha(p) for p in files}
    out=ROOT/('attribution_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    (out/'source').mkdir()
    for p in files:
        if p.suffix=='.py':(out/'source'/p.name).write_bytes(p.read_bytes())
    write(out/'protocol.json',dict(kind='SAVED_OBSERVATION_COMPONENT_DIAGNOSTIC_NOT_ALGORITHM',
        hashes=hashes, variants=NAMES, cases=4, first_candidates_per_case=3,
        intervention='Fixed p and g from original observation; apply same linear orthogonal projector to every true component. No re-estimation under intervention.',
        attribution='Truth labels nearest first-source after fitting; true components only for counterfactual diagnosis; cross terms explicitly retained.',
        stopping='12 predetermined projection cases, 6 score maps each; no position optimization or tuning.',
        evaluation='Same full grid and 100m exclusion; report exact-truth score separately to detect grid coverage effects. No statistical generalization.'))
    raw=dict(np.load(frozen))
    rows=[]
    for row in old['rows']:
        case=row['case'];layout=int(case[1]);truth=raw['layouts'][layout]
        obs=np.load(OLD/(case+'_input.npz'));y,f=obs['y'],obs['f']
        assert signature(y,f)==row['input_signature']
        full_y=np.load(OLD/(f'L{layout}_P2_input.npz'))['y']
        gain=raw[f'layout{layout}_g'];delay=raw[f'layout{layout}_delta']
        h,phase=response(truth,delay,gain,f)
        source=np.einsum('fqm,fm->fq',np.linalg.pinv(h),full_y)
        comp=[]
        for q in range(2):
            los=phase[:,:,q,0]*gain[None,:,q,0]*source[:,q,None]
            ref=np.sum(phase[:,:,q,1:]*gain[None,:,q,1:],axis=2)*source[:,q,None]
            if case.endswith('P0'):ref=np.zeros_like(ref)
            comp.extend([los,ref])
        comp=np.array(comp)
        checks[case+'/reconstruct']=np.linalg.norm(comp.sum(axis=0)-y)/np.linalg.norm(y)<1e-12
        for first in range(3):
            tag=f'{case}_first{first}'
            p=np.array(next(t for t in row['pairs'] if t['id']==f'{first}_0')['coarse'][0])
            g,_,fit=rank_one(y,f,p)
            which=int(np.argmin(np.linalg.norm(truth-p,axis=1)));target=1-which
            removed=np.array([projector(c,f,p,g) for c in comp]);remain=comp-removed
            target_signal=comp[2*target:2*target+2].sum(axis=0)
            projected_target=remain[2*target:2*target+2].sum(axis=0)
            actual=y-fit
            variants=np.array([actual,target_signal,projected_target,
                actual-remain[2*which+1],remain[2*target],
                actual+removed[2*target:2*target+2].sum(axis=0)])
            checks[tag+'/projector_replay']=np.allclose(remain.sum(axis=0),actual,rtol=1e-11,atol=1e-11)
            print('MAPS',tag,flush=True)
            scores=multi_score(variants,f,GRID)
            truth_scores=multi_score(variants,f,truth[target:target+1])[:,0]
            np.savez_compressed(out/(tag+'_maps.npz'),scores=scores,points=GRID,truth=truth,first=p)
            details=[]
            for k,name in enumerate(NAMES):
                top=np.array(peaks(scores[k],exclude=[p]))
                near=np.linalg.norm(GRID-truth[target],axis=1)<=20
                allowed=np.linalg.norm(GRID-p,axis=1)>=100
                validnear=near&allowed
                near_index=int(np.argmax(np.where(validnear,scores[k],-np.inf)))
                details.append(dict(variant=name,candidates=top,
                    min_candidate_error_m=float(np.min(np.linalg.norm(top-truth[target],axis=1))),
                    truth_score=float(truth_scores[k]),max_grid_score=float(np.max(scores[k,allowed])),
                    near20_score=float(scores[k,near_index]),near20_grid=GRID[near_index],
                    near20_point_rank=1+int(np.sum(scores[k,allowed]>scores[k,near_index])),
                    truth_vs_max_ratio=float(truth_scores[k]/np.max(scores[k,allowed])),
                    energy=float(np.linalg.norm(variants[k])**2)))
            oldsecond=np.array([t['coarse'][1] for t in row['pairs'] if t['id'].startswith(str(first)+'_')])
            checks[tag+'/same_candidates']=np.array_equal(oldsecond,np.array(details[0]['candidates']))
            breakdown={}
            for name,point in [('truth',truth[target]),('actual_top1',np.array(details[0]['candidates'])[0])]:
                b=score_breakdown(remain,f,point)
                checks[tag+'/'+name+'/cross_terms']=np.isclose(b['score'],b['reconstructed'],rtol=1e-11,atol=1e-11)
                breakdown[name]=b
            losses=[]
            for c,pc in zip(comp,removed):
                e=np.linalg.norm(c)**2
                losses.append(None if e==0 else float(np.linalg.norm(pc)**2/e))
            combined_loss=float(np.linalg.norm(target_signal-projected_target)**2/np.linalg.norm(target_signal)**2)
            record=dict(case=case,first=first,first_position=p,assigned_first_source=which,target_source=target,
                first_error_m=float(np.linalg.norm(p-truth[which])),component_order=['A_LOS','A_reflections','B_LOS','B_reflections'],
                removed_energy_fraction=losses,target_total_removed_fraction=combined_loss,
                score_variants=details,score_breakdown=breakdown)
            rows.append(record);write(out/(tag+'.json'),record)
            print('DONE',tag,'targetloss',round(combined_loss,4),'candidate errors',[(x['variant'],round(x['min_candidate_error_m'],2)) for x in details],flush=True)
    checks.update({'hash/'+p:sha(p)==h for p,h in hashes.items()})
    write(out/'result.json',dict(rows=rows,checks=checks,passed=bool(all(checks.values())),seconds=time.perf_counter()-start))
    print('REPORT',out,flush=True)
    assert all(checks.values()),[k for k,v in checks.items() if not v]


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
