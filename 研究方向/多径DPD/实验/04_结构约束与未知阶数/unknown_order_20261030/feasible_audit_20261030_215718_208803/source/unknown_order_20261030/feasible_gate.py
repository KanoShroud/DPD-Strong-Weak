"""Audit terminal changes, replay affected IQ, then gated unseen-order check."""
import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits
import cross_order as engine
import reference as ref
from reference_feasible import finish, fit

H=ref.HERE


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def latest(name):
    return Path(read(H/(name+'_latest.json'))['path'])


def folder(stage, extra):
    old=latest('cross')
    known=read(old/'preflight.json')['hashes']
    assert all(ref.sha(p)==v for p,v in known.items())
    paths=[Path(p) for p in known]+[Path(__file__),H/'reference_feasible.py',H/'可行终态批准协议.md']+extra
    for mod in tuple(sys.modules.values()):
        name=getattr(mod,'__file__',None)
        if name and Path(name).resolve().is_relative_to(ref.BASE) and Path(name).suffix=='.py':
            paths.append(Path(name).resolve())
    hashes={str(p.resolve(strict=True)):ref.sha(p) for p in paths}
    assert all(Path(p).is_relative_to(ref.BASE) for p in hashes)
    out=H/(stage+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    for name in hashes:
        p=Path(name);dest=out/'source'/p.relative_to(ref.BASE)
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    ref.write(out/'preflight.json',dict(hashes=hashes,old_cross=str(old)))
    print('输出',out,flush=True)
    return out,hashes


def identity(hashes):
    return all(ref.sha(p)==v for p,v in hashes.items())


def spectral(iq):
    n=iq.shape[1];f=np.fft.fftfreq(n,1/ref.FS);use=abs(f)<=ref.B/2
    return f[use],np.fft.fft(iq,axis=1,norm='ortho')[:,use].T,n


def audit():
    old=latest('cross');data=read(old/'result.json')
    extra=[old/'result.json']+[old/(r['tag']+'_input.npz') for r in data['rows']]
    out,hashes=folder('feasible_audit',extra)
    checks={};rows=[];begin=time.perf_counter()
    for row in data['rows']:
        with np.load(old/(row['tag']+'_input.npz'),allow_pickle=False) as inp: iq=inp['iq']
        checks[row['tag']+'/iq']=hashlib.sha256(np.ascontiguousarray(iq).tobytes()).hexdigest()==row['iq_sha']
        f,y,n=spectral(iq);paths=[];previous=np.array(row['los']['x']);events=[]
        for ev in row['history']:
            if 'fit' not in ev: break
            s=ev['fit'];pick=ev['pick'];paths.append((pick['m'],pick['q']))
            x0=np.r_[previous,pick['rho'],pick['delay']]
            raw=s.get('pg_finish',{}).get('before',s)
            x,new=finish(np.array(raw['x']),raw,x0,f,y,paths,n)
            same=np.array_equal(x,np.array(s['x'])) and new['stationary']==s['stationary']
            checks[row['tag']+f"/{ev['step']}/budget"]=new['iterations']<=300
            checks[row['tag']+f"/{ev['step']}/finite"]=bool(np.isfinite(x).all())
            events.append(dict(step=ev['step'],same=bool(same),delta=float(np.linalg.norm(x-np.array(s['x']))),new=new))
            previous=x
            if not same: break
        changed=any(not e['same'] for e in events)
        rows.append(dict(tag=row['tag'],changed=changed,events=events))
        print(row['tag'],'重跑' if changed else '等价复用',flush=True)
    checks['identity']=identity(hashes)
    ref.write(out/'result.json',dict(rows=rows,checks=checks,all_checks_pass=all(checks.values()),seconds=time.perf_counter()-begin))
    ref.write(H/'feasible_audit_latest.json',dict(path=str(out)))


def evaluate_row(row,p):
    row['geometry_errors_m']=ref.error(row['initial'],p)
    row['los_errors_m']=ref.error(row['los']['p'],p)
    row['errors_m']=ref.error(row['final']['p'],p)
    for ev in row['history']:
        if 'fit' in ev: ev['errors_m']=ref.error(ev['fit']['p'],p)


def execute(unseen=False):
    audit_run=latest('feasible_audit');audit_data=read(audit_run/'result.json')
    assert audit_data['all_checks_pass'] and identity(read(audit_run/'preflight.json')['hashes'])
    old=latest('cross');old_data=read(old/'result.json')
    extra=[audit_run/'result.json',old/'result.json']
    if unseen:
        repair_run=latest('feasible_repair');d=read(repair_run/'result.json')
        assert d['all_checks_pass'] and all(r['stop']=='no_detected_increment' for r in d['rows'])
        gate=read(H/'可行版未见阶数放行.json');assert gate['approved_to_run']
        assert gate['solver_sha']==ref.sha(H/'reference_feasible.py')
        extra += [repair_run/'result.json',H/'可行版未见阶数放行.json']
    else:
        extra += [old/(r['tag']+'_input.npz') for r in old_data['rows']]
        extra += [old/f'L{li}_generation.npz' for li in (20,40)]
    out,hashes=folder('feasible_unseen' if unseen else 'feasible_repair',extra)
    engine.fit=fit  # Explicit backend injection; frontend/search/generator remain frozen.
    changed={r['tag'] for r in audit_data['rows'] if r['changed']}
    rows=[];checks={};begin=time.perf_counter()
    todo=[dict(tag=f'L{li}_L3',layout=li,condition='L3') for li in (20,40)] if unseen else old_data['rows']
    for oldrow in todo:
        tag=oldrow['tag'];li=oldrow['layout']
        if not unseen and tag not in changed:
            row=copy.deepcopy(oldrow);row.update(reused=True,timing_basis='historical_pg_full_pipeline_excludes_new_acceptance')
            checks.update({tag+'/'+k:v for k,v in row['checks'].items()})
            rows.append(row);ref.write(out/(tag+'.json'),row);continue
        with np.load(old/f'L{li}_generation.npz',allow_pickle=False) as gen: p=gen['p'].copy()
        if unseen:
            bank=engine.generate_bank(p,li,True);iq,physical=engine.observation(bank,'L3')
            np.savez_compressed(out/f'L{li}_generation.npz',**bank)
            np.savez_compressed(out/(tag+'_input.npz'),iq=iq,**physical)
            checks[tag+'/generation']=bool(np.max(abs(physical['relative_coefficients']))<1 and np.all(physical['counts']==3))
        else:
            with np.load(old/(tag+'_input.npz'),allow_pickle=False) as inp: iq=inp['iq'].copy()
        iqsha=hashlib.sha256(np.ascontiguousarray(iq).tobytes()).hexdigest()
        if not unseen: checks[tag+'/identity']=iqsha==oldrow['iq_sha']
        row=engine.solve_iq(iq,out,tag)
        checks[tag+'/iq_unchanged']=iqsha==hashlib.sha256(np.ascontiguousarray(iq).tobytes()).hexdigest()
        evaluate_row(row,p)  # Evaluation only after all blind decisions.
        row.update(tag=tag,layout=li,condition=oldrow['condition'],iq_sha=iqsha,reused=False,timing_basis='new_full_pipeline')
        checks.update({tag+'/'+k:v for k,v in row['checks'].items()})
        rows.append(row);ref.write(out/(tag+'.json'),row)
        print(tag,row['stop'],'误差',row['errors_m'],'分量',len(row['paths']),flush=True)
    checks['identity_unchanged']=identity(hashes)
    ref.write(out/'result.json',dict(rows=rows,checks=checks,all_checks_pass=all(checks.values()),seconds=time.perf_counter()-begin,
                                   historical_source=str(old),reused=sum(r['reused'] for r in rows)))
    ref.write(H/('feasible_unseen_latest.json' if unseen else 'feasible_repair_latest.json'),dict(path=str(out)))
    print('完成',all(checks.values()),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['audit','repair','unseen']);args=parser.parse_args()
    with threadpool_limits(limits=2):
        if args.stage=='audit': audit()
        else: execute(args.stage=='unseen')
