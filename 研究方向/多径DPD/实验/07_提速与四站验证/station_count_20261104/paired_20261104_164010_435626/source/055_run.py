"""Four/five station numerical gates and complete-IQ paired observations."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits
import backend as b

HERE=Path(__file__).resolve().parent
fc,ref,rg=b.fc,b.ref,b.rg
OLD=HERE.parent/'acceptance_gate_20261104'


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def kernel(out,engines):
    checks={};rng=np.random.default_rng(2026110404)
    iq=rg.inputs('L2_R00')
    for name,engine in engines.items():
        f,y,n,sigma=fc.cal.spectral(iq[engine.ids])
        x=np.r_[np.array([-900.,-500.,1100.,700.])/ref.UNIT,
                np.ones(engine.G),np.full(engine.G,.1), .1,.05,3.3,-.2,.1,7.2,.13,-.04,15.1]
        paths=[(0,0),(1,1),(3,0)]
        ff,yy=f[::59],y[::59]
        r,j=engine.evaluate(x,ff,yy,paths)
        cost,grad=engine.objective(x,ff,yy,paths)
        eps=1e-5
        fd=np.column_stack([(engine.evaluate(x+eps*np.eye(len(x))[k],ff,yy,paths)[0]-
                            engine.evaluate(x-eps*np.eye(len(x))[k],ff,yy,paths)[0])/(2*eps) for k in range(len(x))])
        checks[name+'/Jacobian_FD']=bool(np.linalg.norm(fd-j)/np.linalg.norm(j)<3e-6)
        checks[name+'/gradient']=bool(np.allclose(grad,j.T@r,rtol=1e-8,atol=1e-10))
        checks[name+'/cost']=bool(abs(cost-.5*r@r)<1e-12)
        checks[name+'/dimension']=j.shape==(len(ff)*engine.M*2,engine.D+9)
        data=engine.core(x,ff,yy,paths)
        h=data[0];coef=(np.linalg.pinv(h,rcond=1e-12)@yy[:,:,None])[:,:,0]
        residual=yy-(h@coef[:,:,None])[:,:,0]
        checks[name+'/svd_residual']=bool(np.allclose(residual,data[3],rtol=1e-8,atol=1e-10))
        z=rng.normal(size=len(x))*5
        p=engine.ref.project(z,n);q=engine.ref.project(rng.normal(size=len(x))*5,n)
        checks[name+'/convex_projection']=bool((z-p)@(q-p)<1e-8)
        use=abs(np.fft.fftfreq(n,1/ref.FS))<=ref.B/2
        _,bank,meta=engine.search(x,paths,f,y,n,use,sigma,1)
        checks[name+'/FFT']=all(meta['checks'].values())
        checks[name+'/search_count']=len(bank)==engine.M*2*(n-1)
        fc.write(out/(name+'_instantiation.json'),dict(M=engine.M,ids=engine.ids,reference=engine.reference,
            sources={key:value.generated_source for key,value in vars(engine).items() if hasattr(value,'generated_source')}))
        if name=='FIVE_ORIGINAL':
            rr,jj=ref.evaluate(x,ff,yy,paths)
            checks['five_original_r']=bool(np.array_equal(r,b.old.s.evaluate(x,ff,yy,paths)[0]))
            checks['five_original_j']=bool(np.allclose(j,jj,rtol=1e-8,atol=1e-10))
        print(name,'KERNEL',all(checks.values()),flush=True)
    # Gauge change preserves the same response column spans and scalar likelihood.
    a,c=engines['FIVE_ORIGINAL'],engines['FIVE_COMMON']
    f,y,_,_=fc.cal.spectral(iq)
    x=np.r_[np.array([-900.,-500.,1100.,700.])/ref.UNIT,rng.normal(size=8),rng.normal(size=8)]
    gg=np.ones((5,2),complex);gg[a.free]=(x[4:12]+1j*x[12:20]).reshape(4,2)
    gg=gg/gg[1]
    xc=np.r_[x[:4],gg[c.free].real.ravel(),gg[c.free].imag.ravel()]
    checks['gauge_cost']=abs(a.objective(x,f,y,[])[0]-c.objective(xc,f,y,[])[0])<1e-12
    fc.write(out/'kernel.json',dict(checks=checks,all_pass=all(checks.values())))
    assert all(checks.values()),{k:v for k,v in checks.items() if not v}


def audit(result,iq,engine):
    f,y,n,_=fc.cal.spectral(iq)
    checks={}
    for k,(state,paths) in enumerate([(result['los'],[])]+[(v['fit'],result['paths'][:v['K']]) for v in result['history'] if 'fit' in v]):
        x=np.array(state['x']);r,j=engine.evaluate(x,f,y,paths)
        pg=float(np.max(abs(x-engine.ref.project(x-j.T@r,n))))
        checks[f'{k}/cost']=abs(.5*r@r-state['cost'])<1e-10
        checks[f'{k}/PG']=abs(pg-state['projected_gradient'])<1e-9
        checks[f'{k}/feasible']=np.linalg.norm(x-engine.ref.project(x,n))<1e-7
        checks[f'{k}/state_stationary_honest']=bool(state['stationary'])==bool(pg<=1e-7)
    checks['decrease']=all(v.get('decrease',True) for v in result['history'])
    return checks


def run(out,args,engines):
    tags=args.tags or (['L2_R00','L3_R00','L3_R05'] if args.mode=='pilot' else [f'{c}_R{i:02d}' for i in range(1,8) for c in ('L2','L3')])
    rows=[];checks={}
    contract=fc.load(rg.FORMAL/'contract.json')['contract']
    for i,tag in enumerate(tags):
        iq=rg.inputs(tag)
        methods=['FIVE_ORIGINAL','FIVE_COMMON','FOUR_COMMON'] if args.mode=='pilot' else ['FIVE_COMMON','FOUR_COMMON']
        if i%2:methods=methods[::-1]
        for name in methods:
            engine=engines[name];selected=iq[engine.ids].copy()
            start=time.perf_counter()
            print('START',tag,name,flush=True)
            try:
                result=engine.estimate(selected)
            except (ValueError,np.linalg.LinAlgError,ArithmeticError) as exc:
                row=dict(case=tag,method=name,status='exception',error=repr(exc),seconds=time.perf_counter()-start)
                rows.append(row);fc.write(out/'results.json',dict(rows=rows,checks=checks,complete=False));print(row,flush=True)
                continue
            fc.write(out/f'{tag}_{name}.json',result)
            ck=audit(result,selected,engine)
            if name=='FIVE_ORIGINAL':
                folder='pilot_20261104_160749_988104' if tag.endswith('R00') else 'paired_20261104_160907_648761'
                saved=read(OLD/folder/(tag+'_GATED.json'))
                ck['frozen_final_x']=bool(np.array_equal(result['final']['x'],saved['final']['x']))
                ck['frozen_paths']=result['paths']==[tuple(p) for p in saved['paths']]
            checks.update({tag+'/'+name+'/'+k:bool(v) for k,v in ck.items()})
            with np.load(contract['conditions'][tag.split('_')[0]]['bank'],allow_pickle=False) as z:truth=z['p']
            errors=np.array(ref.error(result['final']['p'],truth))
            initial_errors=np.array(ref.error(result['los']['p'],truth))
            row=dict(case=tag,method=name,status='returned',seconds=result['seconds'],normal=result['normal'],stop=result['stop'],
                errors=errors,RMSE=float(np.sqrt(np.mean(errors**2))),LOS_RMSE=float(np.sqrt(np.mean(initial_errors**2))),
                K=len(result['paths']),PG=result['final']['projected_gradient'],J=result['final']['cost'],
                frontend=result['frontend_seconds'],fit=sum(v['fit']['seconds'] for v in result['history'] if 'fit' in v),
                search=sum(v['search']['seconds'] for v in result['history']),audit=all(ck.values()))
            rows.append(row);fc.write(out/'results.json',dict(rows=rows,checks=checks,complete=False));print(row,flush=True)
            assert all(ck.values()),ck
    fc.write(out/'results.json',dict(rows=rows,checks=checks,complete=True))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['kernel','pilot','paired'],required=True);ap.add_argument('--tags',nargs='+')
    args=ap.parse_args()
    tags=args.tags or (['L2_R00'] if args.mode=='kernel' else ['L2_R00','L3_R00','L3_R05'] if args.mode=='pilot' else [f'{c}_R{i:02d}' for i in range(1,8) for c in ('L2','L3')])
    out=HERE/(args.mode+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir(exist_ok=False);print('结果目录',out,flush=True)
    engines=dict(FIVE_ORIGINAL=b.Backend(range(5),0),FIVE_COMMON=b.Backend(range(5),1),FOUR_COMMON=b.Backend([1,2,3,4],0))
    hashes=rg.preflight(out,tags)
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.md'))+list(OLD.glob('*.py'))
    files += [Path(fc.load(rg.FORMAL/'contract.json')['contract']['conditions'][c]['bank']) for c in {t.split('_')[0] for t in tags}]
    for tag in tags:
        folder='pilot_20261104_160749_988104' if tag.endswith('R00') else 'paired_20261104_160907_648761'
        p=OLD/folder/(tag+'_GATED.json')
        if p.exists():files.append(p)
    hashes.update({str(p.resolve(strict=True)):ref.sha(p) for p in files})
    snap=out/'source';snap.mkdir()
    for i,p in enumerate(hashes):
        if Path(p).suffix in ('.py','.md'):shutil.copy2(p,snap/f'{i:03d}_{Path(p).name}')
    fc.write(out/'identity.json',dict(hashes=hashes,command=sys.argv))
    try:
        if args.mode=='kernel':kernel(out,engines)
        else:run(out,args,engines)
    finally:
        same=all(ref.sha(p)==v for p,v in hashes.items());fc.write(out/'postflight.json',dict(unchanged=same));assert same


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
