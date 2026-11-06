"""Fixed wrong-position explanations in the ideal frequency model.

This is a truth-assisted diagnostic, not blind localization. Failure to find
an alternative is not a lower bound on the achievable residual.
"""
from datetime import datetime
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
from structural_probe import ROOT, STATIONS, LAYOUTS, B, SEED, wave, response, write, sha
from numpy.polynomial.legendre import leggauss

SOURCE=ROOT/'structure_20261027_202324_816706'


def encode(g,d):
    g=g/g[:1,:,0,None]
    los=g[:,:,0];ratio=g[:,:,1:]/los[:,:,None]
    u=ratio/np.sqrt(1-np.sum(abs(ratio)**2,axis=2))[:,:,None]
    # All paths have Bdelta gaps >=3; diagnostic's allowed gap upper bound is20.
    gaps=np.diff(d,axis=2)
    x=np.r_[los[1:].real.ravel(),los[1:].imag.ravel(),u.real.ravel(),u.imag.ravel(),gaps.ravel()]
    return x


def decode(x):
    los=np.ones((5,2),complex);los[1:]=x[:8].reshape(4,2)+1j*x[8:16].reshape(4,2)
    u=x[16:36].reshape(5,2,2)+1j*x[36:56].reshape(5,2,2)
    ratios=u/np.sqrt(1+np.sum(abs(u)**2,axis=2))[:,:,None]
    g=np.concatenate([los[:,:,None],los[:,:,None]*ratios],axis=2)
    d=np.concatenate([np.zeros((5,2,1)),np.cumsum(x[56:].reshape(5,2,2),axis=2)],axis=2)
    return g,d


def fit_cost(x,p,f,y):
    g,d=decode(x);h,_=response(p,d,g,f)
    # Batched SVD avoids normal equations and determines rank explicitly.
    u,s,vh=np.linalg.svd(h,full_matrices=False)
    inv=np.divide(1,s,out=np.zeros_like(s),where=s>s[:,:1]*1e-12)
    coef=np.einsum('fqi,fi->fq',vh.conj().transpose(0,2,1),inv*np.einsum('fmi,fm->fi',u.conj(),y))
    residual=y-np.einsum('fmq,fq->fm',h,coef)
    return float(np.linalg.norm(residual)**2/np.linalg.norm(y)**2)


def main():
    start=time.perf_counter();out=ROOT/('alternatives_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir(exist_ok=False)
    assert SOURCE.resolve().is_relative_to(ROOT.resolve())
    z=dict(np.load(SOURCE/'inputs.npz'));protocol_src=SOURCE/'protocol.json'
    import json
    assert sha(SOURCE/'inputs.npz')==json.loads((SOURCE/'result.json').read_text(encoding='utf-8'))['input_sha']
    sources=[Path(__file__),ROOT/'structural_probe.py',SOURCE/'inputs.npz',protocol_src]
    hashes={str(p):sha(p) for p in sources}
    write(out/'protocol.json',dict(kind='WRONG_POSITION_EXPLANATION_DIAGNOSTIC_NOT_LOCALIZATION',hashes=hashes,
        offsets_m=[20,200],move='source A along x, source B along y; one source moved at a time',
        starts='true nuisance plus one fixed perturbed nuisance initialization; not available to a blind solver',
        constraints='Ref LOS complex gain1 gauge; reflection ratios have total squared norm<1; Bdelta consecutive gaps [3,20]; no true delay hints used as bounds',
        optimizer=dict(method='L-BFGS-B',maxiter=250,maxfun=40000,ftol=1e-12,gtol=1e-7),
        caution='A feasible tiny-residual alternative can be a counterexample in this sampled ideal model. Nonzero residual is not a certified lower bound.'))
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    rows=[];checks={};rng=np.random.default_rng(SEED+1);f=leggauss(96)[0]*.49*B
    # Regenerate symbol offsets from original fixed generation order, then assert source reconstruction.
    original=np.random.default_rng(SEED)
    for layout,p in enumerate(LAYOUTS):
        bits=original.choice([-1.,1.],(2,128));offset=original.uniform(0,1/(B/1.25),2)
        original.uniform(3,5,(5,2));original.uniform(8,10,(5,2));original.uniform(-np.pi,np.pi,(5,2));original.uniform(.3,.7,(5,2));original.uniform(-np.pi,np.pi,(5,2));original.uniform(-np.pi,np.pi,(5,2))
        checks[f'L{layout}/bits']=bool(np.array_equal(bits,z[f'layout{layout}_bits']))
        g=z[f'layout{layout}_g'];d=z[f'layout{layout}_delta']
        s=np.stack([wave(f,bits[k],offset[k]) for k in range(2)],axis=1);h,_=response(p,d,g,f);y=np.einsum('fmq,fq->fm',h,s)
        x0=encode(g,d);checks[f'L{layout}/truth_fit']=fit_cost(x0,p,f,y)<1e-24
        bounds=[(None,None)]*56+[(3.,20.)]*20
        for q in [0,1]:
            for displacement in [20,200]:
                wrong=p.copy();wrong[q,q]+=displacement
                for init in range(2):
                    x=x0.copy()
                    if init:
                        x[:56]+=rng.normal(0,.1,56);x[56:]=np.clip(x[56:]+rng.normal(0,.3,20),3,20)
                    first=fit_cost(x,wrong,f,y);history=[first];best=[first,x.copy()]
                    def fun(v):
                        value=fit_cost(v,wrong,f,y)
                        if value<best[0]:best[:]=[value,v.copy()]
                        return value
                    def callback(v):history.append(fit_cost(v,wrong,f,y))
                    t=time.perf_counter();opt=minimize(fun,x,method='L-BFGS-B',bounds=bounds,callback=callback,
                        options=dict(maxiter=250,maxfun=40000,ftol=1e-12,gtol=1e-7,maxls=30))
                    gg,dd=decode(best[1]);margin=float(np.min(abs(gg[:,:,0])**2-np.sum(abs(gg[:,:,1:])**2,axis=2)))
                    valid=bool(margin>0 and np.min(np.diff(dd,axis=2))>=3-1e-12)
                    row=dict(layout=layout,source=q,displacement_m=displacement,init=init,first=first,
                        best_relative_residual=float(np.sqrt(best[0])),valid=valid,margin=margin,success=bool(opt.success),
                        message=str(opt.message),iterations=int(opt.nit),evaluations=int(opt.nfev),seconds=time.perf_counter()-t,
                        final_objective=float(opt.fun),gradient_norm=float(np.linalg.norm(opt.jac)),history=history,
                        positions=wrong,best_parameters=best[1])
                    rows.append(row);write(out/f'L{layout}_q{q}_d{displacement}_i{init}.json',row)
                    print(layout,q,displacement,init,'residual',row['best_relative_residual'],'stop',opt.message,flush=True)
        assert all(checks.values())
    checks.update({'hash/'+p:sha(p)==h for p,h in hashes.items()});assert all(checks.values())
    write(out/'result.json',dict(rows=rows,checks=checks,passed=True,seconds=time.perf_counter()-start))
    print('REPORT',out,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
