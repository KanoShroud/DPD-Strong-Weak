"""One bounded analytic-gradient follow-up; no new data or search positions."""
from pathlib import Path
from datetime import datetime
import time
import json
import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
from alternative_fit import SOURCE, encode, decode, fit_cost
from structural_probe import ROOT, LAYOUTS, B, SEED, wave, response, write, sha
from numpy.polynomial.legendre import leggauss


def objective(x,p,f,y):
    g,d=decode(x);h,phase=response(p,d,g,f)
    uu,ss,vh=np.linalg.svd(h,full_matrices=False)
    inv=np.divide(1,ss,out=np.zeros_like(ss),where=ss>ss[:,:1]*1e-12)
    coef=np.einsum('fqi,fi->fq',vh.conj().transpose(0,2,1),inv*np.einsum('fmi,fm->fi',uu.conj(),y))
    residual=y-np.einsum('fmq,fq->fm',h,coef);den=np.linalg.norm(y)**2
    jac=np.zeros((len(f),5,76),complex)
    u=x[16:36].reshape(5,2,2)+1j*x[36:56].reshape(5,2,2)
    for m in range(5):
        for q in range(2):
            t=np.sqrt(1+np.sum(abs(u[m,q])**2));rat=u[m,q]/t
            e=phase[:,m,q,1:];es=e@u[m,q];signal=coef[:,q];g0=g[m,q,0]
            if m:
                z=signal*(phase[:,m,q,0]+e@rat);i=(m-1)*2+q
                jac[:,m,i]=z;jac[:,m,i+8]=1j*z
            for ell in range(2):
                i=(m*2+q)*2+ell
                jac[:,m,16+i]=signal*g0*(e[:,ell]/t-es*u[m,q,ell].real/t**3)
                jac[:,m,36+i]=signal*g0*(1j*e[:,ell]/t-es*u[m,q,ell].imag/t**3)
                affected=slice(ell,None)
                jac[:,m,56+i]=signal*g0*(-2j*np.pi*f/B)*(e[:,affected]@rat[affected])
    grad=-2*np.einsum('fm,fmv->v',residual.conj(),jac).real/den
    return float(np.linalg.norm(residual)**2/den),grad


def main():
    start=time.perf_counter();out=ROOT/('gradient_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir(exist_ok=False)
    z=dict(np.load(SOURCE/'inputs.npz'));assert sha(SOURCE/'inputs.npz')==json.loads((SOURCE/'result.json').read_text(encoding='utf-8'))['input_sha']
    files=[Path(__file__),ROOT/'alternative_fit.py',ROOT/'structural_probe.py',SOURCE/'inputs.npz'];hashes={str(p):sha(p) for p in files}
    write(out/'protocol.json',dict(kind='ONE_BOUNDED_ANALYTIC_GRADIENT_FOLLOWUP',hashes=hashes,
        preserved='Same 16 fixed-position/start combinations, 96 frequencies, source and channel constraints as preceding diagnostic',
        changes='Analytic variable-projection objective gradient, maxiter5000, maxfun20000, same ftol1e-12 gtol1e-7',
        stop='No further optimizer tuning this turn. Nonconverged trials remain inconclusive. Not blind localization.'))
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    rows=[];checks={};rng=np.random.default_rng(SEED+1);original=np.random.default_rng(SEED);f=leggauss(96)[0]*.49*B
    for layout,p in enumerate(LAYOUTS):
        bits=original.choice([-1.,1.],(2,128));offset=original.uniform(0,1/(B/1.25),2)
        original.uniform(3,5,(5,2));original.uniform(8,10,(5,2));original.uniform(-np.pi,np.pi,(5,2));original.uniform(.3,.7,(5,2));original.uniform(-np.pi,np.pi,(5,2));original.uniform(-np.pi,np.pi,(5,2))
        checks[f'L{layout}/bits']=bool(np.array_equal(bits,z[f'layout{layout}_bits']))
        g=z[f'layout{layout}_g'];d=z[f'layout{layout}_delta'];s=np.stack([wave(f,bits[k],offset[k]) for k in range(2)],axis=1)
        h,_=response(p,d,g,f);y=np.einsum('fmq,fq->fm',h,s);x0=encode(g,d)
        bounds=[(None,None)]*56+[(3.,20.)]*20
        for q in range(2):
            for displacement in [20,200]:
                wrong=p.copy();wrong[q,q]+=displacement
                for init in range(2):
                    x=x0.copy()
                    if init:
                        x[:56]+=rng.normal(0,.1,56);x[56:]=np.clip(x[56:]+rng.normal(0,.3,20),3,20)
                    val,grad=objective(x,wrong,f,y);eps=1e-6
                    numeric=np.array([(fit_cost(x+np.eye(76)[j]*eps,wrong,f,y)-fit_cost(x-np.eye(76)[j]*eps,wrong,f,y))/(2*eps) for j in range(76)])
                    key=f'L{layout}_q{q}_d{displacement}_i{init}'
                    checks[key+'/objective']=bool(abs(val-fit_cost(x,wrong,f,y))<1e-14)
                    checks[key+'/gradient']=bool(np.linalg.norm(numeric-grad)/max(np.linalg.norm(numeric),1e-15)<1e-6)
                    assert all(checks.values())
                    first=val;history=[val];best=[val,x.copy()]
                    def fun(v):
                        vv,gg=objective(v,wrong,f,y)
                        if vv<best[0]:best[:]=[vv,v.copy()]
                        return vv,gg
                    def callback(v):history.append(fit_cost(v,wrong,f,y))
                    t=time.perf_counter();opt=minimize(fun,x,jac=True,method='L-BFGS-B',bounds=bounds,callback=callback,
                        options=dict(maxiter=5000,maxfun=20000,ftol=1e-12,gtol=1e-7,maxls=30))
                    gg,dd=decode(best[1]);margin=float(np.min(abs(gg[:,:,0])**2-np.sum(abs(gg[:,:,1:])**2,axis=2)))
                    projected=opt.jac.copy()
                    projected[56:][(opt.x[56:]<=3+1e-8)&(projected[56:]>0)]=0
                    projected[56:][(opt.x[56:]>=20-1e-8)&(projected[56:]<0)]=0
                    row=dict(case=key,layout=layout,source=q,displacement_m=displacement,init=init,first=first,
                        best_relative_residual=float(np.sqrt(best[0])),valid=bool(margin>0 and np.min(np.diff(dd,axis=2))>=3-1e-12),
                        margin=margin,success=bool(opt.success),message=str(opt.message),iterations=int(opt.nit),evaluations=int(opt.nfev),
                        seconds=time.perf_counter()-t,final_objective=float(opt.fun),gradient_norm=float(np.linalg.norm(opt.jac)),
                        projected_gradient_inf=float(np.max(abs(projected))),history=history,positions=wrong,best_parameters=best[1])
                    rows.append(row);write(out/(key+'.json'),row)
                    print(key,'residual',row['best_relative_residual'],'iterations',opt.nit,'projected',row['projected_gradient_inf'],flush=True)
    checks.update({'hash/'+p:sha(p)==h for p,h in hashes.items()});assert all(checks.values())
    write(out/'result.json',dict(rows=rows,checks=checks,passed=True,seconds=time.perf_counter()-start))
    print('REPORT',out,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
