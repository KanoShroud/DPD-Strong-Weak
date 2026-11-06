"""Fixed-lambda profiled objective; position-only mechanism control."""
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
PREV=BASE/'regularization_gate_20261105'
sys.path.insert(0,str(PREV))
from discrepancy import cfg as cfg,ad as ad,stable,INPUT as INPUT,SAVED as SAVED  # noqa: E402
from conditional_precision import group_solve,least_squares,prediction,calibrate  # noqa: E402,F401


def waveform(h,y,c,lam):
    if lam==0 or np.all(c==0):
        return least_squares(h,y),dict(passed=True,iterations=0,gap=0.)
    free=np.flatnonzero(c==0);pen=np.flatnonzero(c>0)
    hp=h[:,:,pen];yp=y
    if len(free):
        hu=h[:,:,free];inv=np.linalg.pinv(hu,rcond=1e-12)
        yp=y-prediction(hu,(inv@y[...,None])[...,0]);hp=hp-hu@(inv@hp)
    sp,info=group_solve(hp,yp,lam*c[pen])
    s=np.zeros((len(y),len(c)),complex);s[:,pen]=sp
    if len(free):s[:,free]=(inv@(y-prediction(h[:,:,pen],sp))[...,None])[...,0]
    return s,info


class PositionProblem:
    def __init__(self,x,paths,f,y):
        self.x=x.copy();self.paths=paths;self.f=f;self.scale=np.linalg.norm(y);self.y=y/self.scale;self.e=stable.Engine()
        g=np.ones((4,2),complex);g[1:]=(x[4:10]+1j*x[10:16]).reshape(3,2)
        self.c=np.zeros(2)
        for k,(m,q) in enumerate(paths):self.c[q]+=abs(g[m,q])*np.linalg.norm(x[16+3*k:18+3*k])
        self.calls=0

    def value(self,p,lam):
        x=self.x.copy();x[:4]=p;data=self.e.core(x,self.f,self.y,self.paths);h=data[0]
        s,info=waveform(h,self.y,self.c,lam);self.calls+=1
        if not info['passed']:raise RuntimeError('conditional solver not certified')
        residual=prediction(h,s)-self.y
        penalty=float(self.c@np.linalg.norm(s,axis=0));rss=float(np.vdot(residual,residual).real)
        gradient=np.zeros(4)
        for k,q,m,z in self.e.derivatives(x,self.paths,data):
            if k>=4:break
            gradient[k]=np.vdot(residual,z*s[:,q,None]).real
        return .5*rss+lam*penalty,gradient,dict(rss=rss,penalty=penalty,source_norms=np.linalg.norm(s,axis=0),inner=info)

    def fit(self,p,lam):
        trace=[];start_calls=self.calls
        def fun(v):
            value,gradient,_=self.value(v,lam);return value,gradient
        def callback(v):
            value,gradient,_=self.value(v,lam);trace.append(dict(p=v*self.e.ref.UNIT,value=value,gradient_max=np.max(abs(gradient))))
        bound=3000/self.e.ref.UNIT
        fit=minimize(fun,p,method='L-BFGS-B',jac=True,bounds=[(-bound,bound)]*4,callback=callback,
            options=dict(maxiter=100,gtol=1e-8,ftol=1e-14,maxls=30))
        value,grad,extra=self.value(fit.x,lam)
        pg=np.max(abs(fit.x-np.clip(fit.x-grad,-bound,bound)))
        return dict(x_position=fit.x,p=fit.x.reshape(2,2)*self.e.ref.UNIT,value=value,gradient=grad,pg=pg,
            stationary=bool(pg<=1e-8),scipy_success=bool(fit.success),message=str(fit.message),iterations=fit.nit,
            calls=self.calls-start_calls,trace=trace,**extra)

    def sensitivity(self,p):
        hessians=[];rgrads=[]
        for step in [1e-3,5e-4]:
            hess=np.zeros((4,4));rgrad=np.zeros(4)
            for k in range(4):
                direction=np.eye(4)[k]*step
                _,gp,ip=self.value(p+direction,0.);_,gm,im=self.value(p-direction,0.)
                hess[:,k]=(gp-gm)/(2*step);rgrad[k]=(ip['penalty']-im['penalty'])/(2*step)
            hessians.append((hess+hess.T)/2);rgrads.append(rgrad)
        h=hessians[-1];eigen=np.linalg.eigvalsh(h)
        stable_h=np.linalg.norm(hessians[0]-h)/max(np.linalg.norm(h),1e-30)<.01
        if eigen[0]<=0:return dict(valid=False,eigenvalues=eigen,hessian_stable=stable_h)
        response=-np.linalg.solve(h,rgrads[-1])
        return dict(valid=bool(stable_h),eigenvalues=eigen,hessian_stable=stable_h,response=response,
            response_m=response*self.e.ref.UNIT,gradient_penalty=rgrads[-1],hessian=h)
