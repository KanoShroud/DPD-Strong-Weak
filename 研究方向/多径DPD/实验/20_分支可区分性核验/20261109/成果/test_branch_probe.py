"""Standalone numerical checks; no repository IQ or ground-truth oracle in estimator."""
from types import SimpleNamespace
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from branch_probe import fit_branch, fixed_nuisance_fit, propose_path, branch_approx_score

C=299792458.
B=12.5e6
FS=50e6
STATIONS=np.array([[3000.,3000.],[-3000.,3000.],[-3000.,-3000.],[3000.,-3000.]])

class Engine:
    def __init__(self):
        self.M=4; self.D=16; self.reference=0
        self.ref=SimpleNamespace(UNIT=C/B,B=B,FS=FS,C=C,project=self.project)
        self._known_gains=np.array([[1+0j,1+0j], [.7+.1j,.6-.2j],[.5-.1j,.4+.3j],[.9+.3j,.7+.2j]])

    @staticmethod
    def project(x,n):
        z=np.asarray(x).copy()
        z[:4]=np.clip(z[:4],-3000/(C/B),3000/(C/B))
        for k in range(16,len(z),3):
            z[k:k+2]/=max(1,np.linalg.norm(z[k:k+2]))
            z[k+2]=np.clip(z[k+2],0,250)
        return z

    def _response(self,x,f,paths):
        p=np.asarray(x[:4]).reshape(2,2)*self.ref.UNIT
        delay=np.linalg.norm(p[None]-STATIONS[:,None],axis=2)/C
        g=np.ones((4,2),complex)
        g[1:]=(x[4:10]+1j*x[10:16]).reshape(3,2)
        phase=np.exp(-2j*np.pi*f[:,None,None]*delay[None])
        base=phase*g[None]
        h=base.copy()
        for k,(m,q) in enumerate(paths):
            i=16+3*k
            amp=x[i]+1j*x[i+1]
            h[:,m,q]+=base[:,m,q]*amp*np.exp(-2j*np.pi*f*x[i+2]/B)
        return h,base

    def core(self,x,f,y,paths):
        h,base=self._response(x,f,paths)
        inv=np.linalg.pinv(h,rcond=1e-12)
        s=(inv@y[:,:,None])[:,:,0]
        r=y-(h@s[:,:,None])[:,:,0]
        return h,inv,s,r,base

    def objective(self,x,f,y,paths):
        x=np.asarray(x,float)
        den=np.vdot(y,y).real
        def calc(z):
            H,_=self._response(z,f,paths)
            inv=np.linalg.pinv(H,rcond=1e-12)
            resid=y-(H@(inv@y[:,:,None]))[:,:,0]
            return np.vdot(resid,resid).real/den
        score=calc(x)
        grad=approx_derivative(lambda z:np.array([calc(z)]),x,method='3-point',rel_step=1e-5).ravel()
        return score,grad

    def rank_one(self,y,f,p):
        # A fixed-position smoke-test initialization: not used by actual repository.
        g=self._known_gains[:,0] if p[0]<0 else self._known_gains[:,1]
        phase=np.exp(-2j*np.pi*f[:,None]*np.linalg.norm(STATIONS-p,axis=1)[None]/C)
        a=phase*g[None]
        wave=np.einsum('fm,fm->f',a.conj(),y)/np.sum(abs(a)**2,axis=1)
        return g,wave,a*wave[:,None]

if __name__=='__main__':
    e=Engine();r=np.random.default_rng(20261109)
    n=512
    f=np.fft.fftfreq(n,1/FS)
    f=f[abs(f)<=B/2]
    p=np.array([[-500.,-1000.],[1300.,900.]])
    gains=e._known_gains[1:]
    x=np.r_[p.ravel()/e.ref.UNIT,gains.real.ravel(),gains.imag.ravel()]
    h,_=e._response(np.r_[x,.3,.2,16.],f,[(0,0)])
    s=r.normal(size=(len(f),2))+1j*r.normal(size=(len(f),2))
    y=np.einsum('fmq,fq->fm',h,s)
    init,los=branch_approx_score(e,p,f,y,n)
    print('initial_loss',los)
    xf,state=fixed_nuisance_fit(e,init,[],f,y,n,maxiter=15)
    assert np.array_equal(xf[:4],x[:4])
    key,amp,meta=propose_path(e,xf,[],f,y,n)
    print('candidate',key,amp,meta)
    assert 0<=amp[2]<=250 and np.linalg.norm(amp[:2])<=1+1e-12
    z,path,rep=fit_branch(e,p,f,y,n,1,15)
    assert np.array_equal(z[:4],x[:4])
    assert len(path)==1 and len(z)==19
    assert np.linalg.norm(z-e.project(z,n))<1e-7
    assert rep['steps'][-1]['state']['cost']<=rep['steps'][0]['state']['cost']+1e-8
    print('PASS: same location, physical projection, non-increasing conditional objective, no truth in path proposal')
