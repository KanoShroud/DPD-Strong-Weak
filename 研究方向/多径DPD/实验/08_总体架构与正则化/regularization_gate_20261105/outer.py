"""Conditional Morozov outer value/derivative, only regular interior points."""
import numpy as np
from conditional import calibrate, prediction


def coefficients(x,paths):
    g=np.ones((4,2),complex);g[1:]=(x[4:10]+1j*x[10:16]).reshape(3,2)
    c=np.zeros(2);dc=np.zeros((2,len(x)))
    for i,(m,q) in enumerate(paths):
        k=16+3*i;rho=x[k]+1j*x[k+1];a=abs(rho);b=abs(g[m,q])
        if min(a,b)<=1e-12:
            raise ValueError('nonsmooth_zero_amplitude')
        c[q]+=a*b
        dc[q,k]=b*rho.real/a;dc[q,k+1]=b*rho.imag/a
        if m:
            j=4+2*(m-1)+q
            dc[q,j]+=a*g[m,q].real/b;dc[q,j+6]+=a*g[m,q].imag/b
    return c,dc


def value_gradient(e,x,f,y,paths,radius2,gradient=True):
    data=e.core(x,f,y,paths);h=data[0]
    c,dc=coefficients(x,paths)
    s,info=calibrate(h,y,c,radius2)
    if info['status']!='interior_discrepancy':
        return None,None,info
    scale=np.linalg.norm(y);value=info['value']/scale
    if not gradient: return value,None,info
    residual=prediction(h,s)-y
    grad=np.linalg.norm(s,axis=0)@dc
    for k,q,m,z in e.derivatives(x,paths,data):
        if m is None:
            derivative=z*s[:,q,None];r=residual
        else:
            derivative=z*s[:,q];r=residual[:,m]
        grad[k]+=np.vdot(r,derivative).real/info['multiplier']
    return value,grad/scale,info
