# -*- coding: utf-8 -*-
"""Audits finite-frequency identifiability algebra; no IQ and no positioning solver."""
import itertools
import json
import numpy as np

c=299792458.0
bs=np.array([[3000,3000],[-3000,3000],[-3000,-3000],[3000,-3000]],float)
df=50e6/32768

def tau(p):
    return np.linalg.norm(bs-np.asarray(p),axis=1)/c

def channel(p,f,g0,g1,delta):
    f=np.asarray(f)
    return np.exp(-2j*np.pi*tau(p)[:,None]*f[None,:])*(g0[:,None]+g1[:,None]*np.exp(-2j*np.pi*delta[:,None]*f[None,:]))

def check():
    p=np.array([450.,-930.]); q=np.array([462.,-937.])
    g0=np.array([1+.22j,.85-.3j,-.67+.1j,.9+.27j]);
    g1=g0*np.array([.42*np.exp(.3j),.46*np.exp(1.1j),.5*np.exp(-.7j),.38*np.exp(.9j)])
    d=np.array([.32,.81,1.23,.63])*1e-6
    two=500000.+np.arange(2)*df
    h=channel(p,two,g0,g1,d)
    g0a=np.empty(4,complex);g1a=np.empty(4,complex)
    for m in range(4):
        A=np.stack([np.ones(2),np.exp(-2j*np.pi*two*d[m])],axis=1)
        target=h[m]*np.exp(2j*np.pi*two*tau(q)[m])
        g0a[m],g1a[m]=np.linalg.solve(A,target)
    err=np.linalg.norm(channel(q,two,g0a,g1a,d)-h)/np.linalg.norm(h)
    ratio=float(np.max(abs(g1a/g0a)))
    assert err<1e-10 and ratio<1 and np.linalg.norm(p-q)>1

    da=np.array([.45,.74,.92,.28])*1e-6
    g0b=g0*np.exp(1j*np.array([.14,-.3,.3,-.15]))
    g1b=g0b*np.array([.2+.25j,-.3+.22j,.2-.25j,-.14-.18j])
    f=480000.+np.arange(8)*df
    h=channel(p,f,g0,g1,d);hp=channel(q,f,g0b,g1b,da)
    m=2;ref=0
    target=hp[m]*h[ref]-h[m]*hp[ref]
    a=[]
    for i,j in itertools.product((0,1),repeat=2):
        t=tau(q)[m]+(da[m] if i else 0)+tau(p)[ref]+(d[ref] if j else 0)
        v=(g1b[m] if i else g0b[m])*(g1[ref] if j else g0[ref]);a.append((t,v))
    for i,j in itertools.product((0,1),repeat=2):
        t=tau(p)[m]+(d[m] if i else 0)+tau(q)[ref]+(da[ref] if j else 0)
        v=-(g1[m] if i else g0[m])*(g1b[ref] if j else g0b[ref]);a.append((t,v))
    nodes=np.array([e[0] for e in a]); coeff=np.array([e[1] for e in a])
    V=np.exp(-2j*np.pi*f[:,None]*nodes[None,:])
    relative=np.linalg.norm(V@coeff-target)/np.linalg.norm(target)
    min_sv=float(np.linalg.svd(V,compute_uv=False)[-1])
    assert relative<1e-9 and len(set(np.round(nodes,15)))==8
    maxspan=2*(6000*np.sqrt(2)/c+3e-6)
    assert maxspan<1/df

    rng=np.random.default_rng(2026110921)
    H0=rng.normal(size=(4,2))+1j*rng.normal(size=(4,2))
    H1=rng.normal(size=(4,2))+1j*rng.normal(size=(4,2))
    Q=np.column_stack([np.kron(H0[:,k].conj(),H1[:,k].conj()) for k in range(2)])
    assert np.linalg.matrix_rank(H0)==np.linalg.matrix_rank(H1)==np.linalg.matrix_rank(Q)==2
    T=sum(-2*np.einsum('i,j,k,l->ijkl',H0[:,k],H1[:,k],H0[:,k].conj(),H1[:,k].conj()) for k in range(2))
    sv=np.linalg.svd(T.reshape(16,16),compute_uv=False)
    assert np.count_nonzero(sv>sv[0]*1e-12)==2
    cond=[]
    for delta in [10e-9,100e-9,1e-6,10e-6]:
        W=np.exp(-2j*np.pi*f[:,None]*np.array([22e-6,22e-6+delta])[None,:])
        s=np.linalg.svd(W,compute_uv=False)
        cond.append(float(s[0]/s[-1]))
    assert cond[0]>cond[-1]
    return dict(status='PASS',no_IQ=True,tests=4,
        two_frequency_position_difference_m=float(np.linalg.norm(p-q)),
        two_frequency_relative_complex_fit_error=float(err),
        alternative_max_echo_amplitude_ratio=ratio,
        eight_tone_cross_relation_relative_error=float(relative),
        eight_tone_cross_relation_min_singular_value=min_sv,
        alias_period_us=float(1e6/df),max_cross_support_us=float(1e6*maxspan),
        anchored_three_mode_kruskal_ranks=[2,2,2],
        near_delay_condition_numbers=cond,
        warning='Exact conditional uniqueness is not finite-IQ stability or a working DPD algorithm')

if __name__=='__main__':
    print(json.dumps(check(),indent=2))
