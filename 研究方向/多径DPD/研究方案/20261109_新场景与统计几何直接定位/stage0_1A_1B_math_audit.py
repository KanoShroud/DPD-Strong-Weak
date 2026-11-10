#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic algebra checks, never uses project IQ or a positioning solver.

Checks (i) non-Hermitian cross-frequency fourth cumulant subspace,
(ii) Gaussian disconnected moment contamination, (iii) Wedin bound,
(iv) variable projection for complex source trispectrum coefficients,
(v) two-tone gain feasibility versus 3+ tone constraints.
"""
import json
import numpy as np

C=299792458.0
BS=np.array([[3000,3000],[-3000,3000],[-3000,-3000],[3000,-3000]],float)

def projector(X):
    u,s,_=np.linalg.svd(X,full_matrices=False)
    r=np.sum(s>1e-9*s[0])
    return u[:,:r]@u[:,:r].conj().T

def paired_subspace_test():
    r=np.random.default_rng(1100261)
    H=[r.normal(size=(4,2))+1j*r.normal(size=(4,2)) for _ in range(4)]
    ZA=np.column_stack([np.kron(H[0][:,q],H[1][:,q]) for q in range(2)])
    ZB=np.column_stack([np.kron(H[2][:,q],H[3][:,q]) for q in range(2)])
    gamma=np.array([-.8+.5j,.42-.19j])
    K=ZA@np.diag(gamma)@ZB.conj().T
    U,s,_=np.linalg.svd(K)
    Proj=U[:,:2]@U[:,:2].conj().T
    err=float(np.linalg.norm(Proj-projector(ZA),'fro'))
    assert np.linalg.matrix_rank(K,tol=1e-9)==2
    assert err<1e-11
    minrankbound=min(abs(gamma))*np.linalg.svd(ZA,compute_uv=False)[-1]*np.linalg.svd(ZB,compute_uv=False)[-1]
    assert s[1]+1e-12>=minrankbound
    randomnoise=r.normal(size=K.shape)+1j*r.normal(size=K.shape)
    eta=.08*s[1]
    E=randomnoise*eta/np.linalg.norm(randomnoise,2)
    Uh,sh,_=np.linalg.svd(K+E)
    Ph=Uh[:,:2]@Uh[:,:2].conj().T
    perr=float(np.linalg.norm(Ph-Proj,'fro'))
    pbound=float(np.sqrt(4)*eta/(s[1]-eta))
    assert perr < pbound+1e-12
    return dict(off_manifold_K_rank=2,left_projector_error=err,
                smallest_nonzero_singular=float(s[1]),rank_lower_bound=float(minrankbound),
                perturbation_eta=float(eta),perturbed_left_projection_error=perr,
                Wedin_type_projection_bound=pbound)

def moment_contamination_test():
    r=np.random.default_rng(1100262)
    Ha=r.normal(size=(4,2))+1j*r.normal(size=(4,2))
    Hb=r.normal(size=(4,2))+1j*r.normal(size=(4,2))
    Z=np.column_stack([np.kron(Ha[:,q],Hb[:,q]) for q in range(2)])
    gam=np.array([-2.,-.9])
    C4=Z@np.diag(gam)@Z.conj().T
    Ra=Ha@Ha.conj().T
    Rb=Hb@Hb.conj().T
    gaussian_part=np.kron(Ra,Rb)
    fullmoment=C4+gaussian_part
    ranks=[int(np.linalg.matrix_rank(C4,tol=1e-9)),int(np.linalg.matrix_rank(gaussian_part,tol=1e-9)),int(np.linalg.matrix_rank(fullmoment,tol=1e-9))]
    assert ranks==[2,4,4],ranks
    return dict(connected_fourth_rank=ranks[0],disconnected_second_order_product_rank=ranks[1],naive_raw_moment_rank=ranks[2],
                note='source K and second-order disconnected moment computed as exact population algebra')

def complex_gamma_profile_test():
    r=np.random.default_rng(1100263)
    ZL=r.normal(size=(16,2))+1j*r.normal(size=(16,2))
    ZR=r.normal(size=(16,2))+1j*r.normal(size=(16,2))
    atoms=np.column_stack([np.outer(ZL[:,q],ZR[:,q].conj()).reshape(-1) for q in range(2)])
    gamma=np.array([-.95+.27j,.36-.12j])
    K=(atoms@gamma).reshape(16,16)
    noise=.02*(r.normal(size=K.shape)+1j*r.normal(size=K.shape))
    Kh=K+noise
    gh=np.linalg.solve(atoms.conj().T@atoms,atoms.conj().T@Kh.reshape(-1))
    residual=float(np.linalg.norm(Kh.reshape(-1)-atoms@gh))
    true_residual=float(np.linalg.norm(Kh.reshape(-1)-atoms@gamma))
    assert residual <= true_residual+1e-11
    grad=np.linalg.norm(atoms.conj().T@(Kh.reshape(-1)-atoms@gh))
    assert grad<1e-9
    return dict(profiled_complex_gamma=[[z.real,z.imag] for z in gh],
                residual=residual,residual_at_true_gamma=true_residual,
                normal_equations_error=float(grad))

def simple_physical_channel(p,f,a,b,d):
    delays=np.linalg.norm(BS-p,axis=1)/C
    return np.exp(-2j*np.pi*delays[:,None]*f[None,:])*(a[:,None]+b[:,None]*np.exp(-2j*np.pi*d[:,None]*f[None,:]))

def two_frequency_null_test():
    p=np.array([340.,-1190.])
    p_prime=p+np.array([.3,-.2])
    freq=np.array([-2.3e6,3.1e6]); extra=np.array([.33,.65,1.1,.75])*1e-6
    a=np.array([1+.3j,.85-.25j,.73+.14j,.96-.13j]);b=.35*a*np.exp(1j*np.array([.4,-1.2,.9,2.0]))
    H=simple_physical_channel(p,freq,a,b,extra)
    a2=np.zeros(4,complex);b2=np.zeros(4,complex)
    tau2=np.linalg.norm(BS-p_prime,axis=1)/C
    for m in range(4):
        M=np.column_stack([np.ones(2),np.exp(-2j*np.pi*freq*extra[m])])
        a2[m],b2[m]=np.linalg.solve(M,H[m]*np.exp(2j*np.pi*freq*tau2[m]))
    H2=simple_physical_channel(p_prime,freq,a2,b2,extra)
    err=float(np.linalg.norm(H2-H)/np.linalg.norm(H))
    ratio=float(np.max(np.abs(b2/a2)))
    assert err<1e-11 and ratio<1
    # Third frequency generically breaks the freely re-fitted two-tone match.
    f3=np.array([-2.3e6,3.1e6,4.2e6])
    H3=simple_physical_channel(p,f3,a,b,extra)
    H3bad=simple_physical_channel(p_prime,f3,a2,b2,extra)
    err3=float(np.linalg.norm(H3bad-H3)/np.linalg.norm(H3))
    assert err3>1e-5
    return dict(position_shift_m=float(np.linalg.norm(p_prime-p)),
                two_tone_exact_relative_error=err, alternative_max_echo_LOS_amplitude_ratio=ratio,
                three_tone_relative_error_with_same_refitted_gains=err3,
                note='two-frequency channel samples cannot constrain position when two free gains per receiver')

if __name__=='__main__':
    data=dict(status='PASS',description='algebra only; no RF IQ, FFT experiments or position estimator',
              off_diagonal=paired_subspace_test(),
              diagonal_moment=moment_contamination_test(),
              complex_coefficient_profile=complex_gamma_profile_test(),
              two_tone_gain_free=two_frequency_null_test())
    print(json.dumps(data,ensure_ascii=False,indent=2))