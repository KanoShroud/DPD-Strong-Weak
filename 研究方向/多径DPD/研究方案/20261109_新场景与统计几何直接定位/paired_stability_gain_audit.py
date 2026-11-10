#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact linear-algebra audit; no user IQ, position solver, or RMSE trial.

Checks paired fourth-order spectral-gap perturbation bounds, real fourth-order
source coefficient profiling, and a conditional cross-relation gain elimination.
"""
import json
import numpy as np

C = 299792458.0
BS = np.array([[3000., 3000.], [-3000., 3000.], [-3000., -3000.], [3000., -3000.]])


def projector(U):
    return U @ U.conj().T


def paired_spectral_gap():
    rng = np.random.default_rng(2026111007)
    normcols = lambda H: H / np.linalg.norm(H, axis=0, keepdims=True)
    Ha = normcols(rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2)))
    Hb = normcols(rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2)))
    Z = np.column_stack([np.kron(Ha[:,q], Hb[:,q]) for q in range(2)])
    gamma = np.array([-1., -.12])
    K = Z @ np.diag(gamma) @ Z.conj().T
    lamb, eigvec = np.linalg.eigh(K)
    delta = float(np.sort(abs(lamb))[-2])
    coherence = float(abs(np.vdot(Ha[:,0], Ha[:,1]) * np.vdot(Hb[:,0], Hb[:,1])))
    lower = float(min(abs(gamma)) * (1-coherence))
    assert delta + 1e-12 >= lower
    raw = rng.normal(size=K.shape) + 1j*rng.normal(size=K.shape)
    raw = (raw + raw.conj().T)/2
    eta = .14*delta
    E = eta * raw / np.linalg.norm(raw, 2)
    Khat = K + E
    lambh, vh = np.linalg.eigh(Khat)
    U = eigvec[:,np.argsort(abs(lamb))[-2:]]
    Uh = vh[:,np.argsort(abs(lambh))[-2:]]
    error = float(np.linalg.norm(projector(Uh)-projector(U), 'fro'))
    upper = float(np.sqrt(2*2)*eta/(delta-eta))
    assert error <= upper + 1e-12 and eta < delta/2
    return dict(name='paired_gap_and_signed_spectrum', delta=delta,
                guaranteed_gap_lower_bound=lower,
                paired_source_coherence=coherence,
                perturbation_spectral_norm=eta,
                projector_error_frobenius=error,
                rigorous_projector_error_upper_bound=upper,
                warning='population matrix perturbation, not a finite-IQ noise model')


def gamma_profile():
    rng = np.random.default_rng(2026111017)
    H0 = rng.normal(size=(4,2))+1j*rng.normal(size=(4,2))
    H1 = rng.normal(size=(4,2))+1j*rng.normal(size=(4,2))
    Z = np.column_stack([np.kron(H0[:,q],H1[:,q]) for q in range(2)])
    true_gamma = np.array([-2.1,.35])
    K = Z@np.diag(true_gamma)@Z.conj().T
    E = rng.normal(size=K.shape)+1j*rng.normal(size=K.shape)
    E = .03*(E+E.conj().T)/2
    Khat = K+E
    Gram = np.abs(Z.conj().T@Z)**2
    rhs = np.real(np.array([np.vdot(Z[:,q], Khat@Z[:,q]) for q in range(2)]))
    gh = np.linalg.solve(Gram,rhs)
    cost = lambda g: float(np.linalg.norm(Khat - Z@np.diag(g)@Z.conj().T,'fro')**2)
    assert np.linalg.eigvalsh(Gram)[0] > 0
    assert cost(gh) <= cost(true_gamma) + 1e-10
    grad = 2*(Gram@gh-rhs)
    assert np.linalg.norm(grad) < 1e-9
    return dict(name='exact_signed_gamma_profiling',
                gamma_truth=true_gamma.tolist(),gamma_profile_solution=gh.tolist(),
                gram_min_eigenvalue=float(np.linalg.eigvalsh(Gram)[0]),
                profile_cost=cost(gh),true_gamma_cost=cost(true_gamma),
                stationarity_gradient_norm=float(np.linalg.norm(grad)),
                warning='profiles only real source fourth coefficients; complex channel gains and delays remain nonlinear')


def relative_delay(p):
    return np.linalg.norm(BS-p,axis=1)/C


def physical_channel(p, delay, a, b, f):
    return (np.exp(-2j*np.pi*relative_delay(p)[:,None]*f[None,:])
            *(a[:,None]+b[:,None]*np.exp(-2j*np.pi*delay[:,None]*f[None,:])))


def cross_relation_matrix(p,delay,v,f):
    """For every other station m: v_m H_0(p) - v_0 H_m(p) = 0.

    At a fixed position and fixed echo delays, the rows are LINEAR in the
    2M complex gains (a_m, b_m). SVD gives a normalized unconstrained bound.
    """
    M=v.shape[0]
    A=np.zeros(((M-1)*len(f),2*M),complex)
    t=relative_delay(p)
    row=0
    for k,fk in enumerate(f):
        for m in range(1,M):
            e0=np.exp(-2j*np.pi*fk*t[0]);em=np.exp(-2j*np.pi*fk*t[m])
            A[row,0] = v[m,k]*e0
            A[row,1] = v[m,k]*e0*np.exp(-2j*np.pi*fk*delay[0])
            A[row,2*m] = -v[0,k]*em
            A[row,2*m+1] = -v[0,k]*em*np.exp(-2j*np.pi*fk*delay[m])
            row+=1
    return A


def channel_gain_elimination():
    p=np.array([650.,-950.]); false_p=np.array([-900.,720.])
    f=np.linspace(-5.8e6,5.8e6,18)
    delta=np.array([.33,.68,1.07,.52])*1e-6
    a=np.array([1.+.12j,.72-.3j,-.92+.1j,.89+.06j])
    b=a*np.array([.27+.05j,.31-.07j,-.2+.13j,.25+.12j])
    h=physical_channel(p,delta,a,b,f)
    gauge=np.exp(1j*(.28+np.linspace(-1.5,1.4,len(f))))*(.8+.2*np.sin(np.linspace(0,3,len(f))))
    v=h*gauge[None,:]
    v=v/np.linalg.norm(v,axis=0,keepdims=True)
    A=cross_relation_matrix(p,delta,v,f)
    wrong_A=cross_relation_matrix(false_p,delta,v,f)
    g=np.empty(2*len(a),complex)
    g[::2]=a;g[1::2]=b
    null=float(np.linalg.norm(A@g))
    min_true=float(np.linalg.svd(A,compute_uv=False)[-1])
    min_wrong=float(np.linalg.svd(wrong_A,compute_uv=False)[-1])
    assert null<2e-12 and min_true<2e-12 and min_wrong>.01
    return dict(name='conditional_linear_gain_elimination',
                frequency_count=len(f),receivers=len(a),rows=A.shape[0],columns=A.shape[1],
                true_gain_cross_relation_residual=null,true_smallest_singular=min_true,
                false_position_smallest_singular_with_true_delays=min_wrong,
                warning='requires already identified source-specific frequency directions; fixed echo delays only. Unconstrained SVD may violate physical gain constraints. No global fast optimizer yet.')


def main():
    checks=[paired_spectral_gap(),gamma_profile(),channel_gain_elimination()]
    return dict(status='PASS',scope='exact matrix algebra, no RF IQ or localization test',checks=checks)


if __name__=='__main__':
    print(json.dumps(main(),ensure_ascii=False,indent=2))
