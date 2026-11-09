#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Theory-only algebraic audits of cyclic cumulant correction and geometry focusing.

No real RF/IQ, no random experiment for localization. NumPy is the only dependency.
"""
from __future__ import annotations
import itertools
import json
import numpy as np

C = 299792458.0
BS = np.array([[3000., 3000.], [-3000., 3000.], [-3000., -3000.], [3000., -3000.]])


def quartets(x):
    """Return E[X0 X1 X2 X3] and all three pairwise moment products.

    x: (number of joint states, four variables); last two entries
    should already be conjugated if that is the statistic of interest.
    """
    m4 = np.mean(np.prod(x, axis=1))
    pairs = [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)]
    twos = [(np.mean(x[:, a] * x[:, b]), np.mean(x[:, c] * x[:, d]))
            for a,b,c,d in pairs]
    return m4, twos


def cyclic_debias_exact():
    """Enumerate both BPSK sources for each cyclo phase; no Monte Carlo.

    Independent sources still exhibit nonconstant conditional second
    moments due to deterministic symbol-clock power variation. A naive
    cumulant based on globally averaged covariance is biased; cyclic
    product correction removes this bias exactly.
    """
    gains = np.array([[1+.2j, .5-.3j], [.6-.1j, .7+.15j],
                      [-.4+.5j, .9-.2j], [1-.5j, -.3+.4j]])
    a1 = np.array([1., 2., 1.3, .7, 1.2, 2.1, 1.6, .8])
    a2 = np.array([1.4, .9, 1.8, .8, 1.1, 1.2, .7, 2.0])
    signs = np.array(list(itertools.product([-1.,1.],repeat=2)))
    m4, two = [], [[],[],[]]
    for t in range(len(a1)):
        x = (signs * [a1[t],a2[t]]) @ gains.T
        v = np.column_stack([x[:,0],x[:,1],x[:,2].conj(),x[:,3].conj()])
        mm, tw = quartets(v)
        m4.append(mm)
        for k,(u,w) in enumerate(tw): two[k].append((u,w))
    m4 = np.asarray(m4)
    moments = np.asarray(two)
    true_avg = np.mean(m4) - sum(np.mean(moments[k,:,0]*moments[k,:,1]) for k in range(3))
    naive = np.mean(m4) - sum(np.mean(moments[k,:,0])*np.mean(moments[k,:,1]) for k in range(3))
    cyclic = np.mean(m4)
    for k in range(3):
        spec1 = np.fft.fft(moments[k,:,0])/len(a1)
        spec2 = np.fft.fft(moments[k,:,1])/len(a1)
        cyclic -= np.sum(spec1 * spec2[(-np.arange(len(a1))) % len(a1)])
    # Additive independent-source cumulant with BPSK kappa4=-2 at each phase.
    expected = -2*(np.mean(a1**4)*gains[0,0]*gains[1,0]*gains[2,0].conj()*gains[3,0].conj()
                   + np.mean(a2**4)*gains[0,1]*gains[1,1]*gains[2,1].conj()*gains[3,1].conj())
    err = max(abs(cyclic-true_avg),abs(cyclic-expected))
    assert err < 2e-13, err
    assert abs(naive-true_avg) > .01
    return {"name":"cycle_average_cumulant_exact", "phase_count":len(a1),
            "true_average_real_imag":[true_avg.real,true_avg.imag],
            "naive_bias_abs":float(abs(naive-true_avg)),
            "cyclic_correction_abs_error":float(abs(cyclic-true_avg)),
            "independent_source_additivity_abs_error":float(abs(cyclic-expected)),
            "note":"exact two-source BPSK enumeration; not finite-IQ estimator"}


def scalar_band_two_source_tensor():
    """Exact BPSK enumeration to audit narrowband rank-2 population model.

    The mixing vectors are assumed flat per band. This is a conditional
    idealization; frequency-varying physical channels need a bias bound.
    """
    rng = np.random.default_rng(2026110925)
    H0 = rng.normal(size=(4,2)) + 1j*rng.normal(size=(4,2))
    H1 = rng.normal(size=(4,2)) + 1j*rng.normal(size=(4,2))
    # Two independent sources each containing three independent innovations;
    # the same innovations feed both frequency-band versions of that source.
    c0 = rng.normal(size=(2,3)) + 1j*rng.normal(size=(2,3))
    c1 = rng.normal(size=(2,3)) + 1j*rng.normal(size=(2,3))
    bits = np.asarray(list(itertools.product((-1.,1.),repeat=6)))
    u0 = np.column_stack([(bits[:,3*q:3*q+3] @ c0[q]) for q in range(2)])
    u1 = np.column_stack([(bits[:,3*q:3*q+3] @ c1[q]) for q in range(2)])
    z0, z1 = u0 @ H0.T, u1 @ H1.T
    m4 = np.einsum('ni,nj,nk,nl->ijkl',z0,z1,z0.conj(),z1.conj()) / len(bits)
    # Needed pair moments in the order (Z0_i, Z1_j, Z0_k*, Z1_l*).
    mm01 = z0.T@z1 / len(bits)
    mm23 = z0.conj().T@z1.conj() / len(bits)
    mm02 = z0.T@z0.conj() / len(bits)
    mm13 = z1.T@z1.conj() / len(bits)
    mm03 = z0.T@z1.conj() / len(bits)
    mm12 = z1.T@z0.conj() / len(bits)
    C4 = m4 - np.einsum('ij,kl->ijkl',mm01,mm23) \
            - np.einsum('ik,jl->ijkl',mm02,mm13) \
            - np.einsum('il,jk->ijkl',mm03,mm12)
    coefficient = -2*np.sum(c0*c1*np.conj(c0)*np.conj(c1),axis=1)
    target = np.einsum('q,iq,jq,kq,lq->ijkl',coefficient,H0,H1,H0.conj(),H1.conj())
    err = float(np.max(abs(C4-target)))
    assert err < 4e-12,err
    unfolding_rank=int(np.linalg.matrix_rank(C4.reshape(16,16),tol=1e-9))
    assert unfolding_rank==2
    return {"name":"flat_subband_fourway_tensor", "sources":2,
            "innovations_per_source":3,"matrix_unfold_rank":unfolding_rank,
            "max_exact_enumeration_error":err,
            "note":"exact for bandwise flat H; not a proof that a finite subband is flat"}


def focusing_lipschitz_bound():
    """Geometry focusing reduces delay spread for the target source, exactly.

    Return both the correct-position bound and a wrong-location bound for
    the same source. No signals are generated or located.
    """
    p_true = np.array([340., -1210.])
    p_false = np.array([-940., 880.])
    d=lambda p: (np.linalg.norm(BS-p,axis=1)-np.linalg.norm(BS[0]-p))/C
    true_d=d(p_true)
    delays=np.array([.34, .81, 1.10, .61])*1e-6
    a0=np.array([1+.2j,.8-.1j,.74+.3j,-.6+.18j])
    a1=np.array([.3+.15j,-.24+.21j,.2-.1j,.25+.07j])
    fcenter=2.1e6
    w=130e3
    f=np.linspace(fcenter-w/2,fcenter+w/2,401)
    bound_details=[]
    for label,p in [('correct',p_true),('wrong',p_false)]:
        eps=true_d-d(p)
        def focused(x):
            return (a0[:,None]*np.exp(-2j*np.pi*eps[:,None]*x)
                    + a1[:,None]*np.exp(-2j*np.pi*(eps+delays)[:,None]*x))
        hf=focused(f[None,:])
        hc=focused(np.array([[fcenter]]))
        dif=np.max(abs(hf-hc),axis=1)
        upper=np.pi*w*(abs(a0)*abs(eps)+abs(a1)*abs(eps+delays))
        assert np.all(dif <= upper + 2e-12), (label,dif,upper)
        if label=='correct':assert np.max(abs(eps)) < 1e-14
        bound_details.append({'candidate':label,'max_frequency_variation':float(np.max(dif)),
                              'max_analytical_upper_bound':float(np.max(upper)),
                              'max_relative_station_delay_error_us':float(np.max(abs(eps))*1e6)})
    return {'name':'geometry_focusing_bandwidth_bound','subband_width_hz':w,
            'details':bound_details,
            'note':'exact spectral bound; does not establish unique position maximum'}


def sample_covariance_bound():
    """Proof audit via finite Toeplitz covariance matrix, not IQ simulation.

    This checks the generic inequality Var(mean U_n) <= K/N
    for a known exponentially decaying covariance sequence.
    """
    n=400; rho=0.62
    lags=np.arange(n)
    covariance=rho**abs(np.subtract.outer(lags,lags))
    exact=float(np.sum(covariance)/n**2)
    upper=(1+rho)/(1-rho)/n
    assert exact<=upper
    return {'name':'abstract_moment_mean_covariance_bound','N':n,
            'exact_variance_for_AR1_covariance':exact,'summability_upper_bound':upper,
            'note':'checks elementary variance inequality, not that PAM obeys assumptions'}


def run():
    results=[cyclic_debias_exact(),scalar_band_two_source_tensor(),
             focusing_lipschitz_bound(),sample_covariance_bound()]
    return {'status':'PASS','scope':'exact moment algebra and deterministic analytic checks only',
            'project_IQ_read':False,'new_position_estimator':False,'checks':results}


if __name__=='__main__':
    print(json.dumps(run(),ensure_ascii=False,indent=2))
