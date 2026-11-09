#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent algebra checks. No project IQ and no localization experiments."""
import json
import numpy as np


def tensor_gauge_invariance():
    rg=np.random.default_rng(202611092056)
    H=[rg.normal(size=(4,2))+1j*rg.normal(size=(4,2)) for _ in range(4)]
    weights=np.array([0.8-0.2j,-1.1+0.6j])
    T=sum(weights[q]*np.einsum('i,j,k,l->ijkl',H[0][:,q],H[1][:,q],H[2][:,q].conj(),H[3][:,q].conj()) for q in range(2))
    noise=0.08*(rg.normal(size=(4,4,4,4))+1j*rg.normal(size=(4,4,4,4)))
    T=T+noise
    frequencies=[-3e6,1.5e6,-2.2e6,0.7e6]
    time_diff=rg.uniform(-20e-6,20e-6,4)
    U=[np.exp(2j*np.pi*freq*time_diff) for freq in frequencies]
    U[2]=U[2].conj();U[3]=U[3].conj()
    T2=T*np.einsum('i,j,k,l->ijkl',*U)
    sv0=np.linalg.svd(T.reshape(16,16),compute_uv=False)
    sv1=np.linalg.svd(T2.reshape(16,16),compute_uv=False)
    err=float(np.max(np.abs(sv0-sv1)))
    norm_err=float(abs(np.linalg.norm(T)-np.linalg.norm(T2)))
    assert err<1e-12 and norm_err<1e-12
    return {'statement':'unitary candidate delay compensation preserves every unfolding singular value and Frobenius norm', 'max_singular_value_delta':err,'frobenius_norm_delta':norm_err}


def three_harmonic_support():
    # Ideal RC/SRRC spectral support is f in [-(1+beta)/(2T),+(1+beta)/(2T)].
    # Product v_i(t)*v_j(t) has Fourier support within +/- (1+beta)/T.
    beta=0.25
    harmonics={k:max(0.,1+beta-abs(k)) for k in range(-4,5)}
    assert [k for k,v in harmonics.items() if v>0] == [-1,0,1]
    beta_zero={k:max(0.,1-abs(k)) for k in range(-3,4)}
    assert beta_zero[1]==0
    return {'ideal_root_raised_cosine_rolloff':beta,'symbol_rate_normalized_spectral_overlap_widths':harmonics,'nonzero_harmonics':[-1,0,1], 'beta_zero_first_cycle_overlap':0.}


def clock_frequency_accuracy():
    # Exactly computed finite-sample Fourier coefficient of ideal mean exp(i*omega*n).
    # Rate error c/N creates nonvanishing bias, whereas c/N^1.5 vanishes.
    vals=[]; c=np.pi
    for n in [128,512,2048,8192]:
        k=np.arange(n)
        badly=np.mean(np.exp(-1j*c*k/n))
        better=np.mean(np.exp(-1j*c*k/n**1.5))
        vals.append({'N':n,'delta_c_over_N_bias':float(abs(badly-1)), 'delta_c_over_N3over2_bias':float(abs(better-1))})
    assert vals[0]['delta_c_over_N_bias']>0.9
    assert abs(vals[-1]['delta_c_over_N_bias']-vals[0]['delta_c_over_N_bias'])<0.01
    assert vals[-1]['delta_c_over_N3over2_bias']<vals[0]['delta_c_over_N3over2_bias']/6
    return {'statement':'clock estimate at O(1/N) can leave O(1) cyclic-coefficient bias','data':vals}


def m_dependent_moment_bound():
    n=4096
    coefficients=np.array([1.,0.4,-0.2,0.1])
    cov=np.convolve(coefficients,coefficients[::-1])
    D=len(coefficients)-1
    var0=np.sum(coefficients**2)
    exact=(n*cov[D]+2*sum((n-k)*cov[D+k] for k in range(1,D+1)))/n**2
    bound=(2*D+1)*var0/n
    assert exact >= 0 and exact <= bound+1e-14
    return {'lag_dependence_radius_D':D,'exact_mean_variance':float(exact),'generic_upper_bound':float(bound)}


def separation_stability():
    # Euclidean distance to closed profile-model sets is 1-Lipschitz in data.
    T=np.array([1.,0.,0.])
    true_model=np.array([[1.,0.,0.],[2.,0.,0.]] )
    wrong_model=np.array([[0.,1.,0.],[0.,0.,1.]])
    margin=min(np.linalg.norm(T-z) for z in wrong_model)
    error=np.array([.08,.13,-.09]);obs=T+error
    Dgood=min(np.linalg.norm(obs-z) for z in true_model)
    Dbad=min(np.linalg.norm(obs-z) for z in wrong_model)
    assert np.linalg.norm(error) < margin/2 and Dgood < Dbad
    return {'margin':float(margin),'statistical_perturbation':float(np.linalg.norm(error)),'true_model_observed_distance':float(Dgood),'wrong_model_observed_distance':float(Dbad),'warning':'only conditional on positive model separation margin'}


def relative_phase_delay_bound():
    # H_m/H_ref cancels an arbitrary common source frequency response.
    # |arg(1+a exp(-i omega delta))| <= arcsin(|a|), provided |a|<1.
    c=299792458.
    bw=12.5e6
    r=.5
    upper=(np.arcsin(r)+np.arcsin(r))/(np.pi*bw)
    rg=np.random.default_rng(202611092100)
    max_seen=0.
    for _ in range(1000):
        am=r*np.exp(1j*rg.uniform(0,2*np.pi))
        a0=r*np.exp(1j*rg.uniform(0,2*np.pi))
        dm=rg.uniform(.15e-6,1.5e-6)
        d0=rg.uniform(.15e-6,1.5e-6)
        phase=lambda a,d,f: np.angle(1+a*np.exp(-2j*np.pi*f*d))
        fminus=-bw/2
        fplus=bw/2
        phase_error=((phase(am,dm,fplus)-phase(am,dm,fminus))
                     -(phase(a0,d0,fplus)-phase(a0,d0,fminus)))/(2*np.pi*bw)
        max_seen=max(max_seen,abs(phase_error))
        assert abs(phase_error)<=upper+1e-15
    return {'statement':'if source-specific channel ratio is observable, bounded echo amplitude gives a LOS-TDOA chord-phase error bound without estimating echo delays',
            'reflection_amplitude_ratio_r':r,'reflection_power_ratio_r2':r*r,
            'bandwidth_hz':bw,'analytic_max_time_difference_error_ns':upper*1e9,
            'distance_difference_error_scale_m':upper*c,
            'max_error_in_1000_algebraic_parameter_checks_ns':max_seen*1e9,
            'WARNING':'requires exact unwrapped source-specific H_m/H_ref, not raw mixed IQ; it is not a location RMSE guarantee'}


def main():
    return {'scope':'deterministic structural proofs and numeric algebra checks, NO project IQ',
            'status':'PASS','tests':[tensor_gauge_invariance(),three_harmonic_support(),clock_frequency_accuracy(),m_dependent_moment_bound(),separation_stability(),relative_phase_delay_bound()]}

if __name__=='__main__':
    print(json.dumps(main(),indent=2))
