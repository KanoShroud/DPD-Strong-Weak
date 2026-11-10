#!/usr/bin/env python3
"""Algebraic checks for a proposed cross-frequency fourth-cumulant DPD.
No IQ input, no position estimator, no stochastic performance simulation.
"""
import json
import numpy as np


def projector(X, r=None):
    if r is not None:
        U, s, _ = np.linalg.svd(X, full_matrices=False)
        assert s[r-1] > 1e-10*s[0]
        X = U[:, :r]
    return X @ np.linalg.pinv(X)


def check(seed=20261110):
    rng = np.random.default_rng(seed)
    M,Q=4,2
    # Source-specific channel matrix at two frequencies, both rank Q.
    Ha = rng.normal(size=(M,Q)) + 1j*rng.normal(size=(M,Q))
    Hb = rng.normal(size=(M,Q)) + 1j*rng.normal(size=(M,Q))
    Z = np.column_stack([np.kron(Ha[:,q],Hb[:,q]) for q in range(Q)])
    gamma = np.array([-2.3,-0.7])  # negative fourth cumulants are allowed
    K = Z@np.diag(gamma)@Z.conj().T
    singular = np.linalg.svd(K,compute_uv=False)
    Ptrue=projector(Z)
    eig,evec=np.linalg.eigh(K)
    big=np.argsort(np.abs(eig))[-Q:]
    Pfrom=evec[:,big]@evec[:,big].conj().T
    assert np.linalg.norm(Ptrue-Pfrom)<1e-12
    # An arbitrary mixture of two matched Kronecker products is NOT itself separable.
    sep_sv=np.linalg.svd((Z[:,0]+(0.4+.2j)*Z[:,1]).reshape(M,M),compute_uv=False)
    assert sep_sv[1]>1e-4
    # A one-source clone gives a zero one-sided inclusion score but fails subspace equality.
    duplicate=np.column_stack([Z[:,0],Z[:,0]])
    inclusion=np.linalg.norm((np.eye(M*M)-Ptrue)@duplicate)
    duplicated_projector_mismatch=np.linalg.norm(Ptrue-projector(duplicate))
    assert inclusion<1e-12 and duplicated_projector_mismatch>.9
    # Covariance whitening: only a Q-dimensional signal subspace survives.
    Pa=np.array([1.3,0.4]); Pb=np.array([0.9,1.7])
    Ba=Ha*np.sqrt(Pa)[None,:];Bb=Hb*np.sqrt(Pb)[None,:]
    Ra=Ba@Ba.conj().T;Rb=Bb@Bb.conj().T
    def whiten(R):
        values,U=np.linalg.eigh(R)
        ids=np.argsort(values)[::-1][:Q]
        Uq=U[:,ids];v=values[ids]
        W=(Uq/np.sqrt(v)[None,:]).conj().T  # diag(1/sqrt(v)) Uq^H
        return W,U
    Wa,Ua=whiten(Ra);Wb,Ub=whiten(Rb)
    assert np.linalg.norm(Wa@Ra@Wa.conj().T-np.eye(Q))<1e-12
    assert np.linalg.norm(Wb@Rb@Wb.conj().T-np.eye(Q))<1e-12
    # Source spectra and same-source trispectra have a frequency-independent ratio.
    T=np.array([1.1,0.7]);sig=np.array([1.2,1.4]);kap=np.array([-2.0,-.6]);
    Ga=np.array([1.4,.9]);Gb=np.array([.7,1.1]);
    Pa_ref=sig**2/T*Ga**2
    Pb_ref=sig**2/T*Gb**2
    Gamma=kap/T * Ga**2 * Gb**2
    ratio=Gamma/(Pa_ref*Pb_ref)
    invariant=kap*T/sig**4
    assert np.allclose(ratio,invariant)
    # Test the normalization with Pa/Pb derived from the same illustrative source model.
    Bas=Ha*np.sqrt(Pa_ref)[None,:]; Bbs=Hb*np.sqrt(Pb_ref)[None,:]
    Was,_=whiten(Bas@Bas.conj().T); Wbs,_=whiten(Bbs@Bbs.conj().T)
    Kpop=Z@np.diag(Gamma)@Z.conj().T
    Wk=np.kron(Was,Wbs)
    Kw=Wk@Kpop@Wk.conj().T
    ew=np.linalg.eigvalsh(Kw)
    assert np.allclose(np.sort(ew)[:2],np.sort(invariant),atol=1e-11)
    # Fatal pitfall: whitening alone destroys receiver-space information.
    Na=Ua[:,:M-Q];Nb=Ub[:,:M-Q]
    Hpa=Ha+0.35*Na@np.array([[1.,0.],[0.,1.]])
    Hpb=Hb+0.2*Nb@np.array([[1.,0.],[0.,1.]])
    invis=max(np.linalg.norm(Wa@(Hpa-Ha)),np.linalg.norm(Wb@(Hpb-Hb)))
    Zwhite=np.column_stack([np.kron(Wa@Ha[:,q],Wb@Hb[:,q]) for q in range(Q)])
    Zwhite_bad=np.column_stack([np.kron(Wa@Hpa[:,q],Wb@Hpb[:,q]) for q in range(Q)])
    whitened_projector_difference=np.linalg.norm(projector(Zwhite)-projector(Zwhite_bad))
    covariance_subspace_mismatch=np.linalg.norm(projector(Ha)-projector(Hpa)) +np.linalg.norm(projector(Hb)-projector(Hpb))
    assert invis<1e-12 and whitened_projector_difference<1e-12
    assert covariance_subspace_mismatch>.02
    return {
      'status':'PASS', 'scope':'pure random-matrix algebra, not IQ localization',
      'assumed_stations':M,'assumed_sources':Q,
      'paired_fourth_cumulant_rank':int(np.linalg.matrix_rank(K)),
      'population_fourth_projector_error':float(np.linalg.norm(Ptrue-Pfrom)),
      'nontrivial_source_mixture_matrix_second_singular':float(sep_sv[1]),
      'source_cloning_one_sided_score':float(inclusion),
      'source_cloning_two_sided_projector_error':float(duplicated_projector_mismatch),
      'frequency_independent_kurtosis_coupling_error':float(np.max(abs(ratio-invariant))),
      'whitened_fourth_eigenvalues':ew.tolist(),
      'whitening_nullspace_invisibility_error':float(invis),
      'wrong_candidate_whitened_projector_error':float(whitened_projector_difference),
      'wrong_candidate_second_order_subspace_error':float(covariance_subspace_mismatch),
      'interpretation':'Fourth paired subspace identifies aligned products in population, but whitening needs explicit covariance-subspace consistency; source cloning must be prevented.'
    }


if __name__=='__main__':
    print(json.dumps(check(),ensure_ascii=False,indent=2))
