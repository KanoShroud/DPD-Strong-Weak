"""Four deterministic algebraic audits, no project IQ or localizer."""
import json
import numpy as np


def run():
    rng = np.random.default_rng(261110)
    a = rng.normal(size=(4,2))+1j*rng.normal(size=(4,2))
    b = rng.normal(size=(4,2))+1j*rng.normal(size=(4,2))
    z = np.column_stack([np.kron(a[:,i],b[:,i]) for i in range(2)])
    c = rng.normal(size=(2,2))+1j*rng.normal(size=(2,2))
    mm = [(z@c[:,i]).reshape(4,4) for i in range(2)]
    ua = np.linalg.svd(np.column_stack(mm),full_matrices=False)[0][:,:2]
    ub = np.linalg.svd(np.column_stack([t.T for t in mm]),full_matrices=False)[0][:,:2]
    t0,t1 = [ua.conj().T@t@ub.conj() for t in mm]
    _,evec = np.linalg.eig(t0@np.linalg.inv(t1))
    ah = ua@evec
    sim = abs(ah.conj().T@a)/(np.linalg.norm(ah,axis=0)[:,None]*np.linalg.norm(a,axis=0)[None,:])
    err_pencil = max(0.,float(1-np.min(np.max(sim,axis=1))))
    assert err_pencil < 1e-11

    anchor, df, eps = -.09137, .00025, .00927
    cycles = [-.4,-.2,0,.2,.4]
    margins = []
    for j in range(8):
        fs = np.array([anchor,anchor+j*df,-anchor-eps,-anchor-j*df+eps])
        for mask in range(1,15):
            val = sum(fs[t] for t in range(4) if mask & (1<<t))
            margins.append(min(abs(val-k-round(val-k)) for k in cycles))
    guard = float(min(margins))
    assert guard > 0 and 1/(df*50e6) > 62.61e-6

    freq = np.linspace(-5.8e6,5.8e6,31)
    delay = .63e-6
    gain = np.array([.7-.1j,.23+.12j])
    basis = np.column_stack([np.ones(len(freq)),np.exp(-2j*np.pi*freq*delay)])
    yy = basis@gain
    gh = np.linalg.lstsq(basis,yy,rcond=None)[0]
    err_ls = float(np.linalg.norm(basis@gh-yy))
    assert err_ls < 1e-12

    bz = np.zeros(len(freq),complex)
    assert np.linalg.norm(basis@np.array([.7-.1j,0])-np.full(len(freq),.7-.1j)) < 1e-12
    return dict(status='PASS',iq_used=False,pencil_error=err_pencil,
                off_cycle_margin=guard,physical_alias_period_us=1e6/(df*50e6),
                two_gain_conditional_ls_residual=err_ls,
                note='Algebra only: no actual mixed-IQ cumulant estimator or DPD optimization')


if __name__ == '__main__':
    print(json.dumps(run(),ensure_ascii=False,indent=2))
