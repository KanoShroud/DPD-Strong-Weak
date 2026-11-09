"""Algebraic gates only. No localization or generator modification."""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out = HERE.parent/'运行数据/algebra'
    out.mkdir(parents=True, exist_ok=False)
    identity = {str(p): sha(p) for p in [Path(__file__), HERE/'执行合同.md']}
    rng = np.random.default_rng(2026110713)
    residuals, penalties = [], []
    for _ in range(100):
        f, m, q = 37, 4, 2
        signal = rng.normal(size=(f,q))+1j*rng.normal(size=(f,q))
        y_old, y_new = np.zeros((f,m), complex), np.zeros((f,m), complex)
        old_reg, new_reg = 0., 0.
        for source in range(q):
            # Four LOS columns and eight reflection columns: math identity only.
            stations = np.tile(np.arange(m), 3)
            coeff = rng.normal(size=12)+1j*rng.normal(size=12)
            delay = rng.uniform(0, 1, size=12)
            phase = np.exp(-2j*np.pi*np.arange(f)[:,None]*delay[None])
            lifted = signal[:,source,None]*coeff[None,:]
            for k, station in enumerate(stations):
                y_old[:,station] += phase[:,k]*coeff[k]*signal[:,source]
                y_new[:,station] += phase[:,k]*lifted[:,k]
            old_reg += np.sum(abs(coeff[4:]))*np.linalg.norm(signal[:,source])
            new_reg += np.linalg.norm(lifted[:,4:], axis=0).sum()
            assert np.linalg.matrix_rank(lifted, tol=1e-9) == 1
        residuals.append(float(np.linalg.norm(y_old-y_new)/np.linalg.norm(y_old)))
        penalties.append(float(abs(old_reg-new_reg)/old_reg))
    # Rank-dropped construction at two arbitrary position phase maps.
    yy = rng.normal(size=(37,4))+1j*rng.normal(size=(37,4))
    phases = [np.exp(-2j*np.pi*np.arange(37)[:,None]*rng.uniform(size=(1,4))) for _ in range(2)]
    lifted = [.5*yy/phase for phase in phases]
    predicted = sum(phase*x for phase,x in zip(phases,lifted))
    relaxation = dict(relative_residual=float(np.linalg.norm(predicted-yy)/np.linalg.norm(yy)),
                      ranks=[int(np.linalg.matrix_rank(x)) for x in lifted],
                      reflection_penalty=0., both_sources_nonzero=all(np.linalg.norm(x)>0 for x in lifted))
    # Absolute-gain dominance is not jointly convex.
    endpoints = [(1., .8), (-1., .8)]
    midpoint = np.mean(endpoints, axis=0)
    dominance = dict(endpoints_feasible=[abs(b)<=abs(g) for g,b in endpoints],
                     midpoint_feasible=bool(abs(midpoint[1])<=abs(midpoint[0])))
    # Exact enumeration, not a finite-sample independence test.
    source = np.array([[-1,-1],[-1,1],[1,-1],[1,1]],float)
    rotated = source@np.array([[1.,1.],[1.,-1.]])/np.sqrt(2)
    def mixed_cumulant(x):
        a,b=x.T
        return float(np.mean(a*a*b*b)-np.mean(a*a)*np.mean(b*b)-2*np.mean(a*b)**2)
    moments = dict(source_cov=(source.T@source/4).tolist(),
                   rotated_cov=(rotated.T@rotated/4).tolist(),
                   independent_mixed_cumulant=mixed_cumulant(source),
                   rotated_mixed_cumulant=mixed_cumulant(rotated),
                   scope='instantaneous_exact_moment_example_not_convolutive_DPD_proof')
    assert max(residuals+penalties) < 1e-12
    assert relaxation['relative_residual'] < 1e-12 and relaxation['ranks']==[4,4]
    assert all(dominance['endpoints_feasible']) and not dominance['midpoint_feasible']
    assert np.allclose(moments['source_cov'],moments['rotated_cov'])
    assert abs(moments['independent_mixed_cumulant'])<1e-12
    assert abs(moments['rotated_mixed_cumulant']+1)<1e-12
    assert all(sha(Path(p))==h for p,h in identity.items())
    result=dict(seed=2026110713,trials=100,observation_identity_max_error=max(residuals),
                regularizer_identity_max_error=max(penalties),rank_dropped_relaxation=relaxation,
                dominance_convexity=dominance,moment_example=moments,source_hashes=identity,
                scope='mathematical_unit_tests_only', gate='HOLD_TARGET_NOT_NEW_OR_RELAXATION_NOT_PHYSICAL')
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
