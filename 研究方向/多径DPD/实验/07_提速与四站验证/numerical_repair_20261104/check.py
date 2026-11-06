"""Numerical correctness gates before any paired localization."""
import numpy as np
from threadpoolctl import threadpool_limits
import stable

cfg, ad = stable.cfg, stable.ad
HERE = stable.HERE


def main():
    old = stable.inside.Engine(); new = stable.Engine()
    rng = np.random.default_rng(2026110461)
    f = np.linspace(-cfg.B/2, cfg.B/2, 47)
    y = rng.normal(size=(47, 4))+1j*rng.normal(size=(47, 4))
    x = np.r_[[-1500, -900, 1200, 1800], rng.normal(size=12), .2, -.15, 3.4, -.3, .1, 14.8]
    x[:4] /= ad.b.ref.UNIT
    paths = [(1, 0), (3, 1)]
    a, ga = old.objective(x, f, y, paths)
    b, gb = new.objective(x, f, y, paths)
    r, j = new.evaluate(x, f, y, paths)
    d = rng.normal(size=len(x)); d /= np.linalg.norm(d)
    fd = (new.evaluate(x+1e-5*d, f, y, paths)[0]-new.evaluate(x-1e-5*d, f, y, paths)[0])/2e-5
    rel = np.linalg.norm(fd-j@d)/np.linalg.norm(j@d)
    checks = dict(same_objective=bool(np.isclose(a, b, rtol=1e-12, atol=1e-12)),
                  same_gradient=bool(np.allclose(ga, gb, atol=1e-10, rtol=1e-9)),
                  jacobian=bool(rel < 3e-6),
                  gradient_identity=bool(np.allclose(gb, j.T@r, atol=1e-10, rtol=1e-9)))
    singular = x.copy(); singular[:2] = cfg.STATIONS[0]/ad.b.ref.UNIT
    val, gg = new.objective(singular, f, y, paths)
    checks['invalid_trial_extended_cost'] = bool(np.isposinf(val) and len(new.invalid_trials) == 1)
    gate = new.fit.__globals__['Gate'](x, f, y, paths, cfg.N)
    checks['invalid_trial_never_accepted'] = not gate.acceptable(singular, val, gg)
    h = np.zeros((2, 4, 2), complex)
    h[:, 0, 0] = 1
    h[:, 1, 1] = [1e-10, 1e-13]
    yy = np.ones((2, 4), complex)
    inv, _, _ = new.linear_projection(h, yy)
    checks['original_rank_cutoff'] = bool(np.allclose(inv, np.linalg.pinv(h, rcond=1e-12)) and np.array_equal(new.basis[2].sum(axis=1), [2, 1]))
    legacy = cfg.HERE/'pilot_20261104_182457_012169'
    fixed = []
    for tag in ('MIX_02_R00_L2_P100', 'OUT_01_R00_L3_P125'):
        result = cfg.read(legacy/'cases'/tag/'MP_decision.json')['result']
        with np.load(legacy/'cases'/tag/'iq.npz') as z:
            f2, y2, _, _ = ad.b.fc.cal.spectral(z['iq'])
        xx = np.array(result['final']['x']); pp = result['paths']
        aa, ag = old.objective(xx, f2, y2, pp)
        bb, bg = new.objective(xx, f2, y2, pp)
        rr, jj = new.evaluate(xx, f2, y2, pp)
        fixed.append(dict(tag=tag, cost_difference=abs(aa-bb), gradient_identity=float(np.linalg.norm(bg-jj.T@rr)),
                          envelope_change=float(np.linalg.norm(bg-ag))))
    checks['legacy_illconditioned_cost'] = all(q['cost_difference'] < 1e-9 for q in fixed)
    checks['legacy_gradient_identity'] = all(q['gradient_identity'] < 1e-7 for q in fixed)
    answer = dict(passed=all(checks.values()), checks=checks, fd_relative=float(rel), fixed_states=fixed,
                  source_hashes={str(HERE/'stable.py'): cfg.sha(HERE/'stable.py'), str(HERE/'check.py'): cfg.sha(HERE/'check.py')})
    cfg.write(HERE/'correctness_v1.json', answer)
    print(answer)
    assert answer['passed']


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
