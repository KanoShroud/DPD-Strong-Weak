"""Fixed-candidate cone likelihood diagnostic, not a calibrated full detector."""
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, nnls

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent/'source_path_gate_20261105'
sys.path.insert(0, str(PREVIOUS))
from diagnose import cfg, ad, stable, data, INPUT  # noqa: E402,F401


def tangent_constraints(e, x, n, tol=1e-8):
    rows, labels = [], []
    for k in range(e.D, len(x), 3):
        rho = x[k:k+2]
        if np.linalg.norm(rho) >= 1-tol:
            row = np.zeros(len(x)); row[k:k+2] = rho
            rows.append(row); labels.append(f'disk_{(k-e.D)//3}')
    limits = [(k, -3000/e.ref.UNIT, 3000/e.ref.UNIT) for k in range(4)]
    limits += [(k, 0., n/cfg.FS*cfg.B) for k in range(e.D+2, len(x), 3)]
    for k, lo, hi in limits:
        if x[k] <= lo+tol:
            row = np.zeros(len(x)); row[k] = -1
            rows.append(row); labels.append(f'lower_{k}')
        if x[k] >= hi-tol:
            row = np.zeros(len(x)); row[k] = 1
            rows.append(row); labels.append(f'upper_{k}')
    return np.asarray(rows).reshape(-1, len(x)), labels


def cone_ls(a, r, c):
    """min ||r-a*d||^2 subject to c*d<=0; original scale retained in output."""
    norm = np.linalg.norm(r)
    assert norm > 0
    scale = np.linalg.norm(a, axis=0)
    scale = np.where(scale > 0, scale, 1.)
    az = a/scale
    # Sufficient statistics preserve the exact quadratic without regularization.
    gram = az.T@az
    rhs = az.T@(r/norm)
    cz = c/scale
    if len(cz):
        cz = cz/np.linalg.norm(cz, axis=1, keepdims=True)
    def fun(v):
        return float(.5*v@gram@v-rhs@v), gram@v-rhs
    opt = minimize(fun, np.zeros(a.shape[1]), jac=True, method='SLSQP',
        constraints=[dict(type='ineq', fun=lambda v: -cz@v, jac=lambda v: -cz)] if len(cz) else [],
        options=dict(maxiter=1000, ftol=1e-12))
    v = opt.x
    g = gram@v-rhs
    cv = cz@v
    active = cv >= -1e-7
    lam = np.zeros(len(cz))
    if np.any(active):
        lam[active] = nnls(cz[active].T, -g, maxiter=10000)[0]
    kkt = float(np.linalg.norm(g+cz.T@lam, ord=np.inf))
    feasibility = float(max(0., np.max(cv))) if len(cv) else 0.
    complementarity = float(np.max(abs(lam*cv))) if len(cv) else 0.
    d = v*norm/scale
    residual = r-a@d
    # Tolerances certify numerical QP solution, not a new localization acceptance.
    certified = feasibility < 1e-8 and kkt < 2e-6 and complementarity < 1e-7
    return d, dict(rss=float(residual@residual), relative_rss=float(residual@residual/norm**2),
        kkt=kkt, feasibility=feasibility, complementarity=complementarity,
        certified=certified, iterations=int(opt.nit), success=bool(opt.success), message=str(opt.message))


def score(a, b, r, c):
    d0, null = cone_ls(a, r, c)
    ab = np.column_stack([a, b])
    cc = np.column_stack([c, np.zeros((len(c), b.shape[1]))])
    d1, alt = cone_ls(ab, r, cc)
    gain = null['rss']-alt['rss']
    return d0, d1, dict(null=null, alternative=alt, gain=gain,
        certified=bool(null['certified'] and alt['certified'] and gain >= -1e-8*(r@r)))


def selftest():
    rng = np.random.default_rng(2026110511)
    a = rng.normal(size=(40, 5)); b = rng.normal(size=(40, 2)); r = rng.normal(size=40)
    _, _, out = score(a, b, r, np.zeros((0, 5)))
    rp = r-a@(np.linalg.pinv(a)@r)
    bp = b-a@(np.linalg.pinv(a)@b)
    expected = float(np.linalg.norm(bp@(np.linalg.pinv(bp)@rp))**2)
    checks = dict(interior_analytic=abs(out['gain']-expected) < 1e-8, interior_kkt=out['certified'])
    for i, r in enumerate([np.array([-2., 1., 3.]), np.array([2., -1., -3.])]):
        a = np.array([[1.], [0.], [0.]])
        b = np.array([[.8, 0.], [.6, 0.], [0., 1.]])
        _, _, out = score(a, b, r, np.array([[-1.]]))
        n = np.array([.6, -.8, 0.])
        expected = r@r-max(r[0], 0.)**2-min(n@r, 0.)**2
        checks[f'boundary_{i}'] = abs(out['gain']-expected) < 1e-8 and out['certified']
    assert all(checks.values()), checks
    return checks
