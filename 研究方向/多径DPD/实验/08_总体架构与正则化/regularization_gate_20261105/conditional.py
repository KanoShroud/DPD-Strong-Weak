"""Exact convex conditional group problem and discrepancy multiplier search."""
import numpy as np
from scipy.optimize import brentq


def prediction(h, s):
    return np.einsum('fmq,fq->fm', h, s)


def least_squares(h, y):
    if h.shape[-1] == 0:
        return np.empty((len(y), 0), complex)
    return (np.linalg.pinv(h, rcond=1e-12)@y[..., None])[..., 0]


def group_solve(h, y, alpha, initial=None, maxiter=1000, tol=1e-9):
    """All alpha strictly positive; column groups span all frequency bins."""
    if not np.all(alpha > 0):
        raise ValueError('Unpenalized groups must be eliminated first')
    nsource = h.shape[-1]
    s = least_squares(h, y) if initial is None else initial.copy()
    norm2 = float(np.vdot(y, y).real)
    if norm2 == 0:
        return np.zeros_like(s), dict(passed=True, iterations=0, gap=0., primal=0., dual=0.)
    a = np.sum(abs(h)**2, axis=1)
    residual = y-prediction(h, s)
    oldvalue = np.inf
    for iteration in range(maxiter):
        for q in range(nsource):
            residual += h[:, :, q]*s[:, q, None]
            b = np.sum(h[:, :, q].conj()*residual, axis=1)
            bn = np.linalg.norm(b)
            if bn <= alpha[q]:
                s[:, q] = 0.
            else:
                hi = alpha[q]*a[:, q].max()/(bn-alpha[q])
                def root(t):
                    if t == 0:
                        return -alpha[q]
                    return t*np.linalg.norm(b/(a[:, q]+t))-alpha[q]
                weight = brentq(root, 0., hi*(1+1e-12), xtol=1e-14, rtol=1e-13)
                s[:, q] = b/(a[:, q]+weight)
            residual -= h[:, :, q]*s[:, q, None]
        penalty = float(alpha@np.linalg.norm(s, axis=0))
        primal = .5*float(np.vdot(residual, residual).real)+penalty
        score = np.linalg.norm(np.sum(h.conj()*residual[:, :, None], axis=1), axis=0)
        shrink = min(1., float(np.min(np.divide(alpha, score, out=np.full_like(alpha, np.inf), where=score>0))))
        dualres = shrink*residual
        dual = float(np.vdot(dualres, y).real-.5*np.vdot(dualres, dualres).real)
        gap = primal-dual
        if primal > oldvalue+1e-10*norm2:
            return s, dict(passed=False, reason='nonmonotone', iterations=iteration+1, gap=gap)
        oldvalue = primal
        if -tol*norm2 <= gap <= tol*norm2:
            return s, dict(passed=True, iterations=iteration+1, gap=gap, primal=primal, dual=dual)
    return s, dict(passed=False, reason='iteration_limit', iterations=maxiter, gap=gap, primal=primal, dual=dual)


def calibrate(h, y, c, radius2):
    """Conditional Morozov value. No truth, noise variance or model order input."""
    c = np.asarray(c, float)
    if np.any(c < 0) or radius2 < 0:
        raise ValueError('Nonnegative c and radius required')
    scale = np.linalg.norm(y)
    if scale == 0:
        return np.zeros((len(y), len(c)), complex), dict(status='zero_data', passed=True, value=0., rss=0., multiplier=0., inner_cycles=0)
    yn = y/scale; target = radius2/scale**2
    penalized = np.flatnonzero(c>0); free = np.flatnonzero(c==0)
    hu = h[:, :, free]; hp = h[:, :, penalized]
    if len(free):
        inv = np.linalg.pinv(hu, rcond=1e-12)
        yp = yn-prediction(hu, (inv@yn[..., None])[..., 0])
        hp = hp-hu@(inv@hp)
    else:
        inv = None; yp = yn

    def restore(sp):
        z = np.zeros((len(y), len(c)), complex)
        z[:, penalized] = sp
        if len(free):
            residual = yn-prediction(h[:, :, penalized], sp)
            z[:, free] = (inv@residual[..., None])[..., 0]
        return z*scale

    sp = least_squares(hp, yp)
    rss_min = float(np.linalg.norm(yp-prediction(hp, sp))**2)
    if rss_min > target+1e-10:
        return restore(sp), dict(status='infeasible_fixed_channel', passed=False, value=float(c@np.linalg.norm(restore(sp), axis=0)),
            rss=rss_min*scale**2, multiplier=None, inner_cycles=0)
    if np.linalg.norm(yp)**2 <= target+1e-10 or len(penalized)==0:
        z = restore(np.zeros_like(sp))
        return z, dict(status='zero_penalty_optimum', passed=True, value=0., rss=float(np.linalg.norm(y-prediction(h,z))**2), multiplier=0., inner_cycles=0)
    if abs(rss_min-target) <= 1e-10:
        return restore(sp), dict(status='least_squares_boundary', passed=False, value=float(c@np.linalg.norm(restore(sp),axis=0)),rss=rss_min*scale**2,multiplier=0.,inner_cycles=0)
    cp = c[penalized]
    upper = float(np.max(np.linalg.norm(np.sum(hp.conj()*yp[:, :, None],axis=1),axis=0)/cp))
    lower = 0.; ledger = []; cycles = 0
    for iteration in range(50):
        lam = .5*(lower+upper)
        sp, info = group_solve(hp, yp, lam*cp, sp)
        cycles += info['iterations']
        rss = float(np.linalg.norm(yp-prediction(hp, sp))**2)
        ledger.append(dict(multiplier_normalized=lam, rss_normalized=rss, gap=info['gap'], inner_pass=info['passed']))
        if not info['passed']:
            return restore(sp), dict(status='inner_not_certified', passed=False, value=float(c@np.linalg.norm(restore(sp),axis=0)),rss=rss*scale**2,multiplier=lam*scale,inner_cycles=cycles,trace=ledger)
        if abs(rss-target) <= 1e-8:
            z = restore(sp)
            return z, dict(status='interior_discrepancy', passed=True, value=float(c@np.linalg.norm(z,axis=0)),rss=rss*scale**2,
                multiplier=lam*scale, inner_cycles=cycles, iterations=iteration+1, trace=ledger)
        if rss > target:
            upper = lam
        else:
            lower = lam
    return restore(sp), dict(status='outer_not_certified',passed=False,value=float(c@np.linalg.norm(restore(sp),axis=0)),rss=rss*scale**2,multiplier=lam*scale,inner_cycles=cycles,trace=ledger)


def selftest():
    rng = np.random.default_rng(2026110519)
    h = rng.normal(size=(24,4,2))+1j*rng.normal(size=(24,4,2))
    y = rng.normal(size=(24,4))+1j*rng.normal(size=(24,4))
    c = np.array([1.2,.8])
    sls = least_squares(h,y); rmin=np.linalg.norm(y-prediction(h,sls))**2
    radius = .5*(rmin+np.linalg.norm(y)**2)
    s, val = calibrate(h,y,c,radius)
    z=np.array([3*np.exp(.4j),.2*np.exp(-.3j)])
    sz,vz=calibrate(h*z,y,c*abs(z),radius)
    _, infeasible=calibrate(h,y,c,rmin*.5)
    _, zero=calibrate(h,y,c,np.linalg.norm(y)**2*1.01)
    sfree,free=calibrate(h,y,np.zeros(2),radius)
    _, onefree=calibrate(h,y,np.array([0.,1.]),radius)
    dh=(rng.normal(size=h.shape)+1j*rng.normal(size=h.shape))*.1
    dc=np.array([.03,-.02]); step=1e-4
    _, plus=calibrate(h+step*dh,y,c+step*dc,radius)
    _, minus=calibrate(h-step*dh,y,c-step*dc,radius)
    predicted=float(dc@np.linalg.norm(s,axis=0)+np.vdot(prediction(h,s)-y,prediction(dh,s)).real/val['multiplier'])
    fd=(plus['value']-minus['value'])/(2*step)
    checks=dict(interior=val['passed'],radius=abs(val['rss']-radius)<1e-8*np.linalg.norm(y)**2,
        complex_scale=abs(val['value']-vz['value'])<1e-6*val['value'] and np.linalg.norm(prediction(h,s)-prediction(h*z,sz))<1e-4,
        infeasible=infeasible['status']=='infeasible_fixed_channel',zero_group=zero['status']=='zero_penalty_optimum',
        all_unpenalized=free['passed'] and np.linalg.norm(sfree-sls)<1e-10,
        one_unpenalized=onefree['passed'],envelope=bool(abs(fd-predicted)<2e-3*max(1.,abs(predicted))))
    return dict(passed=all(checks.values()), checks=checks, directional_difference=fd,directional_prediction=predicted,
        interior=val,one_unpenalized_status=onefree['status'])
