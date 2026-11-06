"""Local perturbation diagnostics, explicitly not blind localization performance."""
import argparse
import time
import warnings

import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

from common import (UNIT, STATIONS, B, C, cases, preflight, write, load,
                    spectra, verify, source_indices, regularity)
from boundary_update import closure
from structural_probe import response

ALPHAS = [0., .05, .1, 1.]


def los_response(x, f):
    p = x[:4].reshape(2, 2)*UNIT
    gains = np.ones((5, 2), complex)
    gains[1:] = x[4:12].reshape(4, 2)+1j*x[12:20].reshape(4, 2)
    displacement = p[None]-STATIONS[:, None]
    distance = np.linalg.norm(displacement, axis=2)
    phase = np.exp(-2j*np.pi*f[:, None, None]*distance[None]/C)
    h = phase*gains[None]
    dh = np.zeros((len(f), 5, 2, 20), complex)
    direction = displacement/distance[:, :, None]
    for q in range(2):
        dh[:, :, q, 2*q:2*q+2] = h[:, :, q, None]*(-2j*np.pi*f[:, None, None]/B)*direction[None, :, q]
        for m in range(1, 5):
            k = (m-1)*2+q
            dh[:, m, q, 4+k] = phase[:, m, q]
            dh[:, m, q, 12+k] = 1j*phase[:, m, q]
    return h, dh


def evaluate(x, f, y):
    h, dh = los_response(x, f)
    inv = np.linalg.pinv(h, rcond=1e-12)
    s = np.einsum('fqm,fm->fq', inv, y)
    r = y-np.einsum('fmq,fq->fm', h, s)
    ds = np.einsum('fmqv,fq->fmv', dh, s)
    j = -ds+np.einsum('fmq,fqv->fmv', h, np.einsum('fqm,fmv->fqv', inv, ds))
    j -= np.einsum('fqm,fqv->fmv', inv.conj(), np.einsum('fmqv,fm->fqv', dh.conj(), r))
    scale = np.sqrt(2)/np.linalg.norm(y)
    return scale*np.r_[r.real.ravel(), r.imag.ravel()], scale*np.concatenate(
        [j.real.reshape(-1, 20), j.imag.reshape(-1, 20)])


def fit(x, f, y):
    cache = [None, None]
    def value(z):
        if cache[0] is None or not np.array_equal(z, cache[0]):
            cache[:] = [z.copy(), evaluate(z, f, y)]
        return cache[1]
    lo = np.r_[np.full(4, -3000/UNIT), np.full(16, -np.inf)]
    opt = least_squares(lambda z: value(z)[0], x, jac=lambda z: value(z)[1],
                        bounds=(lo, -lo), x_scale='jac', max_nfev=300,
                        ftol=1e-12, xtol=1e-12, gtol=1e-12)
    r, j = value(opt.x)
    return opt.x, dict(cost=.5*float(r@r), gradient=float(np.max(abs(j.T@r))),
                      status=int(opt.status), nfev=int(opt.nfev), message=opt.message,
                      interior=bool(np.min(3000/UNIT-abs(opt.x[:4])) > 1e-5))


def gradient(x, f, y):
    r, j = evaluate(x, f, y)
    return j.T@r


def hessian(x, f, y, step):
    h = np.column_stack([(gradient(x+step*np.eye(20)[k], f, y)-
                           gradient(x-step*np.eye(20)[k], f, y))/(2*step) for k in range(20)])
    skew = np.linalg.norm(h-h.T)/max(np.linalg.norm(h), 1e-30)
    return .5*(h+h.T), float(skew)


def prediction(x, f, y, parts, base):
    r, j = evaluate(x, f, y)
    gn = j.T@j
    h1, skew1 = hessian(x, f, y, 1e-4)
    h, skew = hessian(x, f, y, 5e-5)
    eigen = np.linalg.eigvalsh(h)
    fd = np.linalg.norm(h-h1)/max(np.linalg.norm(h), 1e-30)
    regular = bool(eigen[0] > eigen[-1]*1e-10 and base['gradient'] <= 1e-8 and base['interior'])
    slopes, gn_slopes, qchecks = [], [], []
    for e in parts:
        qs = [(gradient(x, f, y+eps*e)-gradient(x, f, y-eps*e))/(2*eps) for eps in [1e-4, 5e-5]]
        qchecks.append(float(np.linalg.norm(qs[1]-qs[0])/max(np.linalg.norm(qs[1]), 1e-30)))
        dr = (evaluate(x, f, y+5e-5*e)[0]-evaluate(x, f, y-5e-5*e)[0])/1e-4
        gn_slopes.append(-np.linalg.solve(gn, j.T@dr))
        slopes.append(-np.linalg.solve(h, qs[1]) if regular else None)
    # Independently check source additivity, without choosing directions using truth errors.
    e = parts.sum(axis=0)
    qtot = (gradient(x, f, y+5e-5*e)-gradient(x, f, y-5e-5*e))/1e-4
    total = -np.linalg.solve(h, qtot) if regular else None
    return dict(base_residual_norm=float(np.linalg.norm(r)), hessian_skew=skew,
                hessian_skew_coarse=skew1, hessian_step_relative=float(fd), eigenvalues=eigen,
                q_step_relative=qchecks, implicit_regular=regular,
                exact_minus_gn_relative=float(np.linalg.norm(h-gn)/np.linalg.norm(gn)),
                slopes=slopes, gn_slopes=gn_slopes, total_slope=total,
                additivity_error=None if total is None else float(np.linalg.norm(total-sum(slopes))))


def vector_error(pred, actual):
    norm = float(np.linalg.norm(actual))
    pnorm = float(np.linalg.norm(pred))
    return dict(absolute_m=float(np.linalg.norm(pred-actual)),
                relative=None if norm < 1e-7 else float(np.linalg.norm(pred-actual)/norm),
                cosine=None if min(norm, pnorm) < 1e-7 else float(np.clip(pred@actual/(norm*pnorm), -1, 1)))


def run_case(data, mode):
    f, actual, parts = spectra(data)
    gain = data['g'][:, :, 0]/data['g'][0, :, 0]
    xtrue = np.r_[data['p'].ravel()/UNIT, gain[1:].real.ravel(), gain[1:].imag.ravel()]
    if mode == 'matched_control':
        h, _ = los_response(xtrue, f)
        losparts = np.fft.fft(data['los'], axis=2, norm='ortho')[:, :, abs(np.fft.fftfreq(8192, 1/50e6)) <= B/2].transpose(2, 1, 0)
        coeff = np.sum(h.conj()*losparts, axis=1)/np.sum(abs(h)**2, axis=1)
        actual = np.einsum('fmq,fq->fm', h, coeff)
        hf, _ = response(data['p'], data['d'], data['g']/data['g'][0, :, 0][None, :, None], f)
        parts = np.einsum('fmq,fq->qfm', hf-h, coeff)
    x0, base = fit(xtrue, f, actual)
    pred = prediction(x0, f, actual, parts, base)
    rows = []
    for alpha in ALPHAS:
        if alpha == 0:
            xx, info = x0, base
        else:
            xx, info = fit(x0, f, actual+alpha*parts.sum(axis=0))
        delta = (xx[:4]-x0[:4])*UNIT
        gn = alpha*sum(pred['gn_slopes'])[:4]*UNIT
        exact = None if pred['total_slope'] is None else alpha*pred['total_slope'][:4]*UNIT
        rows.append(dict(alpha=alpha, delta_m=delta, positions=xx[:4].reshape(2, 2)*UNIT,
                         fit=info, gn_prediction_m=gn, exact_prediction_m=exact,
                         gn_error=vector_error(gn, delta),
                         exact_error=None if exact is None else vector_error(exact, delta)))
    return dict(mode=mode, oracle_local_diagnostic=True, base=base, prediction=pred,
                base_position_error_m=(x0[:4]-xtrue[:4])*UNIT, rows=rows)


def selftest(data):
    rng = np.random.default_rng(2026103002)
    checks = {}
    # Characterization of Euclidean projection onto the CLOSED product set.
    worst_vi = 0.
    worst_firm = 0.
    for _ in range(100):
        x, y = rng.normal(size=(2, 80))*3
        x[:4] *= 100
        y[:4] *= 100
        x[60:] *= 10
        y[60:] *= 10
        px, py = closure(x), closure(y)
        expected = x.copy()
        expected[:4] = np.clip(x[:4], -3000/UNIT, 3000/UNIT)
        for m in range(5):
            for q in range(2):
                ids = [20+4*m+2*q, 21+4*m+2*q, 40+4*m+2*q, 41+4*m+2*q]
                expected[ids] /= max(1., np.linalg.norm(expected[ids]))
        expected[60:] = np.clip(expected[60:], 3, 20)
        assert np.allclose(px, expected, rtol=0, atol=1e-13)
        assert np.allclose(closure(px), px, rtol=0, atol=1e-13)
        worst_vi = max(worst_vi, float((x-px)@(py-px)))
        worst_firm = max(worst_firm, float(np.linalg.norm(px-py)**2-(px-py)@(x-y)))
    checks['projection_formula_idempotence_100'] = True
    checks['projection_variational'] = worst_vi < 1e-9
    checks['projection_firm_nonexpansive'] = worst_firm < 1e-9
    checks['source_partition'] = np.array_equal(np.sort(np.r_[source_indices(0), source_indices(1)]), np.arange(80))
    f, y, _ = spectra(data)
    gains = data['g'][:, :, 0]/data['g'][0, :, 0]
    x = np.r_[data['p'].ravel()/UNIT, gains[1:].real.ravel(), gains[1:].imag.ravel()]
    rr, jj = evaluate(x, f, y)
    from solver_reliability import evaluate as original
    from position_reduction import position_jac
    v = np.r_[x[4:], np.zeros(40), np.full(20, 8.)]
    r0, j0, _ = original(v, data['p'], f, y, 'physical')
    jp = position_jac(v, data['p'], f, y)
    checks['frozen_residual_parity'] = bool(np.allclose(rr, r0, atol=1e-12, rtol=1e-10))
    checks['frozen_jacobian_parity'] = bool(np.allclose(jj, np.c_[jp, j0[:, :16]], atol=1e-11, rtol=1e-9))
    fd = np.column_stack([(evaluate(x+1e-5*np.eye(20)[k], f, y)[0]-evaluate(x-1e-5*np.eye(20)[k], f, y)[0])/2e-5 for k in range(20)])
    checks['analytic_jacobian'] = np.linalg.norm(fd-jj)/np.linalg.norm(jj) < 1e-7
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['short', 'full'], required=True)
    args = parser.parse_args()
    short = args.phase == 'short'
    out, hashes = preflight('task1_'+args.phase, short)
    start = time.perf_counter()
    checks, rows, regular = {}, [], []
    for idx, (tag, inp, folder) in enumerate(cases(short)):
        data = dict(np.load(inp))
        if idx == 0:
            checks.update(selftest(data))
            assert all(checks.values()), checks
        for mode in (['matched_control', 'finite_iq'] if short else ['finite_iq']):
            before = time.perf_counter()
            r = run_case(data, mode)
            r.update(tag=tag, seconds=time.perf_counter()-before)
            if mode == 'matched_control':
                checks[tag+'/zero_residual'] = r['prediction']['base_residual_norm'] < 1e-10
                checks[tag+'/zero_residual_hessian_equals_gn'] = r['prediction']['exact_minus_gn_relative'] < 1e-6
            checks[tag+'/'+mode+'/hessian_fd'] = r['prediction']['hessian_step_relative'] < 1e-5
            checks[tag+'/'+mode+'/hessian_symmetry'] = r['prediction']['hessian_skew'] < 1e-5
            checks[tag+'/'+mode+'/q_fd'] = max(r['prediction']['q_step_relative']) < 1e-5
            rows.append(r)
            write(out/(tag+'_'+mode+'.json'), r)
            print(tag, mode, 'residual', round(r['prediction']['base_residual_norm'], 5),
                  'regular', r['prediction']['implicit_regular'], 's', round(r['seconds'], 2), flush=True)
        for path in ['LOS', 'MP']:
            rec = load(folder/(tag+'_'+path+'.json'))
            for stage, pp, vv in [('seed', rec['los_p'], rec['seeded_v']), ('final', rec['mp']['p'], rec['mp']['v'])]:
                f, _, _ = spectra(data)
                rg = regularity(np.r_[np.asarray(pp).ravel()/UNIT, vv], f)
                rg.update(tag=tag+'_'+path, stage=stage)
                regular.append(rg)
    identities = verify(hashes)
    checks['identities'] = all(identities.values())
    checks['saved_feasibility'] = all(v['feasible'] for v in regular)
    checks['saved_rank_regular'] = all(v['min_rank_ratio'] > 1e-10 for v in regular)
    write(out/'result.json', dict(rows=rows, checks=checks, regularity=regular, identities=identities,
          seconds=time.perf_counter()-start, scope='THEORY_DIAGNOSTICS_NOT_BLIND_PERFORMANCE'))
    write(out.parent/('task1_'+args.phase+'_latest.json'), dict(path=str(out)))
    assert all(checks.values()), checks
    print('任务一完成', len(rows), '检查', len(checks), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2), warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='divide by zero encountered in log10', category=RuntimeWarning)
        main()
