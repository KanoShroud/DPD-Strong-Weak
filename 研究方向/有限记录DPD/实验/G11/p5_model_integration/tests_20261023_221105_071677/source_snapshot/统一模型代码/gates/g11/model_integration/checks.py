"""Small independent reference checks; no accuracy-based parameter selection."""
import numpy as np
from scipy.linalg import lstsq

from .models import build, FS, BAND
from .solvers import profile_hessian, run
from ...g10.r4.paper_checks.assignment_metrics import self_test


KINDS = ('periodic', 'hann', 'finite_fourier', 'slepian')


def relative(a, b):
    return float(np.linalg.norm(np.asarray(a)-b)/max(np.linalg.norm(b), 1e-12))


def all_checks():
    rng = np.random.default_rng(20261024051)
    iq = rng.normal(size=(4, 128))+1j*rng.normal(size=(4, 128))
    p = np.array([[173.3, 117.8], [-137.2, -213.7]])
    points = np.array([[173., 118.], [-140., -220.], [420., 180.]])
    result = {}
    for kind in KINDS:
        model = build(iq, kind)
        fit = model.fit(p, gradient=True)
        fd = []
        for delta in np.eye(4)*.001:
            plus = model.fit((p.ravel()+delta).reshape(2, 2))['value']
            minus = model.fit((p.ravel()-delta).reshape(2, 2))['value']
            fd.append((plus-minus)/.002)
        grad_error = relative(fit['gradient'], fd)
        assert grad_error < .003, (kind, 'gradient', grad_error)
        _, h = model.hessian(p[0], model.z)
        _, ref_h = profile_hessian(model, p[0], model.z)
        h_error = relative(h, ref_h)
        assert h_error < .005, (kind, 'profile hessian', h_error)
        base = model.fit(p[:1])['value']
        scores = model.scan(points, others=p[:1])
        exact = [base-model.fit(np.vstack([p[:1], q]))['value'] for q in points]
        conditional_error = relative(scores, exact)
        assert conditional_error < 1e-7, (kind, conditional_error)
        scores_r = model.scan(points, data=fit['residual'])
        residual_energy = np.vdot(fit['residual'], fit['residual']).real/model.energy
        expected_r = [residual_energy-model.fit(q[None], data=fit['residual'])['value'] for q in points]
        assert relative(scores_r, expected_r) < 1e-7
        swapped = model.fit(p[::-1])
        assert abs(swapped['value']-fit['value']) < 1e-10
        outside = rng.normal(size=iq.shape)+1j*rng.normal(size=iq.shape)
        spectrum = np.fft.fft(outside, axis=-1)
        spectrum[:, abs(np.fft.fftfreq(128, 1/FS)) <= BAND] = 0
        changed = build(iq+10*np.fft.ifft(spectrum, axis=-1), kind)
        assert relative(model.z, changed.z) < 1e-12
        assert abs(changed.fit(p)['value']-fit['value']) < 1e-10
        detail = dict(gradient_relative=grad_error, hessian_relative=h_error,
                      conditional_schur_relative=conditional_error, outside_band=True,
                      permutation=True, residual_matching=True, metadata=model.metadata)
        if kind in ('finite_fourier', 'slepian'):
            a = model.matrix(p)
            reference = lstsq(np.vstack([a, np.sqrt(model.ridge)*np.eye(a.shape[1])]),
                np.r_[model.z.ravel(), np.zeros(a.shape[1])], lapack_driver='gelsy')[0]
            error = relative(fit['coefficients'], reference)
            assert error < 1e-6, (kind, 'augmented LS', error)
            detail['augmented_ls_relative'] = error
            # A coordinate rescaling must also transform the energy penalty.
            scale = np.geomspace(.5, 2., a.shape[1])
            aa = a*scale[None]
            reparam = lstsq(np.vstack([aa, np.sqrt(model.ridge)*np.diag(scale)]),
                np.r_[model.z.ravel(), np.zeros(a.shape[1])], lapack_driver='gelsy')[0]
            assert relative(a@reference, aa@reparam) < 1e-9
            detail['penalty_coordinate_invariance'] = True
        result[kind] = detail
    result['metric_checks'] = self_test()
    return result


def frozen_periodic_checks(iq):
    """Raw versus normalized response, relative-time gauge and CPU/GPU parity."""
    import torch
    from ...g10.r3.solvers import Model
    from ...g10.r4.adaptation_review import profile_derivatives
    from ...g10.r4.bnomp import coefficient, raw_response
    from ...research_discovery.slepian_reference import reconstruct
    rows = []
    for kind in ('periodic', 'hann'):
        m = build(iq, kind)
        # Same window order as the historical P4 criterion.
        raw = reconstruct(m.z, m.n, m.keep)
        old = Model(raw, sample_rate_hz=FS, half_band_hz=BAND)
        p = np.array([[173.3, 117.8], [-137.2, -213.7]])
        fit = m.fit(p, gradient=True)
        value, gradient = old.joint(p, True)
        np.testing.assert_allclose(fit['value'], value, rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(fit['gradient'], gradient, rtol=1e-7, atol=1e-10)
        scores = m.scan(p+3., others=p[:1])
        # Frozen conditional score is unnormalized; compare its own LS decrease.
        expected = [old.joint(p[:1])-old.joint(np.vstack([p[:1], q])) for q in p+3.]
        np.testing.assert_allclose(scores, expected, rtol=1e-7, atol=1e-10)
        a = raw_response(old, p[0]).squeeze(-1)
        q = coefficient(a, old.x)
        ov, og, oh = profile_derivatives(old, p[0], old.x, q)
        fit1, h = m.hessian(p[0], m.z)
        np.testing.assert_allclose(fit1['value'], ov, rtol=1e-10, atol=1e-11)
        np.testing.assert_allclose(fit1['gradient'], og, rtol=1e-7, atol=1e-10)
        np.testing.assert_allclose(h, oh, rtol=1e-6, atol=1e-10)
        torch.cuda.synchronize()
        rows.append(dict(kind=kind, value_error=abs(value-fit['value']),
                         gradient_relative=relative(fit['gradient'], gradient),
                         profile_hessian_relative=relative(h, oh)))
    return rows


def pipeline_smoke(iq):
    """Engineering grid only; this is not a localization/performance experiment."""
    points = np.array([[x, y] for y in (-300., 0., 300.) for x in (-300., 0., 300.)])
    rows = []
    for kind in KINDS:
        for algorithm in ('AP5', 'VP'):
            m = build(iq[:, :128], kind)
            row = run(m, algorithm, points, dict(local_maxiter=3, cycles=2, newton_steps=1))
            assert np.isfinite(row['positions']).all()
            assert row['counts']['scans'] == 2
            if algorithm == 'AP5':
                assert all(x['objective'] <= x['initial_objective']+1e-12 for x in row['branches'])
            else:
                assert all(x['after'] <= x['before']+1e-10 for x in row['history'])
            rows.append(dict(kind=kind, algorithm=algorithm, scope='engineering only, 3x3 grid and limited iterations',
                             result=row))
    return rows
