"""R1 numerical and contract tests, no formal performance assertions."""
import io
import json
import unittest
from unittest.mock import patch
import numpy as np
import torch
from ..test_g10 import Algebra
from ..physics import fit, Model as ReferenceModel, sbl_posterior, grid
from .physics import Model, small_fit, Spectrum
from .scenes import make, geometry_audit, task_list
from .config import LAYOUTS


class R1Tests(unittest.TestCase):
    def test_geometry_and_counts(self):
        self.assertEqual(len(geometry_audit()), 4)
        tasks = list(task_list())
        self.assertEqual(len(tasks), 520)
        self.assertEqual(sum(t[-1] == 'weak' for t in tasks), 40)
        self.assertEqual(len(grid(20.)), 10201)
        self.assertLess(20/np.sqrt(2), 20)

    def test_pairing_realizations(self):
        a, b, c = make(0, 0, 0, n=512), make(0, 0, 15, n=512), make(0, 1, 0, n=512)
        np.testing.assert_array_equal(a['noise'], b['noise'])
        np.testing.assert_array_equal(a['signal'][1], b['signal'][1])
        np.testing.assert_allclose(b['signal'][0], a['signal'][0]*10**(15/20), rtol=1e-13)
        self.assertFalse(np.array_equal(a['noise'], c['noise']))
        np.testing.assert_array_equal(a['positions'], c['positions'])

    def test_fast_fit_and_gradient(self):
        rng = np.random.default_rng(92)
        iq = rng.normal(size=(4, 128))+1j*rng.normal(size=(4, 128))
        m, old = Model(iq), ReferenceModel(iq)
        for pp in (np.asarray(LAYOUTS[0]), np.array([[173., 117.]]), np.array([[173., 117.]]*2)):
            a = m.response(pp)
            actual = small_fit(a, m.x)[1]
            expected = fit(a, m.x)[1]
            np.testing.assert_allclose(actual.cpu(), expected.cpu(), atol=1e-10, rtol=1e-9)
            self.assertAlmostEqual(m.joint(pp), old.joint(pp), places=10)
        pp = np.asarray(LAYOUTS[0])
        v, g = m.joint(pp, True)
        vv, gg = old.joint(pp, True)
        np.testing.assert_allclose(g, gg, rtol=1e-7, atol=1e-10)
        self.assertAlmostEqual(v, vv, places=10)
        np.testing.assert_allclose(m.joint_batch(np.stack([pp, pp+1])),
                                  [old.joint(pp), old.joint(pp+1)], atol=1e-10)

    def test_spectrum_gradient(self):
        scene = make(0, 0, n=512)
        m = Model(scene['signal'][1]+scene['noise'])
        for method in ('MUSIC', 'MVDR'):
            for weight in ('uniform', 'excess_power'):
                spec = Spectrum(m, method, 1, 8, 1e-3, weight)
                p = np.asarray([167., 122.])
                v, g = spec.value_grad(p, 1.)
                finite = np.array([(-spec.evaluate((p+d)[None])[0]+spec.evaluate((p-d)[None])[0])/.002
                                   for d in np.eye(2)*.001])
                np.testing.assert_allclose(g, finite, rtol=1e-5, atol=1e-8)
                self.assertAlmostEqual(v, -spec.evaluate(p[None])[0], places=9)

    def test_exact_dictionary_cache(self):
        rng = np.random.default_rng(94)
        iq = rng.normal(size=(4, 32))+1j*rng.normal(size=(4, 32))
        m, reference = Model(iq), ReferenceModel(iq)
        pp = grid(20.)
        self.assertTrue(m.enable_response_cache(pp, 8))
        for ids in (np.arange(70, 90), np.array([0, 101, 2087])):
            np.testing.assert_allclose(m.response(pp[ids], m.f[3:8]).cpu(),
                                      reference.response(pp[ids], reference.f[3:8]).cpu(), atol=1e-13)
        moved = pp.copy(); moved[[1, 135]] += .123
        m.refresh_response_cache(moved, 8)
        np.testing.assert_allclose(m.response(moved, m.f[4:7]).cpu(),
                                  reference.response(moved, reference.f[4:7]).cpu(), atol=1e-13)
        np.testing.assert_allclose(m.conditional(pp[300:320]), reference.conditional(pp[300:320]), atol=1e-12)
        self.assertGreater(m.counts['dictionary_cache_hits'], 0)

    def test_trace_not_a_deadline(self):
        from .methods import Trace, at_time
        with patch('time.perf_counter', side_effect=[0., 500., 501.]):
            t = Trace(lambda: None)
            t.check()
            t.save([[1, 2]], .5)
        self.assertEqual(at_time(dict(events=t.events), 60), [])
        self.assertEqual(at_time(dict(events=t.events), 600), [[1, 2]])

    def test_sbl_update_reference(self):
        rng = np.random.default_rng(93)
        a = torch.tensor(rng.normal(size=(8, 4, 7))+1j*rng.normal(size=(8, 4, 7)), device='cuda')
        x = torch.tensor(rng.normal(size=(8, 4))+1j*rng.normal(size=(8, 4)), device='cuda')
        gamma = torch.ones(7, dtype=torch.float64, device='cuda')*.1
        mu, diag, cov, old = sbl_posterior(a, x, gamma, .3)
        num = (abs(mu/gamma[None])**2).sum(0)
        den = ((gamma[None]-diag)/gamma[None]**2).sum(0)
        inv = torch.linalg.inv(cov)
        refnum = abs((a.mH@inv@x[..., None]).squeeze(-1)).square().sum(0)
        refden = (a.mH@inv@a).diagonal(dim1=-2, dim2=-1).real.sum(0)
        np.testing.assert_allclose(num.cpu(), refnum.cpu(), rtol=1e-10)
        np.testing.assert_allclose(den.cpu(), refden.cpu(), rtol=1e-10)
        updated = gamma*torch.sqrt(num/den)
        new = sbl_posterior(a, x, updated, .3)[3]
        self.assertLessEqual(float(new.sum()), float(old.sum())+1e-9)


def suite():
    tests = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(Algebra),
                               unittest.defaultTestLoader.loadTestsFromTestCase(R1Tests)])
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(tests)
    return dict(passed=result.wasSuccessful(), tests=result.testsRun, log=stream.getvalue())


if __name__ == '__main__':
    result = suite()
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result['passed'] else 1)
