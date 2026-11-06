"""Algebra, gradients, pair identities, metrics, timeout and source-isolated tests."""
import json
import os
import subprocess
import sys
import unittest
import numpy as np
import torch
from .physics import Model, atom, fit, Spectrum, sbl_posterior
from .scenes import make, observation
from .metrics import evaluate
from .methods import at_time, run


class Algebra(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1234)
        self.iq = self.rng.normal(size=(4, 128))+1j*self.rng.normal(size=(4, 128))
        self.m = Model(self.iq)
        self.p = np.array([[-371.3, 421.7], [648.2, -123.4]])

    def test_projection_lstsq_noiseless_and_rank(self):
        for count in (1, 2):
            a = self.m.response(self.p[:count])
            x = self.m.x
            s, r = fit(a, x)
            aa, xx = a.cpu().numpy(), x.cpu().numpy()
            expected = np.stack([np.linalg.lstsq(ai, xi, rcond=1e-10)[0] for ai, xi in zip(aa, xx)])
            np.testing.assert_allclose(s.cpu(), expected, rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose((a.mH@r[..., None]).cpu(), 0, atol=1e-10)
            clean = (a@s[..., None]).squeeze(-1)
            _, residual = fit(a, clean)
            self.assertLess(float(residual.abs().max().cpu()), 1e-10)
        a = self.m.response(np.repeat(self.p[:1], 2, axis=0))
        _, r = fit(a, self.m.x)
        self.assertTrue(torch.isfinite(r).all())

    def test_joint_gradient(self):
        value, grad = self.m.joint(self.p, True)
        flat = self.p.ravel()
        finite = []
        for i in range(4):
            delta = np.eye(4)[i]*.001
            finite.append((self.m.joint(flat+delta)-self.m.joint(flat-delta))/.002)
        np.testing.assert_allclose(grad, finite, rtol=1e-5, atol=1e-8)
        self.assertGreaterEqual(value, 0)

    def test_conditional_gain(self):
        other = self.p[:1]
        v1 = self.m.joint(other)
        gain = self.m.conditional(self.p[1:], other)[0]
        self.assertAlmostEqual(v1-self.m.joint(self.p), gain, places=10)
        np.testing.assert_allclose(self.m.conditional(self.p, chunk=1),
                                   self.m.conditional(self.p, chunk=64), atol=1e-12)

    def test_joint_population_equivalence(self):
        pp = np.stack([self.p, self.p+3, self.p-5])
        np.testing.assert_allclose(self.m.joint_batch(pp), [self.m.joint(p) for p in pp], rtol=1e-10, atol=1e-12)

    def test_noiseless_offgrid_refinement(self):
        from .methods import refine_joint
        a = self.m.response(self.p)
        s = torch.as_tensor(self.rng.normal(size=(128, 2))+1j*self.rng.normal(size=(128, 2)), device='cuda')
        x = (a@s[..., None]).squeeze(-1).cpu().numpy()
        model = Model(np.fft.ifft(x.T*128))
        result, value = refine_joint(model, self.p+.05, maxiter=60)
        np.testing.assert_allclose(result, self.p, atol=.01)
        self.assertLess(value, 1e-9)

    def test_newton_hessian(self):
        z = torch.tensor(self.p[0], device='cuda', requires_grad=True)
        fun = lambda p: self.m.residual_score(p, self.m.x)
        h = torch.autograd.functional.hessian(fun, z).cpu().numpy()
        finite = []
        for step in np.eye(2)*.001:
            def grad(p):
                q = torch.tensor(p, device='cuda', requires_grad=True)
                return torch.autograd.grad(fun(q), q)[0].cpu().numpy()
            finite.append((grad(self.p[0]+step)-grad(self.p[0]-step))/.002)
        np.testing.assert_allclose(h, np.asarray(finite).T, rtol=1e-5, atol=1e-8)

    def test_spectrum_reference(self):
        for method in ('MUSIC', 'MVDR'):
            spec = Spectrum(self.m, method, 2, 4, 1e-3)
            result = spec.evaluate(self.p)
            a = self.m.response(self.p, spec.freq).cpu().numpy()
            matrix = spec.matrix.cpu().numpy()
            den = np.array([[np.vdot(a[f, :, j], matrix[f]@a[f, :, j]).real
                             for j in range(2)] for f in range(len(a))])
            expected = 1/den.mean(0) if method == 'MUSIC' else (1/den).mean(0)
            np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_sbl_woodbury_complex(self):
        a = self.m.response(self.p)[:4]
        x = self.m.x[:4]
        gamma = torch.tensor([.13, .27], device='cuda', dtype=torch.float64)
        noise = .02
        mu, diagonal, _, evidence = sbl_posterior(a, x, gamma, noise)
        cov = torch.linalg.inv(a.mH@a/noise+torch.diag(1/gamma))
        expected = (cov@a.mH@x[..., None]).squeeze(-1)/noise
        np.testing.assert_allclose(mu.cpu(), expected.cpu(), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(diagonal.cpu(), cov.diagonal(dim1=1, dim2=2).real.cpu(), rtol=1e-10)
        new_gamma = (mu.abs().square()+diagonal).mean(0)
        _, _, _, new_evidence = sbl_posterior(a, x, new_gamma, noise)
        self.assertLessEqual(float(new_evidence.sum()), float(evidence.sum())+1e-10)

    def test_atom_derivative_and_amplitude(self):
        p = torch.tensor(self.p, device='cuda', requires_grad=True)
        a = atom(p, self.m.f[:5], self.m.stations, normalized=False)
        d = torch.linalg.vector_norm(p[:, None]-self.m.stations, dim=-1)
        np.testing.assert_allclose(abs(a.detach().cpu()).numpy()[0], 1/d.detach().cpu().numpy().T)
        self.assertTrue(torch.autograd.grad(a.real.sum(), p)[0].isfinite().all())

    def test_metric_and_timeout(self):
        self.assertEqual(evaluate([[0, 0]], [[20, 0]])['success']['20'], [True])
        self.assertEqual(evaluate([[0, 0]], [[20.001, 0]])['success']['20'], [False])
        self.assertEqual(evaluate([[0, 0], [100, 0]], [[100, 0], [0, 0]])['success']['20'], [True, True])
        self.assertEqual(sum(evaluate([[0, 0], [10, 0]], [[5, 0]])['success']['20']), 1)
        self.assertEqual(evaluate([[0, 0]], [])['matched'], 0)
        result = run(self.iq, 'LS', budget=0.)
        self.assertEqual(result['status'], 'BUDGET_EXHAUSTED')
        self.assertEqual(at_time(result, 60), [])

    def test_paired_centers_and_powers(self):
        a = make('smoke', 0, center='random', n=512)
        b = make('smoke', 0, center='zero', n=512)
        c = make('smoke', 0, center='zero', gap_db=15, n=512)
        np.testing.assert_array_equal(a['noise'], b['noise'])
        np.testing.assert_array_equal(a['positions'], b['positions'])
        np.testing.assert_allclose(abs(a['signal']), abs(b['signal']), rtol=1e-13, atol=1e-15)
        np.testing.assert_array_equal(b['signal'][1], c['signal'][1])
        np.testing.assert_allclose(c['signal'][0], b['signal'][0]*10**(3/20), rtol=1e-13)
        self.assertAlmostEqual(a['meta']['actual_gap_db'], 12.)
        self.assertEqual(observation(a, 'weak')[2], 1)

    def test_non_ascii_subprocess(self):
        env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8:replace')
        r = subprocess.run([sys.executable, '-c', "print('G10 中文日志 测试✓')"],
            text=True, encoding='utf-8', errors='replace', capture_output=True, env=env, check=True)
        self.assertIn('中文日志', r.stdout)


def suite():
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Algebra))
    return dict(passed=result.wasSuccessful(), tests=result.testsRun,
                failures=[str(x) for x in result.failures], errors=[str(x) for x in result.errors])


if __name__ == '__main__':
    result = suite()
    print(json.dumps(result, ensure_ascii=False))
    sys.exit(0 if result['passed'] else 1)
