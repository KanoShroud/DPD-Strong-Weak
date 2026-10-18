"""Algebra and pipeline invariants; not a localization-performance gate."""
import copy
import io
import os
import subprocess
import sys
import unittest
import numpy as np
import torch
from ..metrics import evaluate
from ..r1.methods import run as original_run, ap_solver, Trace
from ..r1.physics import small_fit
from .config import CONFIG
from .solvers import CountModel, local, run, curvature
from .runtime import seal


def synthetic(k=2, n=128):
    rng = np.random.default_rng(2026101501)
    p = np.array([[173., 117.], [813., 213.]])[:k]
    seed_iq = rng.normal(size=(4, n))+1j*rng.normal(size=(4, n))
    m = CountModel(seed_iq, sample_rate_hz=CONFIG['sample_rate_hz'])
    s = rng.normal(size=(n, k))+1j*rng.normal(size=(n, k))
    s[abs(m.f.cpu().numpy()) > 6.25e6] = 0
    x = (m.response(p)@torch.as_tensor(s, device=m.device)[..., None]).squeeze(-1)
    iq = np.fft.ifft(x.detach().cpu().numpy().T*n, axis=1)
    return iq, p


class Checks(unittest.TestCase):
    def test_ls_projection_gradient(self):
        iq, p = synthetic()
        m = CountModel(iq, sample_rate_hz=CONFIG['sample_rate_hz'])
        p = p+np.array([[7., -4.], [-5., 3.]])
        a = m.response(p)
        s, r = small_fit(a, m.x)
        direct = (torch.linalg.pinv(a)@m.x[..., None]).squeeze(-1)
        np.testing.assert_allclose(s.cpu(), direct.cpu(), rtol=1e-9, atol=1e-9)
        self.assertAlmostEqual(m.joint(p), float(abs(r).square().sum()/m.energy), places=12)
        _, g = m.joint(p, True)
        eps = 1e-3
        fd = []
        for j in range(4):
            delta = np.eye(4)[j].reshape(2, 2)*eps
            fd.append((m.joint(p+delta)-m.joint(p-delta))/(2*eps))
        np.testing.assert_allclose(g, fd, rtol=2e-5, atol=1e-9)
        z = torch.tensor(p, dtype=torch.float64, device=m.device, requires_grad=True)
        aa = m.response(z)
        frozen_s = small_fit(aa, m.x)[0].detach()
        frozen_loss = abs(m.x-(aa@frozen_s[..., None]).squeeze(-1)).square().sum()/m.energy
        envelope = torch.autograd.grad(frozen_loss, z)[0].cpu().numpy().ravel()
        np.testing.assert_allclose(g, envelope, rtol=1e-8, atol=1e-10)

    def test_surrogate_upper_bound(self):
        iq, p = synthetic()
        m = CountModel(iq, sample_rate_hz=CONFIG['sample_rate_hz'])
        p = p+5
        a, s, r = m.fitted(p)
        fixed = m.x-a[:, :, 0]*s[:, 0, None]
        for offset in (0., 2., -11.):
            trial = p.copy(); trial[1] += offset
            b = m.response(trial[1:2])
            _, residual = small_fit(b, fixed)
            surrogate = float(abs(residual).square().sum()/m.energy)
            self.assertLessEqual(m.joint(trial), surrogate+1e-12)
            if offset == 0:
                self.assertAlmostEqual(m.joint(p), surrogate, places=12)

    def test_noiseless_offgrid_and_k1(self):
        for k in (1, 2):
            iq, p = synthetic(k)
            m = CountModel(iq, sample_rate_hz=CONFIG['sample_rate_hz'])
            estimate, value, _ = local(m, p+np.array([3., -2.]), CONFIG)
            self.assertLess(np.linalg.norm(estimate-p, axis=1).max(), .05)
            self.assertLess(value, 1e-10)

    def test_curvature_finite_difference(self):
        iq, p = synthetic()
        m = CountModel(iq, sample_rate_hz=CONFIG['sample_rate_hz'])
        diag = curvature(m, p)
        h = np.asarray(diag['hessian'])
        fd = []
        for delta in np.eye(4)*1e-3:
            _, plus = m.joint(p+delta.reshape(2, 2), True)
            _, minus = m.joint(p-delta.reshape(2, 2), True)
            fd.append((plus-minus)/.002)
        np.testing.assert_allclose(h, np.array(fd).T, rtol=2e-4, atol=1e-9)

    def test_local_block_relation_and_rank_reversal(self):
        h = np.array([[2., 0., .7, .1], [0., 3., .2, .8],
                      [.7, .2, 2., 0.], [.1, .8, 0., 2.]])
        e = np.array([1., 2., -1., .5])
        one = -np.linalg.solve(h[:2, :2], h[:2, 2:]@e[2:])
        two = -np.linalg.solve(h[2:, 2:], h[2:, :2]@one)
        transition = np.linalg.solve(h[2:, 2:], h[2:, :2])@np.linalg.solve(h[:2, :2], h[:2, 2:])
        np.testing.assert_allclose(two, transition@e[2:])
        self.assertLess(max(abs(np.linalg.eigvals(transition))), 1.)
        initial, improvement = np.array([10., 12.]), np.array([1., 5.])
        self.assertGreater(improvement[1]-improvement[0], initial[1]-initial[0])
        self.assertLess((initial-improvement)[1], (initial-improvement)[0])

    def test_nomp_original_and_common_prefix(self):
        iq, _ = synthetic()
        old = original_run(iq, 'BNOMP', 2, CONFIG['original_nomp'])
        rows = [run(iq, method, diagnostics=False) for method in ('NOMP', 'NOMP_EXTRA12', 'NOMP_JOINT')]
        for result in rows:
            np.testing.assert_allclose(result['mechanism']['prefix_positions'], old['positions'], rtol=0, atol=1e-8)
            self.assertLessEqual(result['events'][-1]['native_objective'], old['events'][-1]['native_objective']+1e-12)

    def test_ap_containment_and_historical_first_output(self):
        iq, _ = synthetic()
        rows = {method: run(iq, method, diagnostics=False) for method in
                ('AP_1_ALT', 'AP_5_ALT', 'AP_1_JOINT', 'AP_5_JOINT')}
        for suffix in ('ALT', 'JOINT'):
            one, many = rows['AP_1_'+suffix], rows['AP_5_'+suffix]
            np.testing.assert_allclose(one['mechanism']['branches'][0]['final_positions'],
                                       many['mechanism']['branches'][0]['final_positions'], rtol=0, atol=1e-8)
            self.assertLessEqual(many['events'][-1]['native_objective'], one['events'][-1]['native_objective']+1e-12)
        class FirstOutput(Exception):
            pass
        class FirstTrace(Trace):
            def save(self, positions, objective):
                super().save(positions, objective)
                raise FirstOutput()
        trace = FirstTrace(lambda: None)
        m = CountModel(iq, trace.check, sample_rate_hz=CONFIG['sample_rate_hz'])
        with self.assertRaises(FirstOutput):
            ap_solver(m, trace, 2, CONFIG['original_nomp'])
        np.testing.assert_allclose(rows['AP_5_JOINT']['positions'], trace.events[-1]['positions'], rtol=0, atol=1e-7)

    def test_metrics_boundaries_and_checksum(self):
        truth = [[0., 0.], [100., 0.]]
        self.assertTrue(all(evaluate(truth, [[120., 0.], [20., 0.]])['success']['20']))
        self.assertFalse(all(evaluate(truth, [[120.001, 0.], [20., 0.]])['success']['20']))
        self.assertEqual(evaluate(truth, [])['matched'], 0)
        row = dict(config=CONFIG, positions=truth)
        before = seal(row)
        changed = copy.deepcopy(row); changed['positions'][0][0] = 1.
        self.assertNotEqual(before, seal(changed))


def suite():
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    return dict(passed=result.wasSuccessful(), tests=result.testsRun, failures=len(result.failures),
                errors=len(result.errors), log=stream.getvalue())


def encoding_check():
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8:replace')
    result = subprocess.run([sys.executable, '-c', "print('中文日志：G10-R2 恢复检查')"],
                            text=True, capture_output=True, encoding='utf-8', errors='replace', env=env)
    return dict(passed=result.returncode == 0 and '中文日志：G10-R2 恢复检查' in (result.stdout or ''),
                stdout=result.stdout or '', stderr=result.stderr or '', returncode=result.returncode)


if __name__ == '__main__':
    result = suite()
    print(result['log'])
    raise SystemExit(0 if result['passed'] else 1)
