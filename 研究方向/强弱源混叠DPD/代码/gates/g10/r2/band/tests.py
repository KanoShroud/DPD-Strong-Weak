"""Small algebra/engineering checks; no localization ranking is inferred."""
import io
import unittest
import numpy as np
import torch
from ...physics import fit, sbl_posterior
from ...r1.physics import Model as OriginalModel
from ...r1.sbl import statistics
from ...metrics import evaluate
from ..tests import synthetic, encoding_check
from ..solvers import local
from .physics import Model, Spectrum
from .config import DEFAULT_CONFIG, jobs, observations
from .solvers import run
from .scenes import layouts, geometry_valid, make, band_energy
from .stopping import StopController, movement


class Checks(unittest.TestCase):
    def test_band_and_fft_noise(self):
        rng = np.random.default_rng(9)
        iq = (rng.normal(size=(4, 8192))+1j*rng.normal(size=(4, 8192)))/np.sqrt(2)
        m = Model(iq)
        f = np.fft.fftfreq(8192, 1/50e6)
        np.testing.assert_array_equal(m.f.cpu(), f[abs(f) <= 7e6])
        np.testing.assert_allclose(m.x.cpu(), (np.fft.fft(iq).T/8192)[abs(f) <= 7e6], rtol=0, atol=0)
        self.assertEqual(m.n, 2293)
        self.assertEqual(m.sample_count, 8192)
        self.assertEqual(m.frequency_noise_variance, 1/8192)
        ratio = float(abs(m.x).square().mean().cpu())/(1/8192)
        self.assertTrue(.95 < ratio < 1.05)

    def test_full_band_compatibility(self):
        iq, p = synthetic(n=256)
        a, b = OriginalModel(iq, sample_rate_hz=50e6), Model(iq, half_band_hz=25e6)
        self.assertEqual(a.joint(p+5), b.joint(p+5))
        np.testing.assert_array_equal(a.x.cpu(), b.x.cpu())
        np.testing.assert_allclose(a.joint_batch([p, p+5]), b.joint_batch([p, p+5]), atol=1e-14)

    def test_projection_gradient_offgrid(self):
        for k in (1, 2):
            iq, p = synthetic(k, n=256)
            m = Model(iq)
            a = m.response(p+5)
            _, residual = fit(a, m.x)
            self.assertAlmostEqual(m.joint(p+5), float(abs(residual).square().sum()/m.energy), places=11)
            _, g = m.joint(p+5, True)
            fd = [(m.joint(p+5+d.reshape(k, 2))-m.joint(p+5-d.reshape(k, 2)))/.002 for d in np.eye(k*2)*.001]
            np.testing.assert_allclose(g, fd, rtol=2e-5, atol=1e-9)
            q, value, _ = local(m, p+3, DEFAULT_CONFIG)
            self.assertLess(movement(p, q), .05)
            self.assertLess(value, 1e-10)

    def test_segment_covariance(self):
        rng = np.random.default_rng(10)
        iq = rng.normal(size=(4, 512))+1j*rng.normal(size=(4, 512))
        m = Model(iq)
        for segments in (4, 8):
            length = 512//segments
            f = np.fft.fftfreq(length, 1/50e6)
            data = np.fft.fft(iq.reshape(4, segments, length), axis=-1)/length
            data = data[:, :, abs(f) <= 7e6].transpose(2, 0, 1)
            expected = data@data.conj().transpose(0, 2, 1)/segments
            spec = Spectrum(m, 'MVDR', 2, segments, 1e-3)
            np.testing.assert_allclose(spec.cov.cpu(), expected, rtol=1e-13, atol=1e-14)
            self.assertEqual(len(spec.freq), len(expected))
            for method in ('MUSIC', 'MVDR'):
                spec = Spectrum(m, method, 2, segments, 1e-3)
                point = np.array([173., 117.])
                _, gradient = spec.value_grad(point, 1.)
                fd = [(-spec.evaluate((point+d)[None])[0]+spec.evaluate((point-d)[None])[0])/.002 for d in np.eye(2)*.001]
                np.testing.assert_allclose(gradient, fd, rtol=2e-4, atol=1e-8)

    def test_sbl_noise_and_cache(self):
        iq, p = synthetic(n=128)
        m = Model(iq)
        gamma = torch.tensor([.1, .2], device='cuda')
        got = statistics(m, p, gamma, 11)
        a = m.response(p)
        _, _, _, evidence = sbl_posterior(a, m.x, gamma, 1/128)
        self.assertAlmostEqual(got['evidence'], float(evidence.mean()), places=10)
        before = m.response(p).clone()
        self.assertTrue(m.enable_response_cache(p, 11))
        np.testing.assert_allclose(m.response(p).cpu(), before.cpu(), rtol=0, atol=0)
        self.assertEqual(m.counts['dictionary_cache_bytes'], m.n*4*2*16)

    def test_stop_is_set_based_and_observation_only(self):
        self.assertEqual(movement([[0, 0], [100, 0]], [[100, 0], [0, 0]]), 0)
        controller = StopController(DEFAULT_CONFIG)
        g = torch.ones(3, device='cuda')
        for iteration in range(5, 36, 5):
            p = [[0, 0], [100, 0]][::(-1 if iteration % 10 else 1)]
            stop = controller.update(iteration, -10., g, p, iteration*.1)
        self.assertTrue(stop)
        self.assertEqual(controller.first_stop['iteration'], 35)
        iq, _ = synthetic(n=128)
        cfg = dict(sbl_iterations=10, grid=200., sbl_grid=200., sbl_active_atoms=2,
                   sbl_coordinate_iterations=2, dictionary_cache=False)
        a = run(iq, 'SBL', config=dict(cfg, sbl_stop_rule='legacy'))
        b = run(iq, 'SBL', config=dict(cfg, sbl_stop_rule='observe'))
        np.testing.assert_array_equal(a['positions'], b['positions'])
        self.assertEqual([x['native_objective'] for x in a['history']], [x['native_objective'] for x in b['history']])

    def test_geometry_manifest_metrics(self):
        pp = layouts()
        self.assertTrue(all(geometry_valid(p, i) for i, p in enumerate(pp)))
        self.assertEqual(len(list(observations(True))), 260)
        self.assertEqual(len(list(jobs(True))), 2800)
        self.assertEqual(len(list(observations(False))), 32)
        scene = make((0, 40, 0, 'mixed'))
        self.assertGreater(band_energy(scene)['minimum_fraction'], .99)
        self.assertTrue(all(evaluate([[0, 0], [100, 0]], [[120, 0], [20, 0]])['success']['20']))
        self.assertFalse(all(evaluate([[0, 0]], [[20.001, 0]])['success']['20']))
        self.assertEqual(evaluate([[0, 0]], [])['matched'], 0)


def suite():
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    return dict(passed=result.wasSuccessful(), tests=result.testsRun, log=stream.getvalue(), encoding=encoding_check())


if __name__ == '__main__':
    result = suite()
    print(result['log'])
    raise SystemExit(0 if result['passed'] and result['encoding']['passed'] else 1)
