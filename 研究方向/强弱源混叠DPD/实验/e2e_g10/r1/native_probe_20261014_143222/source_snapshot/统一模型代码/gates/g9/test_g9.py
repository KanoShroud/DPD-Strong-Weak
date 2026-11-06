"""无正式性能含义的指标、数据配对和物理代数测试。"""
import unittest

import numpy as np
import torch

from .metrics import evaluate
from .scenes import make
from .solver import Score


class MetricsTests(unittest.TestCase):
    def test_swap(self):
        m = evaluate([[0, 0], [300, 0]], [[300, 0], [0, 0]])
        self.assertEqual(m['gospa_m'], 0)
        self.assertEqual(m['errors_m'], [0, 0])

    def test_empty_far_duplicate_boundary(self):
        for pred in ([], [[1000, 0], [2000, 0]]):
            m = evaluate([[0, 0], [300, 0]], pred)
            expected = 100 if not pred else np.sqrt(20000)
            self.assertAlmostEqual(m['gospa_m'], expected)
        self.assertEqual(evaluate([[0, 0], [300, 0]], [[0, 0], [0, 0]])['success']['100'], [True, False])
        m = evaluate([[0, 0]], [[100, 0]])
        self.assertEqual(m['success']['100'], [True])
        self.assertEqual(m['localization_p'], 0)
        self.assertEqual(m['missed_p']+m['false_p'], 10000)

    def test_squared_assignment(self):
        # 自动枚举两种配对，与实现SSE结果独立对比。
        truth = np.array([[0, 0], [300, 0.]])
        pred = np.array([[90, 40], [-20, 200.]])
        expected = min(np.square(truth-pred).sum(), np.square(truth-pred[::-1]).sum())
        self.assertAlmostEqual(sum(e*e for e in evaluate(truth, pred)['errors_m']), expected)

    def test_missing_and_no_truth(self):
        m = evaluate([[0, 0], [300, 0]], [[0, 0]])
        self.assertEqual(m['matched'], 1)
        self.assertEqual(m['errors_m'], [0, None])
        self.assertAlmostEqual(m['gospa_m'], np.sqrt(5000))
        self.assertEqual(evaluate([], [])['gospa_m'], 0)

    def test_threshold_assignment_is_not_sse_assignment(self):
        # 距离矩阵[[30,80],[90,110]]可由平面中四点构造。
        d = 100.
        x1 = (30**2-90**2+d*d)/(2*d)
        x2 = (80**2-110**2+d*d)/(2*d)
        truth = [[0, 0], [d, 0]]
        pred = [[x1, np.sqrt(30**2-x1*x1)], [x2, np.sqrt(80**2-x2*x2)]]
        m = evaluate(truth, pred)
        np.testing.assert_allclose(m['errors_m'], [30, 110])
        self.assertEqual(m['success']['100'], [True, True])

    def test_same_bytes_and_power(self):
        a = make('smoke', 0, 0, .5, 0, 6, 1024)
        b = make('smoke', 0, 0, .5, 0, 12, 1024)
        np.testing.assert_array_equal(a['noise'], b['noise'])
        np.testing.assert_array_equal(a['signal'][1], b['signal'][1])
        np.testing.assert_allclose(b['signal'][0], a['signal'][0]*10**(6/20))
        self.assertGreaterEqual(np.linalg.norm(a['positions'][0]-a['positions'][1]), 300)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA')
    def test_physics_gradient_and_projection(self):
        rng = np.random.default_rng(4)
        iq = rng.normal(size=(4, 256))+1j*rng.normal(size=(4, 256))
        calc = Score(iq)
        for others in ([], [[300., 120.]]):
            calc.set_sources(others)
            point = np.array([-210., 220.])
            value, gradient = calc.value_gradient(point)
            self.assertAlmostEqual(value, -calc.evaluate([point])[0], places=10)
            numerical = []
            for k in range(2):
                step = np.eye(2)[k]*.001
                numerical.append(-(calc.evaluate([point+step])[0]-calc.evaluate([point-step])[0])/.002)
            np.testing.assert_allclose(gradient, numerical, rtol=1e-5, atol=1e-9)
            a = calc.steering([point])[0].cpu().numpy()
            x = calc.x.cpu().numpy()
            p = np.broadcast_to(np.eye(4), (256, 4, 4)) if calc.p is None else calc.p.cpu().numpy()
            pa = np.einsum('fmn,fn->fm', p, a)
            px = np.einsum('fmn,fn->fm', p, x)
            expected = np.sum(np.abs(np.sum(pa.conj()*px, -1))**2/np.sum(abs(pa)**2, -1))/calc.energy
            self.assertAlmostEqual(-value, expected, places=10)
            score, change = calc.evaluate([point], True)
            np.testing.assert_allclose(score, [-value])
            self.assertTrue(np.isfinite(change).all() and (change >= 0).all())


if __name__ == '__main__':
    unittest.main()
