"""Small deterministic correctness tests; not localization evidence."""
import unittest
import numpy as np
import control as c


class ControlTests(unittest.TestCase):
    def test_real_noise_units(self):
        z = np.array([[1+2j, 3-4j]])
        self.assertAlmostEqual(float(c.ref.real(z)@c.ref.real(z)), 2*float(np.sum(abs(z)**2)))

    def test_branch_order_changes(self):
        x = np.r_[np.zeros(20), .1, .2, 3., .2, .1, 7.]
        z = x.copy()
        z[22], z[25] = 8., 2.
        a = c.branches(x, z, [[1, 0], [1, 0]], 8192)
        self.assertEqual(len(a['delay_sort_changes']), 1)
        self.assertEqual(a['max_delay_movement_Btau'], 5.)
        self.assertEqual(c.branches(x, x, [[1, 0], [1, 0]], 8192)['delay_sort_changes'], [])

    def test_actual_hessian_linear_limit(self):
        rng = np.random.default_rng(39)
        a = rng.normal(size=(12, 4))
        dy = rng.normal(size=12)
        # For residual y-Ax, J=-A; implicit derivative is -H^-1 J^T.
        expected = np.linalg.lstsq(a, dy, rcond=None)[0]
        response = -np.linalg.solve(a.T@a, -a.T)@dy
        np.testing.assert_allclose(response, expected, atol=1e-12)

    def test_no_total_power_constraint(self):
        x = np.r_[np.zeros(20), .8, 0., 3., .8, 0., 7.]
        np.testing.assert_array_equal(c.ref.project(x, 8192), x)


if __name__ == '__main__':
    unittest.main()
