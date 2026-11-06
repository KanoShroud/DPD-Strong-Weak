"""Numerical identity tests only; no DPD or frozen data alteration."""
import unittest
import numpy as np
import audit


def row(d, e, predicted=None, fallback=False):
    d, e = np.asarray(d), np.asarray(e)
    displacement = d @ d / 2
    cross = d @ e
    return dict(d=d, e_wide=e, displacement=displacement, actual_cross=cross,
                loss_difference=displacement+cross, loss_wide=e@e/2,
                loss_selected=(e+d)@(e+d)/2, risk_difference=predicted,
                plugin_cross=None if predicted is None else predicted-displacement,
                fallback=fallback)


class Identities(unittest.TestCase):
    def test_one_case(self):
        r = row([1., 2., 3., 4.], [2., -1., 1., -3.], -1.)
        s = audit.decompose([r])
        self.assertEqual(s['empirical_covariance_n_trace'], 0.)
        self.assertAlmostEqual(s['empirical_loss_difference'], r['loss_selected']-r['loss_wide'])

    def test_random_sample_identity_and_bias_variance_identity(self):
        rng = np.random.default_rng(20261103)
        rows = [row(d, e, .1) for d, e in zip(rng.normal(size=(8, 4)), rng.normal(size=(8, 4)))]
        s = audit.decompose(rows)
        self.assertAlmostEqual(s['mse_wide'], s['wide_sample_mean_squared_norm']+s['wide_empirical_centered_mse_n'])
        self.assertAlmostEqual(s['empirical_loss_difference'], s['displacement']+s['sample_mean_cross']+s['empirical_covariance_n_trace'])

    def test_uses_n_not_n_minus_one(self):
        rows = [row([x, 0., 0., 0.], [x, 0., 0., 0.]) for x in [-1., 1.]]
        self.assertEqual(audit.decompose(rows)['empirical_covariance_n_trace'], 1.)

    def test_fallback_denominators_explicit(self):
        rows = [row([1., 0., 0., 0.], [0., 0., 0., 0.], -.1),
                row([0., 0., 0., 0.], [100., 0., 0., 0.], fallback=True)]
        s = audit.decompose(rows)
        self.assertEqual(s['scored_n'], 1)
        self.assertEqual(s['fallback'], 1)
        self.assertEqual(s['empirical_loss_difference'], .25)
        self.assertEqual(s['scored_mean_empirical_difference'], .5)

    def test_all_unscored_remains_unscored(self):
        s = audit.decompose([row([0., 0., 0., 0.], [100., 0., 0., 0.], fallback=True)])
        self.assertNotIn('mean_predicted_difference', s)
        self.assertEqual(s['empirical_loss_difference'], 0.)


if __name__ == '__main__':
    unittest.main()
