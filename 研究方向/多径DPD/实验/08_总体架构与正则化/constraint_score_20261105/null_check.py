"""Analytic fixed-cone Gaussian check; not an RF false-alarm calibration."""
import numpy as np
from scipy.stats import chi2
from local_cone import HERE, cfg


def main():
    out = HERE/'fixed_design_null_v1.json'
    assert not out.exists()
    rng = np.random.default_rng(2026110511)
    z = rng.normal(size=(200000, 3))
    norm2 = np.sum(z*z, axis=1)
    null = norm2-np.maximum(z[:, 0], 0)**2
    scores = []
    rows = []
    for cosine, sine in [(0., 1.), (.8, .6), (.6, .8), (np.sqrt(.96), .2)]:
        normal = np.array([sine, -cosine, 0.])
        alt = np.minimum(z@normal, 0)**2
        t = null-alt
        assert t.min() > -1e-12 and np.max(t-norm2) < 1e-12
        scores.append(t)
        if cosine == 0:
            assert np.allclose(t, z[:, 1]**2+z[:, 2]**2)
        rows.append(dict(cosine=cosine, sine=sine, mean=float(t.mean()),
            chi2_2_reject_rate=float(np.mean(t > chi2.isf(.05, 2))),
            rank3_bound_reject_rate=float(np.mean(t > chi2.isf(.05, 3)))))
    maximum = np.max(scores, axis=0)
    cfg.write(out, dict(seed=2026110511, samples=len(z), rows=rows,
        common_rank3_bonferroni_threshold=float(chi2.isf(.05/4, 3)),
        four_candidate_reject_rate=float(np.mean(maximum > chi2.isf(.05/4, 3))),
        pointwise_bound_passed=True, source_sha=cfg.sha(__file__),
        scope='Known sigma, fixed designs/cones, exact linear Gaussian model. No data-adaptive RF guarantee.'))
    print(cfg.read(out))


if __name__ == '__main__':
    main()
