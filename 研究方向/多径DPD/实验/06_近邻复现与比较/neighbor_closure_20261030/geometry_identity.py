"""Algebraic model check only: not a localization performance experiment."""
import json

import numpy as np

from collect import OUT

rng = np.random.default_rng(2026103001)
c, bandwidth = 299792458., 12.5e6
stations = np.array([[0., 0.], [3000., 3000.], [-3000., 3000.],
                     [-3000., -3000.], [3000., -3000.]])
freq = np.linspace(-bandwidth/2, bandwidth/2, 129)
rows = []
for case in range(100):
    positions = rng.uniform(-1500., 1500., (2, 2))
    moved = positions + rng.uniform(-.1, .1, (2, 2))
    tau = np.linalg.norm(positions[:, None]-stations[None], axis=2)/c
    tau2 = np.linalg.norm(moved[:, None]-stations[None], axis=2)/c
    delta = np.broadcast_to(np.array([4., 12.])/bandwidth, (2, 5, 2)).copy()
    changed_delta = delta - (tau2-tau)[..., None]
    rho = .3*np.exp(1j*rng.uniform(-np.pi, np.pi, (2, 5, 2)))
    def reflected(t, delay):
        return (rho[..., None]*np.exp(-2j*np.pi*(t[..., None]+delay)[..., None]*freq)).sum(axis=2)
    a, b = reflected(tau, delta), reflected(tau2, changed_delta)
    relative = np.linalg.norm(a-b)/np.linalg.norm(a)
    gaps = np.diff(np.concatenate([np.zeros((2, 5, 1)), changed_delta], axis=2), axis=2)*bandwidth
    los_change = np.linalg.norm(np.exp(-2j*np.pi*tau[..., None]*freq)-
                                np.exp(-2j*np.pi*tau2[..., None]*freq))
    assert relative < 1e-11
    assert np.all((gaps > 3) & (gaps < 20))
    assert np.all(np.sum(abs(rho)**2, axis=2) < 1)
    assert los_change > 1e-8
    rows.append(dict(case=case, reflection_relative_difference=float(relative),
                     los_change_norm=float(los_change), min_gap=float(gaps.min()),
                     max_gap=float(gaps.max())))
(OUT/'geometry_identity.json').write_text(json.dumps(dict(
    kind='ALGEBRAIC_RESPONSE_IDENTITY_NOT_LOCALIZATION_ABLATION',
    seed=2026103001, cases=100, passed=True,
    worst_reflection_relative_difference=max(r['reflection_relative_difference'] for r in rows),
    caution='No RMSE, no identifiability theorem, no claim reflections provide zero statistical information.',
    rows=rows), indent=2), encoding='utf-8')
print('PASS',len(rows),'algebra cases; max relative difference',
      max(r['reflection_relative_difference'] for r in rows))
