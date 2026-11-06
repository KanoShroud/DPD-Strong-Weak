"""Local deterministic information scaling; algebra audit, not localization trial."""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.g8.physics import C, RECEIVERS
from 统一模型代码.gates.g11.precision_diagnostic import sha, write
from 统一模型代码.gates.research_discovery.residual_competition import HOLDOUT


def response(points, frequencies):
    distance = np.linalg.norm(points[None, :, :]-RECEIVERS[:, None, :], axis=-1)
    gain = 1/distance
    gain /= np.linalg.norm(gain, axis=0)
    return gain[None]*np.exp(-2j*np.pi*frequencies[:, None, None]*distance[None]/C)


def main():
    out = HOLDOUT.parent/('round17_information_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    assert not out.exists() and out.resolve().is_relative_to(ROOT/'outputs_e2e')
    files = sorted((HOLDOUT/'inputs').glob('*.npz'))
    identities = {str(p.resolve(strict=True)): sha(p) for p in files}
    out.mkdir()
    write(out/'protocol.json', dict(inputs=identities, gaps=[0, 15, 30],
        purpose='matched deterministic local Fisher scaling, no performance claim',
        noise_variance=1, frequencies=65, central_difference_m=0.001,
        unknowns='two complex coefficients per frequency and four coordinates',
        waveform='fixed synthetic independent complex coefficients, seed20261023017'))
    rng = np.random.default_rng(20261023017)
    frequency = np.linspace(-6.25e6, 6.25e6, 65)
    rows = []
    for path in files:
        with np.load(path, allow_pickle=False) as z:
            positions = z['positions'].copy()
        a = response(positions, frequency)
        projector = np.eye(4)[None]-a@np.linalg.pinv(a)
        coefficient = rng.normal(size=(65, 2))+1j*rng.normal(size=(65, 2))
        derivative = []
        for source in range(2):
            for coordinate in range(2):
                plus, minus = positions.copy(), positions.copy()
                plus[source, coordinate] += 0.001
                minus[source, coordinate] -= 0.001
                da = (response(plus, frequency)-response(minus, frequency))/0.002
                derivative.append((projector@(da[:, :, source]*coefficient[:, source, None])[:, :, None]).ravel())
        j = np.column_stack(derivative)
        base = 2*(j.conj().T@j).real
        assert np.linalg.eigvalsh(base).min() > 0
        base_cov = np.linalg.inv(base)
        for gap in (0, 15, 30):
            rho = 10**(gap/20)
            scaled = j*np.array([rho, rho, 1, 1])[None]
            fisher = 2*(scaled.conj().T@scaled).real
            covariance = np.linalg.inv(fisher)
            error = np.linalg.norm(covariance[2:, 2:]-base_cov[2:, 2:])/np.linalg.norm(base_cov[2:, 2:])
            assert error < 1e-10
            rows.append(dict(case=path.stem, gap=gap, relative_weak_block_error=float(error),
                condition=float(np.linalg.cond(fisher)), weak_trace=float(np.trace(covariance[2:, 2:]))))
    write(out/'results.json', rows)
    for path, identity in identities.items():
        assert sha(path) == identity
    write(out/'audit.json', dict(completed=True, comparisons=len(rows), inputs_unchanged=True,
        algebra_only=True, maximum_relative_error=max(r['relative_weak_block_error'] for r in rows)))
    print(out, flush=True)


if __name__ == '__main__':
    main()
