"""Verify exact conditional-score power polynomial, diagnostic only."""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.research_discovery.window_baselines import weight, VARIANTS
from 统一模型代码.gates.research_discovery.finite_extension_probe import SOURCE
from 统一模型代码.gates.g11.precision_diagnostic import sha, write, set_x
from 统一模型代码.gates.g10.r3.solvers import Model, Trace
from 统一模型代码.gates.g8.physics import RECEIVERS

HOLDOUT = SOURCE.parent/'round14_holdout_20261022_225848_064859'


def main():
    out = SOURCE.parent/('round15_competition_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    assert not out.exists() and out.resolve().is_relative_to(ROOT/'outputs_e2e')
    inputs = {str(p): sha(p) for p in HOLDOUT.rglob('*') if p.is_file()}
    protected = {str(p): sha(p) for p in (ROOT/'统一模型代码/gates').rglob('*.py')}
    out.mkdir()
    gaps = [0, 3, 6, 9, 12, 15, 20, 25, 30]
    write(out/'protocol.json', dict(source=str(HOLDOUT), gaps=gaps, probes=(RECEIVERS+1).tolist(),
        fixed_first='true position, diagnostic only', windows=VARIANTS, inputs=inputs, protected=protected))
    rows = []
    for path in sorted((HOLDOUT/'inputs').glob('*.npz')):
        with np.load(path, allow_pickle=False) as z:
            sig, noise, truth = z['signal'][..., :1024], z['noise'][:, :1024], z['positions']
        for name in VARIANTS:
            w = weight(name, 1024, 50e6)
            trace = Trace(lambda: None, history=False)
            m = Model((sig.sum(0)+noise)*w, trace.check, sample_rate_hz=50e6, half_band_hz=7e6)
            keep = m.frequency_indices.cpu().numpy()
            zs = (np.fft.fft(sig*w, axis=-1)/1024)[..., keep].transpose(0, 2, 1)
            zn = (np.fft.fft(noise*w, axis=-1)/1024)[:, keep].T
            a = m.response(truth[:1]).cpu().numpy()
            proj = np.eye(4)[None]-a@np.linalg.pinv(a, rcond=1e-10)
            e = (proj@zs[0, :, :, None]).squeeze(-1)
            b = (proj@(zs[1]+zn)[:, :, None]).squeeze(-1)
            points = np.vstack([truth[1:], RECEIVERS+1])
            u = proj@m.response(points).cpu().numpy()
            u /= np.linalg.norm(u, axis=1)[:, None, :]
            ue = np.einsum('flq,fl->fq', u.conj(), e)
            ub = np.einsum('flq,fl->fq', u.conj(), b)
            ac = np.sum(abs(ue)**2, axis=0)
            bc = np.sum((ue.conj()*ub).real, axis=0)
            cc = np.sum(abs(ub)**2, axis=0)
            coef = np.stack([ac[1:]-ac[0], bc[1:]-bc[0], cc[1:]-cc[0]], axis=1)
            errors, curves = [], []
            for gap in gaps:
                rho = 10**(gap/20)
                set_x(m, rho*zs[0]+zs[1]+zn)
                actual = m.conditional(points, truth[:1])*float(m.energy)
                delta = actual[1:]-actual[0]
                predicted = rho*rho*coef[:, 0]+2*rho*coef[:, 1]+coef[:, 2]
                error = float(np.max(abs(delta-predicted))/max(float(np.max(abs(actual))), 1e-30))
                assert error < 1e-10
                errors.append(error)
                curves.append(dict(gap=gap, wrong_minus_true=delta.tolist()))
            rows.append(dict(case=path.stem, window=name, coefficients=coef.tolist(),
                curves=curves, max_relative_identity_error=max(errors)))
    write(out/'results.json', rows)
    summary = []
    for name in VARIANTS:
        part = [r for r in rows if r['window'] == name]
        summary.append(dict(window=name, n=len(part),
            cases_with_positive_A=sum(any(c[0] > 0 for c in r['coefficients']) for r in part),
            wrong_probe_beats_truth_15=sum(max(r['curves'][5]['wrong_minus_true']) > 0 for r in part),
            wrong_probe_beats_truth_30=sum(max(r['curves'][-1]['wrong_minus_true']) > 0 for r in part)))
    write(out/'summary.json', summary)
    for path, identity in {**inputs, **protected}.items():
        assert sha(path) == identity, path
    write(out/'audit.json', dict(completed=True, base_cases=8, window_cases=len(rows),
        identity_comparisons=len(rows)*len(gaps)*4, inputs_unchanged=True, sources_unchanged=True))
    print(out, flush=True)


if __name__ == '__main__':
    main()
