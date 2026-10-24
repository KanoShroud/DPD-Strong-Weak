"""Exact quadratic score attribution of G11-P1; no additional localization.

The squared reduced residual is quadratic in the observation. Its position
gradient therefore decomposes into three self terms and three cross terms.
This is an algebraic attribution, not a novel estimator or a Fisher bound.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.g11.precision_diagnostic import (
    P0, Model, components, read, write, resolve, load_case, set_x, derivatives,
    sensitivity, csv_write, PROVENANCE, sha,
)
from 统一模型代码.gates.g8.physics import C


def make_gradient(m, truth, a):
    delta = truth[:, None]-m.stations.cpu().numpy()
    d = np.linalg.norm(delta, axis=-1)
    unit = (delta/d[..., None]).transpose(1, 0, 2)
    b = -1/d.T[None]-2j*np.pi*m.f.cpu().numpy()[:, None, None]/C
    da = a[..., None]*b[..., None]*unit[None]
    inv = np.linalg.pinv(a, rcond=1e-10)

    def gradient(y):
        s = (inv@y[..., None]).squeeze(-1)
        r = y-(a@s[..., None]).squeeze(-1)
        q = (da*s[:, None, :, None]).reshape(len(a), 4, 4)
        return -2*np.einsum('fl,fli->i', r.conj(), q).real  # UNNORMALIZED
    return gradient


def main():
    out = resolve(sys.argv[1])
    report = read(out/'report.json')
    protocol = read(out/'protocol.json')
    assert out.is_relative_to(ROOT/'outputs_e2e/unified/e2e_g11')
    assert not (out/'attribution.json').exists()
    expected = read(P0/'source_manifest.json')
    folder = resolve(read(ROOT/'outputs_e2e/unified/e2e_g10/r4/review_supplement_release.json')['out'])
    rows = []
    checks = dict(max_gradient_relative_error=0., max_decomposition_relative_error=0.,
                  max_linear_derivative_relative_error=0.)
    for old in report['rows']:
        task = (old['layout'], old['repeat'], old['gap'])
        tag, signal, noise, truth, saved = load_case(folder, expected, task)
        m = Model(signal.sum(0)+noise)
        z, w, a, coef, exact, residual = components(m, signal, noise, truth)
        grad = make_gradient(m, truth, a)
        total = z.sum(0)+w
        set_x(m, total)
        energy = float(m.energy.cpu())
        _, g, h = derivatives(m, truth)
        expected_gradient = grad(total)
        rel = np.linalg.norm(g*energy-expected_gradient)/max(np.linalg.norm(expected_gradient), 1e-30)
        assert rel < 1e-8
        checks['max_gradient_relative_error'] = max(checks['max_gradient_relative_error'], rel)
        data = {'A': z[0], 'B': z[1], 'noise': w}
        terms = {name: grad(y) for name, y in data.items()}
        for n1, n2 in (('A', 'B'), ('A', 'noise'), ('B', 'noise')):
            terms[n1+'x'+n2] = grad(data[n1]+data[n2])-terms[n1]-terms[n2]
        rel = np.linalg.norm(sum(terms.values())-expected_gradient)/max(np.linalg.norm(expected_gradient), 1e-30)
        assert rel < 1e-8
        checks['max_decomposition_relative_error'] = max(checks['max_decomposition_relative_error'], rel)
        # Small-perturbation derivative of the score must agree with the GN map.
        eps = 1e-4
        centered = (grad(exact.sum(0)+eps*w)-grad(exact.sum(0)-eps*w))/(2*eps)
        dvec = truth[:, None]-m.stations.cpu().numpy()
        d = np.linalg.norm(dvec, axis=-1)
        unit = (dvec/d[..., None]).transpose(1, 0, 2)
        factor = -1/d.T[None]-2j*np.pi*m.f.cpu().numpy()[:, None, None]/C
        q = (a[..., None]*factor[..., None]*unit[None]*coef[:, None, :, None]).reshape(len(a), 4, 4)
        j = q-a@(np.linalg.pinv(a)@q)
        linear = -2*np.einsum('fli,fl->i', j.conj(), w).real
        rel = np.linalg.norm(centered-linear)/max(np.linalg.norm(linear), 1e-30)
        assert rel < 1e-7
        checks['max_linear_derivative_relative_error'] = max(checks['max_linear_derivative_relative_error'], rel)
        eig = np.linalg.eigvalsh((h+h.T)/2)
        # At-truth Newton prediction is also diagnostic, not guaranteed accurate.
        mapped = {k: (-np.linalg.solve(h, v/energy)).reshape(2, 2).tolist() for k, v in terms.items()}
        exact_term = grad(exact[0])
        assert np.linalg.norm(exact_term) < 1e-12
        sen = sensitivity(m, truth, a, coef, residual, w)
        rows.append(dict(case=tag, task=task, score_terms={k: v.tolist() for k,v in terms.items()},
                         score=expected_gradient.tolist(), mapped_newton_terms_m=mapped,
                         at_truth_hessian_min=float(eig.min()),
                         at_truth_newton_shift_m=(-np.linalg.solve(h,g)).reshape(2,2).tolist(),
                         small_perturbation_shift_m=sen['predicted_shift_m'],
                         actual_shift_m=old['actual_shift_m']))
    flat = []
    for old, r in zip(report['rows'], rows):
        error = np.array(old['actual_shift_m'])[1]
        naive = np.array(r['small_perturbation_shift_m'])[1]
        newton = np.array(r['at_truth_newton_shift_m'])[1]
        flat.append(dict(case=r['case'], layout=old['layout'], repeat=old['repeat'], gap=old['gap'],
             weak_actual_shift_m=float(np.linalg.norm(error)),
             weak_linear_prediction_m=float(np.linalg.norm(naive)),
             weak_newton_prediction_m=float(np.linalg.norm(newton)),
             weak_A_self_term_m=float(np.linalg.norm(r['mapped_newton_terms_m']['A'][1])),
             linear_prediction_error_m=float(np.linalg.norm(naive-error)),
             newton_prediction_error_m=float(np.linalg.norm(newton-error))))
    # Exact A-self gradient scales with A POWER, not just A amplitude.
    ratios = []
    for r in rows:
        if r['task'][2] != 15:
            continue
        zero = next(v for v in rows if v['task'][:2] == r['task'][:2] and v['task'][2]==0)
        ga = np.array(r['score_terms']['A']); gz = np.array(zero['score_terms']['A'])
        scale = 10**1.5
        relative = np.linalg.norm(ga-scale*gz)/max(np.linalg.norm(ga),1e-30)
        assert relative < 1e-8
        ratios.append(relative)
    checks['max_A_self_power_scaling_error'] = max(ratios)
    # Preserve all previous evidence and frozen algorithms/papers.
    assert all(sha(p)==h for p,h in {**protocol['source_code'], **protocol['protected'], **PROVENANCE}.items())
    write(out/'attribution.json', dict(checks=checks, rows=rows,
            scope='Exact score decomposition, at-truth local sensitivity only; no localization',
            script_sha=sha(__file__), source_manifest=PROVENANCE))
    csv_write(out/'attribution.csv', flat)
    print(json.dumps(checks, ensure_ascii=False))
    print('diagnostic example', next(r for r in flat if r['case']=='13_1000_15'))


if __name__ == '__main__':
    main()
