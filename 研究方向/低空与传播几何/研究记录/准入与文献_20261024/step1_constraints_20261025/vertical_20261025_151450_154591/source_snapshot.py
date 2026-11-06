"""Enumerate shared vertical-plane candidates for two fixed reflection lengths."""
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.polynomial import Polynomial as P
from scipy.optimize import least_squares

from check_constraints import PRIOR, ROOT, verify


def solve(sources, receivers, lengths, points):
    scale = 1000.
    ps, rs, ls = sources/scale, receivers/scale, lengths/scale
    q = (ls**2-np.sum((ps-rs)**2, axis=1))/4
    den = P([1., 0., 1.])
    def proj(point):
        return P([point[0], 2*point[1], -point[0]])
    ap, ar = [proj(x) for x in ps], [proj(x) for x in rs]
    D = ap[0]+ar[0]-ap[1]-ar[1]
    C = ap[0]*ar[0]-ap[1]*ar[1]-(q[0]-q[1])*den*den
    F = (ap[0]*D-C)*(ar[0]*D-C)-q[0]*D*D*den*den
    roots = F.roots()
    angles = [float(2*np.arctan(z.real)) for z in roots if abs(z.imag) < 1e-6]
    # Include the half-angle pole and possible D=0 branch explicitly.
    angles += [np.pi]+[float(2*np.arctan(z.real)) for z in D.roots() if abs(z.imag) < 1e-6]
    candidates = []
    for angle in angles:
        n = np.array([np.cos(angle), np.sin(angle), 0.])
        pp, rr = ps@n, rs@n
        diff = pp[0]+rr[0]-pp[1]-rr[1]
        numerator = pp[0]*rr[0]-pp[1]*rr[1]-q[0]+q[1]
        if abs(diff) > 1e-10:
            offsets = [numerator/diff]
        elif abs(numerator) < 1e-8:
            offsets = [z.real for z in np.roots([1., -pp[0]-rr[0], pp[0]*rr[0]-q[0]]) if abs(z.imag) < 1e-8]
        else:
            continue
        for offset in offsets:
            def residual(x):
                normal = np.array([np.cos(x[0]), np.sin(x[0]), 0.])
                image = ps-2*(ps@normal-x[1])[:, None]*normal
                return np.linalg.norm(image-rs, axis=1)-ls
            # Refine only near an algebraic candidate; reject unsatisfied pole candidates first.
            before = float(np.linalg.norm(residual([angle, offset])))
            if before > 1e-5:
                continue
            fit = least_squares(residual, [angle, offset], max_nfev=100,
                                xtol=1e-13, ftol=1e-13, gtol=1e-13)
            n = np.array([np.cos(fit.x[0]), np.sin(fit.x[0]), 0.])
            b = fit.x[1]*scale
            if np.max(points@n-b) < 0:
                n, b = -n, -b
            check = verify(n, b, sources, receivers, lengths, points)
            candidates.append({'algebraic_initial_residual_scaled': before, **check})
    return {'polynomial_coefficients': F.coef.tolist(),
            'polynomial_roots_real_imag': [[float(z.real), float(z.imag)] for z in roots],
            'candidates': candidates, 'valid_count': sum(c['valid'] for c in candidates),
            'degenerate_polynomial': bool(np.max(abs(F.coef)) < 1e-15)}


def main():
    start = time.perf_counter()
    out = ROOT/('vertical_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    paths = [Path(__file__), ROOT/'check_constraints.py', ROOT/'执行合同.md', PRIOR/'result.json', PRIOR/'iq_witness.npz']
    identity = {str(p.resolve(strict=True)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (out/'preflight.json').write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    w = json.loads((PRIOR/'result.json').read_text(encoding='utf-8'))['witness']
    s = np.array(w['stations_m'])
    results = {}
    with np.load(PRIOR/'iq_witness.npz') as d:
        for name in ['original', 'alternate']:
            src = np.array([w[name+'_position_m'], w['unchanged_second_source_m']])
            missing = next(i for i in range(5) if i not in w[name+'_los_support'])
            r = s[[missing, 4]]
            L = d[name+'_lengths_m'][[0, 1], [missing, 4]]
            results[name] = solve(src, r, L, np.vstack([src, s]))
        src = np.array([w['original_position_m'], w['unchanged_second_source_m']])
        r = s[[0, 4]]
        n, b = np.array([1., 0., 0.]), -900.
        image = src-2*(src@n-b)[:, None]*n
        results['positive_control'] = solve(src, r, np.linalg.norm(image-r, axis=1), np.vstack([src, s]))
    checks = {'positive_control': results['positive_control']['valid_count'] > 0,
              'nondegenerate_polynomials': not any(r['degenerate_polynomial'] for r in results.values()),
              'inputs_unchanged': all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p,h in identity.items())}
    (out/'result.json').write_text(json.dumps({'identity': identity, 'results': results, 'checks': checks,
        'all_checks_pass': all(checks.values()), 'wall_seconds': time.perf_counter()-start},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'out': str(out), 'counts': {k:v['valid_count'] for k,v in results.items()}, 'checks': checks}))
    if not all(checks.values()):
        raise RuntimeError('Audit failed')


if __name__ == '__main__':
    main()
