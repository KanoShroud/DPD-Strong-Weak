"""Bounded common-plane delay audit; not full ray tracing or localization."""
import hashlib
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent/'step1_deployment_20261025/run_20261025_142854_848860'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def plane(v):
    a, e, b = v
    return np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)]), b


def verify(n, b, sources, receivers, lengths, all_points):
    rays = []
    for p, r, length in zip(sources, receivers, lengths):
        mirrored = p-2*(p@n-b)*n
        denominator = (r-mirrored)@n
        t = (b-mirrored@n)/denominator if abs(denominator) > 1e-12 else float('nan')
        q = mirrored+t*(r-mirrored)
        actual = np.linalg.norm(p-q)+np.linalg.norm(r-q)
        u = (q-p)/np.linalg.norm(q-p)
        outgoing = (r-q)/np.linalg.norm(r-q)
        rays.append({'point_m': q.tolist(), 'line_fraction': float(t),
                     'path_error_m': float(abs(actual-length)),
                     'reflection_error': float(np.linalg.norm(u-2*(u@n)*n-outgoing))})
    clearance = float(np.min(all_points@n-b))
    valid = bool(clearance > 0 and all(0 < r['line_fraction'] < 1
                  and r['path_error_m'] < 1e-6 and r['reflection_error'] < 1e-8 for r in rays))
    return {'plane_normal': n.tolist(), 'plane_offset_m': float(b),
            'minimum_same_side_clearance_m': clearance, 'rays': rays, 'valid': valid}


def fit_plane(sources, receivers, lengths, all_points):
    # Scale metre coordinates uniformly for numerical conditioning only.
    scale = 1000.
    p, r, L, points = sources/scale, receivers/scale, lengths/scale, all_points/scale
    rng = np.random.default_rng(20261025)
    trials = []
    best = None
    for index in range(32):
        n0 = rng.normal(size=3)
        n0 /= np.linalg.norm(n0)
        x0 = [np.arctan2(n0[1], n0[0]), np.arcsin(n0[2]), np.min(points@n0)-.05]

        def residual(x):
            n, b = plane(x)
            image = p-2*(p@n-b)[:, None]*n
            return np.r_[np.linalg.norm(image-r, axis=1)-L, np.minimum(points@n-b, 0.)]

        fit = least_squares(residual, x0, xtol=1e-12, ftol=1e-12, gtol=1e-12,
                            max_nfev=2000)
        n, b = plane(fit.x)
        check = verify(n, b*scale, sources, receivers, lengths, all_points)
        entry = {'start': index, 'nfev': fit.nfev, 'status': fit.status,
                 'least_squares_residual_norm': float(np.linalg.norm(fit.fun)), **check}
        trials.append(entry)
        if check['valid'] and (best is None or max(z['path_error_m'] for z in check['rays'])
                               < max(z['path_error_m'] for z in best['rays'])):
            best = entry
    return {'trials': trials, 'valid_starts': sum(x['valid'] for x in trials), 'witness': best}


def main():
    start = time.perf_counter()
    out = ROOT/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    paths = [Path(__file__), ROOT/'执行合同.md', PRIOR/'result.json', PRIOR/'iq_witness.npz']
    identity = {str(p.resolve(strict=True)): sha(p) for p in paths}
    (out/'preflight.json').write_text(json.dumps({'sha256': identity, 'python': platform.python_version(),
         'numpy': np.__version__, 'scipy': scipy.__version__}, ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    data = json.loads((PRIOR/'result.json').read_text(encoding='utf-8'))
    excess = []
    for case in data['cases']:
        for candidate in case['candidates']:
            if (not candidate['true_support'] and candidate['in_working_box']
                    and candidate['positive_excess'] and candidate['separation_m'] > 1e-3):
                excess.append({'case': case['id'], 'mode': case['mode'],
                               'true_excess_m': case['excess_m'], **candidate})
    w = data['witness']
    s = np.array(w['stations_m'])
    results = {}
    with np.load(PRIOR/'iq_witness.npz') as iq:
        for name in ['original', 'alternate']:
            src = np.array([w[name+'_position_m'], w['unchanged_second_source_m']])
            missing = next(i for i in range(5) if i not in w[name+'_los_support'])
            receivers = s[[missing, 4]]
            lengths = iq[name+'_lengths_m'][[0, 1], [missing, 4]]
            results[name] = fit_plane(src, receivers, lengths, np.vstack([src, s]))
        src = np.array([w['original_position_m'], w['unchanged_second_source_m']])
        receivers = s[[0, 4]]
        n, b = np.array([1., 0., 0.]), -900.
        image = src-2*(src@n-b)[:, None]*n
        L = np.linalg.norm(image-receivers, axis=1)
        results['positive_control'] = fit_plane(src, receivers, L, np.vstack([src, s]))
    check = {'positive_control_found': results['positive_control']['witness'] is not None,
             'input_unchanged': all(sha(Path(p)) == h for p, h in identity.items()),
             'counts_preserved': len(data['cases']) == 540}
    result = {'identity': identity, 'changed_support_candidates_in_box': excess,
              'common_plane_delay_fits': results, 'checks': check,
              'all_checks_pass': all(check.values()), 'wall_seconds': time.perf_counter()-start}
    (out/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'out': str(out), 'valid_starts': {k:v['valid_starts'] for k,v in results.items()},
                      'candidate_count': len(excess), 'checks': check, 'seconds': result['wall_seconds']}, ensure_ascii=False))
    if not result['all_checks_pass']:
        raise RuntimeError('Control or evidence identity failed')


if __name__ == '__main__':
    main()
