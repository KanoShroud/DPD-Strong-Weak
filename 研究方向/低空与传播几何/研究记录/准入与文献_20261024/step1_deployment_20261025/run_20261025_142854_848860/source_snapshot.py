"""Finite exhaustive support/root audit for fixed deployment families."""
import csv
import hashlib
import importlib.util
import itertools
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent/'step1_four_los_20261025/check_four_los.py'
spec = importlib.util.spec_from_file_location('four_los_audit', OLD)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
base = old.base
XY = [(80., 60.), (-120., 140.), (520., 80.), (-460., -120.)]
HEIGHTS = [30., 75., 120.]
EXCESS = [10., 50., 150.]
MODES = ['flat', 'slope', 'roof']


def plane_z(mode, x, y):
    if mode == 'flat':
        return 4.
    if mode == 'slope':
        return 10.+.01*x+.005*y
    return 0.


def stations(mode):
    s = base.STATIONS.copy()
    if mode != 'roof':
        s[:, 2] = [plane_z(mode, x, y) for x, y in s[:, :2]]
    return s


def regions(mode, p):
    height = p[2]-plane_z(mode, p[0], p[1])
    return bool(height > 1e-6), bool(height > 1e-6 and height <= 150 and np.all(abs(p[:2]) <= 700))


def enumerate_roots(anchors, lengths):
    relative = anchors[1:]-anchors[0]
    _, singular, vh = np.linalg.svd(relative)
    if singular[-1] > 1e-9:
        return old.roots_for_lengths(anchors, lengths), 'nonplanar_quadratic'
    if singular[1] < 1e-9:
        return [], 'collinear_geometry'
    basis, normal = vh[:2].T, vh[-1]
    uv = relative@basis
    rd = lengths[1:]-lengths[0]
    M = np.column_stack([2*uv, 2*rd])
    rhs = np.sum(relative**2, axis=1)-rd**2
    if np.linalg.matrix_rank(M, tol=1e-9) < 3:
        return [], 'singular_augmented_planar'
    fit = np.linalg.solve(M, rhs)
    d0 = fit[2]
    height2 = d0**2-fit[:2]@fit[:2]
    if d0 < 0 or height2 < -1e-7:
        return [], 'planar_no_real_solution'
    height = np.sqrt(max(height2, 0.))
    roots = []
    for sign in ([1., -1.] if height > 1e-9 else [1.]):
        p = anchors[0]+basis@fit[:2]+sign*height*normal
        residual = float(np.max(abs(base.diff(p, anchors)-rd)))
        if residual < 1e-6:
            roots.append({'position': p, 'range_residual_m': residual})
    return roots, 'planar_mirror_pair'


def iq_witness(out, case, alt):
    mode = case['mode']
    s = stations(mode)
    p = np.array(case['source_m'])
    q = np.array(alt['position_m'])
    p2 = np.array([-80., -100., plane_z(mode, -80., -100.)+65.])
    support = [i for i in range(5) if i != case['missing_station']]
    support_alt = alt['support']
    L = np.vstack([base.ranges(p, s), base.ranges(p2, s)])
    L[0, case['missing_station']] += case['excess_m']
    L[1, 4] += 80.
    shift_m = np.linalg.norm(q-s[support_alt[0]])-L[0, support_alt[0]]
    La = L.copy()
    La[0] += shift_m
    for mi in support_alt:
        La[0, mi] = np.linalg.norm(q-s[mi])
    rng = np.random.default_rng(20261027)
    f = np.sort(rng.uniform(-6.25e6, 6.25e6, 257))
    b = rng.normal(size=(2, 257))+1j*rng.normal(size=(2, 257))
    b /= np.linalg.norm(b, axis=1)[:, None]
    ba = b.copy()
    ba[0] *= np.exp(2j*np.pi*f*shift_m/base.C)
    gain = np.exp(1j*rng.uniform(-np.pi, np.pi, (2, 5)))
    gain[0, case['missing_station']] *= .2
    gain[1, 4] *= .2
    t = np.arange(1024)/50e6
    data = []
    for coeff, length in [(b, L), (ba, La)]:
        y = np.zeros((5, 1024), complex)
        for source in range(2):
            for mi in range(5):
                y[mi] += gain[source, mi]*base.waveform(f, coeff[source], t-length[source, mi]/base.C)
        data.append(y)
    rel = float(np.linalg.norm(data[0]-data[1])/np.linalg.norm(data[0]))
    alternate_missing = next(i for i in range(5) if i not in support_alt)
    ray = base.reflection_witness(q, s[alternate_missing], La[0, alternate_missing])
    energy = abs(gain[0])**2
    np.savez_compressed(out/'iq_witness.npz', original=data[0], alternate=data[1],
                        frequency_hz=f, coefficients=b, alternate_coefficients=ba,
                        gains=gain, original_lengths_m=L, alternate_lengths_m=La, t_s=t)
    return {'case_id': case['id'], 'mode': mode, 'original_position_m': p.tolist(),
            'alternate_position_m': q.tolist(), 'unchanged_second_source_m': p2.tolist(),
            'stations_m': s.tolist(), 'original_los_support': support, 'alternate_los_support': support_alt,
            'separation_m': float(np.linalg.norm(p-q)), 'iq_relative_difference': rel,
            'alternate_nlos_excess_m': float(La[0, alternate_missing]-np.linalg.norm(q-s[alternate_missing])),
            'original_los_energy_sum': float(np.sum(energy[support])),
            'original_nlos_energy': float(energy[case['missing_station']]),
            'alternate_los_energy_sum': float(np.sum(energy[support_alt])),
            'alternate_nlos_energy': float(energy[alternate_missing]), 'reflection': ray}


def main():
    start = time.perf_counter()
    out = ROOT/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    assert ROOT.resolve().is_relative_to(Path('F:/DPD_Strong_Weak/outputs_e2e').resolve())
    out.mkdir(exist_ok=False)
    paths = [Path(__file__).resolve(), OLD, base.ROOT/'check_model.py', ROOT/'执行合同.md']
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    identity = {'input_sha256': hashes, 'python': platform.python_version(), 'numpy': np.__version__,
                'scipy': scipy.__version__, 'kind': 'DETERMINISTIC_ORACLE_PATH_GEOMETRY_NOT_PERFORMANCE'}
    (out/'preflight.json').write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    candidates, cases, status_records = [], [], []
    selected = None
    for mode in MODES:
        s = stations(mode)
        for xy_index, (x, y) in enumerate(XY):
            for height in HEIGHTS:
                p = np.array([x, y, plane_z(mode, x, y)+height])
                for missing in range(5):
                    for extra in EXCESS:
                        case = {'id': len(cases), 'mode': mode, 'xy_index': xy_index,
                                'height_above_reference_m': height, 'source_m': p.tolist(),
                                'missing_station': missing, 'excess_m': extra, 'candidates': []}
                        L = base.ranges(p, s)
                        L[missing] += extra
                        for subset in itertools.combinations(range(5), 4):
                            rs, status = enumerate_roots(s[list(subset)], L[list(subset)])
                            known = missing not in subset
                            status_records.append({'case': case['id'], 'support': list(subset), 'true_support': known, 'status': status})
                            for r in rs:
                                pos = r.pop('position')
                                shift = np.linalg.norm(pos-s[subset[0]])-L[subset[0]]
                                absent = next(i for i in range(5) if i not in subset)
                                e = float(L[absent]+shift-np.linalg.norm(pos-s[absent]))
                                half, box = regions(mode, pos)
                                row = {'case': case['id'], 'mode': mode, 'support': list(subset),
                                       'position_m': pos.tolist(), 'true_support': known,
                                       'separation_m': float(np.linalg.norm(pos-p)),
                                       'positive_excess': bool(e > 1e-7), 'nlos_excess_m': e,
                                       'in_halfspace': half, 'in_working_box': box, **r}
                                case['candidates'].append(row)
                                candidates.append(row)
                                if selected is None and not known and e > 1e-7 and box and row['separation_m'] > 1e-3:
                                    selected = (case.copy(), row.copy())
                        cases.append(case)
    summary = {}
    for mode in MODES:
        subset = [c for c in cases if c['mode'] == mode]
        summary[mode] = {'conditions': len(subset)}
        for domain in ['in_halfspace', 'in_working_box']:
            summary[mode][domain] = {label: sum(any(r[domain] and r['positive_excess'] and r['separation_m'] > 1e-3 and rule(r) for r in c['candidates']) for c in subset)
                                     for label, rule in [('known_support_ambiguity', lambda r: r['true_support']),
                                                         ('any_support_ambiguity', lambda r: True),
                                                         ('changed_support_ambiguity', lambda r: not r['true_support'])]}
    witness = iq_witness(out, *selected) if selected is not None else None
    singular = [s for s in status_records if s['status'] in ['collinear_geometry', 'singular_augmented_planar']]
    checks = {'all_true_positions_recovered': all(any(r['true_support'] and r['positive_excess'] and r['separation_m'] < 1e-4 and r['in_halfspace'] for r in c['candidates']) for c in cases),
              'all_finite_roots_original_equations': bool(max(r['range_residual_m'] for r in candidates) < 1e-6),
              'input_unchanged': all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items()),
              'witness_valid_if_found': witness is None or (witness['iq_relative_difference'] < 1e-10 and witness['alternate_nlos_excess_m'] > 0 and witness['reflection']['reflection_law_residual'] < 1e-12)}
    result = {'identity': identity, 'mode_stations_m': {m: stations(m).tolist() for m in MODES},
              'summary': summary, 'cases': cases, 'enumeration_status': status_records,
              'singular_systems': singular, 'witness': witness, 'checks': {k: bool(v) for k, v in checks.items()},
              'all_checks_pass': bool(all(checks.values())), 'wall_seconds': time.perf_counter()-start}
    (out/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    with (out/'all_candidates.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        fields = list(dict.fromkeys(key for row in candidates for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidates)
    print(json.dumps({'output': str(out), 'summary': summary, 'singular_count': len(singular), 'witness': witness, 'checks': checks, 'seconds': result['wall_seconds']}, ensure_ascii=False))
    if not result['all_checks_pass']:
        raise RuntimeError('Numerical consistency failure; preserve evidence.')


if __name__ == '__main__':
    main()
