"""Enumerate LOS-support/position ambiguities; not estimator performance."""
import csv
import hashlib
import importlib.util
import itertools
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent/'step1_identifiability_20261025/check_model.py'
spec = importlib.util.spec_from_file_location('previous_model_audit', PREVIOUS)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
TRUE_SUPPORT = [[0, 1, 2, 3], [2, 3, 4, 0]]
GEOMETRIES = [(0., 1., 0.), (.37, .85, 10.), (-.29, 1.15, -5.)]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def roots_for_lengths(anchors, lengths):
    differences = lengths[1:]-lengths[0]
    matrix = 2*(anchors[1:]-anchors[0])
    rhs = np.sum(anchors[1:]**2, axis=1)-np.sum(anchors[0]**2)-differences**2
    a = np.linalg.solve(matrix, rhs)
    b = np.linalg.solve(matrix, -2*differences)
    v = a-anchors[0]
    polynomial = [b@b-1, 2*(v@b), v@v]
    result = []
    for root in np.roots(polynomial):
        if abs(root.imag) > 1e-7 or root.real < 0:
            continue
        p = a+b*root.real
        residual = float(np.max(abs(base.diff(p, anchors)-differences)))
        if residual < 1e-6:
            # Check independent nonlinear original equations, not merely squared equations.
            ls = least_squares(lambda x: base.diff(x, anchors)-differences, p,
                               jac=lambda x: base.jac(x, anchors), xtol=1e-13, ftol=1e-13, gtol=1e-13)
            result.append({'position': p, 'range_residual_m': residual,
                           'nonlinear_residual_m': float(np.max(abs(ls.fun))),
                           'nonlinear_position_difference_m': float(np.linalg.norm(p-ls.x))})
    return result


def within(p, zlo, zhi):
    return bool(np.all(abs(p[:2]) <= 600) and zlo <= p[2] <= zhi)


def configuration(gi):
    angle, scale, z = GEOMETRIES[gi]
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    stations = base.STATIONS.copy()
    sources = base.SOURCES.copy()
    stations[:, :2] = scale*stations[:, :2]@rotation.T
    sources[:, :2] = scale*sources[:, :2]@rotation.T
    sources[:, 2] += z
    return stations, sources


def observed_lengths(stations, sources):
    lengths = np.array([base.ranges(p, stations) for p in sources])
    for qi in range(2):
        missing = next(i for i in range(5) if i not in TRUE_SUPPORT[qi])
        lengths[qi, missing] += 80+12*missing
    return lengths


def make_iq_witness(out, gi, qi, candidate):
    stations, sources = configuration(gi)
    original_lengths = observed_lengths(stations, sources)
    alternate_sources = sources.copy()
    alternate_sources[qi] = candidate
    delta_m = base.ranges(candidate, stations)[TRUE_SUPPORT[qi][0]]-original_lengths[qi, TRUE_SUPPORT[qi][0]]
    alternate_lengths = original_lengths.copy()
    alternate_lengths[qi] += delta_m
    # Enforce alternate LOS lengths by independent geometry rather than uniform-shift assignment.
    for mi in TRUE_SUPPORT[qi]:
        alternate_lengths[qi, mi] = np.linalg.norm(candidate-stations[mi])
    rng = np.random.default_rng(20261026)
    frequency = np.sort(rng.uniform(-6.25e6, 6.25e6, 257))
    coefficient = rng.normal(size=(2, 257))+1j*rng.normal(size=(2, 257))
    coefficient /= np.linalg.norm(coefficient, axis=1)[:, None]
    shifted_coefficient = coefficient.copy()
    shifted_coefficient[qi] *= np.exp(2j*np.pi*frequency*delta_m/base.C)
    gain = np.exp(1j*rng.uniform(-np.pi, np.pi, (2, 5)))
    for source in range(2):
        for mi in range(5):
            if mi not in TRUE_SUPPORT[source]:
                gain[source, mi] *= .2
    t = np.arange(1024)/50e6
    signals = []
    for coeff, lengths in [(coefficient, original_lengths), (shifted_coefficient, alternate_lengths)]:
        y = np.zeros((5, 1024), complex)
        for source in range(2):
            for mi in range(5):
                y[mi] += gain[source, mi]*base.waveform(frequency, coeff[source], t-lengths[source, mi]/base.C)
        signals.append(y)
    y, ya = signals
    error = float(np.linalg.norm(y-ya)/np.linalg.norm(y))
    mi = next(i for i in range(5) if i not in TRUE_SUPPORT[qi])
    reflect = base.reflection_witness(candidate, stations[mi], alternate_lengths[qi, mi])
    np.savez_compressed(out/'iq_witness.npz', original=y, alternate=ya, t_s=t, frequency_hz=frequency,
                        coefficient=coefficient, shifted_coefficient=shifted_coefficient, gain=gain,
                        original_lengths_m=original_lengths, alternate_lengths_m=alternate_lengths)
    return {'geometry': gi, 'source': qi, 'original_sources_m': sources.tolist(),
            'alternate_sources_m': alternate_sources.tolist(), 'stations_m': stations.tolist(),
            'changed_source_separation_m': float(np.linalg.norm(candidate-sources[qi])),
            'iq_relative_difference': error, 'waveform_time_shift_s': float(delta_m/base.C),
            'remaining_nlos_excess_m': float(alternate_lengths[qi, mi]-np.linalg.norm(candidate-stations[mi])),
            'reflection_witness': reflect}


def main():
    started = time.perf_counter()
    out = ROOT/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    if not ROOT.resolve().is_relative_to(Path('F:/DPD_Strong_Weak/outputs_e2e').resolve()):
        raise RuntimeError('Wrong output root')
    out.mkdir(exist_ok=False)
    inputs = {str(p.resolve()): sha(p) for p in [Path(__file__), PREVIOUS, ROOT/'执行合同.md']}
    identity = {'started_utc': datetime.now(timezone.utc).isoformat(), 'inputs_sha256': inputs,
                'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                'experiment_type': 'MODEL_IDENTIFIABILITY_NOT_LOCALIZATION_PERFORMANCE'}
    (out/'preflight.json').write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    all_rows, case_rows, five_rows = [], [], []
    selected = None
    for gi in range(3):
        stations, sources = configuration(gi)
        lengths = observed_lengths(stations, sources)
        for qi, p in enumerate(sources):
            case = {'geometry': gi, 'source': qi, 'position_m': p.tolist(),
                    'true_support': TRUE_SUPPORT[qi], 'candidates': []}
            for support in itertools.combinations(range(5), 4):
                anchors = stations[list(support)]
                for candidate in roots_for_lengths(anchors, lengths[qi, list(support)]):
                    x = candidate.pop('position')
                    common_shift = np.linalg.norm(x-stations[support[0]])-lengths[qi, support[0]]
                    absent = next(i for i in range(5) if i not in support)
                    excess = float(lengths[qi, absent]+common_shift-np.linalg.norm(x-stations[absent]))
                    same_support = set(support) == set(TRUE_SUPPORT[qi])
                    separation = float(np.linalg.norm(x-p))
                    valid = excess > 1e-7
                    row = {'geometry': gi, 'source': qi, 'support': list(support), 'position_m': x.tolist(),
                           'true_support': same_support, 'separation_m': separation,
                           'nlos_station': absent, 'nlos_excess_m': excess,
                           'positive_excess': valid,
                           'domain_20_120': within(x, 20, 120), 'domain_20_150': within(x, 20, 150),
                           'domain_0_300': within(x, 0, 300), **candidate}
                    case['candidates'].append(row)
                    all_rows.append(row)
                    if selected is None and same_support and valid and row['domain_20_120'] and separation > 1e-3:
                        selected = (gi, qi, x.copy())
            case_rows.append(case)
            # Five LOS known-support case gives a linear augmented system in (x,y,z,d0).
            d = base.ranges(p, stations)
            dr = d[1:]-d[0]
            matrix = np.column_stack([2*(stations[1:]-stations[0]), 2*dr])
            rhs = np.sum(stations[1:]**2, axis=1)-np.sum(stations[0]**2)-dr**2
            sol = np.linalg.solve(matrix, rhs)
            five_rows.append({'geometry': gi, 'source': qi, 'augmented_rank': int(np.linalg.matrix_rank(matrix)),
                              'position_error_m': float(np.linalg.norm(sol[:3]-p)),
                              'original_range_residual_m': float(np.max(abs(base.diff(sol[:3], stations)-dr)))})
    witness = make_iq_witness(out, *selected) if selected is not None else None
    summary = {}
    for domain in ['domain_20_120', 'domain_20_150', 'domain_0_300']:
        summary[domain] = {
            'source_configurations': len(case_rows),
            'known_support_with_alternative': sum(any(x[domain] and x['positive_excess'] and x['true_support'] and x['separation_m'] > 1e-3 for x in c['candidates']) for c in case_rows),
            'unknown_support_with_alternative': sum(any(x[domain] and x['positive_excess'] and x['separation_m'] > 1e-3 for x in c['candidates']) for c in case_rows),
            'with_changed_support_alternative': sum(any(x[domain] and x['positive_excess'] and not x['true_support'] and x['separation_m'] > 1e-3 for x in c['candidates']) for c in case_rows),
        }
    checks = {
        'all_true_solutions_recovered': all(any(x['true_support'] and x['separation_m'] < 1e-5 and x['positive_excess'] for x in c['candidates']) for c in case_rows),
        'all_polynomial_candidates_original_equations': max(x['range_residual_m'] for x in all_rows) < 1e-6,
        'independent_nonlinear_equations': max(x['nonlinear_residual_m'] for x in all_rows) < 1e-6,
        'five_los_positive_control': all(x['augmented_rank'] == 4 and x['position_error_m'] < 1e-5 for x in five_rows),
        'iq_witness_valid_if_present': witness is None or (witness['iq_relative_difference'] < 1e-10 and witness['remaining_nlos_excess_m'] > 0 and witness['reflection_witness']['reflection_law_residual'] < 1e-12),
        'input_identity_unchanged': all(sha(Path(p)) == h for p, h in inputs.items()),
    }
    result = {'identity': identity, 'cases': case_rows, 'summary': summary, 'five_los_known_support': five_rows,
              'iq_witness': witness, 'checks': {k: bool(v) for k, v in checks.items()},
              'all_checks_pass': bool(all(checks.values())), 'wall_seconds': time.perf_counter()-started,
              'scope': 'Oracle source-separated path delays; independent unknown NLOS length/gain; no shared-wall constraint',
              'not_proven': 'Unknown multi-source blind separation and unrestricted all-city geometry not tested'}
    (out/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    with (out/'all_candidates.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print(json.dumps({'output': str(out), 'summary': summary, 'witness': witness, 'checks': checks}, ensure_ascii=False))
    if not result['all_checks_pass']:
        raise RuntimeError('Numerical audit failed; preserve output and inspect.')


if __name__ == '__main__':
    main()
