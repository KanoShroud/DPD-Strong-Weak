"""Small deterministic identifiability audit, not a localization benchmark."""
import csv
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import root

ROOT = Path(__file__).resolve().parent
C = 299792458.0
STATIONS = np.array([[-400., -300., 4.], [450., -250., 15.], [380., 400., 7.],
                     [-350., 350., 23.], [20., -460., 11.]])
SOURCES = np.array([[80., 60., 60.], [-120., 140., 80.]])
SUPPORT = [[0, 1, 2], [2, 3, 4]]
ALTERNATE_Z = [110., 30.]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ranges(p, stations):
    return np.linalg.norm(p - stations, axis=1)


def diff(p, stations):
    d = ranges(p, stations)
    return d[1:] - d[0]


def jac(p, stations):
    u = (p - stations) / ranges(p, stations)[:, None]
    return u[1:] - u[0]


def at_height(p, anchors, z):
    target = diff(p, anchors)
    fit = root(lambda xy: diff(np.r_[xy, z], anchors) - target,
               p[:2], jac=lambda xy: jac(np.r_[xy, z], anchors)[:, :2], tol=1e-10)
    candidate = np.r_[fit.x, z]
    error = np.max(np.abs(diff(candidate, anchors) - target))
    if error > 1e-7:
        raise RuntimeError(f'Continuation residual {error}; do not report failed root as equal.')
    return candidate, error


def algebraic_four(p, anchors):
    """Enumerate four-anchor range-difference roots, validating original equations."""
    r = diff(p, anchors)
    matrix = 2 * (anchors[1:] - anchors[0])
    rhs = np.sum(anchors[1:]**2, axis=1) - np.sum(anchors[0]**2) - r**2
    a = np.linalg.solve(matrix, rhs)
    b = np.linalg.solve(matrix, -2*r)
    v = a - anchors[0]
    roots = np.roots([b @ b - 1, 2*(v @ b), v @ v])
    answers = []
    for distance in roots:
        if abs(distance.imag) > 1e-7 or distance.real < 0:
            continue
        x = a + b * distance.real
        residual = float(np.max(np.abs(diff(x, anchors)-r)))
        if residual < 1e-6:
            answers.append({'position_m': x.tolist(), 'range_difference_residual_m': residual,
                            'in_search_domain': bool(np.all(abs(x[:2]) <= 600) and 20 <= x[2] <= 120)})
    return answers


def reflection_witness(p, r, length):
    """One specular path for a freely located plane; not a shared-scene ray tracer."""
    d = np.linalg.norm(p-r)
    direction = np.cross((r-p)/d, np.array([0., 0., 1.]))
    direction /= np.linalg.norm(direction)
    q = (p+r)/2 + direction * np.sqrt(length**2-d**2)/2
    u_in = (q-p)/np.linalg.norm(q-p)
    u_out = (r-q)/np.linalg.norm(r-q)
    normal = u_in-u_out
    normal /= np.linalg.norm(normal)
    reflected = u_in - 2*np.dot(u_in, normal)*normal
    return {'point_m': q.tolist(), 'plane_normal': normal.tolist(),
            'path_residual_m': float(abs(np.linalg.norm(q-p)+np.linalg.norm(r-q)-length)),
            'reflection_law_residual': float(np.linalg.norm(reflected-u_out)),
            'same_side_of_plane': bool(np.dot(p-q, normal)*np.dot(r-q, normal) > 0)}


def waveform(f, coeff, t):
    return coeff @ np.exp(2j*np.pi*f[:, None]*t[None, :])


def main():
    started = time.perf_counter()
    out = ROOT / ('run_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    if not ROOT.resolve().is_relative_to(Path('F:/DPD_Strong_Weak/outputs_e2e').resolve()):
        raise RuntimeError('Output outside active project outputs.')
    out.mkdir(exist_ok=False)
    inputs = {str(p): sha(p) for p in [Path(__file__).resolve(), ROOT/'执行合同.md']}
    identity = {'utc_started': datetime.now(timezone.utc).isoformat(), 'input_sha256': inputs,
                'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                'matplotlib': matplotlib.__version__, 'output': str(out), 'experiment_type': 'IDENTIFIABILITY_WITNESS_NOT_PERFORMANCE'}
    (out/'preflight.json').write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (out/'执行合同快照.md').write_bytes((ROOT/'执行合同.md').read_bytes())
    curves, ranks, fd_errors = [], [], []
    alternatives = []
    # Three prespecified geometries, no rejection or selection using a localization outcome.
    geometry_specs = [(0., 1., 0.), (0.37, 0.85, 10.), (-0.29, 1.15, -5.)]
    for gi, (angle, scale, height_offset) in enumerate(geometry_specs):
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        stations = STATIONS.copy()
        stations[:, :2] = scale * STATIONS[:, :2] @ rot.T
        positions = SOURCES.copy()
        positions[:, :2] = scale * SOURCES[:, :2] @ rot.T
        positions[:, 2] += height_offset
        for qi, p in enumerate(positions):
            anchors = stations[SUPPORT[qi]]
            J = jac(p, anchors)
            h = 1e-3
            fd = np.column_stack([(diff(p+np.eye(3)[j]*h, anchors)-diff(p-np.eye(3)[j]*h, anchors))/(2*h) for j in range(3)])
            fd_errors.append(float(np.max(abs(fd-J))))
            for z in np.linspace(20., 120., 21):
                x, error = at_height(p, anchors, z)
                curves.append({'geometry': gi, 'source': qi, 'x_m': x[0], 'y_m': x[1], 'z_m': x[2],
                               'range_residual_m': error, 'position_separation_m': np.linalg.norm(x-p)})
            for nlos in [3, 4, 5]:
                subset = SUPPORT[qi] + [i for i in range(5) if i not in SUPPORT[qi]][:nlos-3]
                K = jac(p, stations[subset])
                singular = np.linalg.svd(K, compute_uv=False)
                # Eliminate common unknown emission time and free excess delays at blocked links.
                u = (p-stations)/ranges(p, stations)[:, None]
                blocked = [i for i in range(5) if i not in subset]
                nuisance = np.column_stack([np.ones(5), *[np.eye(5)[:, i] for i in blocked]])
                P = np.eye(5)-nuisance @ np.linalg.pinv(nuisance)
                profiled = P @ u
                profiled_rank = np.linalg.matrix_rank(profiled, tol=1e-10)
                ranks.append({'geometry': gi, 'source': qi, 'los_count': nlos,
                              'rank_range_differences': int(np.linalg.matrix_rank(K, tol=1e-10)),
                              'singular_values': singular.tolist(), 'rank_with_free_nlos_profiled': int(profiled_rank)})
            if gi == 0:
                alt, error = at_height(p, anchors, ALTERNATE_Z[qi])
                alternatives.append(alt)
    alternate = np.array(alternatives)
    rng = np.random.default_rng(20261025)
    frequencies = np.sort(rng.uniform(-6.25e6, 6.25e6, 257))
    coeff = rng.normal(size=(2, 257))+1j*rng.normal(size=(2, 257))
    coeff /= np.linalg.norm(coeff, axis=1)[:, None]
    n, fs = 1024, 50e6
    t = np.arange(n)/fs
    gains = np.empty((2, 5), complex)
    original_length = np.empty((2, 5))
    alternate_length = np.empty((2, 5))
    delta = []
    ray_checks = []
    for qi, p in enumerate(SOURCES):
        d = ranges(p, STATIONS)
        da = ranges(alternate[qi], STATIONS)
        shift = (da[SUPPORT[qi][0]]-d[SUPPORT[qi][0]])/C
        delta.append(shift)
        for mi in range(5):
            direct = mi in SUPPORT[qi]
            original_length[qi, mi] = d[mi] if direct else d[mi] + 80 + 12*mi
            alternate_length[qi, mi] = da[mi] if direct else original_length[qi, mi] + shift*C
            amp = 1. if direct else .2
            gains[qi, mi] = amp*np.exp(1j*rng.uniform(-np.pi, np.pi))
            if not direct:
                for label, position, lengths in [('original', p, original_length), ('alternate', alternate[qi], alternate_length)]:
                    length = lengths[qi, mi]
                    if length <= np.linalg.norm(position-STATIONS[mi]):
                        raise RuntimeError('NLOS path violates positive excess length.')
                    ray_checks.append({'world': label, 'source': qi, 'station': mi,
                                       **reflection_witness(position, STATIONS[mi], length)})
    delta = np.array(delta)
    alt_coeff = coeff*np.exp(2j*np.pi*delta[:, None]*frequencies[None, :])
    y = np.zeros((5, n), complex)
    ya = np.zeros_like(y)
    for qi in range(2):
        for mi in range(5):
            y[mi] += gains[qi, mi]*waveform(frequencies, coeff[qi], t-original_length[qi, mi]/C)
            ya[mi] += gains[qi, mi]*waveform(frequencies, alt_coeff[qi], t-alternate_length[qi, mi]/C)
    iq_error = float(np.linalg.norm(y-ya)/np.linalg.norm(y))
    freq_axis = np.fft.fftfreq(n, 1/fs)
    keep = abs(freq_axis) <= 7e6
    Y = np.fft.fft(y, axis=1)[:, keep]/n
    Ya = np.fft.fft(ya, axis=1)[:, keep]/n
    band_error = float(np.linalg.norm(Y-Ya)/np.linalg.norm(Y))
    four_roots = []
    for qi, p in enumerate(SOURCES):
        extra = [i for i in range(5) if i not in SUPPORT[qi]][0]
        subset = SUPPORT[qi]+[extra]
        four_roots.append({'source': qi, 'stations': subset, 'roots': algebraic_four(p, STATIONS[subset])})
    motion = []
    for length in [1024, 4096, 8192]:
        for speed in [0., 1., 5., 15.]:
            duration = length/fs
            motion.append({'samples': length, 'speed_m_s': speed, 'duration_s': duration,
                           'max_displacement_m': speed*duration,
                           'carrier_hz': 2.4e9,
                           'max_one_link_doppler_hz': 2.4e9*speed/C,
                           'max_one_link_phase_rad': 2*np.pi*2.4e9*speed/C*duration,
                           'max_interstation_phase_rad': 4*np.pi*2.4e9*speed/C*duration})
    tests = {
        'jacobian_finite_difference': max(fd_errors) < 1e-7,
        'three_los_rank_two': all(r['rank_range_differences'] == 2 and r['rank_with_free_nlos_profiled'] == 2 for r in ranks if r['los_count'] == 3),
        'four_five_los_local_rank_three': all(r['rank_range_differences'] == 3 and r['rank_with_free_nlos_profiled'] == 3 for r in ranks if r['los_count'] >= 4),
        'curve_equal_tdoa': max(r['range_residual_m'] for r in curves) < 1e-7,
        'nontrivial_position_difference': bool(np.all(np.linalg.norm(SOURCES-alternate, axis=1)>40)),
        'iq_equivalent': iq_error < 1e-10,
        'band_iq_equivalent': band_error < 1e-10,
        'positive_excess_and_reflection': all(r['path_residual_m'] < 1e-8 and r['reflection_law_residual'] < 1e-12 and r['same_side_of_plane'] for r in ray_checks),
        'four_anchor_true_root_present': all(any(np.linalg.norm(np.array(x['position_m'])-SOURCES[r['source']]) < 1e-5 for x in r['roots']) for r in four_roots),
        'input_identity_unchanged': all(sha(Path(p)) == value for p, value in inputs.items()),
    }
    result = {'identity': identity, 'stations_m': STATIONS.tolist(), 'sources_m': SOURCES.tolist(),
              'los_support': SUPPORT, 'alternate_sources_m': alternate.tolist(),
              'position_separations_m': np.linalg.norm(SOURCES-alternate, axis=1).tolist(),
              'waveform_time_shifts_s': delta.tolist(),
              'source_power_normalization': 'Equal spectral coefficient energy; unknown coefficients, not known waveforms',
              'single_link_reflection_to_direct_power': .04,
              'path_lengths_original_m': original_length.tolist(), 'path_lengths_alternate_m': alternate_length.tolist(),
              'iq_relative_difference': iq_error, 'band_iq_relative_difference': band_error,
              'jacobian_fd_max_abs_error': max(fd_errors),
              'curve_max_range_difference_residual_m': max(r['range_residual_m'] for r in curves),
              'ranks': ranks, 'four_anchor_global_candidates': four_roots, 'reflection_checks': ray_checks,
              'motion_scale_checks': motion, 'tests': tests,
              'all_tests_pass': all(tests.values()),
              'scope': 'Unknown independent path delays/gains; physical per-link specular witnesses, no jointly shared building-plane constraint',
              'not_proven': ['No general impossibility for mapless shared-reflector models', 'Four LOS is not asserted globally sufficient',
                             'Not a localization algorithm performance or novelty test']}
    with (out/'equal_tdoa_curves.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    np.savez_compressed(out/'iq_witness.npz', original=y, alternate=ya, coefficients=coeff,
                        alternate_coefficients=alt_coeff, frequencies_hz=frequencies, gains=gains,
                        t_s=t, path_lengths_original_m=original_length, path_lengths_alternate_m=alternate_length)
    fig = plt.figure(figsize=(12, 5), layout='constrained')
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(*STATIONS.T, marker='^', s=70, c='black', label='5 receivers')
    colors = ['#16736b', '#b16b2c']
    for qi in range(2):
        curve = np.array([[r['x_m'], r['y_m'], r['z_m']] for r in curves if r['geometry'] == 0 and r['source'] == qi])
        ax.plot(*curve.T, color=colors[qi], label=f'Source {qi+1}: equal-TDOA curve')
        ax.scatter(*SOURCES[qi], marker='*', s=140, c=colors[qi])
        ax.scatter(*alternate[qi], marker='x', s=80, c=colors[qi])
        for mi in SUPPORT[qi]:
            a = np.vstack([SOURCES[qi], STATIONS[mi]])
            ax.plot(*a.T, color=colors[qi], alpha=.3, linewidth=.8)
    ax.set(xlabel='x (m)', ylabel='y (m)', zlabel='height (m)', title='Three direct links leave a continuum')
    ax.legend(fontsize=7, loc='upper left')
    ax2 = fig.add_subplot(122)
    for qi in range(2):
        curve = np.array([[r['x_m'], r['y_m'], r['z_m']] for r in curves if r['geometry'] == 0 and r['source'] == qi])
        ax2.plot(curve[:, 2], np.linalg.norm(curve-SOURCES[qi], axis=1), color=colors[qi], label=f'Source {qi+1}')
    ax2.set(xlabel='Alternative height (m)', ylabel='Position separation from original (m)', title='All points have identical LOS TDOAs')
    ax2.grid(alpha=.25)
    ax2.legend()
    fig.savefig(out/'ambiguity.png', dpi=180)
    fig.savefig(out/'ambiguity.svg')
    plt.close(fig)
    result['wall_seconds'] = time.perf_counter()-started
    (out/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(out), 'tests': tests, 'iq_error': iq_error,
                      'position_separations_m': result['position_separations_m'], 'seconds': result['wall_seconds']}, ensure_ascii=False))
    if not result['all_tests_pass']:
        raise RuntimeError('Audit test failed; retain run and investigate.')


if __name__ == '__main__':
    main()
