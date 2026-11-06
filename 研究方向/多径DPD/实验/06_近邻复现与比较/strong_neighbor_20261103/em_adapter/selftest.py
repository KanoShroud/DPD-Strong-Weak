"""RF bridge formula tests. Random frequency data, NOT localization performance."""
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import time

import numpy as np
from threadpoolctl import threadpool_limits

import bridge as b


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    here = Path(__file__).resolve().parent
    old = b.BASE/'strong_neighbor_20261102/em_core/core.py'
    spec = importlib.util.spec_from_file_location('prior_em_core', old)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    sources = [here/'bridge.py', here/'selftest.py', old,
               b.BASE/'unknown_order_20261030/reference.py',
               b.BASE/'risk_step_20261030/pilot.py', b.BASE/'structural_probe.py',
               b.BASE/'neighbor_closure_20261030/Friedlander2014.txt']
    before = {str(p.resolve(strict=True)): sha(p) for p in sources}
    out = here/('test_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    start = time.perf_counter()
    checks, detail = {}, {}
    rng = np.random.default_rng(2026110301)
    n = 8192
    full_f = np.fft.fftfreq(n, 1/b.ref.FS)
    f = full_f[abs(full_f) <= b.ref.B/2]
    # Independent numerical fixture; no true physical paths, source coordinates,
    # generated IQ or localization outputs are read by this test.
    x = np.r_[np.array([[-170., 80.], [280., -130.]]).ravel()/b.ref.UNIT,
              rng.uniform(.5, 1.2, 8), rng.uniform(-.2, .2, 8),
              .8, 0., 2.3, .8, 0., 6.7, -.17, .13, 11.9,
              .03, .21, 19.7]
    paths = [(1, 0), (1, 0), (0, 1), (4, 1)]
    y = rng.normal(size=(len(f), 5)) + 1j*rng.normal(size=(len(f), 5))
    a, da, m, q = b.components(x, f, paths)
    h, dh = b.aggregate(a, da, m, q)
    hr, dhr, _ = b.ref.response(x, f, paths)
    checks['M5_Q2_full_frequency_shape'] = h.shape == (2049, 5, 2)
    checks['RF_matrix_matches_frozen_reference'] = np.allclose(h, hr, rtol=1e-12, atol=1e-12)
    checks['RF_derivative_matches_frozen_reference'] = np.allclose(dh, dhr, rtol=1e-12, atol=1e-12)
    checks['reference_gain_gauge_station0_one'] = np.allclose(a[0], np.exp(-2j*np.pi*f*np.linalg.norm(x[:2]*b.ref.UNIT-b.ref.STATIONS[0])/b.ref.C))
    s = b.profile_wave(h, y)
    mu = b.means(a, s, q)
    beta = b.station_beta(m)
    z = b.expectation(y, mu, m, beta)
    norm2 = float(np.sum(abs(y)**2))
    cost, grad = b.surrogate(x, f, paths, s, z, beta, norm2)
    rr, jj = b.ref.evaluate(x, f, y, paths)
    checks['touches_observed_profile_objective'] = abs(cost-.5*rr@rr) < 1e-12
    checks['touches_original_coordinate_VP_gradient'] = np.allclose(grad, jj.T@rr, rtol=1e-9, atol=1e-11)
    checks['one_shared_waveform_per_source'] = s.shape == (len(f), 2)
    checks['beta_normalized_per_station'] = np.allclose(np.bincount(m, weights=beta), 1.)
    checks['E_step_sums_to_observation'] = all(np.allclose(z[m == station].sum(axis=0), y[:, station]) for station in range(5))
    checks['E_step_matches_prior_core_stationwise'] = all(np.allclose(z[m == station], core.expectation(y[:, station], mu[m == station], beta[m == station])) for station in range(5))
    checks['individual_power_bound_not_total_bound'] = bool(np.array_equal(b.ref.project(x, n), x) and sum(x[k]**2+x[k+1]**2 for k in (20, 23)) > 1)
    sx = x.copy()
    sx[22] = n/b.ref.FS*b.ref.B + 1
    sx[25] = -1
    sx[20:22] = [3, 4]
    px = b.ref.project(sx, n)
    checks['public_alias_delay_interval'] = px[22] == n/b.ref.FS*b.ref.B and px[25] == 0.
    checks['disk_not_box_reflection_projection'] = np.allclose(px[20:22], [.6, .8])
    checks['original_coordinate_projection_idempotent'] = np.array_equal(b.ref.project(px, n), px)
    try:
        b.expectation(y, mu, m, beta/5)
        checks['reject_global_beta_normalization'] = False
    except ValueError:
        checks['reject_global_beta_normalization'] = True
    # Full 32-coordinate finite differences on evenly strided frequency subset.
    ids = np.arange(0, len(f), 37)
    fs, ys = f[ids], y[ids]
    aa, dd, mm, qq = b.components(x, fs, paths)
    hs, _ = b.aggregate(aa, dd, mm, qq)
    ss = b.profile_wave(hs, ys)
    zz = b.expectation(ys, b.means(aa, ss, qq), mm, beta)
    nn = float(np.sum(abs(ys)**2))
    cc, gg = b.surrogate(x, fs, paths, ss, zz, beta, nn)
    step = 1e-5
    eye = np.eye(len(x))
    numeric = np.array([(b.surrogate(x+step*e, fs, paths, ss, zz, beta, nn)[0]-b.surrogate(x-step*e, fs, paths, ss, zz, beta, nn)[0])/(2*step) for e in eye])
    detail['surrogate_gradient_relative_error'] = float(np.linalg.norm(numeric-gg)/np.linalg.norm(numeric))
    checks['all_coordinate_surrogate_finite_difference'] = detail['surrogate_gradient_relative_error'] < 2e-7
    num_a = np.stack([(b.components(x+step*e, fs, paths)[0]-b.components(x-step*e, fs, paths)[0])/(2*step) for e in eye], axis=-1)
    detail['component_jacobian_relative_error'] = float(np.linalg.norm(num_a-dd)/np.linalg.norm(num_a))
    checks['all_coordinate_component_finite_difference'] = detail['component_jacobian_relative_error'] < 2e-7
    # SI-unit gradient check: p in metres and tau in seconds, not optimizer scale.
    units = np.ones(len(x))
    units[:4] = b.ref.UNIT
    units[22::3] = 1/b.ref.B
    raw = x*units
    eta = np.array([1e-4]*4 + [1e-5]*16 + [1e-5, 1e-5, 1e-12]*len(paths))
    physical_numeric = np.array([(b.surrogate((raw+eta[k]*e)/units, fs, paths, ss, zz, beta, nn)[0]-b.surrogate((raw-eta[k]*e)/units, fs, paths, ss, zz, beta, nn)[0])/(2*eta[k]) for k, e in enumerate(eye)])
    # Per-coordinate scale avoids high-frequency delay entries masking position errors.
    physical_error = abs(physical_numeric-gg/units)/np.maximum(abs(gg/units), 1e-6)
    detail['SI_gradient_max_relative_error'] = float(max(physical_error))
    checks['SI_position_and_delay_gradient'] = bool(max(physical_error) < 2e-4)
    worst_gap = -np.inf
    for _ in range(20):
        trial = b.ref.project(x+rng.normal(scale=.2, size=len(x)), n)
        ta, _, _, _ = b.components(trial, fs, paths)
        trial_s = ss+.01*(rng.normal(size=ss.shape)+1j*rng.normal(size=ss.shape))
        trial_mu = b.means(ta, trial_s, qq)
        qqcost = float(np.sum(abs(zz-trial_mu)**2/beta[:, None])/nn)
        obs = b.observed_cost(ys, trial_mu, mm)
        worst_gap = max(worst_gap, obs-qqcost)
    detail['maximum_observed_minus_surrogate'] = worst_gap
    checks['20_joint_parameter_waveform_majorization'] = worst_gap <= 1e-12
    new_s = b.shared_wave_mstep(aa, qq, zz, beta)
    new_mu = b.means(aa, new_s, qq)
    qnew = float(np.sum(abs(zz-new_mu)**2/beta[:, None])/nn)
    checks['shared_wave_surrogate_Mstep_decreases'] = qnew <= cc+1e-12
    checks['shared_wave_Mstep_observed_decreases'] = b.observed_cost(ys, new_mu, mm) <= cc+1e-12
    # Compare analytic shared-wave update with independent dense per-frequency LS.
    independent = []
    for k in range(len(fs)):
        mat = np.zeros((len(mm), 2), complex)
        mat[np.arange(len(mm)), qq] = aa[:, k]/np.sqrt(beta)
        independent.append(np.linalg.lstsq(mat, zz[:, k]/np.sqrt(beta), rcond=None)[0])
    checks['shared_wave_update_independent_LS'] = np.allclose(new_s, independent, rtol=1e-12, atol=1e-12)
    states = []
    xx = x.copy()
    for _ in range(6):
        xx, info = b.bounded_mm_step(xx, f, y, paths, n)
        states.append(info)
    checks['6_MM_steps_all_accepted'] = all(v['accepted'] for v in states)
    checks['6_MM_steps_profiled_objective_nonincrease'] = all(v['final_cost'] <= v['old_cost']+1e-12 for v in states)
    checks['6_MM_steps_respect_surrogate_bound'] = all(v['final_cost'] <= v['surrogate_cost']+1e-12 and v['surrogate_cost'] <= v['old_cost'] for v in states)
    checks['MM_terminal_feasible'] = np.linalg.norm(xx-b.ref.project(xx, n)) < 1e-12
    detail['correctness_only_MM_steps'] = states
    after = {str(p): sha(Path(p)) for p in before}
    checks['inputs_unchanged'] = before == after
    checks = {key: bool(value) for key, value in checks.items()}
    result = dict(kind='formula_tests_not_localization_performance', checks=checks,
                  checks_count=len(checks), all_pass=all(checks.values()), detail=detail,
                  seconds=time.perf_counter()-start, source_sha256=before,
                  python=platform.python_version(), numpy=np.__version__,
                  seed=2026110301, fft_n=n, fs=b.ref.FS, bandwidth=b.ref.B,
                  frequency_count=len(f), candidate_support=paths,
                  no_truth_initialization=True, no_formal_input_loaded=True)
    (out/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(out)
    print(json.dumps(dict(all_pass=result['all_pass'], checks=len(checks), seconds=result['seconds'], failed=[k for k,v in checks.items() if not v]), ensure_ascii=False))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    with threadpool_limits(1):
        main()
