"""Read-only replay plus bounded LOS continuation; never edits frozen estimators."""
from datetime import datetime
import importlib.util
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
CAL = BASE / 'risk_calibration_20261102'
FORMAL = CAL / 'formal_20261102_181736_622970'
sys.path.insert(0, str(CAL))
spec = importlib.util.spec_from_file_location('frozen_calibration', CAL/'runner.py')
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)
ref, engine, risk = cal.ref, cal.engine, cal.risk
from storage import load, write  # noqa: E402


def norm2(x):
    return float(np.sum(np.abs(x)**2))


def case_data(folder, contract):
    info = load(folder/'input.json')
    with np.load(folder/'input.npz', allow_pickle=False) as z:
        assert set(z.files) == {'iq'}
        iq = z['iq']
    assert cal.digest(iq) == info['iq_sha']
    condition = info['case']['condition']
    with np.load(contract['conditions'][condition]['bank'], allow_pickle=False) as z:
        bank = dict(z)
    old, _ = engine.observation(bank, condition)
    clean = old-bank['noise']
    assert cal.digest(clean) == contract['conditions'][condition]['clean_sha']
    assert np.array_equal(cal.generate(contract, info['case']), iq)
    return iq, clean, bank, info


def noise_audit(folder, contract):
    iq, clean, bank, info = case_data(folder, contract)
    noise = iq-clean
    n = iq.shape[1]
    use = abs(np.fft.fftfreq(n, 1/ref.FS)) <= ref.B/2
    nt, st, yt = [np.fft.fft(a, axis=1, norm='ortho') for a in (noise, clean, iq)]
    variance = float(bank['variance'])
    sigma = float(np.mean(abs(yt[:, ~use])**2))
    leakage = float(np.mean(abs(st[:, ~use])**2))
    rnoise = ref.real(nt[:, use].T)
    identity = abs(norm2(noise)-norm2(nt))/norm2(noise)
    real_identity = abs(norm2(rnoise)-2*norm2(nt[:, use]))/norm2(rnoise)
    assert identity < 1e-12 and real_identity < 1e-12
    d = dict(case=info['case']['id'], variance_generation_diagnostic_only=variance,
             sigma2_observed=sigma, sigma_ratio=sigma/variance,
             signal_oob_fraction=leakage/variance,
             noise_oob_ratio=float(np.mean(abs(nt[:, ~use])**2)/variance),
             noise_inband_ratio=float(np.mean(abs(nt[:, use])**2)/variance),
             real_coordinate_variance_ratio=float(np.mean(rnoise*rnoise)/variance),
             fft_relative_energy_error=identity, real_energy_error=real_identity, models={})
    solved = load(folder/'solve.json')
    for model, key in [('LOS', 'los'), ('WIDE', 'final')]:
        a = load(folder/(model+'_risk.json'))
        if not a['applicable']:
            d['models'][model] = dict(applicable=False)
            continue
        l = np.asarray(a['L'])
        response = l@rnoise
        positions = np.asarray(solved[key]['p'])
        truth = bank['p']
        # Evaluation-only label alignment; never changes the observed risk score.
        swap = norm2(positions[::-1]-truth) < norm2(positions-truth)
        error = positions-truth[::-1] if swap else positions-truth
        d['models'][model] = dict(applicable=True, error_solver_order=error,
            linear_noise_response_m=response.reshape(2, 2),
            error_minus_linear_noise_m=error-response.reshape(2, 2),
            local_variance_estimated=float(sigma*norm2(l)/2),
            local_variance_generation_diagnostic_only=float(variance*norm2(l)/2),
            actual_loss=norm2(error)/2, linear_noise_energy=norm2(response)/2)
    return d


def geometry_replay(folder, contract):
    iq, _, bank, info = case_data(folder, contract)
    extracted, _, _ = engine.extract(iq)
    peaks = extracted['CC_ABS']
    lags = np.array([sorted(np.array(peaks[f'{m}_{n}']['lag_s'][:6])*ref.C)
                     for m, n in engine.PAIRS])
    pool = engine.triple_candidates(lags)
    pp, cost, reward, status, nit = engine.geometry_fit(pool, lags)
    ids, best = engine.joint_select(reward)
    solved = load(folder/'solve.json')
    assert np.allclose(pp[ids], solved['initial'], rtol=0, atol=1e-8)
    # Truth is accessed only after the frozen blind frontend is replayed.
    distance = cdist(pp, bank['p'])
    nearest = np.argmin(distance, axis=0)
    near_reward = float(np.maximum(reward[nearest[0]], reward[nearest[1]]).sum())
    result = dict(case=info['case']['id'], candidate_count=len(pp), chosen=ids,
        initial=pp[ids], nearest_truth_candidates=nearest,
        coverage_m=distance[nearest, np.arange(2)], nearest_positions=pp[nearest],
        blind_joint_reward=best, diagnostic_near_truth_pair_reward=near_reward,
        reward_margin=best-near_reward, replay_matches_original=True)
    return result, dict(pool=pool, refined=pp, reward=reward, cost=cost, chosen=ids,
                        lags=lags, status=status, iterations=nit)


def continue_los(folder):
    solved = load(folder/'solve.json')
    with np.load(folder/'input.npz', allow_pickle=False) as z:
        f, y, n, sigma = cal.spectral(z['iq'])
    old = solved['los']
    x = np.asarray(old['x'])
    remaining = 300-int(old['iterations'])
    if old['stationary'] or remaining < 1:
        return dict(skipped=True, reason='already_stationary_or_no_budget'), None
    # Exactly the original LOS least_squares objective/options, with remaining budget.
    evaluate = ref.los_fit.__globals__['evaluate']
    before = risk.state_check(x, f, y, [], n)
    cache = [None, None]
    def update(z):
        if cache[0] is None or not np.array_equal(z, cache[0]):
            cache[:] = [z.copy(), evaluate(z, f, y, None)]
        return cache[1]
    lo = np.r_[np.full(4, -3000/ref.UNIT), np.full(16, -np.inf)]
    started = time.perf_counter()
    opt = least_squares(lambda z: update(z)[0], x, jac=lambda z: update(z)[1],
        bounds=(lo, -lo), x_scale='jac', max_nfev=remaining,
        ftol=1e-12, xtol=1e-12, gtol=1e-10)
    after = risk.state_check(opt.x, f, y, [], n)
    decrease = after['normalized_cost'] <= before['normalized_cost']+1e-12
    assert int(old['iterations'])+opt.nfev <= 300
    accepted = after['stationary'] and after['feasible'] and decrease
    result = dict(skipped=False, before=before, after=after, accepted=bool(accepted),
        original_nfev=old['iterations'], extra_nfev=opt.nfev,
        total_nfev=int(old['iterations'])+opt.nfev, seconds=time.perf_counter()-started,
        position_change_m=float(np.max(np.linalg.norm((opt.x[:4]-x[:4]).reshape(2,2)*ref.UNIT, axis=1))),
        optimizer_message=opt.message, x=opt.x, sigma2_est=sigma,
        interpretation='diagnostic_continuation_not_replacement_formal_result')
    if accepted:
        diagnostic, l = risk.diagnose(opt.x, f, y, [], n, sigma, np.random.default_rng(2026110207))
        result['diagnostic'] = diagnostic
        return result, l
    return result, None


def main():
    start = time.perf_counter()
    contract = load(FORMAL/'contract.json')['contract']
    cal.verify(contract)
    out = HERE/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    files = [Path(__file__), HERE/'执行协议.md', FORMAL/'contract.json', FORMAL/'summary.json']
    files += sorted((FORMAL/'cases').glob('*/*.json'))+sorted((FORMAL/'cases').glob('*/input.npz'))
    hashes = {str(p.resolve(strict=True)): ref.sha(p) for p in files}
    write(out/'preflight.json', dict(hashes=hashes, original_contract=contract,
        scope='32 noise audits; 8 frozen frontend replays; 5 bounded LOS continuations',
        python=sys.version, numpy=np.__version__))
    (out/'source').mkdir()
    for p in [Path(__file__), HERE/'执行协议.md']:
        (out/'source'/p.name).write_bytes(p.read_bytes())
    print('结果目录', out, flush=True)
    noise_rows, geometry_rows, continuation_rows = [], [], []
    for folder in sorted((FORMAL/'cases').iterdir()):
        if not folder.is_dir():
            continue
        d = noise_audit(folder, contract)
        noise_rows.append(d)
        print(folder.name, 'noise ratio', round(d['sigma_ratio'], 5), flush=True)
    write(out/'noise_audit.json', noise_rows)
    for folder in sorted((FORMAL/'cases').glob('HIGH_R*')):
        d, arrays = geometry_replay(folder, contract)
        geometry_rows.append(d)
        np.savez_compressed(out/(folder.name+'_geometry.npz'), **arrays)
        print(folder.name, 'coverage', d['coverage_m'], 'margin', d['reward_margin'], flush=True)
        r, l = continue_los(folder)
        r['case'] = folder.name
        continuation_rows.append(r)
        write(out/(folder.name+'_continuation.json'), r)
        if l is not None:
            np.savez_compressed(out/(folder.name+'_continued_influence.npz'), L=l)
        print('continued', folder.name, r.get('accepted'), r.get('position_change_m'), flush=True)
    write(out/'geometry_replay.json', geometry_rows)
    write(out/'continuations.json', continuation_rows)
    assert all(ref.sha(p)==v for p,v in hashes.items())
    cal.verify(contract)
    write(out/'result.json', dict(noise=noise_rows, geometry=geometry_rows,
        continuations=continuation_rows, seconds=time.perf_counter()-start,
        identity_unchanged=True, formal_results_replaced=False, selector_executed=False))
    write(HERE/'latest.json', dict(path=str(out)))
    print('完成', time.perf_counter()-start, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
