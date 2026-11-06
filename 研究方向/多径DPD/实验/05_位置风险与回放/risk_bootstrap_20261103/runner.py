"""Finite-noise fitted-world calibration. Not a deployed risk selector."""
import argparse
from datetime import datetime
import importlib.util
from pathlib import Path
import shutil
import sys

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
FORMAL = BASE/'risk_calibration_20261102/formal_20261102_181736_622970'
spec = importlib.util.spec_from_file_location('bootstrap_frozen_cal', FORMAL.parent/'runner.py')
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)
from storage import load, save, write, lock, recover, stage, cost  # noqa: E402

CONDITIONS = ('L0', 'L2', 'L3', 'HIGH')
REPLICATES = 16
CAPACITY = 64


def loss(positions, focus):
    """Unordered two-source mean squared Euclidean error, in m^2."""
    a, b = np.asarray(positions), np.asarray(focus)
    return float(min(np.sum((a-b)**2), np.sum((a-b[::-1])**2))/2)


def states(solved):
    rows = [solved['los']]
    for event in solved['history']:
        if 'fit' in event and event['decrease']:
            rows.append(event['fit'])
    if not np.array_equal(rows[-1]['x'], solved['final']['x']):
        raise AssertionError('Accepted prefix does not end at frozen output')
    if len(rows)-1 != len(solved['paths']):
        raise AssertionError('Accepted path count differs')
    return rows


def curve(solved, focus):
    rr = states(solved)
    return np.array([loss(rr[min(k, len(rr)-1)]['p'], focus) for k in range(CAPACITY+1)])


def fitted_mean(x, paths, iq):
    f, y, n, sigma2 = cal.spectral(iq)
    h, _, _ = cal.ref.response(np.asarray(x), f, paths)
    s = np.einsum('fqm,fm->fq', np.linalg.pinv(h, rcond=1e-12), y)
    mu = np.einsum('fmq,fq->fm', h, s)
    allf = np.fft.fftfreq(n, 1/cal.ref.FS)
    use = abs(allf) <= cal.ref.B/2
    z = np.zeros((5, n), complex)
    z[:, use] = mu.T
    mean = np.fft.ifft(z, axis=1, norm='ortho')
    rebuilt = np.fft.fft(mean, axis=1, norm='ortho')
    assert np.allclose(rebuilt[:, use].T, mu, rtol=1e-12, atol=1e-12)
    return mean, sigma2, dict(inband_relative_fit_residual=float(np.linalg.norm(y-mu)/np.linalg.norm(y)),
                             fitted_mean_energy=float(np.sum(abs(mean)**2)),
                             band_bins=int(use.sum()), n=n, sigma2_complex=sigma2,
                             fft='ortho', outband_fitted_mean='zero_not_true_finite_record')


def generate(mean, sigma2, condition_index, repeat):
    rng = np.random.default_rng(np.random.SeedSequence([2026110341, condition_index, repeat]))
    noise = np.sqrt(sigma2/2)*(rng.normal(size=mean.shape)+1j*rng.normal(size=mean.shape))
    return mean+noise


def solver_gate(solved):
    if solved['status'] != 'returned':
        return False
    x = np.asarray(solved['final']['x'])
    closure = np.linalg.norm(x-cal.ref.project(x, cal.engine.N))
    return bool(solved['stop'] == 'no_detected_increment'
                and all(solved['checks'].values()) and solved['final']['stationary']
                and np.isfinite(x).all() and closure < 1e-7)


def make_contract():
    old = load(FORMAL/'contract.json')['contract']
    cal.verify(old)
    hashes = dict(old['hashes'])
    files = [FORMAL/'contract.json', HERE/'runner.py', HERE/'test_bootstrap.py', HERE/'执行协议.md']
    files += [cal.PROJECT/'运行入口/E2E/G11/G11有限噪声风险核验一键运行.py']
    for cond in CONDITIONS:
        for rep in range(8):
            folder = FORMAL/'cases'/f'{cond}_R{rep:02d}'
            files.append(folder/'solve.json')
            if rep == 0:
                files += [folder/n for n in ('input.npz', 'input.json', 'WIDE_risk.json')]
    for mod in tuple(sys.modules.values()):
        fn = getattr(mod, '__file__', None)
        if fn and Path(fn).suffix == '.py' and Path(fn).resolve().is_relative_to(BASE):
            files.append(Path(fn))
    hashes.update({str(p.resolve(strict=True)): cal.ref.sha(p) for p in files})
    return dict(hashes=hashes, original=old, replicates=REPLICATES, capacity=CAPACITY,
                donor='R00', validation_receivers=list(range(1, 8)), seed=2026110341,
                policy='accepted_prefix_cap_0_to_64_early_stop_is_original_terminal',
                threads=2, no_actual_selector=True)


def prepare(out, contract):
    result = {}
    for cond in CONDITIONS:
        folder = FORMAL/'cases'/(cond+'_R00')
        solved, wide = load(folder/'solve.json'), load(folder/'WIDE_risk.json')
        eligible = solved['stop'] == 'no_detected_increment' and wide['applicable']
        rec = dict(eligible=eligible, stop=solved['stop'], wide_applicable=wide['applicable'],
                   original_order=len(solved['paths']))
        if eligible:
            with np.load(folder/'input.npz', allow_pickle=False) as z:
                assert set(z.files) == {'iq'}
                iq = z['iq'].copy()
            assert cal.digest(iq) == load(folder/'input.json')['iq_sha'] == solved['iq_sha']
            mean, sigma2, diagnostic = fitted_mean(solved['final']['x'], solved['paths'], iq)
            np.savez_compressed(out/(cond+'_fitted_world.npz'), mean=mean, sigma2=sigma2,
                                focus=np.asarray(solved['final']['p']))
            rec.update(diagnostic=diagnostic, fitted_world_sha=cal.ref.sha(out/(cond+'_fitted_world.npz')))
        result[cond] = rec
    return result


def replicate(out, cond, rep):
    folder = out/'replicates'/f'{cond}_B{rep:02d}'
    folder.mkdir(parents=True, exist_ok=True)
    with np.load(out/(cond+'_fitted_world.npz'), allow_pickle=False) as z:
        mean, sigma2, focus = z['mean'], float(z['sigma2']), z['focus']
    iq = generate(mean, sigma2, CONDITIONS.index(cond), rep)
    # This file is the only statistical input passed into the frozen estimator.
    np.savez_compressed(folder/'input.npz', iq=iq)
    solved = cal.solve(iq, folder, f'{cond}_B{rep:02d}')
    save(folder/'solve.json', solved)
    okay = solver_gate(solved)
    values = curve(solved, focus) if solved['status'] == 'returned' else None
    return dict(condition=cond, repeat=rep, numerical_gate=okay, status=solved['status'],
                stop=solved.get('stop'), order=len(solved.get('paths', [])),
                loss_curve_m2=values, seconds=solved.get('seconds'),
                input_sha=cal.digest(iq), solve_file=str(folder/'solve.json'),
                solve_sha=cal.ref.sha(folder/'solve.json'),
                noisy_mean_not_used=True, original_support_not_solver_input=True)


def summarize(out, donors, stopped):
    result = dict(stopped=stopped, donors=donors, conditions={}, cost=cost(out),
                  actual_selector_run=False, independent_validation=False)
    for cond in CONDITIONS:
        rows = [load(p) for p in sorted((out/'replicates').glob(f'{cond}_B*/result.json'))]
        if not rows:
            continue
        valid = [r for r in rows if r['loss_curve_m2'] is not None]
        curves = np.asarray([r['loss_curve_m2'] for r in valid]).reshape(-1, CAPACITY+1)
        delta = curves-curves[:, -1:]
        result['conditions'][cond] = dict(n_requested=len(rows), n_finite=len(valid),
              numerical_pass=sum(r['numerical_gate'] for r in rows),
              stops=[r['stop'] for r in rows], orders=[r['order'] for r in rows],
              risk_curve_m2=curves.mean(0) if len(valid) else None,
              paired_delta_m2=delta.mean(0) if len(valid) else None,
              paired_mc_se_m2=delta.std(0, ddof=1)/np.sqrt(len(valid)) if len(valid)>1 else None,
              all_rows_present=len(valid)==len(rows),
              warning='Monte Carlo SE only; no simultaneous confidence or selected-model benefit claim')
    save(out/'prediction_frozen.json', result)
    return result


def evaluate_saved(out, prediction, contract):
    # Only after all prediction files are frozen. No truth enters generation or solve.
    assert (out/'prediction_frozen.json').is_file()
    evaluation = {}
    for cond in CONDITIONS:
        with np.load(contract['original']['conditions'][cond]['bank'], allow_pickle=False) as z:
            truth = z['p'].copy()
        curves = np.asarray([curve(load(FORMAL/'cases'/f'{cond}_R{r:02d}'/'solve.json'), truth)
                             for r in range(1, 8)])
        delta = curves-curves[:, -1:]
        row = dict(n=7, risk_curve_m2=curves.mean(0), paired_delta_m2=delta.mean(0),
                   descriptive_paired_se_m2=delta.std(0, ddof=1)/np.sqrt(7),
                   individual_loss_curves_m2=curves, reference_R00_excluded=True)
        pred = prediction['conditions'].get(cond)
        if pred and pred['all_rows_present']:
            # A fixed cap recommended by this one donor, not a per-record selector.
            k = int(np.argmin(pred['risk_curve_m2']))
            row.update(plugin_minimum_cap_diagnostic=k,
                       fixed_cap_receiver_rmse_m=float(np.sqrt(curves[:, k].mean())),
                       receiver_wide_rmse_m=float(np.sqrt(curves[:, -1].mean())),
                       fixed_cap_receiver_delta_m2=float(delta[:, k].mean()),
                       transfer='conditional old-data diagnostic; not independent algorithm validation')
        evaluation[cond] = row
    save(out/'evaluation_only.json', evaluation)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--continue-long', action='store_true')
    args = parser.parse_args()
    from test_bootstrap import run_checks
    checks = run_checks(sys.modules[__name__])
    if not all(checks.values()):
        raise AssertionError(checks)
    if args.resume:
        out = args.resume.resolve(strict=True)
        assert out.parent == HERE and out.name.startswith('run_')
        contract = load(out/'contract.json')
    else:
        contract = make_contract()
        out = HERE/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        out.mkdir(exist_ok=False)
        save(out/'contract.json', contract)
        save(out/'unit_checks.json', checks)
        for p, h in contract['hashes'].items():
            if Path(p).suffix in ('.py', '.md'):
                dest = out/'source'/(h+Path(p).suffix)
                dest.parent.mkdir(exist_ok=True)
                dest.write_bytes(Path(p).read_bytes())
        write(HERE/'latest.json', dict(path=str(out)))
    cal.verify(contract['original'])
    cal.check_paths(contract['hashes'])
    if shutil.disk_usage(HERE).free < 8*1024**3:
        raise RuntimeError('Less than 8 GiB free; do not reduce evidence silently')
    print('输出目录：', out, flush=True)
    with lock(out):
        recover(out)
        donors, _ = stage(out, 'fitted_worlds', out/'donors.json', lambda: prepare(out, contract))
        eligible = [c for c in CONDITIONS if donors[c]['eligible']]
        for c in eligible:
            assert cal.ref.sha(out/(c+'_fitted_world.npz')) == donors[c]['fitted_world_sha']
        stopped, rows = None, []
        for rep in range(REPLICATES):
            for cond in eligible:
                target = out/'replicates'/f'{cond}_B{rep:02d}'/'result.json'
                row, fresh = stage(out, f'{cond}_B{rep:02d}', target, lambda c=cond,r=rep: replicate(out,c,r))
                rows.append(row)
                print(cond, rep, '新增' if fresh else '复用', row['stop'], 'K', row['order'],
                      '累计秒', round(cost(out)['seconds'], 1), flush=True)
                if not row['numerical_gate']:
                    stopped = 'numerical_or_frontend_gate_failed'
                    break
                if cost(out)['seconds'] > 3600 and not args.continue_long:
                    stopped = 'one_hour_budget_handoff'
                    break
            if stopped:
                break
            if rep == 1:
                projected = max(r['seconds'] for r in rows)*REPLICATES*len(eligible)
                save(out/'pilot_gate.json', dict(predicted_full_seconds_conservative=projected,
                      numerical_pass=True, all_eligible_donors=eligible, first_two_per_donor=True))
                if projected > 3600 and not args.continue_long:
                    stopped = 'projected_above_one_hour_handoff'
                    break
        prediction = summarize(out, donors, stopped)
        if stopped is None and len(rows) == len(eligible)*REPLICATES:
            evaluate_saved(out, prediction, contract)
        else:
            save(out/'evaluation_deferred.json', dict(reason=stopped, physical_truth_not_read=True))
        cal.check_paths(contract['hashes'])
        save(out/'postflight.json', dict(all_hashes_unchanged=True, count=len(contract['hashes']),
                                        stopped=stopped, seconds=cost(out)['seconds']))
        print('完成本阶段：', stopped or 'finite_pilot_complete', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
