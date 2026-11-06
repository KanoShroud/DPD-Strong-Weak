"""Observation-derived fixed topology local sensitivity, not blind performance."""
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
FORMAL = BASE/'risk_calibration_20261102/formal_20261102_181736_622970'
sys.path.insert(0, str(BASE/'unknown_order_20261030'))
import reference as ref  # noqa: E402
from reference_feasible import fit  # noqa: E402

spec = importlib.util.spec_from_file_location('support_frozen_risk', BASE/'risk_reference_pilot_20261102/pilot.py')
risk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(risk)
READS = {}


def encode(obj):
    def conv(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, allow_nan=False, default=conv)


def read(p):
    p = Path(p).resolve(strict=True)
    READS[str(p)] = ref.sha(p)
    obj = json.loads(p.read_text(encoding='utf-8'))
    if set(obj) == {'payload', 'sha256'}:
        assert hashlib.sha256(encode(obj['payload']).encode('utf-8')).hexdigest() == obj['sha256']
        return obj['payload']
    return obj


def save(p, obj):
    Path(p).write_text(encode(obj), encoding='utf-8')


def spectral(ident):
    p = FORMAL/'cases'/ident/'input.npz'
    READS[str(p)] = ref.sha(p)
    meta = read(p.with_suffix('.json'))
    with np.load(p, allow_pickle=False) as z:
        assert set(z.files) == {'iq'}
        iq = z['iq'].copy()
    assert hashlib.sha256(np.ascontiguousarray(iq).tobytes()).hexdigest() == meta['iq_sha']
    f = np.fft.fftfreq(iq.shape[1], 1/ref.FS)
    use = abs(f) <= ref.B/2
    ft = np.fft.fft(iq, axis=1, norm='ortho')
    return f[use], ft[:, use].T, iq.shape[1], float(np.mean(abs(ft[:, ~use])**2))


def branches(start, end, paths, n):
    changes = []
    for link in sorted(set(map(tuple, paths))):
        ids = [i for i, p in enumerate(paths) if tuple(p) == link]
        before = sorted(ids, key=lambda i: start[22+3*i])
        after = sorted(ids, key=lambda i: end[22+3*i])
        if before != after:
            changes.append(dict(link=link, before=before, after=after))
    delays = end[22::3]
    radius = np.linalg.norm(end[20:].reshape(-1, 3)[:, :2], axis=1)
    return dict(delay_sort_changes=changes,
                max_delay_movement_Btau=float(np.max(abs(delays-start[22::3]))),
                position_box_active=bool(np.any(3000/ref.UNIT-abs(end[:4]) <= 1e-8)),
                reflection_disk_active=int(np.sum(1-radius <= 1e-8)),
                delay_boundary_active=int(np.sum((delays <= 1e-8) | (n/ref.FS*ref.B-delays <= 1e-8))))


def solve_target(out, ident, center, target, scale, repeat, kind):
    x0, f, y0, paths, n, influence = (center[k] for k in ('x', 'f', 'y', 'paths', 'n', 'L'))
    y = y0+scale*(target-y0)
    before = hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest()
    x, info = fit(x0.copy(), f, y, paths, n)
    assert before == hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest()
    observed = (x[:4]-x0[:4])*ref.UNIT
    predicted = influence@ref.real(scale*(target-y0))
    status = risk.state_check(x, f, y, paths, n)
    initial_r, _ = ref.evaluate(x0, f, y, paths)
    correctness = dict(finite=status['finite'], dimensions=status['dimension_ok'],
                       feasible=status['feasible'], iteration_budget=info['iterations'] <= 300,
                       objective_decrease=status['normalized_cost'] <= .5*initial_r@initial_r+1e-10,
                       start_exact=bool(np.array_equal(x0, center['x'])))
    row = dict(id=ident, condition=ident.split('_')[0], repeat=repeat, kind=kind, scale=scale,
               x=x, position=x[:4].reshape(2, 2)*ref.UNIT, fit=info, status=status,
               paths=paths, initial_x=x0, prediction_m=predicted, actual_change_m=observed,
               response_error_m=float(np.linalg.norm(observed-predicted)),
               response_relative_error=float(np.linalg.norm(observed-predicted)/max(np.linalg.norm(predicted), 1e-30)),
               branch=branches(x0, x, paths, n), correctness=correctness, observation_sha=before)
    # Saved before any truth-position read. No performance-based continuation gate.
    save(out/(ident+'.json'), row)
    print(ident, 'stationary', status['stationary'], 'response_rel', row['response_relative_error'],
          'seconds', round(info['seconds'], 2), flush=True)
    return row


def empirical(rows, truth):
    positions = np.array([r['position'] for r in rows])
    # One donor-derived assignment supplied by the evaluator, not per-fit truth sorting.
    e = positions-truth
    mean = e.mean(0)
    mse = float(np.mean(np.sum(e*e, axis=(1, 2))/2))
    mean_term = float(np.sum(mean*mean)/2)
    variance = float(np.mean(np.sum((e-mean)**2, axis=(1, 2))/2))
    assert np.isclose(mse, mean_term+variance)
    return dict(n=len(rows), rmse_m=np.sqrt(mse), mse_m2=mse,
                sample_mean_error=mean, squared_sample_mean_m2=mean_term,
                empirical_centered_mse_divide_n_m2=variance,
                stationary=sum(r['status']['stationary'] for r in rows),
                mean_response_error_m=float(np.mean([r['response_error_m'] for r in rows])),
                max_response_relative_error=max(r['response_relative_error'] for r in rows),
                delay_sort_changes=sum(bool(r['branch']['delay_sort_changes']) for r in rows))


def evaluate(out, contract, centers, rows):
    result = {}
    for cond, c in centers.items():
        p = Path(contract['conditions'][cond]['bank'])
        READS[str(p)] = ref.sha(p)
        with np.load(p, allow_pickle=False) as z:
            truth = z['p'].copy()
        donor_p = c['x'][:4].reshape(2, 2)*ref.UNIT
        if np.sum((donor_p-truth[::-1])**2) < np.sum((donor_p-truth)**2):
            truth = truth[::-1]
        rr = [r for r in rows if r['condition'] == cond and r['kind'] == 'receiver']
        old = []
        for r in rr:
            previous = read(FORMAL/'cases'/f"{cond}_R{r['repeat']:02d}"/'solve.json')
            pp = np.asarray(previous['final']['p'])
            if np.sum((pp-donor_p[::-1])**2) < np.sum((pp-donor_p)**2):
                pp = pp[::-1]
            old.append(dict(id=r['id'], position=pp, loss_m2=float(np.sum((pp-truth)**2)/2),
                            order=len(previous['paths']), paths=previous['paths'],
                            note='Different blind start and support; descriptive only, not matched causal arm.'))
        result[cond] = dict(fixed=empirical(rr, truth), old_adaptive_descriptive=old,
                            old_adaptive_rmse_m=float(np.sqrt(np.mean([a['loss_m2'] for a in old]))),
                            donor_not_scored=True, common_label_truth=truth)
    save(out/'evaluation_only.json', result)
    return result


def main():
    begin = time.perf_counter()
    out = HERE/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    assert not out.exists() and out.is_relative_to(HERE)
    out.mkdir()
    contract = read(FORMAL/'contract.json')['contract']
    expected = contract['hashes']
    assert all(ref.sha(p) == h for p, h in expected.items())
    for cond in ('L2', 'L3'):
        for rep in range(8):
            for name in ('input.npz', 'input.json', 'solve.json'):
                p = FORMAL/'cases'/f'{cond}_R{rep:02d}'/name
                READS[str(p)] = ref.sha(p)
    for mod in tuple(sys.modules.values()):
        filename = getattr(mod, '__file__', None)
        if filename and Path(filename).suffix == '.py' and Path(filename).resolve().is_relative_to(BASE):
            READS[str(Path(filename).resolve())] = ref.sha(filename)
    for suffix in ('*.md', '*.py'):
        for p in HERE.glob(suffix):
            READS[str(p)] = ref.sha(p)
    # Snapshot and validate provenance before numerical operations.
    hashes = {**expected, **READS}
    for p, h in hashes.items():
        assert Path(p).resolve(strict=True).is_relative_to(BASE.parents[2])
        assert ref.sha(p) == h
        if Path(p).suffix in ('.py', '.md'):
            dest = out/'source'/(h+Path(p).suffix)
            dest.parent.mkdir(exist_ok=True)
            dest.write_bytes(Path(p).read_bytes())
    save(out/'preflight.json', dict(hashes=hashes, python=sys.version, numpy=np.__version__,
                                  scipy=scipy.__version__, platform=platform.platform(), threads=2,
                                  donor='R00', receivers=list(range(1, 8)), scales=[.1, .01],
                                  stage='fixed_group_only_not_strict_two_arm'))
    print('输出', out, flush=True)
    centers, rows, stopped = {}, [], None
    for idx, cond in enumerate(('L2', 'L3')):
        donor = read(FORMAL/'cases'/f'{cond}_R00'/'solve.json')
        f, y, n, sigma2 = spectral(f'{cond}_R00')
        x0 = np.asarray(donor['final']['x'])
        paths = donor['paths']
        x, info = fit(x0.copy(), f, y, paths, n)
        d, influence = risk.diagnose(x, f, y, paths, n, sigma2, np.random.default_rng(20261103010+idx))
        save(out/(cond+'_donor.json'), dict(source=f'{cond}_R00', fit=info, diagnostic=d,
                                           x=x, paths=paths, drift_m=float(np.linalg.norm((x[:4]-x0[:4])*ref.UNIT))))
        if influence is None:
            stopped = 'donor_inapplicable:'+cond
            break
        np.savez_compressed(out/(cond+'_influence.npz'), L=influence)
        centers[cond] = dict(x=x, paths=paths, f=f, y=y, n=n, L=influence)
        ff, yy, nn, _ = spectral(cond+'_R01')
        assert np.array_equal(ff, f) and nn == n
        row = solve_target(out, cond+'_R01', centers[cond], yy, 1., 1, 'receiver')
        rows.append(row)
        if not all(row['correctness'].values()) or not row['status']['stationary']:
            stopped = 'short_test_numerical_failure:'+cond
            break
    estimate = time.perf_counter()-begin+20*np.mean([r['fit']['seconds'] for r in rows]) if rows else None
    save(out/'short_test.json', dict(case_ids=[r['id'] for r in rows], stopped=stopped,
                                    projected_total_seconds=estimate, truth_not_loaded=True))
    if not stopped and estimate > 3600:
        stopped = 'projected_above_one_hour_handoff_required'
    if not stopped:
        for cond, c in centers.items():
            for rep in range(2, 8):
                _, yy, _, _ = spectral(f'{cond}_R{rep:02d}')
                rows.append(solve_target(out, f'{cond}_R{rep:02d}', c, yy, 1., rep, 'receiver'))
            _, yy, _, _ = spectral(cond+'_R01')
            for scale in (.1, .01):
                pair = []
                for sign in (1, -1):
                    ident = cond+f'_small_{scale}_{sign:+d}'
                    r = solve_target(out, ident, c, yy, sign*scale, 1, 'local_perturbation')
                    pair.append(r)
                    rows.append(r)
                observed = (np.asarray(pair[0]['position'])-np.asarray(pair[1]['position'])).ravel()/2
                expected_response = np.asarray(pair[0]['prediction_m'])
                save(out/(cond+f'_centered_{scale}.json'), dict(scale=scale, actual=observed,
                      predicted=expected_response, error_m=float(np.linalg.norm(observed-expected_response)),
                      relative_error=float(np.linalg.norm(observed-expected_response)/max(np.linalg.norm(expected_response), 1e-30)),
                      both_stationary=all(r['status']['stationary'] for r in pair)))
    save(out/'decisions_before_truth.json', dict(rows=rows, stopped=stopped))
    evaluation = evaluate(out, contract, centers, rows) if len(centers) == 2 and len(rows) >= 2 else None
    allhashes = {**hashes, **READS}
    unchanged = all(ref.sha(p) == h for p, h in allhashes.items())
    save(out/'postflight.json', dict(hashes=allhashes, unchanged=unchanged, count=len(allhashes)))
    save(out/'result.json', dict(stopped=stopped, seconds=time.perf_counter()-begin,
                               local_fit_count=len(centers)+len(rows), receiver_refits=sum(r['kind'] == 'receiver' for r in rows),
                               rows=rows, evaluation=evaluation, unchanged=unchanged,
                               strict_adaptive_arm_run=False, new_blind_localizations=0))
    assert unchanged
    save(HERE/'latest.json', dict(path=str(out)))
    print('完成', stopped, '秒', time.perf_counter()-begin, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
