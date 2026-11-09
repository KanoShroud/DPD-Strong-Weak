from 研究方向.多径DPD.代码.mp_dpd import code_path
# ruff: noqa: F401
# Preserve imported globals for legacy function cloning and reexports.
"""Frozen-IQ initialization counterfactuals; not a new deployed estimator."""
import argparse
from datetime import datetime
import importlib.util
from pathlib import Path
import shutil
import types

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
FORMAL = BASE/'risk_calibration_20261102/formal_20261102_181736_622970'
GEOMETRY = BASE/'risk_followup_20261102/run_20261102_211453_060232'
spec = importlib.util.spec_from_file_location('attribution_cal', code_path('io/spectral_pipeline.py'))
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)
ref, engine = cal.ref, cal.engine
from 研究方向.多径DPD.代码.mp_dpd.io.storage import load, save, read, write, stage, lock, recover, cost  # noqa: E402

FAILURES = ('HIGH_R00', 'HIGH_R05')


def loss(p, truth):
    p, truth = np.asarray(p), np.asarray(truth)
    return float(min(np.sum((p-truth)**2), np.sum((p-truth[::-1])**2))/2)


def pair_distance(p, q):
    return min(np.linalg.norm(p-q, axis=1).max(),
               np.linalg.norm(p-q[::-1], axis=1).max())


def alternate_pair(points, reward):
    """Highest original reward outside the first pair's c/B neighbourhood."""
    chosen, best = engine.joint_select(reward)
    ii, jj = np.triu_indices(len(points), k=1)
    values = np.maximum(reward[ii], reward[jj]).sum(axis=1)
    order = np.argsort(-values, kind='stable')
    native = points[chosen]
    for rank, k in enumerate(order):
        ids = [int(ii[k]), int(jj[k])]
        if pair_distance(points[ids], native) > ref.UNIT:
            return dict(first=chosen, first_reward=best, second=ids,
                        second_reward=float(values[k]), raw_rank=rank+1,
                        skipped_near_duplicates=rank, scale_m=ref.UNIT)
    raise ValueError('No distinct pair; keep this as a failed coverage gate')


def iq_for(tag):
    folder = FORMAL/'cases'/tag
    info = load(folder/'input.json')
    with np.load(folder/'input.npz', allow_pickle=False) as z:
        assert set(z.files) == {'iq'}
        iq = z['iq']
    assert cal.digest(iq) == info['iq_sha']
    assert iq.shape == (5, 8192)
    return iq


def truth_for(tag, contract):
    # Evaluation only. Not called by candidate selection or solver wrappers.
    cond = tag.split('_')[0]
    with np.load(contract['conditions'][cond]['bank'], allow_pickle=False) as z:
        return z['p']


def frontend(iq):
    extracted, _, _ = engine.extract(iq)
    peaks = extracted['CC_ABS']
    lags = np.array([sorted(np.asarray(peaks[f'{m}_{n}']['lag_s'][:6])*ref.C)
                     for m, n in engine.PAIRS])
    pool = engine.triple_candidates(lags)
    pp, cc, reward, status, iterations = engine.geometry_fit(pool, lags)
    return dict(pool=pool, refined=pp, cost=cc, reward=reward, lags=lags,
                status=status, iterations=iterations)


def los_from_initial(iq, initial):
    initial = np.asarray(initial, dtype=float)
    f, y, n, _ = cal.spectral(iq)
    gains = np.column_stack([engine.rank_one(y, f, p)[0] for p in initial])
    x0 = np.r_[np.asarray(initial).ravel()/ref.UNIT,
               gains[1:].real.ravel(), gains[1:].imag.ravel()]
    _, state = engine.los_fit(x0, f, y, n)
    return state


def states(solved):
    answer = [solved['los']]
    answer += [v['fit'] for v in solved['history'] if 'fit' in v and v['decrease']]
    assert len(answer) == len(solved['paths'])+1
    assert np.array_equal(answer[-1]['x'], solved['final']['x'])
    return answer


def objective_check(state, paths, iq):
    x = np.asarray(state['x'])
    f, y, n, _ = cal.spectral(iq)
    h, _, _ = ref.response(x, f, paths)
    s = np.einsum('fqm,fm->fq', np.linalg.pinv(h, rcond=1e-12), y)
    residual = y-np.einsum('fmq,fq->fm', h, s)
    direct = float(np.sum(abs(residual)**2)/np.sum(abs(y)**2))
    r, j = ref.evaluate(x, f, y, paths)
    pg = float(np.max(abs(x-ref.project(x-j.T@r, n))))
    closure = float(np.linalg.norm(x-ref.project(x, n)))
    assert abs(direct-state['cost']) < 1e-11
    assert abs(direct-.5*r@r) < 1e-12
    assert closure < 1e-7
    return dict(cost=direct, projected_gradient=pg, feasible=True,
                stationary=pg <= 1e-7)


def prepare(out, contract):
    records = []
    for folder in sorted((FORMAL/'cases').iterdir()):
        if not folder.is_dir():
            continue
        old = load(folder/'solve.json')
        records.append(dict(case=folder.name, K=len(old['paths']), stop=old['stop'],
            cost=old['final']['cost'], stationary=old['final']['stationary'],
            original_seconds=old['seconds'],
            rmse_m=np.sqrt(loss(old['final']['p'], truth_for(folder.name, contract)))))
    assert len(records) == 32
    write(out/'frozen_32_summary.json', records)
    decisions = []
    for rep in range(8):
        tag = f'HIGH_R{rep:02d}'
        iq = iq_for(tag)
        a = frontend(iq)
        with np.load(GEOMETRY/f'{tag}_geometry.npz', allow_pickle=False) as z:
            assert np.allclose(a['refined'], z['refined'], rtol=0, atol=1e-7)
            assert np.allclose(a['reward'], z['reward'], rtol=0, atol=1e-9)
        d = alternate_pair(a['refined'], a['reward'])
        old = load(FORMAL/'cases'/tag/'solve.json')
        assert np.allclose(a['refined'][d['first']], old['initial'], rtol=0, atol=1e-7)
        d.update(case=tag, initial=a['refined'][d['second']],
                 iq_sha=cal.digest(iq), truth_used=False)
        decisions.append(d)
        np.savez_compressed(out/f'{tag}_geometry.npz', **a)
    save(out/'observation_decisions_before_truth.json', decisions)
    evaluated = []
    for d in decisions:
        truth = truth_for(d['case'], contract)
        with np.load(out/f"{d['case']}_geometry.npz", allow_pickle=False) as z:
            points = z['refined']
        evaluated.append(dict(case=d['case'], first_rmse=np.sqrt(loss(points[d['first']], truth)),
                              second_rmse=np.sqrt(loss(d['initial'], truth))))
    write(out/'candidate_evaluation_only.json', evaluated)
    return dict(cases=32, decisions=decisions, all_checks_pass=True)


def short_check(out):
    tag = 'HIGH_R05'
    iq = iq_for(tag)
    old = load(FORMAL/'cases'/tag/'solve.json')
    check = los_from_initial(iq, old['initial'])
    assert abs(check['cost']-old['los']['cost']) < 1e-10
    assert np.max(abs(np.asarray(check['p'])-old['los']['p'])) < 1e-4
    f, y, n, sigma = cal.spectral(iq)
    use = abs(np.fft.fftfreq(n, 1/ref.FS)) <= ref.B/2
    pick, _, meta = engine.search(np.asarray(check['x']), [], f, y, n, use, sigma, 1)
    original = old['history'][0]['pick']
    assert all(meta['checks'].values())
    assert [pick[k] for k in ('m', 'q', 'sample_delay')] == [original[k] for k in ('m', 'q', 'sample_delay')]
    assert np.allclose(pick['rho'], original['rho'], rtol=1e-5, atol=1e-8)
    assert engine.fit is cal.frozen_fit
    assert pair_distance(np.array([[1., 2.], [3., 4.]]), np.array([[3., 4.], [1., 2.]])) == 0
    # Two separate derivations of the normalized complex objective.
    obj = objective_check(check, [], iq)
    return dict(all_checks_pass=True, replay_case=tag, loss=check['cost'],
                first_pick=pick, cost_identity=obj, fit='reference_feasible.fit')


def alternate_solve(out, d):
    tag = d['case']
    iq = iq_for(tag)
    with np.load(out/f'{tag}_geometry.npz', allow_pickle=False) as z:
        expected = z['reward']
    calls = []
    def fixed_select(reward):
        assert np.allclose(reward, expected, rtol=0, atol=1e-9)
        native, value = engine.joint_select(reward)
        assert native == d['first'] and abs(value-d['first_reward']) < 1e-9
        calls.append(1)
        return d['second'], d['second_reward']
    scope = {**engine.solve_iq.__globals__, 'joint_select': fixed_select,
             'fit': cal.frozen_fit}
    solver = types.FunctionType(engine.solve_iq.__code__, scope, name='diagnostic_alternate_initial')
    assert solver.__code__ is engine.solve_iq.__code__
    assert scope['search'] is engine.search and scope['los_fit'] is engine.los_fit
    assert scope['fit'] is cal.frozen_fit
    attempt = out/(tag+'_attempt_'+datetime.now().strftime('%H%M%S_%f'))
    attempt.mkdir(exist_ok=False)
    print('开始完整替代分支', tag, flush=True)
    solved = solver(iq, attempt, tag)
    assert len(calls) == 1 and cal.digest(iq) == d['iq_sha']
    assert np.allclose(solved['initial'], d['initial'], rtol=0, atol=1e-7)
    solved.update(status='returned', attempt=str(attempt),
                  role='observation_selected_counterfactual_not_deployed_algorithm',
                  truth_used=False, iq_sha=d['iq_sha'])
    print('完成替代分支', tag, solved['stop'], 'K', len(solved['paths']),
          'J', solved['final']['cost'], flush=True)
    return solved


def comparison(tag, old, new, iq, contract):
    a, b = states(old), states(new)
    truth = truth_for(tag, contract)
    costs = []
    for k in range(min(len(a), len(b))):
        costs.append(dict(K=k, original_cost=a[k]['cost'], alternative_cost=b[k]['cost'],
            original_rmse=np.sqrt(loss(a[k]['p'], truth)), alternative_rmse=np.sqrt(loss(b[k]['p'], truth)),
            original_stationary=a[k]['stationary'], alternative_stationary=b[k]['stationary']))
    before = objective_check(old['final'], old['paths'], iq)
    after = objective_check(new['final'], new['paths'], iq)
    embedded = None
    if len(new['paths']) <= len(old['paths']):
        paths = list(new['paths'])
        x = list(new['final']['x'])
        for link in old['paths'][len(paths):]:
            paths.append(link)
            x.extend([0., 0., 1.])
        state = dict(x=x, cost=new['final']['cost'])
        embedded = objective_check(state, paths, iq)
    gate = bool(new['stop'] == 'no_detected_increment' and after['stationary']
                and all(new['checks'].values()) and after['cost'] < before['cost']-1e-10)
    return dict(case=tag, original_K=len(old['paths']), alternative_K=len(new['paths']),
        original_stop=old['stop'], alternative_stop=new['stop'],
        original=before, alternative=after, zero_amplitude_embedding=embedded,
        original_rmse_m=np.sqrt(loss(old['final']['p'], truth)),
        alternative_rmse_m=np.sqrt(loss(new['final']['p'], truth)),
        common_order_curve=costs, stage_B_gate=gate,
        interpretation='a_better_feasible_explanation_not_global_identifiability_or_new_algorithm')


def los_screen(out, decisions, contract):
    blind, fitted = [], []
    for d in decisions:
        tag = d['case']
        second, _ = stage(out, tag+'_alternative_LOS', out/(tag+'_alternative_LOS.json'),
                           lambda d=d: los_from_initial(iq_for(d['case']), d['initial']))
        first = load(FORMAL/'cases'/tag/'solve.json')['los']
        # Pure observation rule. No location-error-dependent acceptance.
        selected = 2 if second['cost'] < first['cost'] else 1
        blind.append(dict(case=tag, selected=selected, first_cost=first['cost'],
                          second_cost=second['cost'], first_stationary=first['stationary'],
                          second_stationary=second['stationary']))
        fitted.append((first, second))
    save(out/'LOS_screen_decisions_before_truth.json', blind)
    evaluated = []
    for d, (first, second) in zip(blind, fitted):
        truth = truth_for(d['case'], contract)
        evaluated.append(dict(**d, first_rmse_m=np.sqrt(loss(first['p'], truth)),
             second_rmse_m=np.sqrt(loss(second['p'], truth)),
             selected_rmse_m=np.sqrt(loss((first if d['selected']==1 else second)['p'], truth))))
    return dict(rows=evaluated, deployable_selector=False,
                reason='development_diagnostic_only_and_original_LOS_stopping_reported')


# Historical main retained only in the evidence archive.


# Historical batch entry disabled; use the public mp_dpd API.
