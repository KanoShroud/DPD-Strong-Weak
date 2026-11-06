"""Evaluation-only exact decomposition of frozen risk replay; no solver import."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
REPLAY = BASE / 'risk_replay_20261102/formal_20261102_223046_550434'
FORMAL = BASE / 'risk_calibration_20261102/formal_20261102_181736_622970'
READS = {}
CHECKS = []


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    path = Path(path).resolve(strict=True)
    READS[str(path)] = sha(path)
    obj = json.loads(path.read_text(encoding='utf-8'))
    if set(obj) == {'payload', 'sha256'}:
        check(hashlib.sha256(encode(obj['payload']).encode()).hexdigest() == obj['sha256'],
              'payload:' + str(path))
        return obj['payload']
    return obj


def check(ok, name):
    if not ok:
        raise AssertionError(name)
    CHECKS.append(name)


def close(x, y, name, atol=2e-9):
    check(np.allclose(x, y, rtol=1e-9, atol=atol), name)


def write(path, value):
    path.write_text(encode(value), encoding='utf-8')


def decompose(rows):
    d = np.asarray([r['d'] for r in rows])
    e = np.asarray([r['e_wide'] for r in rows])
    empirical = np.array([r['loss_difference'] for r in rows])
    displacement = np.mean(np.sum(d*d, axis=1)/2)
    mean_cross = float(np.mean(d, axis=0) @ np.mean(e, axis=0))
    covariance = float(np.mean(np.sum((d-d.mean(0))*(e-e.mean(0)), axis=1)))
    close(empirical.mean(), displacement+mean_cross+covariance, 'group exact decomposition')
    scored = [r for r in rows if r['risk_difference'] is not None]
    result = dict(n=len(rows), fallback=sum(r['fallback'] for r in rows),
                  displacement=displacement, sample_mean_cross=mean_cross,
                  empirical_covariance_n_trace=covariance,
                  empirical_loss_difference=empirical.mean(),
                  mean_d=d.mean(0), mean_e_wide=e.mean(0),
                  wide_sample_mean_squared_norm=np.sum(e.mean(0)**2)/2,
                  wide_empirical_centered_mse_n=np.mean(np.sum((e-e.mean(0))**2, axis=1)/2),
                  mse_wide=np.mean([r['loss_wide'] for r in rows]),
                  mse_selected_or_fallback=np.mean([r['loss_selected'] for r in rows]))
    if scored:
        risk = np.array([r['risk_difference'] for r in scored])
        error = np.array([r['loss_difference']-r['risk_difference'] for r in scored])
        result.update(scored_n=len(scored), mean_predicted_difference=risk.mean(),
                      scored_mean_empirical_difference=np.mean([r['loss_difference'] for r in scored]),
                      scored_mean_displacement=np.mean([r['displacement'] for r in scored]),
                      mean_plugin_cross=np.mean([r['plugin_cross'] for r in scored]),
                      mean_realized_cross=np.mean([r['actual_cross'] for r in scored]),
                      mean_prediction_error=error.mean(), prediction_error_rms=np.sqrt(np.mean(error**2)),
                      mean_absolute_prediction_error=np.mean(np.abs(error)),
                      mean_predicted_improvement=-risk.mean(),
                      absolute_error_exceeds_absolute_risk_gap=int(np.sum(np.abs(error)>np.abs(risk))),
                      note='Prediction errors concern realized loss; not proof of expectation bias.')
    return result


def main():
    start = time.perf_counter()
    out = HERE / ('analysis_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    check(not out.exists(), 'new output')
    check(REPLAY.resolve(strict=True).is_relative_to(BASE), 'read root')
    check(out.is_relative_to(HERE) and not out.is_relative_to(REPLAY), 'write separation')
    out.mkdir()
    contract = read(REPLAY/'contract.json')
    expected = contract['hashes']
    for path, digest in expected.items():
        check(sha(path) == digest, 'frozen pre:' + path)
    pre = {**expected}
    summary = read(REPLAY/'summary.json')
    check(summary['complete'] and summary['completed'] == 32, 'formal complete')
    rows, states, label_warnings, support_groups = [], [], [], {}
    for inv in contract['inventory']:
        ident = inv['case']
        cond = ident.split('_')[0]
        record = read(REPLAY/'cases'/ident/'record.json')
        decision = read(REPLAY/'cases'/ident/'decision_before_evaluation.json')
        close(decision['selected_position'], record['decision']['selected_position'], 'decision unchanged')
        check(decision == record['decision'], 'decision full equality')
        solved = read(FORMAL/'cases'/ident/'solve.json')
        bank = Path(contract['original']['conditions'][cond]['bank']).resolve(strict=True)
        READS[str(bank)] = sha(bank)
        with np.load(bank, allow_pickle=False) as z:
            truth = z['p'].copy()
        pw = np.asarray(solved['final']['p'])
        swapped = np.sum((pw[::-1]-truth)**2) < np.sum((pw-truth)**2)
        # Apply ONE common permutation to all states and truth evaluation only.
        target = truth[::-1] if swapped else truth
        e = (pw-target).ravel()
        counts = np.zeros((5, 2), dtype=int)
        supports = {0: counts.copy().tolist()}
        for event in solved['history']:
            if 'fit' not in event:
                continue
            counts[event['pick']['m'], event['pick']['q']] += 1
            supports[int(counts.sum())] = counts.copy().tolist()
        kfinal = len(solved['paths'])
        wide = decision['scores'][kfinal]
        check(wide['order'] == kfinal, 'final order')
        evals = {v['order']: v for v in record['evaluation_only_states']}
        for r in decision['scores']:
            k = r['order']
            check(read(REPLAY/'cases'/ident/f'K{k:02d}.json') == r, 'standalone state equals record')
            p = np.asarray(r['position'])
            loss = float(np.sum((p-target)**2)/2)
            bestloss = float(min(np.sum((p-truth)**2), np.sum((p[::-1]-truth)**2))/2)
            if not np.isclose(loss, bestloss, rtol=1e-9, atol=1e-9):
                label_warnings.append(dict(case=ident, order=k, common_label_loss=loss,
                                           optimal_assignment_loss=bestloss))
            close(bestloss, evals[k]['loss_m2'], 'state loss replication')
            dd = (p-pw).ravel()
            disp = float(dd@dd/2)
            rd = float(r['risk']-wide['risk']) if r['applicable'] and wide['applicable'] else None
            states.append(dict(case=ident, condition=cond, order=k, position=p,
                               common_label_loss=loss, optimal_loss=bestloss,
                               applicable=r['applicable'], risk=r.get('risk'),
                               risk_difference=rd, loss_difference=loss-float(e@e/2),
                               displacement=disp, actual_cross=float(dd@e),
                               plugin_cross=rd-disp if rd is not None else None,
                               d=dd, e_wide=e, support_counts=supports[k],
                               source_order_truth_permutation=[1, 0] if swapped else [0, 1]))
        ps = np.asarray(decision['selected_position'])
        d = (ps-pw).ravel()
        lossw, losss = float(e@e/2), float(np.sum((ps-target)**2)/2)
        close(lossw, record['losses']['WIDE'], 'wide matched evaluation')
        close(losss, record['losses']['SELECTED_OR_FALLBACK'], 'selected common label evaluation')
        rd = decision['selected_risk']-wide['risk'] if not decision['fallback'] else None
        disp, cross = float(d@d/2), float(d@e)
        close(losss-lossw, disp+cross, 'per-case decomposition')
        rows.append(dict(case=ident, condition=cond, selected_order=decision['selected_order'],
                         final_order=kfinal, fallback=decision['fallback'], d=d, e_wide=e,
                         displacement=disp, actual_cross=cross, risk_difference=rd,
                         plugin_cross=rd-disp if rd is not None else None,
                         loss_difference=losss-lossw, loss_wide=lossw, loss_selected=losss,
                         risk_coverage=[decision['applicable_count'], decision['candidate_count']],
                         source_order_truth_permutation=[1, 0] if swapped else [0, 1]))
        support_groups[ident] = supports
    check(len(rows) == 32 and len(states) == 834, '32 cases 834 states')
    for c in ('L0', 'L2', 'L3', 'HIGH'):
        permutations = {tuple(r['source_order_truth_permutation']) for r in rows if r['condition'] == c}
        check(len(permutations) == 1, 'consistent source labels across repeats:' + c)
    groups = {c: decompose([r for r in rows if r['condition'] == c]) for c in ('L0', 'L2', 'L3', 'HIGH')}
    groups['HIGH_eligible6_diagnostic'] = decompose([r for r in rows if r['condition'] == 'HIGH' and not r['fallback']])
    groups['ALL32_mixed_conditions_descriptive'] = decompose(rows)
    for c in ('L0', 'L2', 'L3', 'HIGH'):
        close(groups[c]['mse_wide'], summary['conditions'][c]['models']['WIDE']['mse'], 'summary wide')
        close(groups[c]['mse_selected_or_fallback'], summary['conditions'][c]['models']['SELECTED_OR_FALLBACK']['mse'], 'summary selected')
    common = {}
    for c in ('L0', 'L2', 'L3'):
        rr = [r for r in rows if r['condition'] == c]
        limit = min(r['final_order'] for r in rr)
        table = []
        for k in range(limit+1):
            ss = [s for s in states if s['condition'] == c and s['order'] == k]
            check(len(ss) == 8, 'common order 8')
            valid = [s for s in ss if s['risk_difference'] is not None]
            supports = [encode(s['support_counts']) for s in ss]
            item = dict(order=k, n=8, applicable=len(valid), unique_station_source_count_patterns=len(set(supports)),
                        mean_loss=np.mean([s['optimal_loss'] for s in ss]),
                        mean_loss_difference=np.mean([s['loss_difference'] for s in ss]),
                        note='Order fixed retrospectively; support, delay, gain and wide reference remain data-dependent.')
            if len(valid) == 8:
                item['mean_predicted_difference'] = np.mean([s['risk_difference'] for s in ss])
                item['mean_prediction_error'] = item['mean_loss_difference']-item['mean_predicted_difference']
            table.append(item)
        common[c] = table
    oracle = {}
    for c in ('L0', 'L2', 'L3', 'HIGH'):
        vals = []
        for row in [r for r in rows if r['condition'] == c]:
            ss = [s for s in states if s['case'] == row['case']]
            eligible = [s for s in ss if s['applicable']]
            vals.append(dict(case=row['case'], all_min_loss=min(s['optimal_loss'] for s in ss),
                             eligible_min_loss=min((s['optimal_loss'] for s in eligible), default=None),
                             wide_loss=row['loss_wide']))
        oracle[c] = vals
    post = {p: sha(p) for p in pre}
    check(pre == post, 'frozen pre post identical')
    for path, digest in READS.items():
        check(sha(path) == digest, 'read artifact post:' + path)
    result = dict(material_passport='ANALYZED', new_localization_runs=0, source_label_warnings=label_warnings,
                  groups=groups, common_order_descriptive=common, oracle_diagnostic_only=oracle,
                  cases=rows, states=states, common_order_is_fixed_support=False,
                  checks=len(CHECKS), seconds=time.perf_counter()-start,
                  environment=dict(python=sys.version, numpy=np.__version__, platform=platform.platform()),
                  limitations=['Eight conditional noise repeats do not identify causal selection effects.',
                               'Squared sample mean is not an unbiased estimator of squared bias.',
                               'No independent validation; no new selector, cutoff or localization.'])
    result['coverage'] = dict(cases=len(rows), states=len(states),
                              applicable_states=sum(s['applicable'] for s in states),
                              fallback_cases=sum(r['fallback'] for r in rows))
    write(out/'analysis.json', result)
    write(out/'identity.json', dict(frozen_pre=pre, frozen_post=post, read_artifacts=READS,
                                   script_sha256=sha(Path(__file__)), checks=CHECKS))
    write(HERE/'latest.json', dict(output=str(out), material_passport='ANALYZED'))
    print(encode(dict(output=str(out), groups=groups, label_warnings=label_warnings,
                      checks=len(CHECKS), seconds=result['seconds'])))


if __name__ == '__main__':
    main()
