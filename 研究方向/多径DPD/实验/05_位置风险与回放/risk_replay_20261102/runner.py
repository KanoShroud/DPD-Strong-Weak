"""Frozen-rule replay of 32 saved trajectories; no localization rerun."""
import argparse
from datetime import datetime
import hashlib
import importlib.util
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
PROJECT = next(p for p in HERE.parents if (p/'AGENTS.md').exists())
CAL = BASE/'risk_calibration_20261102'
FORMAL = CAL/'formal_20261102_181736_622970'
FOLLOW = BASE/'risk_followup_20261102'
PROBE = FOLLOW/'selector_20261102_211914_296566'
sys.path.insert(0, str(CAL))
spec = importlib.util.spec_from_file_location('frozen_calibration_replay', CAL/'runner.py')
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)
from storage import load, save, read, write, stage, lock, recover, cost, append, encode  # noqa: E402

ENTRY = PROJECT/'运行入口/E2E/G11/G11位置风险完整回放一键运行.py'
PROBE_IDS = ('L0_R00', 'L2_R00', 'L3_R00')


class CorrectnessFailure(AssertionError):
    def __init__(self, message, diagnostic):
        super().__init__(message)
        self.diagnostic = diagnostic


def sha(p):
    return cal.ref.sha(Path(p))


def norm2(x):
    return float(np.sum(np.abs(x)**2))


def states(solved):
    rows = [(solved['los'], [])]
    paths = []
    for event in solved['history']:
        if 'fit' not in event:
            continue
        if not event['decrease']:
            raise AssertionError('Saved trajectory contains unaccepted update.')
        pick = event['pick']
        paths = paths + [(pick['m'], pick['q'])]
        rows.append((event['fit'], paths.copy()))
    if paths != [tuple(p) for p in solved['paths']]:
        raise AssertionError('Saved path identity mismatch.')
    if not np.array_equal(rows[-1][0]['x'], solved['final']['x']):
        raise AssertionError('Saved final is not last accepted state.')
    return rows


def state_id(state, paths):
    return hashlib.sha256(encode(dict(x=state['x'], p=state['p'], paths=paths)).encode()).hexdigest()


def case_ids():
    return [case['id'] for case in cal.cases(False)]


def make_contract():
    original = load(FORMAL/'contract.json')['contract']
    cal.verify(original)
    pre = read(PROBE/'preflight.json')
    cal.check_paths(pre['hashes'])
    if pre['contract'] != original or pre['rule'] != 'minimum_existing_plugin_risk_tie_lower_order':
        raise AssertionError('Probe protocol mismatch.')
    files = [HERE/'runner.py', HERE/'report_replay.py', HERE/'selftest.py',
             HERE/'执行协议.md', ENTRY, FORMAL/'contract.json', FORMAL/'events.jsonl',
             PROBE/'preflight.json', FOLLOW/'selector_probe.py']
    inventories = []
    for ident in case_ids():
        folder = FORMAL/'cases'/ident
        solved = load(folder/'solve.json')
        traj = states(solved)
        files.extend(folder/name for name in ('solve.json', 'input.json', 'input.npz', 'LOS_risk.json', 'WIDE_risk.json'))
        inventories.append(dict(case=ident, count=len(traj), stop=solved['stop'],
                                orders=[len(p) for _, p in traj]))
        if ident in PROBE_IDS:
            files.append(PROBE/(ident+'_scores.json'))
            files.extend(PROBE/f'{ident}_K{len(p):02d}_diagnostic.json' for _, p in traj)
    hashes = dict(original['hashes'])
    hashes.update({str(p.resolve(strict=True)): sha(p) for p in files})
    return dict(original=original, hashes=hashes, inventory=inventories,
                rule='minimum_existing_plugin_risk_tie_lower_order',
                fallback='original_final_when_no_reliable_wide_or_no_eligible_state',
                independent_validation=False, new_localization_runs=0,
                use_original_endpoints_only=True, seed=2026110209,
                expected_total=32, threads=2)


def verify(contract):
    cal.verify(contract['original'])
    cal.check_paths(contract['hashes'])


def get_context(ident):
    folder = FORMAL/'cases'/ident
    meta = load(folder/'input.json')
    solved = load(folder/'solve.json')
    with np.load(folder/'input.npz', allow_pickle=False) as z:
        if set(z.files) != {'iq'}:
            raise AssertionError('Unexpected solver-input field.')
        iq = z['iq']
    if cal.digest(iq) != meta['iq_sha'] or solved['iq_sha'] != meta['iq_sha']:
        raise AssertionError('IQ/solver identity mismatch.')
    f, y, n, sigma = cal.spectral(iq)
    wide = load(folder/'WIDE_risk.json')
    if wide['applicable'] and not np.isclose(sigma, wide['sigma2_est'], rtol=1e-12, atol=0):
        raise AssertionError('Noise scale mismatch.')
    return dict(case=ident, solved=solved, traj=states(solved), wide=wide,
                f=f, y=y, n=n, sigma=sigma, iq_sha=meta['iq_sha'],
                wide_sha=sha(folder/'WIDE_risk.json'))


def score_value(p, pw, influence, lw, sigma):
    return norm2(p-pw)/2 + sigma*float(np.sum(influence*lw)) - sigma*norm2(lw)/2


def check_diagnostic(diag, ident, order):
    if diag.get('derivative_checks') and not all(diag['derivative_checks'].values()):
        raise CorrectnessFailure(f'Derivative correctness failure: {ident}/K{order}', diag)


def score_state(ctx, state, paths, reuse=True):
    ident, wide = ctx['case'], ctx['wide']
    order = len(paths)
    p = np.asarray(state['p'])
    pw = np.asarray(ctx['solved']['final']['p'])
    row = dict(order=order, position=p, state_sha=state_id(state, paths),
               iq_sha=ctx['iq_sha'], wide_sha=ctx['wide_sha'], applicable=False,
               compute_seconds=0., equivalent_diagnostic_seconds=0.)
    if ctx['solved']['stop'] != 'no_detected_increment' or not wide['applicable']:
        row.update(reason='no_reliable_wide_reference', provenance='not_scored')
        return row
    if reuse and order not in (0, len(ctx['solved']['paths'])) and ident in PROBE_IDS:
        # These artifacts were saved BEFORE truth access; do not read *_result.json.
        scores = read(PROBE/(ident+'_scores.json'))
        cached = next(r for r in scores if r['order'] == order)
        if any('loss' in k or 'truth' in k for k in cached):
            raise AssertionError('Probe score contains evaluation data.')
        if not np.array_equal(cached['position'], state['p']):
            raise AssertionError('Probe state position mismatch.')
        diag = read(PROBE/f'{ident}_K{order:02d}_diagnostic.json')
        check_diagnostic(diag, ident, order)
        if cached['state'] != diag['state'] or cached['state']['dimension_ok'] is not True:
            raise AssertionError('Probe diagnostic mismatch.')
        row.update(applicable=cached['applicable'], reason=cached.get('reason'),
                   diagnostic=diag, provenance='verified_saved_pre_evaluation_probe_score',
                   equivalent_diagnostic_seconds=float(cached['original_diagnostic_seconds']))
        if row['applicable']:
            row['risk'] = float(cached['risk'])
        return row
    if order == 0 or order == len(ctx['solved']['paths']):
        endpoint = load(FORMAL/'cases'/ident/('LOS_risk.json' if order == 0 else 'WIDE_risk.json'))
        diag = endpoint['diagnostic']
        influence = np.asarray(endpoint['L']) if endpoint['applicable'] else None
        provenance = 'original_endpoint_influence'
    else:
        begin = time.perf_counter()
        diag, influence = cal.risk.diagnose(np.asarray(state['x']), ctx['f'], ctx['y'], paths,
            ctx['n'], ctx['sigma'], np.random.default_rng(2026110209+order))
        row['compute_seconds'] = time.perf_counter()-begin
        provenance = 'new_read_only_saved_state_diagnostic'
    check_diagnostic(diag, ident, order)
    row.update(diagnostic=diag, provenance=provenance, applicable=influence is not None,
               reason=diag.get('reason'), equivalent_diagnostic_seconds=float(diag['seconds']))
    if influence is not None:
        if norm2(p[::-1]-pw) < norm2(p-pw):
            row.update(applicable=False, reason='estimated_source_label_change')
        else:
            row['risk'] = score_value(p, pw, influence, np.asarray(wide['L']), ctx['sigma'])
            if not np.isfinite(row['risk']):
                raise CorrectnessFailure(f'Nonfinite score: {ident}/K{order}', row)
    return row


def protected_score(out, ctx, state, paths):
    try:
        return score_state(ctx, state, paths)
    except CorrectnessFailure as exc:
        target = Path(out)/'failures'/(ctx['case']+f'_K{len(paths):02d}_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
        save(target, dict(error=str(exc), diagnostic=exc.diagnostic,
                          state_sha=state_id(state,paths), iq_sha=ctx['iq_sha']))
        raise


def choose(rows, solved):
    eligible = [r for r in rows if r['applicable']]
    selected = min(eligible, key=lambda r: (r['risk'], r['order'])) if eligible else None
    return dict(selected_order=selected['order'] if selected else None,
                selected_risk=selected['risk'] if selected else None,
                selected_position=selected['position'] if selected else solved['final']['p'],
                fallback=selected is None, output_policy='original_final_fallback' if selected is None else 'risk_minimum',
                candidate_count=len(rows), applicable_count=len(eligible),
                full_trajectory_coverage=len(rows) == len(eligible),
                decision_source='observations_only_before_truth_access', scores=rows)


def evaluate(decision, ctx, contract):
    # This function must run only AFTER decision_before_evaluation.json is saved.
    condition = ctx['case'].split('_')[0]
    with np.load(contract['original']['conditions'][condition]['bank'], allow_pickle=False) as z:
        truth = z['p']
    losses = {}
    errors = {}
    positions = {'SELECTED_OR_FALLBACK':decision['selected_position'],
                 'WIDE':ctx['solved']['final']['p'], 'LOS':ctx['solved']['los']['p']}
    for name, value in positions.items():
        p = np.asarray(value)
        if norm2(p[::-1]-truth) < norm2(p-truth):
            p = p[::-1]
        errors[name] = np.linalg.norm(p-truth, axis=1)
        losses[name] = norm2(p-truth)/2
    oracle = []
    for row in decision['scores']:
        p = np.asarray(row['position'])
        oracle.append(dict(order=row['order'], applicable=row['applicable'],
                           loss_m2=min(norm2(p-truth),norm2(p[::-1]-truth))/2))
    return dict(case=ctx['case'], condition=condition, decision=decision,
                losses=losses, source_errors=errors, evaluation_only_states=oracle,
                baseline_solve_seconds=ctx['solved']['seconds'], stop=ctx['solved']['stop'],
                risk_equivalent_seconds=sum(r['equivalent_diagnostic_seconds'] for r in decision['scores']))


def budget(contract):
    samples = []
    for ident in PROBE_IDS:
        for r in read(PROBE/(ident+'_scores.json')):
            if r['order'] > 0 and not r['reused']:
                samples.append((20+3*r['order'], r['original_diagnostic_seconds']))
    d, seconds = np.array(samples).T
    coefficient = float(np.median(seconds/(d*d)))
    total, new, skipped, reused = 0., 0., 0, 0
    count = 0
    for item in contract['inventory']:
        count += item['count']
        if item['stop'] != 'no_detected_increment':
            skipped += item['count']
            continue
        for order in item['orders']:
            estimate = coefficient*(20+3*order)**2
            total += estimate
            if order in (0, item['orders'][-1]) or item['case'] in PROBE_IDS:
                reused += 1
            else:
                new += estimate
    return dict(states=count, unscored_no_reliable_wide=skipped,
                reusable_states=reused, new_diagnostics=count-skipped-reused,
                coefficient_seconds_per_dimension_squared=coefficient,
                new_compute_hours_estimate=new/3600, expected_range_hours=[new/3600*.7,new/3600*1.6],
                equivalent_diagnostic_hours_estimate=total/3600,
                basis='median actual probe diagnostic time / parameter_dimension^2; no new localization')


def initialize(contract, resume=None):
    pointer = HERE/'latest_formal.json'
    if resume or pointer.exists():
        out = Path(resume or read(pointer)['path']).resolve(strict=True)
        if not out.is_relative_to(HERE) or load(out/'contract.json') != contract:
            raise RuntimeError('恢复目录或冻结身份不匹配。')
        return out
    out = HERE/('formal_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    save(out/'contract.json', contract)
    mapping = {}
    for source, digest in contract['hashes'].items():
        p = Path(source)
        if p.suffix in ('.py', '.md'):
            target = out/'source'/(digest[:24]+p.suffix)
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(p, target)
            mapping[source] = str(target.relative_to(out))
    write(out/'source_map.json',mapping)
    write(pointer,dict(path=str(out)))
    return out


def execute(out, contract):
    report = load_report()
    with lock(out):
        recover(out)
        verify(contract)
        print('完整回放结果目录：', out, flush=True)
        for i, ident in enumerate(case_ids()):
            folder = out/'cases'/ident
            folder.mkdir(parents=True, exist_ok=True)
            final = folder/'record.json'
            if final.exists():
                load(final)
                print(f'[{i+1}/32] {ident} 已完成，跳过。', flush=True)
                continue
            start = time.perf_counter()
            before = cost(out)['seconds']
            ctx = get_context(ident)
            rows = []
            for state, paths in ctx['traj']:
                k = len(paths)
                target = folder/f'K{k:02d}.json'
                row, _ = stage(out, f'{ident}/K{k}', target, lambda st=state, pa=paths:protected_score(out,ctx,st,pa))
                if row['state_sha'] != state_id(state,paths) or row['iq_sha'] != ctx['iq_sha'] or row['wide_sha'] != ctx['wide_sha']:
                    raise AssertionError('Cached state identity mismatch.')
                rows.append(row)
                print(f'[{i+1}/32] {ident} K{k}/{len(ctx["traj"])-1} 评分可用={row["applicable"]} 累计{cost(out)["seconds"]/3600:.2f}h', flush=True)
            decision, _ = stage(out, ident+'/decision', folder/'decision_before_evaluation.json', lambda:choose(rows,ctx['solved']))
            result, _ = stage(out, ident+'/evaluation', final, lambda:evaluate(decision,ctx,contract))
            overhead = max(0., time.perf_counter()-start-(cost(out)['seconds']-before))
            append(out,dict(kind='end',id=ident+'_overhead_'+datetime.now().isoformat(),stage='overhead',seconds=overhead))
            report.summarize(out,contract)
        verify(contract)
        report.summarize(out,contract)
        print('32条完整回放完成。请交Codex回读；不自动修改公式、不启动640条。', flush=True)


def load_report():
    spec = importlib.util.spec_from_file_location('risk_replay_report',HERE/'report_replay.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--resume')
    parser.add_argument('--release', action='store_true')
    args = parser.parse_args()
    if args.release:
        contract = make_contract()
        test = load(HERE/'test_result.json')
        if not test['all_pass'] or test['contract'] != contract:
            raise RuntimeError('最终版本短测证据未通过或已过期。')
        save(HERE/'release.json', dict(contract=contract,budget=budget(contract),tests=test['directory']))
        print('已冻结发布。正式32条未启动。',flush=True)
        return
    released = load(HERE/'release.json')
    contract = released['contract']
    verify(contract)
    if args.check:
        print(encode(released['budget']), flush=True)
        print('冻结身份通过；未创建/运行正式批。',flush=True)
        return
    if args.report_only:
        pointer = HERE/'latest_formal.json'
        if not args.resume and not pointer.exists():
            raise RuntimeError('尚无正式结果，不能只读重绘。')
        out = Path(args.resume or read(pointer)['path']).resolve(strict=True)
        if not out.is_relative_to(HERE) or load(out/'contract.json') != contract:
            raise RuntimeError('报告目录身份不匹配。')
        with lock(out):
            load_report().summarize(out,contract)
        return
    execute(initialize(contract,args.resume),contract)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
