"""Approved finite native-stop review; frozen algorithms/tolerances unchanged.

Eight predetermined observations (four layouts, realization 200, gaps 0/15).
Only the safety cap changes: NOMP 12->200, BNOMP_REF 100->10000. Both retain
their original data-dependent stopping rule; no truth enters either solver.
No automatic continuation or threshold selection if a cap is still reached.
"""
from pathlib import Path
from datetime import datetime
from collections import Counter
import copy
import csv
import io
import sys
import time
import traceback
import numpy as np
import torch
from ..metrics import evaluate
from ..r1.runtime import Ledger
from .config import ROOT, BASE
from .runtime import (read, write, sha, seal, check_contract, load, job_key,
                      require_exclusive, resource_check, lock)
from .solvers import run

TASKS = tuple((g, 200, gap, 'mixed') for g in range(4) for gap in (0, 15))
CAPS = {'NOMP': 200, 'BNOMP_REF': 10000}
REVISION = 'G10_R4_NATIVE_STOP_REVIEW_V1'


def history_audit():
    """Read complete mixed-source records only; do not relabel old outcomes."""
    roots = {
        'R1': ROOT/'outputs_e2e/unified/e2e_g10/r1/fs50m_n8192/formal_20261014_205445',
        'R2_BAND': ROOT/'outputs_e2e/unified/e2e_g10/r2/band/independent_20261016_151236',
        'R3': ROOT/'outputs_e2e/unified/e2e_g10/r3/independent_20261016_210849_013753'}
    result = {}
    for label, root in roots.items():
        root = root.resolve(strict=True)
        kept, identities = [], {}
        for path in sorted((root/'rows').glob('*NOMP*.json')):
            row = read(path)
            method = row.get('method', row.get('job', {}).get('method'))
            task = row.get('task', row.get('job', {}).get('task'))
            kind = row.get('kind', task[-1] if task else None)
            if method not in ('BNOMP', 'NOMP') or kind != 'mixed':
                continue
            if row['metrics'] != evaluate(row['truth'], row['result']['positions']):
                raise RuntimeError('Historical metric replay differs')
            kept.append(row); identities[path.name] = sha(path)
        if len(kept) != 120:
            raise RuntimeError('Expected complete 120-case native NOMP history: '+label)
        counts = Counter(r['result']['stop_reason'] for r in kept)
        capped = [r for r in kept if r['result']['stop_reason'] == 'ITERATION_LIMIT']
        result[label] = dict(out=str(root), cases=len(kept), stops=dict(counts),
            both20=sum(all(r['metrics']['success']['20']) for r in kept),
            capped_but_successful=sum(all(r['metrics']['success']['20']) for r in capped),
            rows=identities)
    return result


def stages(result, method):
    return {n: [h for h in result['history'] if
        (h.get('discovered', len(h['positions']))) == n] for n in (1, 2)}


def prefix_check(old, new, method):
    a, b = stages(old, method), stages(new, method)
    checked = []
    # Before the first changed cap is crossed, identical updates are required.
    for n in (1, 2):
        if n == 2:
            # Identical first-source terminal state implies identical discovery.
            if len(a[1]) != len(b[1]):
                break
        if len(b[n]) < len(a[n]):
            raise RuntimeError('Cap-only extension stopped earlier on identical prefix')
        for x, y in zip(a[n], b[n]):
            np.testing.assert_array_equal(x['positions'], y['positions'])
            assert x['native_objective'] == y['native_objective']
        checked.append(dict(discovered=n, identical_cycles=len(a[n])))
    return checked


def stopping_summary(result, method, cfg):
    output = []
    for n, hist in stages(result, method).items():
        last = hist[-1]
        changes = []
        for i, h in enumerate(hist):
            before = h.get('energy_before')
            if before is None and i:
                before = hist[i-1]['native_objective']
            if before is not None:
                changes.append(float(before-h['native_objective']))
        if method == 'BNOMP_REF':
            stop = result['mechanism']['native_cycles'][n-1]['stop_reason']
            satisfied = abs(changes[-1]) <= cfg['bnomp_energy_tolerance']
        else:
            # Native trace omits the before-value of the first cycle. At an
            # early exit after exactly two cycles the original control flow
            # certifies two stable checks; do not mislabel it as unverified.
            first_omitted = len(hist) == 2 and len(hist) < cfg['cycles']
            satisfied = ((len(changes) >= 2 or first_omitted)
                         and all(d < cfg['objective_tolerance'] for d in changes[-2:])
                         and all(h['max_shift_m'] < cfg['position_tolerance'] for h in hist[-2:]))
            stop = 'NATIVE_STABLE' if satisfied else 'ITERATION_LIMIT'
        output.append(dict(discovered=n, cycles=len(hist), native_stop=stop,
            stopping_conditions_met=bool(satisfied), last_energy_changes=changes[-3:],
            first_cycle_energy_available=bool(method=='BNOMP_REF'),
            early_two_cycle_stop_certified_by_native_control_flow=bool(method=='NOMP' and len(hist)==2),
            last_max_shift_m=last.get('max_shift_m'),
            positions=last['positions'], checkpoints=[dict(cycle=i+1, positions=h['positions'],
                objective=h['native_objective']) for i,h in enumerate(hist)
                if i+1 in (12, 50, 100, 200, 500, 1000, 2000, 5000, 10000)]))
    return output


def summarize(out, contract, rows=None):
    rows = [read(out/'rows'/(job_key(j)+'.json')) for j in contract['jobs']] if rows is None else rows
    summary = {}
    for method in CAPS:
        rs = [r for r in rows if r['job']['method'] == method]
        changes = [r['baseline']['result']['native_objective']-r['result']['native_objective'] for r in rs]
        moves = [np.linalg.norm(np.array(r['result']['positions'])-
            np.array(r['baseline']['result']['positions']), axis=1).max() for r in rs]
        summary[method] = dict(cases=len(rs), old_both20=sum(all(r['baseline']['metrics']['success']['20']) for r in rs),
            new_both20=sum(all(r['metrics']['success']['20']) for r in rs),
            old_weak_rmse=float(np.sqrt(np.mean([r['baseline']['metrics']['errors_m'][1]**2 for r in rs]))),
            new_weak_rmse=float(np.sqrt(np.mean([r['metrics']['errors_m'][1]**2 for r in rs]))),
            native_stopped=sum(all(x['stopping_conditions_met'] for x in r['native_stopping']) for r in rs),
            max_cycles=max(x['cycles'] for r in rs for x in r['native_stopping']),
            stage_cycles=[{str(x['discovered']):x['cycles'] for x in r['native_stopping']} for r in rs],
            old_mean_seconds=float(np.mean([r['baseline']['result']['wall_solver_seconds'] for r in rs])),
            new_mean_seconds=float(np.mean([r['result']['wall_solver_seconds'] for r in rs])),
            median_seconds=float(np.median([r['result']['wall_solver_seconds'] for r in rs])),
            p95_seconds=float(np.quantile([r['result']['wall_solver_seconds'] for r in rs],.95)),
            max_position_movement_m=float(max(moves)), max_objective_improvement=float(max(changes)),
            median_objective_improvement=float(np.median(changes)))
    # Eight paired samples estimate native solver cost only. Other five methods
    # still use the previous 16-case timing, explicitly not fresh measurements.
    prior = read(Path(contract['reference'])/'budget_estimate.json')
    mean_solver = sum(x['mean_based_seconds'] for m,x in prior['by_method'].items() if m not in CAPS)
    p95_solver = sum(x['p95_based_seconds'] for m,x in prior['by_method'].items() if m not in CAPS)
    for method in CAPS:
        group = [r['result'] for r in rows if r['job']['method'] == method]
        total = [r['wall_solver_seconds']+r['certificate_seconds'] for r in group]
        mean_solver += 2400*float(np.mean(total))
        p95_solver += 2400*float(np.quantile(total,.95))
    other = prior['generation_io_seconds']+prior['recording_save_audit_seconds']
    forecast = dict(conditional=True, formal_cases=2400,
        mean_hours=(1.25*(mean_solver+other)+600)/3600,
        p95_scenario_hours=(1.25*(p95_solver+other)+600)/3600,
        basis='Eight native-stop probes, other five methods from previous preparation; no automatic adoption of new caps.')
    report = dict(scope='approved native-stop diagnostic, not new independent performance evidence',
        summaries=summary, provisional_cost=forecast, formal_released=False)
    write(out/'comparison_report.json', report)
    buff = io.StringIO(newline='')
    writer=csv.writer(buff)
    writer.writerow(['layout','repeat','gap','method','old_success','new_success','old_weak_error_m',
        'new_weak_error_m','stage1_cycles','stage2_cycles','seconds','stop'])
    for r in rows:
        writer.writerow(r['job']['task'][:3]+[r['job']['method'],all(r['baseline']['metrics']['success']['20']),
            all(r['metrics']['success']['20']),r['baseline']['metrics']['errors_m'][1],r['metrics']['errors_m'][1],
            *[x['cycles'] for x in r['native_stopping']],r['result']['wall_solver_seconds'],r['result']['stop_reason']])
    (out/'per_case.csv').write_text(buff.getvalue(),encoding='utf-8-sig')
    return report


def main():
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):
            stream.reconfigure(encoding='utf-8',errors='replace')
    require_exclusive(); resource_check()
    pointer = read(BASE/'preparation.json')
    ref = Path(pointer['out']).resolve(strict=True)
    if sha(ref/'contract.json') != pointer['contract_sha']:
        raise RuntimeError('Reference contract changed')
    original = check_contract(ref)
    audit = read(ref/'final_audit_report.json')
    plan = []
    for index,task in enumerate(TASKS):
        methods = ('NOMP','BNOMP_REF') if index % 2 else ('BNOMP_REF','NOMP')
        plan.extend(dict(task=list(task),method=m) for m in methods)
    cache = {}
    for j in plan:
        _, iq, truth, identity = load(original,j['task'])
        path = ref/'rows'/(job_key(j)+'.json')
        if sha(path) != audit['rows'][path.name]:
            raise RuntimeError('Baseline row identity changed')
        cache[job_key(j)] = (iq,truth,identity,read(path))
    out = (BASE/('stop_review_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    if out.exists() or not out.is_relative_to(BASE.resolve()):
        raise RuntimeError('Fresh isolated output required')
    out.mkdir()
    contract = dict(revision=REVISION,reference=str(ref),reference_contract_sha=sha(ref/'contract.json'),
        reference_audit_sha=sha(ref/'final_audit_report.json'),jobs=plan,caps=CAPS,
        unchanged_tolerances=True,unchanged_algorithms=True,formal_started=False,
        source=str(Path(__file__).relative_to(ROOT)),source_sha=sha(__file__),
        schedule_seconds=30*60, no_automatic_extension=True)
    write(out/'contract.json',contract)
    (out/'stop_review.py').write_bytes(Path(__file__).read_bytes())
    write(out/'historical_stops.json',history_audit())
    print('停止核验目录：',out,flush=True)
    with lock(out):
        clock=Ledger(out)
        try:
            # One already-used native run as common warmup, outside formal timing.
            j=plan[0]; iq,_,_,_=cache[job_key(j)]
            start=time.perf_counter()
            run(iq,'NOMP',original['configurations']['NOMP'],clock.check)
            torch.cuda.synchronize()
            write(out/'warmup.json',dict(seconds=time.perf_counter()-start))
            for index,job in enumerate(plan,1):
                if clock.total()>contract['schedule_seconds']:
                    raise RuntimeError('Finite diagnostic schedule reached; no next solve started')
                require_exclusive()
                iq,truth,identity,baseline=cache[job_key(job)]
                cfg=copy.deepcopy(original['configurations'][job['method']])
                key='bnomp_cycles' if job['method']=='BNOMP_REF' else 'cycles'
                cfg[key]=CAPS[job['method']]
                differences=[k for k in cfg if cfg[k]!=baseline['result']['config'][k]]
                assert differences==[key]
                clock.flush(dict(tag=job_key(job),ledger_start_seconds=clock.total()))
                result=run(iq,job['method'],cfg,clock.check)
                prefix=prefix_check(baseline['result'],result,job['method'])
                stopping=stopping_summary(result,job['method'],cfg)
                row=dict(job=job,input_sha=identity,truth=truth.tolist(),result=result,
                    metrics=evaluate(truth,result['positions']),baseline=baseline,
                    changed_configuration_keys=differences,prefix_checks=prefix,native_stopping=stopping)
                row['record_sha']=seal(row)
                write(out/'rows'/(job_key(job)+'.json'),row)
                clock.flush()
                print(f"停止核验 {index}/{len(plan)} | {job_key(job)} | "
                    f"循环{[x['cycles'] for x in stopping]} | {result['wall_solver_seconds']:.2f}s | {result['stop_reason']}",flush=True)
            report=summarize(out,contract)
            check_contract(ref)
            if sha(ref/'contract.json')!=contract['reference_contract_sha'] or sha(ref/'final_audit_report.json')!=contract['reference_audit_sha']:
                raise RuntimeError('Frozen reference changed')
            hashes={}
            for j in plan:
                path=out/'rows'/(job_key(j)+'.json');r=read(path)
                assert r['record_sha']==seal(r)
                assert r['metrics']==evaluate(r['truth'],r['result']['positions'])
                assert r['input_sha']==cache[job_key(j)][2]
                hashes[path.name]=sha(path)
            write(out/'final_audit_report.json',dict(passed=True,solves=len(plan),rows=hashes,
                only_cap_changed=True,prefix_equivalent=True,source_and_input_unchanged=True))
            clock.flush()
            write(out/'status.json',dict(status='COMPLETED_STOP_REVIEW',seconds=clock.total(),formal_started=False))
            write(BASE/'stop_review.json',dict(out=str(out),contract_sha=sha(out/'contract.json')))
            print(report,flush=True)
        except BaseException as exc:
            clock.flush(keep=True)
            write(out/'status.json',dict(status='STOPPED',error=str(exc),traceback=traceback.format_exc(),formal_started=False))
            raise


def repair_report(source):
    """Correct only reporting in a new child directory. No solver invocation."""
    source=Path(source).resolve(strict=True)
    contract=read(source/'contract.json')
    old_audit=read(source/'final_audit_report.json')
    out=source/'report_v2'
    if out.exists():
        raise FileExistsError('Fresh report required')
    out.mkdir()
    rows,checked={},{}
    for job in contract['jobs']:
        name=job_key(job)+'.json';path=source/'rows'/name
        if sha(path)!=old_audit['rows'][name]:
            raise RuntimeError('Original row changed')
        row=read(path)
        assert row['record_sha']==seal(row)
        assert row['metrics']==evaluate(row['truth'],row['result']['positions'])
        row['native_stopping']=stopping_summary(row['result'],job['method'],row['result']['config'])
        rows[name]=row
        checked[name]=dict(source_sha=sha(path),native_stopping=row['native_stopping'])
    report=summarize(out,contract,list(rows.values()))
    write(out/'stopping_replay.json',checked)
    write(out/'final_audit_report.json',dict(passed=True,source=str(source),
        source_audit_sha=sha(source/'final_audit_report.json'),corrected_summary_only=True,
        metrics_and_solver_outputs_unchanged=True,solvers_called=0,
        report_source=str(Path(__file__).relative_to(ROOT)),report_source_sha=sha(__file__)))
    write(out/'correction.json',dict(reason='NOMP first-cycle before-energy is not present in native trace. '
        'An early two-cycle exit certifies both stable checks by native control flow; v1 incorrectly required two recorded deltas.',
        previous_false_native_stop_count=0,corrected_native_stop_count=report['summaries']['NOMP']['native_stopped'],
        console_encoding='Original module stdout used Windows default; JSON evidence is UTF-8. Entry now explicitly emits UTF-8.',
        affected='native_stopped reporting only; no changes to IQ, algorithms, tolerances, positions or timing'))
    (out/'stop_review.py').write_bytes(Path(__file__).read_bytes())
    write(BASE/'stop_review.json',dict(out=str(source),report=str(out),contract_sha=sha(source/'contract.json')))
    print('报告重放完成，未调用定位器：',out)
    return report


if __name__=='__main__':
    main()
