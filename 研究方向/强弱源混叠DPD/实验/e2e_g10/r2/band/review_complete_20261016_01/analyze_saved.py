"""Read-only R2 completion/SBL audit; never invoke a localization solver."""
from collections import Counter
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.g10.r2.band.runtime import read, sha, verify, load_input, validate
from 统一模型代码.gates.g10.metrics import evaluate
from 统一模型代码.gates.g10.r1.methods import at_time
from 统一模型代码.gates.g10.runtime import write


def summarize(rows):
    result = dict(n=len(rows), both=sum(all(r['metrics']['success']['20']) for r in rows),
        seconds_mean=float(np.mean([r['result']['seconds'] for r in rows])),
        seconds_sum=float(sum(r['result']['seconds'] for r in rows)),
        iteration_quantiles=np.quantile([r['result']['counts']['iterations'] for r in rows], [0, .5, .95, 1]).tolist(),
        iteration_mean=float(np.mean([r['result']['counts']['iterations'] for r in rows])),
        stops=dict(Counter(r['result']['stop_reason'] for r in rows)),
        gospa_mean=float(np.mean([r['metrics']['gospa_m'] for r in rows])))
    for name, index in (('strong', 0), ('weak', -1)):
        if rows[0]['kind'] == 'weak' and name == 'strong' or rows[0]['kind'] == 'strong' and name == 'weak':
            continue
        errors = [r['metrics']['errors_m'][index] for r in rows]
        result[name+'_success'] = sum(r['metrics']['success']['20'][index] for r in rows)
        result[name+'_rmse_m'] = float(np.sqrt(np.mean(np.square(errors))))
        result[name+'_p95_m'] = float(np.quantile(errors, .95))
        result[name+'_max_m'] = float(max(errors))
    return result


def main():
    base = ROOT/'outputs_e2e/unified/e2e_g10/r2/band'
    run = Path(read(base/'active_run.json')['out']).resolve(strict=True)
    state, contract, audit = (read(run/name) for name in ('status.json', 'contract.json', 'final_audit_report.json'))
    assert state['status'] == 'COMPLETED' and audit['complete'] and audit['n_rows'] == 2800
    verify(contract['sources']); verify(contract['band_sources'])
    assert sha(run/'input_manifest.json') == contract['input_manifest_sha']
    for name, expected in audit['rows'].items():
        assert sha(run/'rows'/name) == expected, name
    previous = read(run/'main_batch_report/final_audit_report.json')
    assert len(previous['rows']) == 2540
    for name, expected in previous['rows'].items():
        assert audit['rows'][name] == expected, 'Non-SBL row changed: '+name
    rows = []
    for job in contract['jobs']:
        if job['method'] != 'SBL':
            continue
        task = job['task']
        tag = f'{task[0]}_{task[1]}_{task[2]:g}_{task[3]}_SBL_v0'
        row = read(run/'rows'/f'{tag}.json')
        _, _, truth, count, identity = load_input(run, task)
        validate(row, job, identity)
        assert row['truth'] == truth.tolist() and row['result']['count'] == count
        rows.append(row)
    assert len(rows) == 260
    mixed = [r for r in rows if r['kind'] == 'mixed']
    groups = {kind: summarize([r for r in rows if r['kind'] == kind]) for kind in ('weak', 'strong', 'mixed')}
    conditions = {str(gap): summarize([r for r in mixed if r['task'][2] == gap]) for gap in (0, 3, 6, 9, 12, 15)}
    layouts = {str(g): summarize([r for r in mixed if r['task'][0] == g]) for g in range(4)}
    anytime = {}
    for seconds in (1, 5, 15, 30, 60, 120):
        metrics = [evaluate(r['truth'], at_time(r['result'], seconds)) for r in mixed]
        anytime[str(seconds)] = dict(both=sum(all(m['success']['20']) for m in metrics),
                                    weak=sum(m['success']['20'][-1] for m in metrics))
    main_seconds = read(run/'main_batch_report/status.json')['seconds']
    result = dict(run=str(run), complete=True, rows=2800, sbl_rows=260,
        audit=dict(all_row_hashes_pass=True, non_sbl_2540_unchanged=True,
                   source_identity_pass=True, sbl_input_metric_replay_pass=True),
        total_wall_seconds=state['seconds'], slow_batch_wall_seconds=state['seconds']-main_seconds,
        sbl=groups, gaps=conditions, layouts=layouts, anytime_mixed=anytime,
        total_sbl_solver_seconds=sum(g['seconds_sum'] for g in groups.values()),
        interruptions=read(run/'ledger.json')['interruptions'],
        maximum_weak_error_case=max(mixed, key=lambda r:r['metrics']['errors_m'][-1])['tag'],
        analysis_only_no_solver_execution=True)
    write(Path(__file__).parent/'analysis.json', result)
    print(result)


if __name__ == '__main__':
    main()
