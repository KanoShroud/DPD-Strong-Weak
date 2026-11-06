"""Read-only P7 audit. No locator, model fit, or candidate search is called.

The saved B/C choice is retrospective mechanism evidence, not a new method
validation. Its cost is the sum of both archived complete run times.
"""
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from 统一模型代码.gates.g11.model_integration.p7 import (  # noqa: E402
    METHODS, get_input, jobs, key, read, read_row, sha, validate_row, verify,
)
from 统一模型代码.gates.g10.r4.paper_checks.assignment_metrics import evaluate  # noqa: E402

FORMAL = ROOT / 'outputs_e2e/unified/e2e_g11/p7_staged/formal_20261024_112623_403181'


def summarize(rows):
    e = np.array([r['metrics']['errors_m'] for r in rows], float)
    assert np.isfinite(e).all()
    return dict(samples=len(rows), rmse_A=float(np.sqrt(np.mean(e[:, 0]**2))),
                rmse_B=float(np.sqrt(np.mean(e[:, 1]**2))),
                rmse_all=float(np.sqrt(np.mean(e**2))),
                both20=sum(bool(r['metrics']['both20']) for r in rows),
                p95_B=float(np.percentile(e[:, 1], 95)), max_B=float(e[:, 1].max()),
                mean_saved_complete_seconds=float(np.mean([r['result']['seconds'] for r in rows])))


def main():
    contract = read(FORMAL / 'contract.json')
    verify(FORMAL, contract)
    out = Path(__file__).parent / ('replay_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    rows, identities, failures = {}, {}, []
    for task, method in jobs(contract):
        path = FORMAL / 'rows' / (key(task) + '_' + method + '.json')
        row = read_row(path)
        iq, truth = get_input(FORMAL, task)
        validate_row(row, iq, truth, task, method)
        assert sha(FORMAL / 'spectra' / row['spectrum']) == row['spectrum_sha']
        rows[(key(task), method)] = row
        identities[str(path.relative_to(FORMAL))] = sha(path)
        if not row['metrics']['both20']:
            branch_success = [evaluate(truth, b['positions'])['both20'] for b in row['result']['branches']]
            failures.append(dict(**task, method=method, errors=row['metrics']['errors_m'],
                                 any_successful_refined_branch=any(branch_success)))
    assert len(rows) == 968
    tables, combined, contrasts = [], [], []
    for n in (1024, 4096):
        for method in METHODS:
            group_rows = [r for r in rows.values() if r['task']['n'] == n and r['method'] == method]
            tables.append(dict(n=n, method=method, **summarize(group_rows)))
        selected, decisions, cost = [], [], []
        for task in contract['tasks']:
            if task['n'] != n:
                continue
            ident = key(task)
            b, c, d = [rows[(ident, m)] for m in ('B_TERMINAL', 'C_RESIDUAL', 'D_STAGED')]
            assert b['input_sha'] == c['input_sha'] == d['input_sha']
            assert b['result']['data_sha'] == c['result']['data_sha'] == d['result']['data_sha']
            assert b['result']['model'] == c['result']['model'] == d['result']['model']
            # Decide from saved objective only, before consulting performance.
            chosen = min((b, c), key=lambda r: (r['result']['objective'], r['method']))
            selected.append(chosen)
            cost.append(b['result']['seconds'] + c['result']['seconds'])
            decisions.append(dict(task=task, chosen=chosen['method'],
                                  objective_B=b['result']['objective'], objective_C=c['result']['objective']))
            if b['metrics']['both20'] != d['metrics']['both20']:
                good, bad = (b, d) if b['metrics']['both20'] else (d, b)
                contrasts.append(dict(task=task, good=good['method'], bad=bad['method'],
                                      objective_bad_minus_good=bad['result']['objective']-good['result']['objective']))
        stat = summarize(selected)
        # A selected-row runtime is not the runtime of doing both methods.
        del stat['mean_saved_complete_seconds']
        combined.append(dict(n=n, **stat, chosen_counts=dict(Counter(d['chosen'] for d in decisions)),
                             mean_sum_of_B_and_C_saved_times=float(np.mean(cost)), decisions=decisions,
                             status='POST_HOC_REPLAY_NOT_AN_INDEPENDENT_ALGORITHM_EXPERIMENT'))
    limits = []
    for n in (1024, 4096, 8192):
        duration = n / 50e6
        limits.append(dict(n=n, duration_us=duration*1e6, nominal_time_bandwidth=duration*12.5e6,
                           baseline_delay_us=1000/299792458*1e6,
                           maximum_station_delay_span_over_T=(1000/299792458)/duration))
    ledger = read(FORMAL / 'ledger.json')
    result = dict(source=str(FORMAL), source_contract_sha=sha(FORMAL/'contract.json'),
                  script_sha=sha(Path(__file__)), verified_rows=len(rows),
                  verified_input_files=len(contract['inputs']), verified_code_files=len(contract['sources']),
                  selection_stops=dict(Counter(r['result']['stop_reason'] for r in rows.values())),
                  branch_stops=dict(Counter(b['stop_reason'] for r in rows.values() for b in r['result']['branches'])),
                  statistics=tables, failures=failures, BC_replay=combined,
                  discordant_B_D=contrasts, dimensionless_scales=limits,
                  archived_ledger=ledger, identity=identities,
                  scope='saved-result audit only; no new localization or calibration')
    verify(FORMAL, contract)
    for relative, expected in identities.items():
        assert sha(FORMAL/relative) == expected
    destination = out/'audit.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(destination)
    print(json.dumps(dict(rows=len(rows), statistics=tables,
                          combined=[{k: v for k, v in c.items() if k != 'decisions'} for c in combined],
                          contrasts=len(contrasts), all_bad_objectives_higher=all(c['objective_bad_minus_good'] > 0 for c in contrasts),
                          all_failed_branches_fail=not any(f['any_successful_refined_branch'] for f in failures)),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
