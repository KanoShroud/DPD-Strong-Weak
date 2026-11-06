"""Validate pilot identities and native equivalence; freeze an unrun formal plan."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.g10.r4.paper_checks import review_supplement as study  # noqa: E402
from 统一模型代码.gates.g10.r4.paper_checks.assignment_metrics import evaluate  # noqa: E402
from 统一模型代码.gates.g10.runtime import read, write, sha  # noqa: E402


def main():
    pointer = read(study.ACTIVE)
    out = Path(pointer['out']).resolve(strict=True)
    assert sha(out/'contract.json') == pointer['contract_sha']
    c = read(out/'contract.json')
    study.verify_contract(out, c)
    summary = read(out/'summary.json')
    assert c['pilot'] and summary['status'] == 'COMPLETED'
    assert summary['completed_solves'] == 40
    source = Path(c['source'])
    audit = read(source/'final_audit_report.json')
    identities, checked = {}, 0
    for path in sorted((out/'pairs').glob('*.json')):
        pair = read(path)
        iq, truth, identity = study.consume(out, c, pair['stage'], pair['task'])
        assert identity == pair['iq_sha'] and truth.tolist() == pair['truth']
        for row in pair['rows']:
            assert row['metrics'] == evaluate(truth, row['result']['positions'])
            assert row['result']['review_complete_seconds'] > 0
            if pair['stage'] == 'mechanism' and row['method'] == 'VP_VP':
                key = '_'.join(map(str, pair['task']))
                reference = source/'rows'/(key+'_PROFILE_DIAG.json')
                assert sha(reference) == audit['rows'][reference.name]
                old = read(reference)['result']
                np.testing.assert_allclose(row['result']['positions'], old['positions'], atol=1e-5, rtol=0)
                assert row['result']['stop_reason'] == old['stop_reason']
                assert row['result']['counts'] == old['counts']
                checked += 1
        identities[str(path)] = sha(path)
    assert checked == 8
    formal = study.create(False, 1.)
    fc = read(formal/'contract.json')
    assert fc['expected_solves'] == 1920
    # New formal layouts are fixed now, and are not reused pilot layouts.
    for item in fc['layouts']:
        assert item not in c['layouts']
    release = dict(ready=True, formal=str(formal), formal_contract_sha=sha(formal/'contract.json'),
                   pilot=str(out), pilot_contract_sha=sha(out/'contract.json'),
                   pilot_summary_sha=sha(out/'summary.json'), pilot_rows=identities,
                   native_vp_equivalence_cases=checked,
                   forecast_seconds=summary['forecast_formal_seconds'],
                   proposed_budget_hours=1.,
                   boundary_report=str(study.BASE/'review_boundary_20261020_122047_490569/report.json'),
                   status='PREPARED_NOT_RUN')
    write(formal/'release.json', release)
    write(study.BASE/'review_supplement_release.json',
          dict(out=str(formal), release_sha=sha(formal/'release.json')))
    print(json.dumps(release, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
