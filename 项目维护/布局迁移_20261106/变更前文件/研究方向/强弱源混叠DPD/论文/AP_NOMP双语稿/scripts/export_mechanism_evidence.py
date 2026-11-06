"""Export already-completed BNOMP diagnostics; never invoke a solver."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    pointer = read(ROOT/'outputs_e2e/unified/e2e_g10/r4/adaptation_review.json')
    source = Path(pointer['out']).resolve(strict=True)
    assert source.is_relative_to((ROOT/'outputs_e2e').resolve())
    assert sha(source/'contract.json') == pointer['contract_sha']
    audit = read(source/'final_audit_report.json')
    assert audit['passed'] and audit['solves'] == 40
    identities = {str(source/'rows'/name): value for name, value in audit['rows'].items()}
    assert all(sha(p) == v for p, v in identities.items())
    out = PAPER/'data/mechanism_evidence'
    out.mkdir(exist_ok=True)
    for name in ('contract.json', 'comparison_report.json', 'derivative_checks.json',
                 'final_audit_report.json'):
        data = (source/name).read_bytes()
        target = out/name
        if target.exists() and target.read_bytes() != data:
            raise RuntimeError('Refuse to replace a different evidence snapshot: '+name)
        target.write_bytes(data)
        identities[str(source/name)] = sha(source/name)
    report = read(out/'comparison_report.json')
    rows = []
    for name, item in report['summaries'].items():
        rows.append(dict(variant=name, cases=item['cases'], both20=item['both20'],
                         weak_rmse_m=item['weak_rmse'], seconds=item['mean_seconds'],
                         derivative_calls=item['mean_derivative_calls']))
    with (out/'mechanism_summary.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    curvature = []
    for item in read(out/'derivative_checks.json')['records']:
        fixed, vp = np.array(item['fixed_hessian']), np.array(item['profile_hessian'])
        rates = np.linalg.eigvals(np.eye(2)-np.linalg.solve(fixed, vp))
        assert np.max(abs(rates.imag)) < 1e-12
        curvature.append(dict(layout=item['task'][0], position=item['position'],
                              fixed_trial_m=item['fixed_trial_m'],
                              profile_trial_m=item['profile_trial_m'],
                              local_matrix_spectral_radius=float(max(abs(rates))),
                              note='Saved non-optimal diagnostic state, not a measured convergence rate.'))
    (out/'curvature_summary.json').write_text(json.dumps(curvature, ensure_ascii=False, indent=2), encoding='utf-8')
    assert all(sha(p) == v for p, v in identities.items())
    (out/'export_audit.json').write_text(json.dumps(dict(
        passed=True, source=str(source), identities=identities, new_localization_solves=0,
        scope='Eight development observations; diagnostic, not formal performance ranking.'),
        ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(exported=rows, curvature=curvature), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
