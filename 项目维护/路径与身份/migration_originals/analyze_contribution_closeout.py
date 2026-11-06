"""Read saved evidence only; attribute error and recovery without new solves."""
import csv
import hashlib
import json
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data = PAPER / 'data/assignment_cost_20261019'
    previous = json.loads((data / 'report.json').read_text(encoding='utf-8'))
    for name, digest in previous['source_files'].items():
        assert sha(Path(name)) == digest, name
    source = data / 'core_per_case.csv'
    with source.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    by_method = {}
    for row in rows:
        by_method.setdefault(row['method'], {})[row['case']] = row
    methods = ('AP_1_JOINT', 'AP_5_JOINT', 'PROFILE_DIAG')
    assert all(len(by_method[m]) == 2400 for m in methods)
    assert all(set(by_method[m]) == set(by_method[methods[0]]) for m in methods)
    groups = {}
    for group in ('all', 'layout4', 'other19'):
        groups[group] = {}
        for method in methods:
            selected = [r for r in by_method[method].values()
                        if group == 'all' or (int(r['layout']) == 3) == (group == 'layout4')]
            sse = sum(float(r['weak_error_m']) ** 2 for r in selected)
            groups[group][method] = dict(n=len(selected), weak_sse=sse,
                                        weak_rmse=(sse / len(selected)) ** 0.5)
    ap1 = by_method['AP_1_JOINT']
    ap5 = by_method['AP_5_JOINT']
    vp = by_method['PROFILE_DIAG']
    recovered = [r for key, r in ap5.items()
                 if r['both20'] == 'True' and ap1[key]['both20'] != 'True']
    failed = [r for r in ap5.values() if r['both20'] != 'True']

    def identities(items):
        return dict(observations=len(items),
                    layouts=sorted({int(r['layout']) + 1 for r in items}),
                    layout_repeat_count=len({(r['layout'], r['repeat']) for r in items}),
                    cases=[r['case'] for r in items])

    spectral = ROOT / 'outputs_e2e/unified/e2e_g10/r4/spectral_window_20261018_175147_595913/report.json'
    stop = ROOT / 'outputs_e2e/unified/e2e_g10/r4/joint_profile_pilot_20261018_190843_836406/analysis.json'
    spectral_report = json.loads(spectral.read_text(encoding='utf-8'))
    assert spectral_report['status'] == 'COMPLETED' and spectral_report['identities_unchanged']
    spectral_counts = {}
    for r in spectral_report['rows']:
        key = f"{r['method']}/{r['variant']}/{r['mode']}/{r['gap_db']}"
        entry = spectral_counts.setdefault(key, dict(n=0, success=0))
        entry['n'] += 1
        entry['success'] += bool(r['both20'])
    benefit = {g: groups[g]['AP_1_JOINT']['weak_sse'] - groups[g]['PROFILE_DIAG']['weak_sse']
               for g in groups}
    result = dict(sources={str(p): sha(p) for p in (source, data / 'report.json', spectral, stop)},
                  groups=groups, ap1_minus_vp_weak_sse=benefit,
                  ap5_recovered=identities(recovered), ap5_failed=identities(failed),
                  vp_failed=identities([r for r in vp.values() if r['both20'] != 'True']),
                  spectral_counts=spectral_counts,
                  stop_diagnostic=json.loads(stop.read_text(encoding='utf-8')),
                  scope='Saved results only; no positioning, no new sampling or timing.')
    out = PAPER / 'data/contribution_closeout_20261019'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('sources', 'spectral_counts', 'stop_diagnostic')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
