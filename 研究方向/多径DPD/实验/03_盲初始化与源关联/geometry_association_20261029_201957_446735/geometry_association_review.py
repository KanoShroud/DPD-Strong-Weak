"""Read saved ideal-lag diagnostics; no new localization or tuning."""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.spatial.distance import cdist
from geometry_association_gate import SOURCE, STATIONS, C, B, EXACT_M, sha, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--prefix', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve(strict=True)
    hashes = {str(p): sha(p) for p in [out/'result.json', out/'protocol.json']}
    result = json.loads((out/'result.json').read_text(encoding='utf-8'))
    protocol = json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    checks = {str(p): sha(p) == h for p, h in protocol['hashes'].items()}
    stats = []
    for li in range(2):
        for reflections in [0, 2]:
            rows = [r for r in result['rows'] if r['layout'] == li and r['reflections'] == reflections and r['mode'] == 'exact']
            err = np.array([r['metrics']['two_output_errors_m'] for r in rows])
            stats.append(dict(layout=li, reflections=reflections, n=len(rows),
                              exact_two=sum(r['metrics']['exact_count'] == 2 for r in rows),
                              both20=sum(r['metrics']['exact_coverage']['both20'] for r in rows),
                              numerical_rmse_m=float(np.sqrt(np.mean(err**2))),
                              max_error_m=float(err.max()),
                              false20=sum(r['metrics']['exact_false20'] for r in rows),
                              max_combination_count=max(r['association']['combinations'] for r in rows)))
    pressure = []
    for row in result['rows']:
        ar = np.load(out/(row['tag']+'.npz'))
        for index in row['association']['exact']:
            ranges = np.linalg.norm(ar['p'][index]-STATIONS, axis=1)
            # Independent check: actual range differences occur in each input set.
            checks[row['tag']+f'/strict_{index}'] = all(
                min(abs((ranges[m]-ranges[0])-ar[f'lags_0_{m}'])) <= 2*EXACT_M for m in range(1, 5))
        checks[row['tag']+'/nondegenerate_exhaustive'] = row['association']['ranks']['unresolved_rank_lt2'] == 0
        checks[row['tag']+'/finite'] = bool(all(np.isfinite(ar[k]).all() for k in ['p', 'alpha', 'edm', 'tdoa_rms_m']))
        if row['mode'] != 'exact':
            source_tag = row['tag'].split('_'+row['mode'])[0]
            truth = np.load(SOURCE/(source_tag+'.npz'))['p']
            d = cdist(STATIONS, truth)
            match = {f'{m}_{n}': [float(min(abs((d[n,q]-d[m,q])-ar[f'lags_{m}_{n}']))) for q in range(2)]
                     for m in range(5) for n in range(m+1, 5)}
            rankings = {}
            for name, indices in row['association']['top'].items():
                p = ar['p'][indices[:2]]
                if len(p) == 2:
                    e = cdist(p, truth)
                    errors = np.diag(e) if np.sum(np.diag(e)**2) <= np.sum(np.diag(e[:, ::-1])**2) else np.diag(e[:, ::-1])
                    rankings[name] = dict(unrefined_top2_rmse_m=float(np.sqrt(np.mean(errors**2))),
                                          output=p, errors_m=errors)
            pressure.append(dict(tag=row['tag'], mode=row['mode'], true_lag_nearest_residual_m=match,
                                 raw_legal_coverage=row['metrics']['raw_legal_coverage'],
                                 top=row['metrics']['top'], unrefined_rankings=rankings))
    prefix = json.loads((args.prefix/'result.json').read_text(encoding='utf-8'))
    by_tag = {r['tag']: r for r in result['rows']}
    for r in prefix['rows']:
        checks['prefix/'+r['tag']] = r['metrics'] == by_tag[r['tag']]['metrics']
    checks.update({'unchanged/'+p: sha(p) == h for p,h in hashes.items()})
    assert all(checks.values()), [k for k,v in checks.items() if not v]
    review = dict(summary=stats, stress=pressure, checks=checks,
                  completed=result['completed'], total_wall_seconds=result['seconds'],
                  stress_range_std_m=[.01*C/B,.05*C/B], source_sha=hashes)
    write(out/'review.json', review)
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(summary=stats, checks=len(checks), passed=all(checks.values())),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
