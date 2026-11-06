"""Read-only paired review plus a small independent fixed-K LP certificate."""
from datetime import datetime
import csv
import json
from collections import Counter

import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.spatial.distance import cdist

from geometry_association_gate import ROOT, sha, write
from flood_band_pair import pruned_lp

RUNS = {
    'development': {
        'candidate': 'consensus_development_20261029_223235_334946',
        'flood': 'flood_fixed_development_20261029_234717_293684',
        'xia': 'xia_common_development_20261029_235420_205614',
        'data': 'independent_backend_20261029_222111_026721'},
    'validation': {
        'candidate': 'consensus_validation_20261029_223520_495993',
        'flood': 'flood_fixed_validation_20261029_235358_862467',
        'xia': 'xia_common_validation_20261029_235513_195842',
        'data': 'consensus_validation_20261029_223520_495993'}}


def dense_fixed(cost, capacity, k):
    """Direct full-variable formulation of the author's CVX model, eta=0."""
    n, j = cost.shape
    size = n*j+n+j
    eq = np.zeros((n+1, size))
    ub = np.zeros((j+n*j, size))
    for i in range(n):
        eq[i, i*j:(i+1)*j] = 1
        eq[i, n*j+i] = 1
        for q in range(j):
            ub[q, i*j+q] = 1
            ub[j+i*j+q, i*j+q] = 1
            ub[j+i*j+q, n*j+n+q] = -1
    eq[n, n*j+n:] = 1
    sol = linprog(np.r_[cost.ravel(), np.ones(n), np.zeros(j)],
                  A_eq=eq, b_eq=np.r_[np.ones(n), k],
                  A_ub=ub, b_ub=np.r_[np.full(j, capacity), np.zeros(n*j)],
                  bounds=(0, 1), method='highs')
    assert sol.success
    return float(sol.fun)


def metric(errors):
    errors = np.asarray(errors)
    return dict(n=len(errors), rmse_m=float(np.sqrt(np.mean(errors**2))),
                max_m=float(errors.max()), p95_m=float(np.percentile(errors, 95)),
                success20=int(np.sum(np.all(errors <= 20, axis=1))))


def match_distance(a, b):
    d = cdist(a, b)
    i, j = linear_sum_assignment(d*d)
    return float(d[i, j].max())


def main():
    checks = {}
    sources = {}
    summary = []
    pairs = []
    traces = []
    certificates = []
    for seed in range(10):
        cost = np.random.default_rng(9130+seed).uniform(0, 4, (8, 6))
        # Include a cost exactly equal to trash; pruning must still be equivalent.
        cost[0, 0] = 1
        _, _, _, meta = pruned_lp(cost, capacity=3, known_k=2)
        direct = dense_fixed(cost, 3, 2)
        delta = abs(meta['objective']-2-direct)  # eta*sum(q) is constant 2.
        checks[f'fixed_k_dense_{seed}'] = delta < 1e-8
        certificates.append(dict(seed=seed, difference=delta, sparse=meta, dense_objective=direct))
    for phase, runs in RUNS.items():
        loaded = {}
        for name in ['candidate', 'flood', 'xia']:
            folder = ROOT/runs[name]
            file = (folder/'result.json').resolve(strict=True)
            sources[str(file)] = sha(file)
            data = json.loads(file.read_text(encoding='utf-8'))
            loaded[name] = {row['tag']: row for row in data['rows']}
            checks[f'{phase}/{name}/saved_checks'] = all(data['checks'].values())
            protocol = folder/'protocol.json'
            sources[str(protocol)] = sha(protocol)
            for p, digest in json.loads(protocol.read_text(encoding='utf-8'))['hashes'].items():
                checks[f'identity/{p}'] = sha(p) == digest
        checks[f'{phase}/tags'] = all(set(loaded[k]) == set(loaded['candidate']) for k in loaded)
        for path in ['LOS', 'MP']:
            tags = [tag for tag in loaded['candidate'] if tag.endswith('_'+path)]
            for name in ['candidate', 'flood']:
                for stage in ['geometry', 'los', 'mp']:
                    errors = [loaded[name][tag][stage+'_errors_m'] for tag in tags]
                    summary.append(dict(phase=phase, path=path, method=name, stage=stage,
                                        **metric(errors), complete=len(tags), total=len(tags)))
            for stage in ['terminal', 'retained']:
                present = [loaded['xia'][tag][stage+'_evaluation'] for tag in tags
                           if loaded['xia'][tag][stage+'_evaluation']['complete']]
                stats = metric([r['errors_m'] for r in present]) if present else {}
                summary.append(dict(phase=phase, path=path, method='xia', stage=stage,
                                    **stats, complete=len(present), total=len(tags),
                                    rmse_is_conditional=len(present) != len(tags)))
        for tag, current in loaded['candidate'].items():
            other = loaded['flood'][tag]
            xia = loaded['xia'][tag]
            checks[f'{phase}/{tag}/IQ'] = current['iq_sha'] == other['iq_sha'] == xia['iq_sha']
            pairs.append(dict(phase=phase, tag=tag, path=current['path'],
                initial_displacement_m=match_distance(current['initial'], other['initial']),
                final_displacement_m=match_distance(current['mp']['p'], other['mp']['p']),
                mp_mse_change=float(np.mean(np.square(other['mp_errors_m']))-np.mean(np.square(current['mp_errors_m']))),
                candidate_stop=current['mp']['reason'], flood_stop=other['mp']['reason'],
                lp_max_residual=max(other['association']['lp']['checks'].values())))
            file = ROOT/runs['data']/(tag.rsplit('_', 1)[0]+'_input.npz')
            sources[str(file)] = sha(file)
            truth = np.load(file)['p']
            for step in xia['steps']:
                points = np.asarray(step['points'])
                score = np.asarray(step['scores'])
                dec = step['decoded']
                threshold = dec.get('threshold')
                keep = score >= threshold if threshold is not None else np.zeros(len(score), dtype=bool)
                pre = cdist(points, truth).min(0)
                post = cdist(points[keep], truth).min(0).tolist() if keep.any() else [None, None]
                traces.append(dict(phase=phase, tag=tag, iteration=step['iteration'],
                    before_filter_m=pre.tolist(), after_filter_m=post, status=dec['status'],
                    lost_source_within20=bool(any(pre[i] <= 20 and (post[i] is None or post[i] > 20) for i in range(2))),
                    true_source_separation_m=float(np.linalg.norm(truth[0]-truth[1]))))
    assert all(checks.values()), [k for k, ok in checks.items() if not ok]
    out = ROOT/('neighbor_review_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    (out/'neighbor_pair_review.py').write_bytes((ROOT/'neighbor_pair_review.py').read_bytes())
    write(out/'audit.json', dict(checks=checks, source_hashes=sources, certificates=certificates))
    write(out/'summary.json', dict(summary=summary, paired=pairs, xia_trace=traces))
    with (out/'metrics.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        fields = list(dict.fromkeys(k for row in summary for k in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(summary)
    # No timing claims: these runs were not interleaved with the saved candidate.
    print('OUTPUT', out)
    print('CHECKS', len(checks), all(checks.values()))
    for row in summary:
        print(row)
    for phase in RUNS:
        p = [v for v in pairs if v['phase'] == phase]
        print(phase, 'max_initial_difference', max(v['initial_displacement_m'] for v in p),
              'max_final_difference', max(v['final_displacement_m'] for v in p),
              'stop_counts', dict(Counter(v['flood_stop'] for v in p)))
    print('XIA initial max coverage', max(max(v['before_filter_m']) for v in traces if v['iteration'] == 0))
    print('XIA filter-loss trials', len({(v['phase'], v['tag']) for v in traces if v['lost_source_within20']}))
    print('minimum source separation', min(v['true_source_separation_m'] for v in traces))


if __name__ == '__main__':
    main()
