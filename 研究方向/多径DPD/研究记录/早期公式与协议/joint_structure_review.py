"""Read-only diagnostic replay; no estimator retuning or new localization."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
from joint_structure_gate import (channel, projector, truth_x, los_map, GRID, score)
from structural_probe import sha, write


def local_information(x, f, pn):
    h, j = channel(x, f)
    projected = np.einsum('fmn,fnj->fmj', pn, j)/np.linalg.norm(h)
    matrix = np.concatenate([projected.real.reshape(-1, 40), projected.imag.reshape(-1, 40)])
    nuisance = matrix[:, 2:]
    norms = np.linalg.norm(nuisance, axis=0)
    keep = norms > max(float(norms.max()), 1)*1e-12
    nuisance = nuisance[:, keep]/norms[keep]
    u, sv, _ = np.linalg.svd(nuisance, full_matrices=False)
    answers = []
    for tol in [1e-6, 1e-8, 1e-10]:
        rank = int(np.sum(sv > sv[0]*tol))
        a = matrix[:, :2]-u[:, :rank]@(u[:, :rank].T@matrix[:, :2])
        val = np.linalg.svd(a, compute_uv=False)
        answers.append(dict(tolerance=tol, nuisance_rank=rank, position_singular=val,
                            position_rank=int(np.sum(val > max(val[0], 1)*1e-10)),
                            relative_min=float(val[-1]/max(np.linalg.norm(matrix[:, :2]), 1e-30))))
    return answers


def summarize(rows):
    stats = []
    for layout in [0, 1]:
        for paths in [1, 3]:
            cases = [r for r in rows if r['layout'] == layout and r['paths'] == paths]
            complete = [r for r in cases if r['metrics'] is not None]
            err = np.array([r['metrics']['source_errors_m'] for r in complete])
            branches = [b for r in cases for b in r['oracle']['branches'] if 'x' in b]
            local = [b for r in cases for b in r['oracle_local']]
            stats.append(dict(layout=layout, reflections=paths-1, samples=len(cases),
                              complete=len(complete), missing=len(cases)-len(complete),
                              both20=sum(r['metrics']['both20'] for r in complete),
                              conditional_RMSE_m=None if not len(complete) else float(np.sqrt(np.mean(err**2))),
                              conditional_P95_m=None if not len(complete) else float(np.quantile(err, .95)),
                              coverage20=sum(max(r['coarse_nearest_m']) <= 20 for r in cases),
                              branches=len(branches), feasible=sum(b['feasible'] for b in branches),
                              stationary=sum(b['stationary'] for b in branches),
                              iteration_limit=sum(b['status'] == 9 for b in branches),
                              local_controls=len(local), local20=sum(b['position_error'] <= 20 for b in local),
                              local_stationary=sum(b['stationary'] for b in local),
                              local_RMSE_m=float(np.sqrt(np.mean([b['position_error']**2 for b in local])))))
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--prefix', type=Path)
    args = parser.parse_args()
    out = args.output.resolve(strict=True)
    source = out/'result.json'
    identity = sha(source)
    result = json.loads(source.read_text(encoding='utf-8'))
    protocol = json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    checks = {str(p): sha(p) == h for p, h in protocol['hashes'].items()}
    assert all(checks.values()), checks
    begin = time.perf_counter()
    diagnostics = []
    for r in result['rows']:
        z = dict(np.load(out/(r['tag']+'.npz')))
        pn = projector(z['h'])
        each = []
        fifth = max(float(los_map(z['f'], pn, np.array(r['oracle']['candidates'])).max()), 0)
        for q in range(2):
            x = truth_x(z, q)
            nearest = int(np.argmin(np.sum((GRID-z['p'][q])**2, axis=1)))
            near = np.linalg.norm(GRID-z['p'][q], axis=1) <= 20
            best20 = float(z['map'][near].min())
            exact_los = float(los_map(z['f'], pn, z['p'][q:q+1])[0])
            # This is a truth-centered explanation only, never a candidate input.
            offsets = np.stack(np.meshgrid(np.arange(-10., 10.1, 2), np.arange(-10., 10.1, 2)), axis=-1).reshape(-1, 2)
            localmap = los_map(z['f'], pn, z['p'][q]+offsets)
            each.append(dict(source=q, exact_full_score=score(x, z['f'], pn)[0],
                             exact_LOS_score=exact_los, nearest_grid_score=float(z['map'][nearest]),
                             nearest_grid_rank=int(1+np.sum(z['map'] < z['map'][nearest])),
                             best_within20m_score=best20,
                             best_within20m_rank=int(1+np.sum(z['map'] < best20)),
                             fifth_coarse_score=fifth, truth_centered_points_beating_fifth=int(np.sum(localmap < fifth)),
                             truth_centered_count=len(localmap), information=local_information(x, z['f'], pn)))
        true_pair = np.stack([channel(truth_x(z, q), z['f'])[0] for q in range(2)], axis=2)
        checks[r['tag']+'/true_pair'] = bool(np.linalg.norm(projector(true_pair)-pn) < 1e-10)
        checks[r['tag']+'/full_zero'] = max(abs(t['exact_full_score']) for t in each) < 1e-12
        diagnostics.append(dict(tag=r['tag'], sources=each))
    if args.prefix:
        prefix = json.loads((args.prefix/'result.json').read_text(encoding='utf-8'))
        for a, b in zip(prefix['rows'], result['rows']):
            checks['prefix/'+a['tag']] = (a['identity'] == b['identity'] and
                                        a['oracle']['output'] == b['oracle']['output'])
    mix = [t for r in result['rows'] for t in r['mixing']]
    mixes = dict(tests=len(mix), columns=2*len(mix),
                 full_legal_columns=sum(c['all_constraints'] for t in mix for c in t['columns']),
                 excess_support=sum(t['exceeds_three_paths'] for t in mix),
                 geometry_rejected=sum(not t['earliest_geometry']['exact_feasible'] for t in mix),
                 observation_error_max=max(t['observation_error'] for t in mix),
                 projector_error_max=max(t['subspace_error'] for t in mix))
    info = [t for d in diagnostics for s in d['sources'] for t in s['information']]
    summary = dict(groups=summarize(result['rows']), mixing=mixes,
                   local_position_full_rank=sum(t['position_rank'] == 2 for t in info),
                   local_position_tests=len(info), min_relative_singular=min(t['relative_min'] for t in info))
    checks['result_unchanged'] = sha(source) == identity
    assert all(checks.values()), checks
    write(out/'review.json', dict(checks=checks, diagnostics=diagnostics, summary=summary,
                                 result_sha=identity, code_sha=sha(Path(__file__)), seconds=time.perf_counter()-begin,
                                 warning='No full-grid multipath profiling, no finite-IQ/noise/actual blind localization evidence.'))
    (out/'source'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
