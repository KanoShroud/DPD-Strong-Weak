"""Saved-output audit, no optimizer rerun or changes to frozen initialization."""
from pathlib import Path
import argparse
import json
import warnings
import numpy as np
from threadpoolctl import threadpool_limits

from structural_probe import C, B, sha, write
from solver_reliability import evaluate, project
from position_reduction import position_jac
from blind_initialization import feasible, residual, signature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--prefix', type=Path, required=True)
    args = parser.parse_args()
    out = args.directory.resolve(strict=True)
    data = json.loads((out/'result.json').read_text(encoding='utf-8'))
    protocol = json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    assert data['complete_matrix'] and data['checks_passed']
    checks = {'hash/'+p: sha(p) == h for p, h in protocol['hashes'].items()}
    rows = []
    for row in data['rows']:
        z = np.load(out/(row['case']+'_input.npz'))
        y, f = z['y'], z['f']
        checks[row['case']+'/input'] = signature(y, f) == row['input_signature']
        branches = []
        for b in row['branches']:
            p, v = np.asarray(b['final_p']), np.asarray(b['final_v'])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                r, jn, metrics = evaluate(v, p, f, y, 'physical')
            j = np.column_stack([position_jac(v, p, f, y), jn])
            grad = j.T@r
            x = np.r_[p.ravel()/(C/B), v]
            target = x-grad
            target[:4] = np.clip(target[:4], -3000/(C/B), 3000/(C/B))
            target[4:] = project(target[4:])
            direction = target-x
            value = .5*float(r@r)
            probes = []
            # Interior convex combinations with the closure projection.
            # These are diagnostics, not replacements for saved estimates.
            for alpha in [.5, .1, .01, .001]:
                trial = x+alpha*direction
                pp, vv = trial[:4].reshape(2, 2)*(C/B), trial[4:]
                okay = feasible(pp, vv)
                cost = None
                if okay:
                    rr, _ = residual(vv, pp, f, y)
                    cost = .5*float(rr@rr)
                probes.append(dict(alpha=alpha, feasible=okay, cost=cost))
            checks[row['case']+'/'+b['pair_id']+'/saved_cost'] = bool(np.isclose(value, b['final']['cost'], rtol=1e-10, atol=1e-14))
            checks[row['case']+'/'+b['pair_id']+'/monotone'] = all(
                h['next_cost'] < h['cost'] for h in b['final']['history'] if h['accepted'])
            branches.append(dict(pair_id=b['pair_id'], value=value, feasible=feasible(p, v),
                min_gap=float(v[56:].min()), max_gap=float(v[56:].max()),
                near_gap_boundary=int(np.sum((v[56:]-3 < 1e-7) | (20-v[56:] < 1e-7))),
                reflection_power=metrics['max_reflection_power_ratio'],
                projected_mapping_inf=float(np.max(abs(direction))),
                directional_derivative=float(grad@direction), probes=probes,
                feasible_descent_found=any(t['feasible'] and t['cost'] < value-1e-12 for t in probes)))
        # Selection can be reconstructed without the evaluation fields.
        checks[row['case']+'/selection'] = row['chosen'] == min(range(len(row['branches'])), key=lambda i: row['branches'][i]['final']['cost'])
        chosen = row['branches'][row['chosen']]
        initial = next(p for p in row['pairs'] if p['id'] == chosen['pair_id'])
        rows.append(dict(case=row['case'], seconds=row['seconds'],
                         best_coarse_rmse=min(p['coarse_evaluation']['rmse_m'] for p in row['pairs']),
                         best_los_rmse=min(p['los_evaluation']['rmse_m'] for p in row['pairs']),
                         selected_initial=initial['los_evaluation'], final=row['evaluation'],
                         chosen=row['chosen'], branches=branches))
    prefix = json.loads((args.prefix/'L0_P0.json').read_text(encoding='utf-8'))
    full = data['rows'][0]
    checks['prefix_repeat_input'] = prefix['input_signature'] == full['input_signature']
    checks['prefix_repeat_output'] = prefix['output'] == full['output']
    checks['prefix_repeat_selection'] = prefix['chosen'] == full['chosen']
    checks['prefix_repeat_objectives'] = [b['final']['cost'] for b in prefix['branches']] == [b['final']['cost'] for b in full['branches']]
    checks['no_truth_file_access_in_solver'] = 'np.load' not in __import__('inspect').getsource(__import__('blind_initialization').solve)
    # The last syntactic check supplements, rather than replaces, manual
    # transitive input review of solve and its helpers.
    result = dict(rows=rows, checks=checks, passed=all(checks.values()),
                  source_sha=sha(Path(__file__)), result_sha=sha(out/'result.json'),
                  note='Four fixed projected-gradient probes per saved endpoint; no output replacement or localization rerun.')
    write(out/'audit.json', result)
    (out/'source'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    for row in rows:
        b = row['branches'][row['chosen']]
        print(row['case'], 'initial', row['selected_initial']['rmse_m'], 'final', row['final']['rmse_m'],
              'gap boundary', b['near_gap_boundary'], 'power', b['reflection_power'],
              'feasible descent', b['feasible_descent_found'])
    print('CHECKS', len(checks), result['passed'])
    assert result['passed']


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
