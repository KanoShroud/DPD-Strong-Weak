"""Independent range/Jacobian checks; no new localization experiment."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RUN = ROOT/'run_20261025_193552_712855'


def main():
    result = json.loads((RUN/'result.json').read_text(encoding='utf-8'))
    prior_path = next(Path(p) for p in result['identity'] if p.endswith('result.json'))
    prior = json.loads(prior_path.read_text(encoding='utf-8'))
    stations = np.array(prior['stations_m'])
    candidate = result['candidates'][0]
    positions = np.array(candidate['positions_m'])
    # Recompute reflected distances via mirrored receivers, not mirrored sources.
    residuals = []
    rows = []
    for q, source_id in enumerate([9, 12]):
        observed = result['observations_relative_length_m'][q]
        paths = []
        for record in sorted([r for r in prior['records'] if r['scene']=='two_AB'
                              and r['source_id']==source_id], key=lambda r:r['station_id']):
            paths.extend((record['station_id'], p) for p in sorted(record['paths'], key=lambda p:p['length_m']))
        labels = candidate['A_labels'] if q==0 else []
        b_index = 0
        offsets = []
        for index, (station_id, path) in enumerate(paths):
            if q==0:
                label = labels[index]
            elif path['kind']=='direct':
                label = -1
            else:
                label = candidate['B_labels'][b_index]
                b_index += 1
            anchor = stations[station_id].copy()
            if label>=0:
                plane = candidate['planes'][label]
                n = np.array(plane['normal'])
                anchor -= 2*(anchor@n-plane['offset_m'])*n
            length = np.linalg.norm(positions[q]-anchor)
            j = sum(1 for m, _ in paths[:index] if m==station_id)
            offsets.append(length-observed[station_id][j])
            rows.append((q, station_id, label))
        residuals.append(float(np.ptp(offsets)))
    # Independent numerical Jacobian, differently ordered parameters and step.
    vector = np.r_[positions.ravel()/1000,
                   [v for plane in candidate['planes'] for v in
                    [np.arctan2(plane['normal'][1],plane['normal'][0]),plane['offset_m']/1000]],0.,0.]
    def lengths(x):
        values = []
        for q, m, label in rows:
            anchor = stations[m]/1000
            if label>=0:
                theta, offset = x[6+2*label:8+2*label]
                n = np.array([np.cos(theta),np.sin(theta),0.])
                anchor = anchor-2*(anchor@n-offset)*n
            values.append(np.linalg.norm(x[3*q:3*q+3]-anchor)-x[10+q])
        return np.array(values)
    h = 3e-6
    J = np.column_stack([(lengths(vector+np.eye(12)[i]*h)-lengths(vector-np.eye(12)[i]*h))/(2*h)
                         for i in range(12)])
    nuisance = J[:,6:]
    profiled = J[:,:6]-nuisance@np.linalg.lstsq(nuisance,J[:,:6],rcond=1e-8)[0]
    values = np.linalg.svd(profiled,compute_uv=False)
    rank = int(np.sum(values>values[0]*1e-8))
    per_station = [[p for p in itertools.permutations([-1,0,1],len(group)) if -1 not in p[1:]]
                   for group in result['observations_relative_length_m'][0]]
    assignments = sum(sum(label==-1 for group in combo for label in group)>=3
                      for combo in itertools.product(*per_station))
    # Deliberately identify which inputs this audit uses as oracle information.
    checks = {
        'saved_input_hashes': all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v
                                  for p,v in result['identity'].items()),
        'mirrored_receiver_original_range_equivalence': max(residuals)<1e-6,
        'independent_fd_position_rank_six': rank==6,
        'profile_singular_values_match': bool(np.allclose(values,result['local_diagnosis']['shared_unknown_planes']['singular_values'],atol=1e-8,rtol=1e-7)),
        'enumeration_count': assignments*result['plane_sets_count']==result['A_assignments_tested'],
        'no_unresolved_branch': result['rank_deficient_A']==0 and result['unresolved_B_partitions']==0,
        'finite_full_path_witness': candidate['finite_witness']['status']=='full_first_order_paths_match',
    }
    audit = {'scope':'Independent numerical verification of one oracle diagnostic, not blind IQ recovery',
             'range_difference_errors_m':residuals,'profiled_singular_values':values.tolist(),
             'A_assignments_per_plane_set':assignments,'checks':checks,'pass':all(checks.values()),
             'audit_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'result_sha256':hashlib.sha256((RUN/'result.json').read_bytes()).hexdigest()}
    (RUN/'independent_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
    (RUN/'audit_result.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(audit,ensure_ascii=False))
    if not audit['pass']:
        raise RuntimeError('Independent audit failed')


if __name__=='__main__':
    main()
