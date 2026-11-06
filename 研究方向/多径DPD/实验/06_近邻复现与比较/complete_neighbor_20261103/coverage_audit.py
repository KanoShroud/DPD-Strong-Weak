"""Offline truth-labelled coverage/score diagnostic, never an estimator output."""
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import native_xia as run

HERE = Path(__file__).resolve().parent


def main():
    native = Path(json.loads((HERE/'latest_native.json').read_text(encoding='utf-8'))['path']).resolve(strict=True)
    assert json.loads((native/'postflight.json').read_text(encoding='utf-8'))['passed']
    pre = json.loads((native/'preflight.json').read_text(encoding='utf-8'))
    for name, digest in pre['hashes'].items():
        assert run.sha(Path(name)) == digest
    contract = run.load(run.FORMAL/'contract.json')['contract']
    files = [Path(__file__), HERE/'native_xia.py', native/'summary.json']
    files += sorted(native.glob('*/evaluation.json'))+sorted(native.glob('*/decision_before_truth.json'))
    files += [Path(v['bank']) for v in contract['conditions'].values()]
    hashes = {str(p.resolve(strict=True)): run.sha(p) for p in files}
    out = HERE/('coverage_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    run.save(out/'preflight.json', dict(hashes=hashes, native=str(native),
        truth_used_only_for_offline_diagnostic=True, new_localization_runs=0,
        score_cuts='four R00 cases, each true source x/y, -300:2:300 m',
        changes_to_estimator=False))
    rows, cuts = [], []
    offsets = np.arange(-300., 301., 2.)
    for case in run.cal.cases():
        decision = run.load(native/case['id']/'decision_before_truth.json')
        evaluation = run.load(native/case['id']/'evaluation.json')
        with np.load(contract['conditions'][case['condition']]['bank'], allow_pickle=False) as z:
            truth = z['p']
        initial = np.asarray(decision['output']['steps'][0]['points'])
        dist = np.linalg.norm(initial[:,None]-truth[None], axis=-1)
        nearest = dist.min(axis=0)
        assert np.allclose(nearest,evaluation['initial_uniform_coverage_m'])
        row = dict(case=case, nearest_m=nearest,
                   count_within20=(dist<=20).sum(axis=0),
                   count_within_c_over_B=(dist<=299792458./run.B).sum(axis=0))
        rows.append(row)
        if case['repeat'] != 0:
            continue
        with np.load(run.FORMAL/'cases'/case['id']/'input.npz',allow_pickle=False) as z:
            iq=z['iq']
        f,y,n,_=run.cal.spectral(iq)
        truth_score=run.xia.scores(truth,run.STATIONS,y.T,f,n)
        max_initial=float(np.max(decision['output']['steps'][0]['scores']))
        for source in range(2):
            for axis in range(2):
                grid=np.tile(truth[source],(len(offsets),1))
                grid[:,axis]+=offsets
                values=run.xia.scores(grid,run.STATIONS,y.T,f,n)
                center=len(offsets)//2
                assert np.isclose(values[center],truth_score[source],rtol=1e-12,atol=0)
                # Diagnostic center-height crossings, not FWHM of a proven true peak.
                half=truth_score[source]/2
                left_candidates=np.flatnonzero(values[:center]<half)
                right_candidates=np.flatnonzero(values[center+1:]<half)
                left=float(offsets[left_candidates[-1]]) if len(left_candidates) else None
                right=float(offsets[center+1+right_candidates[0]]) if len(right_candidates) else None
                cuts.append(dict(case=case,source=source,axis=axis,offsets_m=offsets,
                    scores=values,truth_score=float(truth_score[source]),
                    truth_score_over_initial_max=float(truth_score[source]/max_initial),
                    highest_cut_offset_m=float(offsets[np.argmax(values)]),
                    center_halfheight_left_m=left,center_halfheight_right_m=right,
                    center_halfheight_width_m=right-left if left is not None and right is not None else None))
    distances=np.asarray([r['nearest_m'] for r in rows])
    summary=dict(c_over_B_m=299792458./run.B,
        square_cell_scale_m=6000/np.sqrt(1000),
        ideal_interior_poisson_mean_nearest_m=6000/(2*np.sqrt(1000)),
        # Interior area approximation only; exact coverage measured above.
        ideal_interior_probability_within20=1-(1-np.pi*20**2/6000**2)**1000,
        nearest_m_min=float(distances.min()),nearest_m_median=float(np.median(distances)),
        nearest_m_max=float(distances.max()),
        source_record_pairs_with_particle_within20=int((distances<=20).sum()),
        source_record_pairs_with_particle_within_c_over_B=int((distances<=299792458./run.B).sum()),
        total_source_record_pairs=distances.size,
        independent_random_particle_seeds=8,
        condition_repeats_share_particle_seed=True,
        new_localization_runs=0, changes_to_estimator=False)
    run.save(out/'analysis.json',dict(summary=summary,rows=rows,cuts=cuts))
    assert all(run.sha(Path(name))==digest for name,digest in hashes.items())
    assert all(run.sha(Path(name))==digest for name,digest in pre['hashes'].items())
    run.save(out/'postflight.json',dict(hashes_unchanged=True,all_checks_pass=True))
    run.save(HERE/'latest_coverage.json',dict(path=str(out)))
    print(summary,flush=True)
    print(out,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
