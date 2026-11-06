"""Saved-output audit and truth-labelled failure tracing; no new localization."""
import json
import hashlib
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'complete_neighbor_20261103'))
import native_xia as old  # noqa: E402
from storage import read, load, save  # noqa: E402


def main():
    out = Path(read(HERE/'latest.json')['path']).resolve(strict=True)
    post = read(out/'postflight.json')
    assert post['all_cases'] and post['completed']==32
    pre = read(out/'preflight.json')
    assert all(old.sha(Path(p))==h for p,h in pre['hashes'].items())
    contract = load(old.FORMAL/'contract.json')['contract']
    source_files = [HERE/'audit.py', out/'summary.json', out/'preflight.json', out/'postflight.json']
    source_files += sorted(out.glob('*/decision_before_truth.json'))+sorted(out.glob('*/evaluation.json'))
    identity = {str(p):old.sha(p) for p in source_files}
    rows, bad, labels = [], [], []
    summary = read(out/'summary.json')
    for case in old.cal.cases():
        folder = out/case['id']
        row = load(folder/'evaluation.json')
        dec = load(folder/'decision_before_truth.json')
        assert row['case']==case and not dec['truth_used']
        assert all(s['particle_count']==pre['design']['particles'] for s in dec['steps'])
        assert all(s['numerical_direct_error']<5e-11 for s in dec['steps'])
        rows.append(row)
        ee = row['evaluation']['Xia_scale_terminal']
        if ee['complete'] and max(ee['errors_m'])<=20:
            continue
        with np.load(contract['conditions'][case['condition']]['bank'],allow_pickle=False) as z:
            truth = z['p']
        with np.load(old.FORMAL/'cases'/case['id']/'input.npz',allow_pickle=False) as z:
            f,y,n,_ = old.cal.spectral(z['iq'])
        truth_scores = old.xia.scores(truth,old.STATIONS,y.T,f,n)
        trace=[]
        for s in dec['steps']:
            artifact=folder/f"round_{s['round']}_retained.npz"
            identity[str(artifact)]=old.sha(artifact)
            with np.load(artifact,allow_pickle=False) as z:
                points=z['points']; scores=z['scores']
            assert len(points)==s['saved_retained']
            assert len(scores)==len(points) and np.isfinite(scores).all()
            nearest=np.linalg.norm(points[:,None]-truth[None],axis=-1).min(axis=0) if len(points) else None
            threshold=s['output'].get('threshold')
            # Compare known truth only here, after the entire batch is immutable.
            threshold_truth=(truth_scores>=threshold).tolist() if threshold is not None else None
            rng=np.random.default_rng()
            rng.bit_generator.state=s['rng_before_generation']
            nearest_raw=np.full(2,np.inf)
            nearest_points=np.zeros((2,2))
            left=s['particle_count']
            offset=0
            pool_hash=hashlib.sha256()
            while left:
                take=min(left,65536)
                if s['round']==1:
                    pool=rng.uniform(-3000,3000,(take,2))
                else:
                    center=np.asarray(dec['steps'][s['round']-2]['output']['positions'])
                    owner=np.arange(offset,offset+take)%2
                    pool=center[owner]+rng.normal(size=(take,2))*10.
                pool_hash.update(np.ascontiguousarray(pool).tobytes())
                distance=np.linalg.norm(pool[:,None]-truth[None],axis=-1)
                ii=distance.argmin(axis=0)
                dd=distance[ii,np.arange(2)]
                for q in range(2):
                    if dd[q]<nearest_raw[q]:
                        nearest_raw[q]=dd[q];nearest_points[q]=pool[ii[q]]
                offset+=take;left-=take
            assert pool_hash.hexdigest()==s['points_sha']
            near_score=old.xia.scores(nearest_points,old.STATIONS,y.T,f,n)
            trace.append(dict(round=s['round'], status=s['output']['status'],
                true_score=truth_scores, true_score_survives_threshold=threshold_truth,
                threshold=threshold, raw_nearest_m=nearest_raw, retained_nearest_m=nearest,
                raw_nearest_score=near_score, nearest_scores_survive=(near_score>=threshold).tolist()
                if threshold is not None else None, full_output=s['output'].get('positions')))
        bad.append(dict(case=case, evaluation=ee, rounds=trace))
        labels.append(case['id'])
    assert all(old.sha(Path(p))==h for p,h in identity.items())
    assert all(old.sha(Path(p))==h for p,h in pre['hashes'].items())
    cover=np.asarray([r['initial_nearest_m'] for r in rows])
    payload=dict(completed=32, failure_ids=labels, failures=bad,
        hashes=identity, hashes_unchanged=True,
        initial_nearest_max_m=float(cover.max()), initial_source_pairs_covered=int((cover<=pre['design']['radius_m']).sum()),
        total_source_record_pairs=64, distinct_pool_seeds=8,
        new_localizations=0, truth_only_for_offline_evaluation=True,
        direct_score_error_max=max(s['numerical_direct_error'] for p in out.glob('*/decision_before_truth.json')
                                   for s in load(p)['steps']))
    save(out/'readback_and_failure_trace.json',payload)
    print(json.dumps({k:v for k,v in payload.items() if k not in ('failures','hashes')},ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
