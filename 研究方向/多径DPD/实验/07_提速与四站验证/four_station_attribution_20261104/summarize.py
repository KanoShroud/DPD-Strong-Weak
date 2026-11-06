"""Reproducible summary and Xia saved-trace audit, no new localization."""
import sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'four_station_physical_20261104'))
import scene as cfg  # noqa: E402
import adapters as ad  # noqa: E402


def main():
    run = HERE/'run_v3'
    identity = cfg.read(run/'identity.json')
    assert all(cfg.sha(p) == h for p, h in identity.items())
    old = cfg.read(run/'offline.json')
    pairs = [cfg.read(p) for p in sorted(run.glob('pair_[0-9][0-9].json'))]
    assert len(old) == 52 and len(pairs) == 22
    e = ad.Engine()
    physics = []
    for layout in cfg.layouts('pilot'):
        p = layout['p']; diff, jac = e.predicted(p)
        sv = np.linalg.svd(jac, compute_uv=False)
        d = np.linalg.norm(cfg.STATIONS[:, None]-p[None], axis=2)
        amp1 = np.sqrt(10**.6*cfg.NOISE_PSD*cfg.B/np.mean((cfg.C/(4*np.pi*1e9*d))**2, axis=0))[None]*cfg.C/(4*np.pi*1e9*d)
        amp24 = np.sqrt(layout['pt_w'])[None]*cfg.C/(4*np.pi*cfg.FC*d)
        physics.append(dict(layout=layout['index'], kind=layout['kind'],
            unweighted_tdoa_jacobian_condition=sv[:, 0]/sv[:, 1],
            two_source_tdoa_separation_in_inverse_bandwidth=abs(diff[0]-diff[1])/(cfg.C/cfg.B),
            carrier_recalibration_relative_error=float(np.max(abs(amp1-amp24)/amp24)),
            nominal_equal_tx_sourceA_snr_db=float(6+10*np.log10(layout['pt_w'][1]/layout['pt_w'][0]))))
    pilot = cfg.HERE/'pilot_20261104_182457_012169'
    xia_rows = []
    for path in sorted((pilot/'cases').iterdir()):
        ev = cfg.read(path/'evaluation.json')
        rr = cfg.read(path/'XIA_decision.json')['result']
        with np.load(pilot/'banks'/f"{ev['layout']:02d}_00.npz") as b:
            truth = b['p']
        with np.load(path/'iq.npz') as b:
            iq = b['iq']
        f, y, n, _ = ad.b.fc.cal.spectral(iq)
        score = ad.xia.scores(truth, cfg.STATIONS, y.T, f, n)
        last = rr['steps'][-1]['output']
        first = last.get('first_position')
        category = 'complete'
        if rr['p'] is None:
            if len(rr['steps']) > 1:
                category = 'second_round_one_cluster'
            elif last['trace'][0][1] == last['retained']:
                category = 'first_round_one_local_cluster'
            else:
                category = 'first_round_radius_expansion'
        xia_rows.append(dict(tag=ev['tag'], category=category, status=rr['status'],
            round=len(rr['steps']), truth_score_over_first_threshold=score/rr['steps'][0]['output']['threshold'],
            first_nearest_truth_m=None if first is None else float(np.linalg.norm(truth-first, axis=1).min()),
            final_trace=last.get('trace'), errors_m=ev['methods']['XIA']['errors_m']))
    gross = [r for r in old if r['stages']['MP'] > 100]
    summary = dict(counts=dict(saved=52, independent_worlds=4, new_full_mp_pairs=22,
        gross_MP_over100=len(gross), gross_already_GEO_over100=sum(r['stages']['GEO'] > 100 for r in gross),
        gross_with_both_refined_candidates_within20=sum(max(r['refined_nearest_m']) < 20 for r in gross)),
        diagnostics_not_changed_success_threshold=True,
        solver_seconds=sum(r['seconds'] for r in pairs), physics=physics,
        xia_counts={k: sum(r['category'] == k for r in xia_rows) for k in sorted({r['category'] for r in xia_rows})},
        source_hashes={str(Path(__file__).resolve()): cfg.sha(__file__), str(HERE/'numerical_audit.py'): cfg.sha(HERE/'numerical_audit.py')},
        formal_allowed=cfg.read(cfg.HERE/'release.json')['formal_allowed'])
    cfg.write(HERE/'xia_trace_audit.json', xia_rows)
    cfg.write(HERE/'summary.json', summary)
    print(summary)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
