"""Observation-only Morozov feasibility certificates on saved blind states."""
import sys
import time
from pathlib import Path
import numpy as np
from scipy.stats import f as f_distribution
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE/'numerical_repair_20261104'))
import stable  # noqa: E402
cfg, ad = stable.cfg, stable.ad
INPUT = BASE/'inside_pilot_20261104/batch_v2'
SAVED = BASE/'numerical_repair_20261104/paired_v1'


def main():
    out = HERE/'discrepancy_v1'
    out.mkdir(exist_ok=False)
    start = time.perf_counter()
    stable.inside.rr.check_hashes(cfg.read(SAVED/'identity.json'))
    hashes = ad.source_hashes()
    hashes.update({str(p): cfg.sha(p) for p in [Path(__file__).resolve(), HERE/'protocol.md']})
    rows = []
    for layout in range(100, 106):
        for condition in ['L0', 'L1_P075', 'L2_P075', 'L3_P150']:
            tag = f'IN_{layout}_R00_{condition}'
            src = INPUT/'cases'/tag
            for name, digest in cfg.read(src/'receipt.json').items():
                assert cfg.sha(src/name) == digest
            path = SAVED/(tag+'_ORTH_SAFE.json')
            hashes[str(path)] = cfg.sha(path)
            hashes[str(src/'iq.npz')] = cfg.sha(src/'iq.npz')
            with np.load(src/'iq.npz', allow_pickle=False) as z:
                iq = z['iq']
            previous = cfg.read(path)
            assert previous['iq_sha'] == cfg.digest(iq)
            result = previous['result']
            e = stable.Engine()
            freq, y, n, sigma2 = ad.b.fc.cal.spectral(iq)
            nf, m = y.shape
            dout = m*(n-nf)
            radius = m*nf*sigma2*f_distribution.isf(.05, 2*m*nf, 2*dout)
            reduced = (m-2)*nf*sigma2*f_distribution.isf(.05, 2*(m-2)*nf, 2*dout)
            states = {}
            for arm, state, paths in [('LOS', result['los'], []), ('MP', result['final'], result['paths'])]:
                x = np.asarray(state['x'])
                cost, grad = e.objective(x, freq, y, paths)
                h, inv, s, residual, *_ = e.core(x, freq, y, paths)
                rss = float(np.vdot(residual, residual).real)
                # Absolute received-reflection component norm in the adopted frequency model.
                gains = np.ones((m, 2), complex)
                gains[1:] = (x[4:4+2*(m-1)]+1j*x[4+2*(m-1):e.D]).reshape(m-1, 2)
                penalty = sum(abs(gains[mm, q])*np.linalg.norm(x[e.D+3*k:e.D+3*k+2])*np.linalg.norm(s[:, q]) for k, (mm, q) in enumerate(paths))
                feasible = float(np.max(abs(x-e.ref.project(x, n)))) < 1e-7
                states[arm] = dict(rss=rss, cost=cost, saved_cost=state['cost'], penalty=float(penalty),
                    observed_radius_ratio=rss/radius, fixed_H_radius_ratio=rss/reduced,
                    within_raw_radius=bool(rss<=radius and feasible), within_fixed_H_radius=bool(rss<=reduced and feasible),
                    feasible=feasible, pg=float(np.max(abs(x-e.ref.project(x-grad, n)))),
                    p=state['p'], paths=len(paths), normal=bool(state['stationary']),
                    objective_replay_pass=bool(abs(cost-state['cost'])<1e-8))
            row = dict(tag=tag, layout=layout, condition=condition, sigma2_observed=sigma2,
                radius2=radius, fixed_H_diagnostic_radius2=reduced, states=states,
                zero_penalty_feasible=states['LOS']['within_raw_radius'],
                estimator_inputs='IQ and saved blind decisions only; no bank metadata',
                mp_outer_normal=bool(result['normal']), source_sha=cfg.sha(path))
            cfg.write(out/(tag+'_decision.json'), row)
            rows.append(row)
    cfg.write(out/'blind_decisions.json', rows)
    cfg.write(out/'identity.json', hashes)
    # Evaluation-only truth access follows all observation-only decisions.
    evaluations = []
    for row in rows:
        bank = INPUT/'banks'/f'{row["layout"]}_00.npz'
        assert cfg.sha(bank) == cfg.read(bank.with_suffix('.json'))['sha']
        with np.load(bank, allow_pickle=False) as z:
            truth = z['p']
        er = dict(tag=row['tag'], condition=row['condition'], zero_penalty_feasible=row['zero_penalty_feasible'])
        for arm in ['LOS', 'MP']:
            errors = stable.inside.rr.evaluation(row['states'][arm]['p'], truth)
            er[arm] = dict(errors=errors, mse=float(np.mean(np.square(errors))))
        evaluations.append(er)
    subsets = {}
    for condition in ['ALL', 'L0', 'L1_P075', 'L2_P075', 'L3_P150']:
        for feasible_only in [False, True]:
            group = [r for r in evaluations if (condition=='ALL' or r['condition']==condition) and (not feasible_only or r['zero_penalty_feasible'])]
            subsets[f'{condition}_'+('LOS_feasible' if feasible_only else 'all')] = dict(count=len(group),
                rmse={arm:float(np.sqrt(np.mean([r[arm]['mse'] for r in group]))) if group else None for arm in ['LOS','MP']})
    cfg.write(out/'evaluation.json', evaluations)
    stable.inside.rr.check_hashes(hashes)
    summary = dict(completed=len(rows), zero_penalty_feasible=sum(r['zero_penalty_feasible'] for r in rows),
        multipath_zero_penalty_feasible=sum(r['zero_penalty_feasible'] and r['condition']!='L0' for r in rows),
        fixed_H_diagnostic_los_feasible=sum(r['states']['LOS']['within_fixed_H_radius'] for r in rows),
        all_replays_pass=all(r['states'][a]['objective_replay_pass'] and r['states'][a]['feasible'] for r in rows for a in ['LOS','MP']),
        subsets=subsets, elapsed_s=time.perf_counter()-start, new_localization_runs=0,
        calibration_guarantee=False, deployment_passed=False)
    cfg.write(out/'summary.json', summary)
    print(summary, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
