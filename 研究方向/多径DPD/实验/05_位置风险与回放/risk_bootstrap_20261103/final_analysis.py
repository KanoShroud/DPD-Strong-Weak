"""Offline completed-run analysis; no fitting, selection-rule changes, or new data."""
from pathlib import Path
import importlib.util
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('bootstrap_final_readback', HERE/'runner.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def align(a, focus):
    a = np.asarray(a)
    return a[::-1] if np.sum((a[::-1]-focus)**2) < np.sum((a-focus)**2) else a


def main():
    out = Path(sys.argv[1]).resolve(strict=True) if len(sys.argv)>1 else Path(json.loads((HERE/'latest.json').read_text(encoding='utf-8'))['path'])
    contract = r.load(out/'contract.json')
    r.cal.check_paths(contract['hashes'])
    pred = r.load(out/'prediction_frozen.json')
    evaluation = r.load(out/'evaluation_only.json')
    assert pred['stopped'] is None
    assert sum(a['n_requested'] for a in pred['conditions'].values()) == 48
    evidence = [out/n for n in ('contract.json','donors.json','prediction_frozen.json','evaluation_only.json','postflight.json','readback_audit.json')]
    evidence += list((out/'replicates').glob('*/result.json'))
    evidence += list((out/'replicates').glob('*/solve.json'))
    hashes = {str(p): r.cal.ref.sha(p) for p in evidence}
    checks, result = {}, {}
    for cond in ('L0','L2','L3'):
        p = pred['conditions'][cond]
        with np.load(out/(cond+'_fitted_world.npz'), allow_pickle=False) as z:
            focus = z['focus'].copy()
        with np.load(contract['original']['conditions'][cond]['bank'], allow_pickle=False) as z:
            truth = align(z['p'], focus)
        reps = [r.load(f) for f in sorted((out/'replicates').glob(cond+'_B*/result.json'))]
        curves = np.asarray([r.curve(r.load(item['solve_file']),focus) for item in reps])
        checks[cond+'/all16_numerically_pass'] = len(reps)==16 and all(a['numerical_gate'] for a in reps)
        checks[cond+'/risk_recomputed'] = np.array_equal(curves.mean(0),p['risk_curve_m2'])
        risk = curves.mean(0)
        delta = curves-curves[:,-1:]
        own_p, own_q, correction = [], [], []
        label_ok = True
        for rep in range(1,8):
            solved = r.load(r.FORMAL/'cases'/f'{cond}_R{rep:02d}'/'solve.json')
            own_p.append(r.curve(solved,truth))
            own_q.append(r.curve(solved,focus))
            rr = r.states(solved)
            pw = align(rr[-1]['p'],focus)
            v = []
            for k in range(65):
                pk = align(rr[min(k,len(rr)-1)]['p'],focus)
                label_ok = label_ok and np.array_equal(pk,align(rr[min(k,len(rr)-1)]['p'],truth))
                label_ok = label_ok and np.array_equal(pw,align(rr[-1]['p'],truth))
                v.append(float(np.sum((focus-truth)*(pk-pw))))
            correction.append(v)
        own_p, own_q = np.asarray(own_p), np.asarray(own_q)
        dp = own_p-own_p[:,-1:]
        dq = own_q-own_q[:,-1:]
        predicted = delta.mean(0)
        observed = dp.mean(0)
        distribution = dq.mean(0)-predicted
        target = observed-dq.mean(0)
        checks[cond+'/all_receiver_metrics_recomputed'] = np.array_equal(own_p.mean(0),evaluation[cond]['risk_curve_m2'])
        checks[cond+'/exact_matched_transfer_decomposition'] = bool(np.allclose(observed-predicted, distribution+target,rtol=1e-12,atol=1e-12))
        if label_ok:
            checks[cond+'/stable_label_vector_identity'] = bool(np.allclose(target,np.mean(correction,0),rtol=1e-8,atol=1e-10))
        chosen = int(np.argmin(risk))
        oracle_fixed = int(np.argmin(own_p.mean(0)))
        ks = sorted(set([0, max(0,chosen-1), chosen, oracle_fixed,64]))
        probes = []
        for k in ks:
            probes.append(dict(cap=k,bootstrap_delta_m2=predicted[k],
                    bootstrap_paired_mc_se_m2=delta[:,k].std(ddof=1)/4,
                    receiver_delta_m2=observed[k],receiver_descriptive_se_m2=dp[:,k].std(ddof=1)/np.sqrt(7),
                    empirical_distribution_difference_m2=distribution[k],
                    empirical_reference_target_difference_m2=target[k]))
        result[cond] = dict(repeats=16, chosen_cap=chosen,
              exactly_tied_minimum_caps=np.flatnonzero(risk==risk.min()),
              every_bootstrap_smaller_cap_mean_nonnegative=bool(np.all(predicted>=0)),
              chosen_equals_wide_every_bootstrap=bool(np.all(curves[:,chosen]==curves[:,-1])),
              chosen_equals_wide_every_receiver=bool(np.all(own_p[:,chosen]==own_p[:,-1])),
              receiver_chosen_rmse_m=np.sqrt(own_p[:,chosen].mean()), receiver_wide_rmse_m=np.sqrt(own_p[:,-1].mean()),
              receiver_loss_differences_m2=dp[:,chosen],
              posthoc_best_fixed_cap=oracle_fixed,posthoc_best_fixed_cap_rmse_m=np.sqrt(own_p[:,oracle_fixed].mean()),
              retrospective_not_algorithm_score=True,stable_labels=label_ok, probes=probes,
              bootstrap_solver_seconds=sum(a['seconds'] for a in reps))
    checks['read_inputs_unchanged'] = all(r.cal.ref.sha(p)==h for p,h in hashes.items())
    r.cal.check_paths(contract['hashes'])
    assert all(checks.values()), checks
    r.save(out/'final_analysis.json',dict(conditions=result,checks=checks,all_pass=True,
                 input_hashes=hashes,source_sha=r.cal.ref.sha(Path(__file__)),
                 total_seconds=pred['cost']['seconds'],new_localization_runs=0,
                 no_new_selector=True,interpretation='Descriptive conditional transfer; no causal isolation or independent validation'))
    for cond,a in result.items():
        print(cond, 'cap',a['chosen_cap'],'ties',min(a['exactly_tied_minimum_caps']),max(a['exactly_tied_minimum_caps']),
              'receiver',a['receiver_chosen_rmse_m'],a['receiver_wide_rmse_m'])
        for row in a['probes']:
            print(row)
    print('analysis checks',len(checks),all(checks.values()))


if __name__=='__main__':
    main()
