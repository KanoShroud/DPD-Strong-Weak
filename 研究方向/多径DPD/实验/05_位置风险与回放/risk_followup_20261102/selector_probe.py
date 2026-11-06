"""Three preregistered saved trajectories; no new localization or truth scoring."""
from datetime import datetime
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits
import audit as a

IDS = ('L0_R00', 'L2_R00', 'L3_R00')


def main():
    started = time.perf_counter()
    contract = a.load(a.FORMAL/'contract.json')['contract']
    a.cal.verify(contract)
    out = a.HERE/('selector_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    files = [Path(__file__), a.HERE/'audit.py', a.HERE/'执行协议.md']
    for ident in IDS:
        files += list((a.FORMAL/'cases'/ident).glob('*.json'))
        files.append(a.FORMAL/'cases'/ident/'input.npz')
    hashes = {str(p.resolve(strict=True)):a.ref.sha(p) for p in files}
    a.write(out/'preflight.json',dict(hashes=hashes,contract=contract,cases=IDS,
        rule='minimum_existing_plugin_risk_tie_lower_order',
        truth_not_used_for_scoring=True, new_localization_runs=0))
    (out/'source').mkdir()
    for p in [Path(__file__),a.HERE/'audit.py',a.HERE/'执行协议.md']:
        (out/'source'/p.name).write_bytes(p.read_bytes())
    results=[]
    print('结果目录',out,flush=True)
    for ident in IDS:
        case_start=time.perf_counter()
        folder=a.FORMAL/'cases'/ident
        solved=a.load(folder/'solve.json')
        wide=a.load(folder/'WIDE_risk.json')
        assert solved['stop']=='no_detected_increment' and wide['applicable']
        lw=np.array(wide['L']);sigma=wide['sigma2_est']
        pw=np.array(solved['final']['p'])
        with np.load(folder/'input.npz',allow_pickle=False) as z:
            f,y,n,s=a.cal.spectral(z['iq'])
        assert np.isclose(s,sigma,rtol=1e-12,atol=0)
        states=[(solved['los'],[])]
        paths=[]
        for ev in solved['history']:
            if 'fit' not in ev:
                continue
            assert ev['decrease']
            pick=ev['pick'];paths=paths+[(pick['m'],pick['q'])]
            states.append((ev['fit'],paths.copy()))
        assert len(paths)==len(solved['paths'])
        assert np.array_equal(states[-1][0]['x'],solved['final']['x'])
        rows=[]
        for state, paths in states:
            order=len(paths);x=np.array(state['x']);p=np.array(state['p'])
            diagnostic_seconds=0.
            if order==0:
                previous=a.load(folder/'LOS_risk.json')
                diag=previous['diagnostic'];l=np.array(previous['L']) if previous['applicable'] else None
                reused=True
            elif order==len(solved['paths']):
                diag=wide['diagnostic'];l=lw;reused=True
            else:
                dstart=time.perf_counter()
                diag,l=a.risk.diagnose(x,f,y,paths,n,sigma,np.random.default_rng(2026110209+order))
                diagnostic_seconds=time.perf_counter()-dstart;reused=False
            a.write(out/f'{ident}_K{order:02d}_diagnostic.json',diag)
            if diag.get('derivative_checks') and not all(diag['derivative_checks'].values()):
                raise AssertionError('Derivative correctness failure: '+ident+'/'+str(order))
            row=dict(order=order,position=p,applicable=bool(l is not None),
                     diagnostic_seconds=diagnostic_seconds,reused=reused,
                     original_diagnostic_seconds=diag.get('seconds'),
                     reason=diag.get('reason'),state=diag['state'])
            if l is not None:
                if a.norm2(p[::-1]-pw)<a.norm2(p-pw):
                    row.update(applicable=False,reason='estimated_source_label_change')
                else:
                    row['risk']=a.norm2(p-pw)/2+sigma*float(np.sum(l*lw))-sigma*a.norm2(lw)/2
            rows.append(row)
            a.write(out/(ident+'_scores.json'),rows)
            print(ident,'K',order,'applicable',row['applicable'],'risk',row.get('risk'),flush=True)
        eligible=[r for r in rows if r['applicable']]
        selected=min(eligible,key=lambda r:(r['risk'],r['order']))
        decision=dict(case=ident,selected_order=selected['order'],selected_risk=selected['risk'],
            candidate_count=len(rows),applicable_count=len(eligible),
            full_trajectory_coverage=len(rows)==len(eligible),scores=rows,
            decision_source='observations_only_before_truth_access',
            baseline_solve_seconds=solved['seconds'])
        a.write(out/(ident+'_decision_before_evaluation.json'),decision)
        # Truth is read only after decision artifact is committed.
        condition=ident.split('_')[0]
        with np.load(contract['conditions'][condition]['bank'],allow_pickle=False) as z:
            truth=z['p']
        for row in rows:
            p=np.array(row['position'])
            row['actual_loss_m2']=min(a.norm2(p-truth),a.norm2(p[::-1]-truth))/2
        decision.update(selected_loss_m2=selected['actual_loss_m2'],
            final_loss_m2=rows[-1]['actual_loss_m2'],los_loss_m2=rows[0]['actual_loss_m2'],
            evaluation_only_best_order=min(rows,key=lambda r:r['actual_loss_m2'])['order'],
            evaluation_only_best_loss_m2=min(r['actual_loss_m2'] for r in rows),
            seconds=time.perf_counter()-case_start)
        results.append(decision)
        a.write(out/(ident+'_result.json'),decision)
    a.cal.verify(contract)
    assert all(a.ref.sha(p)==h for p,h in hashes.items())
    a.write(out/'result.json',dict(rows=results,seconds=time.perf_counter()-started,
        identity_unchanged=True,new_localization_runs=0,
        independent_validation=False,formal_results_replaced=False))
    a.write(a.HERE/'latest_selector.json',dict(path=str(out)))
    print('完成',time.perf_counter()-started,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
