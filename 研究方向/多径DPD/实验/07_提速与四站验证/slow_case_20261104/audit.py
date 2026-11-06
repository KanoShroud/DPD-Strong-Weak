"""Final saved-result audit; no additional optimizer run."""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
OFF=HERE/'offline_20261104_153033_128538'
RUN=HERE/'replay_20261104_153327_274965'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    checks={}
    for folder in (OFF,RUN):
        checks[f'{folder.name}/postflight']=read(folder/'postflight.json')['unchanged']
        for path,sha in read(folder/'identity.json')['hashes'].items():
            checks[f'{folder.name}/{path}']=hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
    checks.update(read(OFF/'offline_complete.json')['checks'])
    result=read(RUN/'results.json')
    checks.update(result['checks'])
    checks['paired_complete']=result['complete'] and len(result['rows'])==4
    original=[r for r in result['rows'] if r['policy']=='ORIGINAL']
    control=[r for r in result['rows'] if r['policy']=='CHECK_ORIGINAL_GATE']
    comparison=read(OFF/'case_stage_comparison.json')
    slow=next(r for r in comparison['totals'] if r['case']=='L3_R05')
    stage=next(r for r in comparison['stages'] if r['case']=='L3_R05' and r['K']==38)
    last=read(OFF/'trajectory.json')['summary'][-1]
    summary=dict(original_mean_seconds=float(np.mean([r['seconds'] for r in original])),
        control_mean_seconds=float(np.mean([r['seconds'] for r in control])),
        relative_inner_reduction=float(1-np.mean([r['seconds'] for r in control])/np.mean([r['seconds'] for r in original])),
        original_solver_calls=original[0]['solver_calls'],control_solver_calls=control[0]['solver_calls'],
        added_checks=control[0]['extra_gate_calls'],position_difference_m=control[0]['position_difference_m'],
        cost_difference=control[0]['cost']-original[0]['cost'],
        final_stage_share=stage['seconds']/slow['seconds'],
        earliest_gate_index=last['first_acceptable']['index'],
        tail_acceptable_states=last['acceptable_points'],audit_count=len(checks),all_pass=all(checks.values()))
    # Generation facts are read after all solvers; never passed to either replay.
    contract_path=HERE.parent/'risk_calibration_20261102/formal_20261102_181736_622970/contract.json'
    condition=read(contract_path)['payload']['contract']['conditions']['L3']
    bank=Path(condition['bank']).resolve(strict=True)
    before=hashlib.sha256(bank.read_bytes()).hexdigest()
    with np.load(bank,allow_pickle=False) as z:
        gaps=np.diff(np.sort(z['delays'],axis=-1),axis=-1)
        summary['generation_bank_min_candidate_gap_B']=float(gaps.min())
    assert hashlib.sha256(bank.read_bytes()).hexdigest()==before
    summary['generation_bank_sha']=before
    summary['generation_bank_path']=str(bank)
    assert summary['all_pass']
    (HERE/'最终审计.json').write_text(json.dumps(dict(summary=summary,checks=checks),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
