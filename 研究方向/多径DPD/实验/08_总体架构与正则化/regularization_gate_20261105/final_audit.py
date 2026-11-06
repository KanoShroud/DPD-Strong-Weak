"""Read-back integrity, counts and mathematical outcome consistency."""
import numpy as np
from discrepancy import HERE,cfg,stable


def main():
    folders=['discrepancy_v1','conditional_v1','ranking_v1','outer_v1','outer_precision_v1','fixed_support_v1','zero_plateau_v1']
    hashes=0
    for folder in folders:
        identity=cfg.read(HERE/folder/'identity.json');stable.inside.rr.check_hashes(identity);hashes+=len(identity)
    rows=cfg.read(HERE/'fixed_support_v1/decisions.json')
    assert len(rows)==12
    assert all(r['result']['feasible'] for r in rows)
    assert all(np.all(np.diff([t['value'] for t in r['result']['trace']])<=1e-12) for r in rows)
    assert all(r['result']['status']!='projected_stationary' for r in rows)
    evaluated=cfg.read(HERE/'fixed_support_v1/evaluation.json');summary=cfg.read(HERE/'fixed_support_v1/summary.json')
    for key in ['before','after']:
        value=float(np.sqrt(np.mean([np.square(r[key]) for r in evaluated])))
        assert abs(summary[key+'_rmse']-value)<1e-10
    plateau=cfg.read(HERE/'zero_plateau_v1/decisions.json')
    assert len(plateau)==192 and sum(r['feasible'] for r in plateau)==176
    assert all(r['penalty']==0 for r in plateau)
    assert cfg.read(HERE/'html_qa.json')['passed']
    cfg.write(HERE/'final_audit.json',dict(passed=True,stage_identities=len(folders),hash_entries_checked=hashes,
        fixed_support_monotone=True,physical_feasibility=True,counts_and_rmse_recomputed=True,
        zero_plateau_recomputed=True,html_qa=True))
    print('PASS',len(folders),'stages',hashes,'hash entries')


if __name__=='__main__':main()
