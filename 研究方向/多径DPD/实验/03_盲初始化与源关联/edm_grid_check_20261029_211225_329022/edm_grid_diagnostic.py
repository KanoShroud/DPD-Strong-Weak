"""Posthoc numerical-resolution check, not an altered comparison version."""
from datetime import datetime
from pathlib import Path
import json
import time

import numpy as np
from threadpoolctl import threadpool_limits

from robust_geometry import ROOT, SOURCE, edm, select, PAIRS, C, sha, write


def main():
    # Deterministic prefix case: first single-source MP case with a >20m error.
    prefix=ROOT/'robust_geometry_20261029_210210_073890'
    name='L0_R0_MP_B_snr6'
    front='CC_ABS'
    files=[Path(__file__),ROOT/'robust_geometry.py',ROOT/'geometry_association_gate.py',
           SOURCE/(name+'.json'),prefix/(name+'_'+front+'.npz'),prefix/(name+'_'+front+'.json')]
    hashes={str(f.resolve(strict=True)):sha(f) for f in files}
    out=ROOT/('edm_grid_check_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    write(out/'protocol.json',dict(hashes=hashes,posthoc=True,not_used_to_choose_main_method=True,
          check='Original 10m grid vs 5m grid at first failing prefix single-source multipath CC case'))
    data=json.loads((SOURCE/(name+'.json')).read_text(encoding='utf-8'))
    lags=np.array([np.sort(np.array(data['extraction'][front][f'{m}_{n}']['lag_s'][:6])*C) for m,n in PAIRS])
    before=np.load(prefix/(name+'_'+front+'.npz'))
    start=time.perf_counter()
    p,s,meta=edm(lags,5.)
    elapsed=time.perf_counter()-start
    assert np.array_equal(meta['u'],before['EDM_RF_u'])
    delta=abs(s-before['EDM_RF_scores'])
    a=select(before['EDM_RF_p'],before['EDM_RF_scores'],1)
    b=select(p,s,1)
    checks={'same_selected_combination':a==b,
            'selected_location_stable':bool(np.max(abs(p[b]-before['EDM_RF_p'][a]))<.01),
            'all_combinations_objective_stable':bool(np.max(delta)<1e-7)}
    checks.update({f:sha(f)==h for f,h in hashes.items()})
    write(out/'result.json',dict(tag=name,front=front,seconds=elapsed,checks=checks,
          max_normalized_cost_difference=float(np.max(delta)),
          p10=before['EDM_RF_p'][a],p5=p[b],selected10=a,selected5=b,
          selected_distance_m=float(np.linalg.norm(p[b]-before['EDM_RF_p'][a]))))
    assert all(checks.values()),checks
    print('PASS',out,elapsed,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
