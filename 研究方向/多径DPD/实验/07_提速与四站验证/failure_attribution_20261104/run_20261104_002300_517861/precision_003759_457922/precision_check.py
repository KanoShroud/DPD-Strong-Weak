"""One remaining-budget LOS continuation, separately reported from frozen results."""
from datetime import datetime
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

import diagnose as d


def continue_one(state, iq):
    x = np.asarray(state['x'])
    f, y, n, _ = d.cal.spectral(iq)
    evaluate = d.engine.los_fit.__globals__['evaluate']
    before = d.objective_check(state, [], iq)
    cache = [None, None]
    def update(z):
        if cache[0] is None or not np.array_equal(cache[0], z):
            cache[:] = [z.copy(), evaluate(z, f, y, None)]
        return cache[1]
    used = state['iterations']
    extra = 0
    if not before['stationary'] and used < 300:
        lo = np.r_[np.full(4, -3000/d.ref.UNIT), np.full(16, -np.inf)]
        opt = least_squares(lambda z: update(z)[0], x, jac=lambda z: update(z)[1],
            bounds=(lo, -lo), x_scale='jac', max_nfev=300-used,
            ftol=1e-12, xtol=1e-12, gtol=1e-10)
        x, extra = opt.x, int(opt.nfev)
    r, j = update(x)
    cost = float(.5*r@r)
    pg = float(np.max(abs(x-d.ref.project(x-j.T@r, n))))
    closure = float(np.linalg.norm(x-d.ref.project(x, n)))
    ok = bool(np.isfinite(x).all() and np.isfinite(cost) and pg <= 1e-7 and closure < 1e-7
              and cost <= state['cost']+1e-12 and used+extra <= 300)
    return dict(p=x[:4].reshape(2, 2)*d.ref.UNIT, x=x, before=before,
                cost=cost, projected_gradient=pg, accepted=ok,
                original_nfev=used, extra_nfev=extra, total_nfev=used+extra,
                max_position_change_m=float(np.linalg.norm(
                    (x[:4]-np.asarray(state['x'])[:4]).reshape(2,2)*d.ref.UNIT, axis=1).max()))


def main():
    parent = Path(d.read(d.HERE/'latest.json')['path'])
    oldresult = d.read(parent/'result.json')
    assert oldresult['stage_B'] is not None
    hashes = d.read(parent/'preflight.json')['hashes']
    assert all(d.ref.sha(p) == h for p, h in hashes.items())
    out = parent/('precision_'+datetime.now().strftime('%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    extra_files = [Path(__file__), d.HERE/'共同精度核验补充.md', parent/'result.json']
    extra_files += list(parent.glob('HIGH_R*_alternative_LOS.json'))
    hashes.update({str(p):d.ref.sha(p) for p in extra_files})
    d.write(out/'preflight.json', dict(hashes=hashes))
    shutil.copyfile(Path(__file__), out/'precision_check.py')
    shutil.copyfile(d.HERE/'共同精度核验补充.md', out/'共同精度核验补充.md')
    started = time.perf_counter()
    blind = []
    for row in oldresult['stage_B']['rows']:
        tag = row['case']
        iq = d.iq_for(tag)
        a = continue_one(d.load(d.FORMAL/'cases'/tag/'solve.json')['los'], iq)
        b = continue_one(d.load(parent/(tag+'_alternative_LOS.json')), iq)
        pick = 2 if b['cost'] < a['cost'] else 1
        blind.append(dict(case=tag, first=a, second=b, selected=pick,
                          previous_selected=row['selected'], unchanged=pick==row['selected']))
    d.save(out/'decision_before_evaluation.json', blind)
    contract = d.load(d.FORMAL/'contract.json')['contract']
    evaluation = []
    for row in blind:
        truth = d.truth_for(row['case'], contract)
        evaluation.append(dict(case=row['case'], selected=row['selected'],
            first_rmse_m=np.sqrt(d.loss(row['first']['p'], truth)),
            second_rmse_m=np.sqrt(d.loss(row['second']['p'], truth))))
    assert all(d.ref.sha(p) == h for p, h in hashes.items())
    result = dict(rows=blind, evaluation=evaluation,
        all_precision_pass=all(r[k]['accepted'] for r in blind for k in ('first','second')),
        all_rankings_unchanged=all(r['unchanged'] for r in blind),
        max_position_change_m=max(r[k]['max_position_change_m'] for r in blind for k in ('first','second')),
        seconds=time.perf_counter()-started, identity_unchanged=True,
        frozen_outputs_replaced=False)
    d.write(out/'result.json', result)
    d.write(d.HERE/'precision_latest.json', dict(path=str(out)))
    print('共同精度核验', out, '全部精度', result['all_precision_pass'],
          '排序不变', result['all_rankings_unchanged'], '秒', result['seconds'], flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
