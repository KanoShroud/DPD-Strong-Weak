"""Post-decision diagnostics. Oracle truth appears ONLY in attribution."""
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

from pilot import HERE, UNIT, FS, B, cols, real, evaluate, cases, sha, write


def summary(rows):
    answer = []
    for mode in ('LOS', 'MP'):
        sub = [r for r in rows if r['mode'] == mode]
        for method in ('LOS', 'RISK', 'FIT'):
            err = np.asarray([r['los_errors_m'] if method == 'LOS' else r['methods'][method]['errors_m'] for r in sub])
            base = np.asarray([r['los_errors_m'] for r in sub])
            delta = np.sum(base**2-err**2, axis=1)
            states = [r['los'] if method == 'LOS' else r['methods'][method]['result'] for r in sub]
            answer.append(dict(mode=mode, method=method, count=len(sub), rmse=float(np.sqrt(np.mean(err**2))),
                               source_rmse=np.sqrt(np.mean(err**2, axis=0)), p95=float(np.percentile(err,95)),
                               improved=int(np.sum(delta > 1e-8)), harmed=int(np.sum(delta < -1e-8)),
                               mean_stage_seconds=float(np.mean([s['seconds'] for s in states])),
                               stationary=sum(s['stationary'] for s in states),
                               iteration_limit=sum(s['iterations'] >= 300 for s in states),
                               added=0 if method == 'LOS' else sum(r['methods'][method]['added'] for r in sub),
                               alias_half_period=0 if method == 'LOS' else sum(r['methods'][method].get('alias_half_period',False) for r in sub),
                               cost_increase=0 if method == 'LOS' else sum(not r['methods'][method].get('cost_better_than_los',True) for r in sub)))
    return answer


def oracle_local(row, inp, rng):
    """Exact risk of two fixed linear estimators with arbitrary omitted mean."""
    pick = row['picks']['RISK']
    if pick is None:
        return dict(tag=row['tag'], not_selected=True), {}
    x = np.asarray(row['los']['x'])
    with np.load(inp, allow_pickle=False) as data:
        noiseless = (data['los']+(data['reflected'] if row['mode']=='MP' else 0)).sum(axis=0)
        iq = noiseless+data['noise']
        truth = data['p'].copy()
    n = iq.shape[1]
    f0 = np.fft.fftfreq(n, 1/FS)
    keep = abs(f0) <= B/2
    f = f0[keep]
    y = np.fft.fft(iq, axis=1, norm='ortho')[:, keep].T
    y0 = np.fft.fft(noiseless, axis=1, norm='ortho')[:, keep].T
    r, j, (h, inv, s) = evaluate(x, f, y, mean_tangent=True)
    un = np.linalg.svd(j[:, 4:], full_matrices=False)[0]
    a = j[:, :4]-un@(un.T@j[:, :4])
    lp = np.linalg.pinv(a, rcond=1e-10)
    m, q = pick['station'], pick['source']
    b = np.zeros((len(f), 5, 2), complex)
    b[:, m, 0] = h[:, m, q]*s[:, q]*np.exp(-2j*np.pi*f*pick['sample_delay']/FS)
    b[:, m, 1] = 1j*b[:, m, 0]
    b -= np.einsum('fmq,fqk->fmk', h, np.einsum('fqm,fmk->fqk', inv, b))
    br = cols(b)
    br -= un@(un.T@br)
    d = lp@br
    residual_b = br-a@d
    ca = np.linalg.inv(residual_b.T@residual_b)
    la = ca@residual_b.T
    mean = np.einsum('fmq,fq->fm', h, s)
    z = real(y0-mean)
    p0 = x[:4].reshape(2,2)*UNIT
    ii,jj = linear_sum_assignment(np.linalg.norm(p0[:,None]-truth[None],axis=2)**2)
    matched = np.empty_like(truth)
    matched[ii] = truth[jj]
    true_increment = (matched-p0).ravel()
    bias0 = UNIT*(lp@z)-true_increment
    correction = UNIT*d@(la@z)
    sigma2 = row['sigma2_estimated']
    variance_added = sigma2*UNIT**2*float(np.trace(d@ca@d.T))
    general = float(2*bias0@correction-correction@correction-variance_added)
    restricted = float(correction@correction-variance_added)
    noise_p = rng.normal(size=(20000,4))@np.linalg.cholesky(sigma2*UNIT**2*(lp@lp.T)).T
    noise_a = rng.normal(size=(20000,2))@np.linalg.cholesky(sigma2*ca).T
    err0 = bias0+noise_p
    err1 = bias0-correction+noise_p-UNIT*noise_a@d.T
    improvement = np.sum(err0**2-err1**2,axis=1)
    se = float(np.std(improvement,ddof=1)/np.sqrt(len(improvement)))
    checks = {row['tag']+'/arbitrary_omission_risk': abs(improvement.mean()-general)<=6*se+1e-10,
              row['tag']+'/orthogonal_noise_blocks': np.linalg.norm(lp@residual_b)<1e-7*max(1.,np.linalg.norm(lp)*np.linalg.norm(residual_b))}
    return dict(tag=row['tag'], oracle_attribution_only=True,
                restricted_model_theoretical_improvement_m2=restricted,
                general_theoretical_improvement_m2=general,
                conditional_linear_mc_improvement_m2=float(improvement.mean()), mc_se=se,
                baseline_bias_m=bias0, modeled_correction_m=correction,
                added_variance_m2=variance_added,
                naive_plugin_score_m2=pick['predicted_gain_m2'],
                actual_single_realization_improvement_m2=float(np.sum(np.asarray(row['los_errors_m'])**2-np.asarray(row['methods']['RISK']['errors_m'])**2)),
                interpretation='Conditional fixed-design linear diagnostic with known mean/truth, NOT estimator input or fresh nonlinear localization Monte Carlo'), checks


def main():
    start=time.perf_counter()
    pointer=json.loads((HERE/'full_latest.json').read_text(encoding='utf-8'))
    full=Path(pointer['path'])
    result=json.loads((full/'result.json').read_text(encoding='utf-8'))
    assert result['all_checks_pass'] and len(result['rows'])==48
    selected=[r for r in result['rows'] if r['tag'].startswith(('L20_R200_', 'L40_R300_'))]
    lookup={tag: inp for tag,_,inp,_ in cases(True)}
    paths=[full/'result.json', Path(__file__), HERE/'pilot.py',*lookup.values()]
    hashes={str(p.resolve(strict=True)):sha(p) for p in paths}
    out=HERE/('attribution_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    (out/'attribution.py').write_bytes(Path(__file__).read_bytes())
    write(out/'preflight.json',dict(hashes=hashes, seed=2026103013, oracle_truth_only_attribution=True))
    rows=[]
    checks={}
    rng=np.random.default_rng(2026103013)
    for row in selected:
        v,c=oracle_local(row,lookup[row['tag']],rng)
        rows.append(v)
        checks.update(c)
    checks['identity_unchanged']=all(sha(p)==h for p,h in hashes.items())
    stats=summary(result['rows'])
    output=dict(summary=stats, diagnostic=rows, checks=checks, all_pass=all(checks.values()),
                paired_solve_seconds=result['seconds'], wall_seconds=time.perf_counter()-start,
                mean_common_search_seconds=float(np.mean([r['search']['seconds'] for r in result['rows']])),
                noise_ratio_min=min(r['noise_estimate_ratio_audit'] for r in result['rows']),
                noise_ratio_max=max(r['noise_estimate_ratio_audit'] for r in result['rows']))
    write(out/'result.json',output)
    write(HERE/'attribution_latest.json',dict(path=str(out)))
    print(json.dumps(output,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else bool(x)))
    assert all(checks.values())


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
