"""Frozen-support focused-risk audit. Not a calibrated risk selector."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OLD = BASE / 'unknown_order_20261030'
sys.path.insert(0, str(OLD))
import reference as ref  # noqa: E402
from reference_feasible import fit  # noqa: E402


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def latest(name):
    return Path(read(OLD / (name + '_latest.json'))['path']).resolve(strict=True)


def load_sources():
    cross, repair, unseen = (latest(k) for k in ('cross', 'feasible_repair', 'feasible_unseen'))
    rows = []
    files = [Path(__file__), HERE / '执行协议.md']
    for root in (cross, repair, unseen):
        pre = read(root / 'preflight.json')['hashes']
        for p, h in pre.items():
            assert Path(p).resolve(strict=True).is_relative_to(BASE)
            assert ref.sha(p) == h, p
        files += [root / 'preflight.json', root / 'result.json']
    for root in (repair, unseen):
        data = read(root / 'result.json')
        assert data['all_checks_pass']
        for row in data['rows']:
            tag = row['tag']
            inp = root / (tag + '_input.npz')
            if not inp.exists():
                inp = cross / (tag + '_input.npz')
            files += [root / (tag + '.json'), inp]
            rows.append((root, inp, row))
    for module in tuple(sys.modules.values()):
        p = getattr(module, '__file__', None)
        if p and Path(p).suffix == '.py' and Path(p).resolve().is_relative_to(BASE):
            files.append(Path(p).resolve())
    hashes = {str(p.resolve(strict=True)): ref.sha(p) for p in files}
    return rows, hashes


def load_iq(inp, row):
    # Never read physical path labels, truth position, or the generator's variance.
    with np.load(inp, allow_pickle=False) as data:
        iq = data['iq'].copy()
    assert hashlib.sha256(np.ascontiguousarray(iq).tobytes()).hexdigest() == row['iq_sha']
    n = iq.shape[1]
    f0 = np.fft.fftfreq(n, 1 / ref.FS)
    use = abs(f0) <= ref.B / 2
    ft = np.fft.fft(iq, axis=1, norm='ortho')
    return f0[use], ft[:, use].T, n, float(np.mean(abs(ft[:, ~use]) ** 2))


def state_check(x, f, y, paths, n):
    r, j = ref.evaluate(x, f, y, paths)
    pg = float(np.max(abs(x - ref.project(x - j.T @ r, n))))
    gaps = [float(np.min(3000 / ref.UNIT - abs(x[:4])))]
    for k in range(20, len(x), 3):
        gaps += [1 - float(np.linalg.norm(x[k:k+2])), float(x[k+2]), n/ref.FS*ref.B-float(x[k+2])]
    closure = float(np.linalg.norm(x - ref.project(x, n)))
    return dict(finite=bool(np.isfinite(x).all() and np.isfinite(r).all()),
                dimension_ok=len(x) == 20 + 3 * len(paths),
                projected_gradient=pg, stationary=pg <= 1e-7,
                closure=closure, feasible=closure < 1e-7,
                minimum_boundary_gap=min(gaps), interior=min(gaps) > 1e-8,
                normalized_cost=float(.5 * r @ r))


def raw(x, f, y, paths):
    r, j = ref.evaluate(x, f, y, paths)
    norm = np.linalg.norm(y)
    return r * norm, j * norm


def gradient(x, f, y, paths):
    r, j = raw(x, f, y, paths)
    return j.T @ r


def hessian(x, f, y, paths, step):
    h = np.empty((len(x), len(x)))
    for k in range(len(x)):
        d = np.zeros_like(x)
        d[k] = step
        h[:, k] = (gradient(x+d, f, y, paths)-gradient(x-d, f, y, paths))/(2*step)
    asym = float(np.linalg.norm(h-h.T) / max(np.linalg.norm(h), 1e-30))
    return (h+h.T)/2, asym


def rel(a, b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-30))


def influence(h, j):
    scale = np.maximum(np.linalg.norm(j, axis=0), 1e-12)
    hs = h / scale[:, None] / scale[None, :]
    ev = np.linalg.eigvalsh(hs)
    pd = bool(ev[0] > max(abs(ev[-1]), abs(ev[0])) * 1e-9)
    info = dict(positive_definite=pd, scaled_min_eigenvalue=float(ev[0]),
                scaled_max_eigenvalue=float(ev[-1]))
    if not pd:
        return None, info
    # Position rows only: no inverse or full data-by-data covariance materialized.
    e = np.eye(len(h))[:, :4]
    v = np.linalg.solve(hs, e / scale[:, None]) / scale[:, None]
    l = -ref.UNIT * v.T @ j.T
    info['solve_residual'] = rel(h @ v, e)
    return l, info


def diagnose(x, f, y, paths, n, sigma2, rng):
    begin = time.perf_counter()
    status = state_check(x, f, y, paths, n)
    result = dict(state=status, path_count=len(paths), derivative_checks={}, applicable=False)
    if not all(status[k] for k in ('finite', 'dimension_ok', 'feasible', 'stationary', 'interior')):
        result.update(reason='nonstationary_or_boundary_or_invalid', seconds=time.perf_counter()-begin)
        return result, None
    r, j = raw(x, f, y, paths)
    h1, a1 = hessian(x, f, y, paths, 1e-4)
    h2, a2 = hessian(x, f, y, paths, 5e-5)
    l1, condition1 = influence(h1, j)
    l2, condition2 = influence(h2, j)
    checks = dict(hessian_symmetry=max(a1, a2) < 1e-4,
                  hessian_step_stability=rel(h1, h2) < 1e-3)
    dy = (rng.normal(size=y.shape)+1j*rng.normal(size=y.shape)) / np.sqrt(2)
    dy *= np.linalg.norm(y)/np.linalg.norm(dy)
    eps = 1e-5
    numeric = (gradient(x,f,y+eps*dy,paths)-gradient(x,f,y-eps*dy,paths))/(2*eps)
    predicted = j.T @ ref.real(dy)
    mixed_error = rel(numeric, predicted)
    checks['mixed_observation_derivative'] = mixed_error < 1e-5
    hresponse, _, _ = ref.response(x, f, paths)
    wave = np.einsum('fqm,fm->fq', np.linalg.pinv(hresponse, rcond=1e-12), y)
    y0 = np.einsum('fmq,fq->fm', hresponse, wave)
    rz, jz = raw(x, f, y0, paths)
    hz, az = hessian(x, f, y0, paths, 1e-4)
    zero_error = rel(hz, jz.T@jz)
    checks['zero_residual_GN_limit'] = zero_error < 1e-4
    checks['zero_residual_construction'] = np.linalg.norm(rz)/np.linalg.norm(y0) < 1e-10
    result.update(derivative_checks=checks, hessian_asymmetry=[a1,a2],
                  hessian_relative_change=rel(h1,h2), mixed_relative_error=mixed_error,
                  zero_residual_GN_relative_error=zero_error, zero_hessian_asymmetry=az,
                  curvature_vs_residual_GN=rel(h2, j.T@j),
                  hessian_conditions=[condition1,condition2])
    if l1 is not None and l2 is not None:
        stability = rel(l1,l2)
        checks['influence_step_stability'] = stability < 1e-2
        result.update(influence_relative_change=stability,
                      local_variance_m2=float(sigma2*np.sum(l2*l2)/2),
                      local_newton_position_step_m=float(np.linalg.norm(l2@r)))
        result['applicable'] = all(checks.values())
    else:
        result['reason'] = 'nonpositive_or_near_singular_actual_hessian'
    result['seconds'] = time.perf_counter()-begin
    return result, l2 if result['applicable'] else None


def linear_check():
    rng = np.random.default_rng(2026110201)
    g = rng.normal(size=(18,9))
    t = np.linalg.pinv(g)[:4]
    l = np.linalg.pinv(g[:,:6])[:4]
    theta = rng.normal(size=9)
    sigma = .2
    y = g@theta[:,None] + sigma*rng.normal(size=(18,20000))
    score = (np.sum(((l-t)@y)**2,axis=0)+2*sigma**2*np.sum(l*t)-sigma**2*np.sum(t*t))/2
    loss = np.sum((l@y-theta[:4,None])**2,axis=0)/2
    diff = score-loss
    se = float(diff.std(ddof=1)/np.sqrt(len(diff)))
    return dict(repeats=len(diff), mean_score=float(score.mean()), mean_loss=float(loss.mean()),
                paired_difference=float(diff.mean()), paired_se=se,
                passed=abs(float(diff.mean())) < 6*se)


def local_pilot(x, f, y, paths, n, sigma2, rng):
    begin = time.perf_counter()
    center, base = fit(x.copy(),f,y,paths,n)
    diag, l = diagnose(center,f,y,paths,n,sigma2,rng)
    result = dict(base_fit=base, center_drift_m=float(np.linalg.norm((center[:4]-x[:4])*ref.UNIT)),
                  center_diagnostic=diag, solve_count=1)
    if l is None:
        result.update(reason='base_not_applicable',seconds=time.perf_counter()-begin)
        return result
    noise = np.sqrt(sigma2/2)*(rng.normal(size=y.shape)+1j*rng.normal(size=y.shape))
    delta = .1*noise
    plus, a = fit(center.copy(),f,y+delta,paths,n)
    minus, b = fit(center.copy(),f,y-delta,paths,n)
    observed = (plus[:4]-minus[:4])*ref.UNIT/2
    expected = l@ref.real(delta)
    response_error = rel(observed,expected)
    okay = a['stationary'] and b['stationary'] and response_error < .1
    result.update(plus=a,minus=b,solve_count=3,predicted_position_response_m=expected,
                  actual_centered_response_m=observed,response_relative_error=response_error,
                  local_response_pass=bool(okay),seconds=time.perf_counter()-begin)
    return result


def main(limit=4, skip_local=False):
    rows, hashes = load_sources()
    out = HERE/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    for filename in hashes:
        p = Path(filename)
        if p.suffix == '.py' or p == HERE/'执行协议.md':
            dst = out/'source'/p.relative_to(BASE)
            dst.parent.mkdir(parents=True,exist_ok=True)
            dst.write_bytes(p.read_bytes())
    ref.write(out/'preflight.json',dict(hashes=hashes,seed=2026110202,
               python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
               limit=limit, skip_local=skip_local,
               task='read_only_saved_risk_audit_and_fixed_support_local_pilot'))
    print('本轮输出',out,flush=True)
    begin = time.perf_counter()
    audit = []
    for root, inp, row in rows:
        f,y,n,sigma2 = load_iq(inp,row)
        paths=[]
        states=[dict(order=0,kind='los',**state_check(np.array(row['los']['x']),f,y,[],n))]
        for ev in row['history']:
            if 'fit' not in ev:
                continue
            pick=ev['pick']; paths.append((pick['m'],pick['q']))
            states.append(dict(order=len(paths),kind='increment',
                          **state_check(np.array(ev['fit']['x']),f,y,paths,n)))
        states.append(dict(order=len(row['paths']),kind='final',
                          **state_check(np.array(row['final']['x']),f,y,row['paths'],n)))
        audit.append(dict(tag=row['tag'],states=states,input=str(inp),record=str(root/(row['tag']+'.json'))))
        print('轨迹审计',row['tag'],len(states),'状态',flush=True)
    ref.write(out/'trajectory_audit.json',audit)
    linear = linear_check()
    ref.write(out/'linear_check.json',linear)
    assert linear['passed']
    rng=np.random.default_rng(2026110202)
    selected=('L20_L0','L20_L2','L20_L3','L20_HIGH')
    diagnostics=[]; local=[]
    for tag in selected[:limit]:
        root,inp,row=next(z for z in rows if z[2]['tag']==tag)
        f,y,n,sigma2=load_iq(inp,row)
        analyses={}; influences={}
        for name, state, paths in [('LOS',row['los'],[]),('WIDE',row['final'],row['paths'])]:
            x=np.array(state['x'])
            print('开始非零残差检查',tag,name,'参数',len(x),flush=True)
            d,l=diagnose(x,f,y,paths,n,sigma2,rng)
            analyses[name]=d; influences[name]=l
            ref.write(out/(tag+'_'+name+'_diagnostic.json'),d)
            if l is not None:
                np.savez_compressed(out/(tag+'_'+name+'_influence.npz'),L=l,x=x)
            print('局部公式可用',tag,name,d['applicable'],'秒',round(d['seconds'],2),flush=True)
        r=dict(tag=tag,models=analyses,estimated_sigma2=sigma2)
        if all(influences[k] is not None for k in ('LOS','WIDE')):
            lm,lw=influences['LOS'],influences['WIDE']
            distance=float(np.sum((np.array(row['los']['p'])-np.array(row['final']['p']))**2)/2)
            wide=float(sigma2*np.sum(lw*lw)/2)
            risk=distance+sigma2*np.sum(lm*lw)-wide
            r['plugin_scores_m2']=dict(LOS=risk,WIDE=wide,LOS_minus_WIDE=risk-wide,
                                      position_disagreement=distance,
                                      label='uncalibrated_nonlinear_selected_support_plugin')
        # Offline evaluation only after construction of all observation-only scores.
        r['evaluation_only_saved_losses_m2']=dict(LOS=float(np.mean(np.array(row['los_errors_m'])**2)),
                                                 WIDE=float(np.mean(np.array(row['errors_m'])**2)))
        diagnostics.append(r)
        ref.write(out/'diagnostics.json',diagnostics)
        if not skip_local and tag in ('L20_L0','L20_L3'):
            for name,state,paths in [('LOS',row['los'],[]),('WIDE',row['final'],row['paths'])]:
                if influences[name] is None:
                    local.append(dict(tag=tag,model=name,skipped=True,reason='original_not_applicable',solve_count=0))
                    continue
                print('局部固定支持短测',tag,name,flush=True)
                a=local_pilot(np.array(state['x']),f,y,paths,n,sigma2,rng)
                a.update(tag=tag,model=name,skipped=False)
                local.append(a)
                ref.write(out/'local_pilot.json',local)
                print('局部响应',tag,name,a.get('local_response_pass'),a.get('response_relative_error'),flush=True)
    identity=all(ref.sha(p)==v for p,v in hashes.items())
    output=dict(audit=audit,linear=linear,diagnostics=diagnostics,local=local,identity_unchanged=identity,
                seconds=time.perf_counter()-begin,new_blind_localization_runs=0,
                fixed_support_local_solves=sum(r['solve_count'] for r in local),
                risk_calibrated=False,formal_repeats_run=False)
    ref.write(out/'result.json',output)
    assert identity
    ref.write(HERE/'latest.json',dict(path=str(out)))
    print('完成：校准仍待验证；局部求解',output['fixed_support_local_solves'],'秒',round(output['seconds'],2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--limit',type=int,choices=(1,2,3,4),default=4)
    parser.add_argument('--skip-local',action='store_true')
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        main(args.limit,args.skip_local)
