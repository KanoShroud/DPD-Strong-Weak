"""Sequential score-test multipath reference, not an IARD reproduction."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE/'risk_step_20261030'))
from pilot import B, C, FS, UNIT, STATIONS, cols, real, sha, write, error, cases, fit as los_fit  # noqa: E402


def response(x, f, paths):
    p = x[:4].reshape(2, 2)*UNIT
    v = p[None]-STATIONS[:, None]
    dist = np.linalg.norm(v, axis=2)
    if dist.min() < 1e-8:
        raise ValueError('Station coincidence')
    g = np.ones((5, 2), complex)
    g[1:] = (x[4:12]+1j*x[12:20]).reshape(4, 2)
    phase = np.exp(-2j*np.pi*f[:, None, None]*dist[None]/C)
    base = phase*g[None]
    fac = np.ones_like(base)
    for i, (m, q) in enumerate(paths):
        a, b, delay = x[20+3*i:23+3*i]
        fac[:, m, q] += (a+1j*b)*np.exp(-2j*np.pi*f*delay/B)
    h = base*fac
    dh = np.zeros((*h.shape, len(x)), complex)
    for q in range(2):
        for xy in range(2):
            dh[:, :, q, 2*q+xy] = h[:, :, q]*(-2j*np.pi*f[:, None]/B)*(v[:, q, xy]/dist[:, q])[None]
        for m in range(1, 5):
            k = 4+2*(m-1)+q
            dh[:, m, q, k] = phase[:, m, q]*fac[:, m, q]
            dh[:, m, q, k+8] = 1j*dh[:, m, q, k]
    for i, (m, q) in enumerate(paths):
        k = 20+3*i
        z = base[:, m, q]*np.exp(-2j*np.pi*f*x[k+2]/B)
        dh[:, m, q, k] = z
        dh[:, m, q, k+1] = 1j*z
        dh[:, m, q, k+2] = z*(x[k]+1j*x[k+1])*(-2j*np.pi*f/B)
    return h, dh, base


def evaluate(x, f, y, paths, tangent=False):
    h, dh, base = response(x, f, paths)
    inv = np.linalg.pinv(h, rcond=1e-12)
    s = np.einsum('fqm,fm->fq', inv, y)
    r = y-np.einsum('fmq,fq->fm', h, s)
    ds = np.einsum('fmqk,fq->fmk', dh, s)
    jp = ds-np.einsum('fmq,fqk->fmk', h, np.einsum('fqm,fmk->fqk', inv, ds))
    if tangent:
        return real(r), cols(jp), (h, inv, s, base)
    j = -jp-np.einsum('fqm,fqk->fmk', inv.conj(), np.einsum('fmqk,fm->fqk', dh.conj(), r))
    return real(r)/np.linalg.norm(y), cols(j)/np.linalg.norm(y)


def project(x, n):
    z = x.copy()
    z[:4] = np.clip(z[:4], -3000/UNIT, 3000/UNIT)
    for k in range(20, len(z), 3):
        z[k:k+2] /= max(1., np.linalg.norm(z[k:k+2]))
        z[k+2] = np.clip(z[k+2], 0., n/FS*B)
    return z


def fit(x0, f, y, paths, n, budget=300, recenter=True):
    start = time.perf_counter()
    x0 = project(x0, n)
    r0, j0 = evaluate(x0, f, y, paths)
    scale = np.maximum(np.linalg.norm(j0, axis=0), 1e-10)
    cost_scale = max(.5*r0@r0, 1e-12)
    cache = [None, None]
    calls = 0
    def unpack(z):
        return x0+z/scale
    def obj(z):
        nonlocal calls
        if cache[0] is None or not np.array_equal(cache[0], z):
            cache[:] = [z.copy(), evaluate(unpack(z), f, y, paths)]
            calls += 1
        r, j = cache[1]
        return .5*r@r/cost_scale, (j.T@r)/scale/cost_scale
    def cons(z):
        v = unpack(z)[20:].reshape(-1, 3)
        return 1-np.sum(v[:, :2]**2, axis=1)
    def jac(z):
        x = unpack(z)
        a = np.zeros((len(paths), len(x)))
        for i in range(len(paths)):
            k = 20+3*i
            a[i, k:k+2] = -2*x[k:k+2]/scale[k:k+2]
        return a
    low = np.full(len(x0), -np.inf)
    high = -low.copy()
    low[:4], high[:4] = -3000/UNIT, 3000/UNIT
    low[22::3], high[22::3] = 0., n/FS*B
    opt = minimize(obj, np.zeros(len(x0)), jac=True, method='SLSQP',
                   bounds=list(zip((low-x0)*scale, (high-x0)*scale)),
                   constraints=[dict(type='ineq', fun=cons, jac=jac)] if paths else [],
                   options=dict(maxiter=budget, ftol=1e-12))
    x = unpack(opt.x)
    r, j = evaluate(x, f, y, paths)
    mapping = float(np.max(abs(x-project(x-j.T@r, n))))
    result = dict(x=x, p=x[:4].reshape(2, 2)*UNIT, cost=float(.5*r@r),
                  projected_gradient=mapping, stationary=mapping <= 1e-7,
                  iterations=int(opt.nit), success=bool(opt.success), message=str(opt.message),
                  calls=calls, seconds=time.perf_counter()-start,
                  closure_error=float(np.linalg.norm(x-project(x, n))))
    # One numerical recentering ONLY after optimizer tolerance termination.
    # Total iterations remain <=300; no localization truth used for this decision.
    if recenter and opt.success and not result['stationary'] and opt.nit < budget:
        xx, other = fit(x, f, y, paths, n, budget-int(opt.nit), False)
        attempts = [result.copy(), other.copy()]
        if other['cost'] <= result['cost']:
            x, result = xx, other
        result['attempts'] = attempts
        result['iterations'] = sum(a['iterations'] for a in attempts)
        result['calls'] = sum(a['calls'] for a in attempts)
        result['seconds'] = time.perf_counter()-start
    return x, result


def search(x, paths, f, y, n, use, sigma2, step):
    start = time.perf_counter()
    r, j, (h, inv, s, base) = evaluate(x, f, y, paths, True)
    u, singular, _ = np.linalg.svd(j, full_matrices=False)
    rank = int(np.sum(singular > singular[0]*1e-10))
    u = u[:, :rank]
    rp = r-u@(u.T@r)
    vectors = np.column_stack([u, rp])
    nf = len(f)*5
    vc = (vectors[:nf]+1j*vectors[nf:]).reshape(len(f), 5, -1)
    ph = np.einsum('fmq,fqm->fm', h, inv).real
    threshold = float(chi2.isf(.05/(step*(step+1)*10*(n-1)), 2))
    checks, banks = {}, []
    for m in range(5):
        for q in range(2):
            signal = base[:, m, q]*s[:, q]
            padded = np.zeros((n, vectors.shape[1]), complex)
            padded[use] = vc[:, m].conj()*signal[:, None]
            dot = np.sqrt(2)*np.fft.fft(padded, axis=0)
            cross = np.stack([dot.real, -dot.imag], axis=-1)
            norm2 = float(2*np.sum(abs(signal)**2*(1-ph[:, m])))
            gram = norm2*np.eye(2)[None]-np.einsum('nki,nkj->nij', cross[:, :rank], cross[:, :rank])
            val = np.linalg.eigvalsh(gram)
            valid = val[:, 0] > norm2*1e-10
            valid[0] = False
            amp = np.zeros((n, 2))
            amp[valid] = np.linalg.solve(gram[valid], cross[valid, rank, :, None])[..., 0]
            evidence = np.einsum('ni,ni->n', amp, cross[:, rank])/sigma2
            for delay in (1, n//3, n-1):
                bc = np.zeros((len(f), 5, 2), complex)
                bc[:, m, 0] = signal*np.exp(-2j*np.pi*f*delay/FS)
                bc[:, m, 1] = 1j*bc[:, m, 0]
                bc -= np.einsum('fmq,fqk->fmk', h, np.einsum('fqm,fmk->fqk', inv, bc))
                direct = cols(bc)
                checks[f'fft_{m}_{q}_{delay}'] = bool(np.allclose(vectors.T@direct, cross[delay], rtol=2e-7, atol=1e-8*np.linalg.norm(direct)))
            banks.append(np.column_stack([np.full(n-1, m), np.full(n-1, q), np.arange(1, n), valid[1:], evidence[1:], amp[1:]]))
    bank = np.vstack(banks)
    ids = np.flatnonzero((bank[:, 3] == 1) & (bank[:, 4] > threshold))
    pick = None
    if len(ids):
        a = bank[ids[np.argmax(bank[ids, 4])]]
        pick = dict(m=int(a[0]), q=int(a[1]), sample_delay=int(a[2]), evidence=float(a[4]),
                    rho=[float(a[5]), float(a[6])], delay=float(a[2]/FS*B))
    return pick, bank, dict(rank=rank, columns=len(x), threshold=threshold, checks=checks,
                            seconds=time.perf_counter()-start, candidates=len(bank))


def selftest(f, y, n, x):
    checks = {}
    rng = np.random.default_rng(2026103021)
    paths = [(1, 0), (1, 0), (0, 1), (3, 1)]
    xx = np.r_[x, .1, .07, 1.3, -.2, .1, 7.1, .05, -.12, 22.8, 0., 0., 3.9]
    smallf, smally = f[::29], y[::29]
    rr, jj = evaluate(xx, smallf, smally, paths)
    eps = 1e-5
    num = np.column_stack([(evaluate(xx+eps*np.eye(len(xx))[k], smallf, smally, paths)[0]-evaluate(xx-eps*np.eye(len(xx))[k], smallf, smally, paths)[0])/(2*eps) for k in range(len(xx))])
    checks['multi_path_exact_vp_jacobian'] = bool(np.linalg.norm(num-jj)/np.linalg.norm(num) < 2e-6)
    from pilot import evaluate as old_evaluate
    oldr, oldj = old_evaluate(x, f, y)
    newr, newj = evaluate(x, f, y, [])
    checks['los_equivalence'] = bool(np.allclose(oldr, newr) and np.allclose(oldj, newj))
    one = xx[:23]
    oldr, oldj = old_evaluate(one, f, y, paths[0])
    newr, newj = evaluate(one, f, y, paths[:1])
    checks['one_path_equivalence'] = bool(np.allclose(oldr, newr) and np.allclose(oldj, newj))
    worst = 0.
    for _ in range(100):
        a = rng.normal(size=len(xx))*5
        pa, pb = project(a, n), project(rng.normal(size=len(xx))*5, n)
        worst = max(worst, float((a-pa)@(pb-pa)))
    checks['convex_product_projection'] = worst < 1e-9
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full',action='store_true')
    args = parser.parse_args()
    if args.full:
        short = Path(json.loads((HERE/'short_latest.json').read_text(encoding='utf-8'))['path'])
        gate = json.loads((short/'result.json').read_text(encoding='utf-8'))
        assert gate['all_checks_pass'] and gate['all_order_stopped']
        assert sha(Path(__file__)) == json.loads((short/'preflight.json').read_text(encoding='utf-8'))['hashes'][str(Path(__file__).resolve())]
    selected = list(cases(not args.full))
    previous = BASE/'risk_step_20261030'
    frozen = Path(json.loads((previous/'full_latest.json').read_text(encoding='utf-8'))['path'])
    known = json.loads((frozen/'preflight.json').read_text(encoding='utf-8'))['hashes']
    files = [Path(__file__), HERE/'执行协议.md', HERE/'数值停止核查补充.md', previous/'pilot.py', BASE/'structural_probe.py']+[p for _, old, inp, _ in selected for p in (old, inp)]
    hashes = {str(p.resolve(strict=True)):sha(p) for p in files}
    assert all(Path(p).is_relative_to(BASE) for p in hashes)
    assert all(h == known[p] for p,h in hashes.items() if p in known)
    out = HERE/(('full_' if args.full else 'short_')+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    (out/'reference.py').write_bytes(Path(__file__).read_bytes())
    write(out/'preflight.json', dict(hashes=hashes, capacity=64, seed=2026103021, no_oracle=True))
    print('输出',out,flush=True)
    begin = time.perf_counter()
    rows, checks = [], {}
    for tag, old, inp, mode in selected:
        saved = json.loads(old.read_text(encoding='utf-8'))
        with np.load(inp, allow_pickle=False) as data:
            iq = (data['los']+(data['reflected'] if mode=='MP' else 0)).sum(axis=0)+data['noise']
        checks[tag+'/identity'] = hashlib.sha256(np.ascontiguousarray(iq).tobytes()).hexdigest() == saved['iq_sha']
        n = iq.shape[1]
        f0 = np.fft.fftfreq(n, 1/FS)
        use = abs(f0) <= B/2
        fft = np.fft.fft(iq, axis=1, norm='ortho')
        f, y = f0[use], fft[:, use].T
        sigma2 = float(np.mean(abs(fft[:, ~use])**2))
        x0 = np.r_[np.asarray(saved['initial']).ravel()/UNIT, np.asarray(saved['initial_v'])[:16]]
        x, los = los_fit(x0, f, y, n)
        if not rows:
            checks.update(selftest(f, y, n, x))
        assert all(checks.values())
        paths, history = [], []
        state = los
        stop = 'capacity'
        case_start = time.perf_counter()
        for step in range(1, 65):
            pick, bank, meta = search(x, paths, f, y, n, use, sigma2, step)
            checks.update({tag+f'/{step}/'+k:v for k,v in meta['checks'].items()})
            np.savez_compressed(out/f'{tag}_step{step}_candidates.npz',bank=bank)
            event = dict(step=step, pick=pick, search=meta)
            if pick is None:
                stop = 'no_detected_increment'
                history.append(event)
                break
            xx = np.r_[x, pick['rho'], pick['delay']]
            pp = paths+[(pick['m'],pick['q'])]
            xx, new = fit(xx, f, y, pp, n)
            event['fit'] = new
            event['decrease'] = new['cost'] <= state['cost']+1e-10
            history.append(event)
            checks[tag+f'/{step}/finite'] = bool(np.isfinite(xx).all())
            checks[tag+f'/{step}/feasible'] = new['closure_error'] < 1e-7
            if not event['decrease']:
                stop = 'numerical_non_decrease'
                break
            x, paths, state = xx, pp, new
            write(out/(tag+'_progress.json'),dict(history=history,paths=paths,state=state))
            print(tag,'路径',len(paths),'目标',round(state['cost'],7),'梯度',format(state['projected_gradient'],'.2g'),flush=True)
            if not state['stationary']:
                stop = 'numerical_nonstationary'
                break
            if not all(checks.values()):
                stop = 'correctness_failure'
                break
        # Evaluation only after all stopping and estimator decisions.
        with np.load(inp,allow_pickle=False) as data:
            truth = data['p']
        for ev in history:
            if 'fit' in ev:
                ev['errors_m'] = error(ev['fit']['p'],truth)
        powers = [float(x[k:k+2]@x[k:k+2]) for k in range(20,len(x),3)]
        row = dict(tag=tag,mode=mode,los=los,los_errors_m=error(los['p'],truth),
                   final=state,errors_m=error(state['p'],truth),paths=paths,stop=stop,
                   power_boundary=sum(v>=1-1e-10 for v in powers),
                   delay_alias=sum(v>n/FS*B/2 for v in x[22::3]),
                   incremental_seconds=time.perf_counter()-case_start,history=history)
        rows.append(row)
        write(out/(tag+'.json'),row)
        print(tag,stop,'误差',np.round(row['errors_m'],3),flush=True)
    checks['frozen_identity'] = all(sha(p)==h for p,h in hashes.items())
    result=dict(rows=rows,checks=checks,all_checks_pass=all(checks.values()),seconds=time.perf_counter()-begin,
                all_order_stopped=all(r['stop']=='no_detected_increment' for r in rows))
    write(out/'result.json',result)
    write(HERE/('full_latest.json' if args.full else 'short_latest.json'),dict(path=str(out)))
    print('完成',len(rows),'数值检查',result['all_checks_pass'],'阶数停止',result['all_order_stopped'],flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
