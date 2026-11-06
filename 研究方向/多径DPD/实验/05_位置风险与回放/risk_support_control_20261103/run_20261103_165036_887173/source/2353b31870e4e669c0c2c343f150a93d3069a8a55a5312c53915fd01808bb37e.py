"""Single nonlinear reflection-addition audit; no adaptive-loop claim."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares, minimize, linear_sum_assignment
from scipy.stats import chi2
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
from structural_probe import B, C, STATIONS  # noqa: E402

UNIT = C/B
FS = 50e6


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, obj):
    def conv(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    p.write_text(json.dumps(obj, default=conv, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def real(z):
    return np.sqrt(2)*np.r_[z.real.ravel(), z.imag.ravel()]


def cols(z):
    return np.sqrt(2)*np.concatenate([z.real.reshape(-1, z.shape[-1]), z.imag.reshape(-1, z.shape[-1])])


def response(x, f, path=None):
    p = x[:4].reshape(2, 2)*UNIT
    g = np.ones((5, 2), complex)
    g[1:] = (x[4:12]+1j*x[12:20]).reshape(4, 2)
    vector = p[None]-STATIONS[:, None]
    distance = np.linalg.norm(vector, axis=2)
    if np.min(distance) < 1e-8:
        raise ValueError('Candidate at station; derivative undefined')
    direction = vector/distance[:, :, None]
    phase = np.exp(-2j*np.pi*f[:, None, None]*distance[None]/C)
    h = phase*g[None]
    factor = np.ones_like(h)
    if path is not None:
        m, q = path
        rho = x[20]+1j*x[21]
        extra = np.exp(-2j*np.pi*f*x[22]/B)
        factor[:, m, q] += rho*extra
        h *= factor
    dh = np.zeros((*h.shape, len(x)), complex)
    for q in range(2):
        for xy in range(2):
            dh[:, :, q, 2*q+xy] = h[:, :, q]*(-2j*np.pi*f[:, None]/B)*direction[None, :, q, xy]
        for m in range(1, 5):
            k = 4+2*(m-1)+q
            dh[:, m, q, k] = phase[:, m, q]*factor[:, m, q]
            dh[:, m, q, k+8] = 1j*dh[:, m, q, k]
    if path is not None:
        m, q = path
        base = phase[:, m, q]*g[m, q]*extra
        dh[:, m, q, 20] = base
        dh[:, m, q, 21] = 1j*base
        dh[:, m, q, 22] = base*rho*(-2j*np.pi*f/B)
    return h, dh


def evaluate(x, f, y, path=None, mean_tangent=False):
    h, dh = response(x, f, path)
    inv = np.linalg.pinv(h, rcond=1e-12)
    s = np.einsum('fqm,fm->fq', inv, y)
    r = y-np.einsum('fmq,fq->fm', h, s)
    ds = np.einsum('fmqk,fq->fmk', dh, s)
    pw_ds = ds-np.einsum('fmq,fqk->fmk', h, np.einsum('fqm,fmk->fqk', inv, ds))
    if mean_tangent:
        return real(r), cols(pw_ds), (h, inv, s)
    j = -pw_ds-np.einsum('fqm,fqk->fmk', inv.conj(), np.einsum('fmqk,fm->fqk', dh.conj(), r))
    norm = np.linalg.norm(y)
    return real(r)/norm, cols(j)/norm


def closure(x, n):
    z = x.copy()
    z[:4] = np.clip(z[:4], -3000/UNIT, 3000/UNIT)
    if len(z) == 23:
        z[20:22] /= max(1., np.linalg.norm(z[20:22]))
        z[22] = np.clip(z[22], 0., n/FS*B)
    return z


def fit(x, f, y, n, path=None):
    start = time.perf_counter()
    cache = [None, None]
    def update(z):
        if cache[0] is None or not np.array_equal(z, cache[0]):
            cache[:] = [z.copy(), evaluate(z, f, y, path)]
        return cache[1]
    if path is None:
        lo = np.r_[np.full(4, -3000/UNIT), np.full(16, -np.inf)]
        opt = least_squares(lambda z: update(z)[0], x, jac=lambda z: update(z)[1],
                            bounds=(lo, -lo), x_scale='jac', max_nfev=300,
                            ftol=1e-12, xtol=1e-12, gtol=1e-10)
        nit = int(opt.nfev)
    else:
        lo = np.r_[np.full(4, -3000/UNIT), np.full(18, -np.inf), 0.]
        hi = np.r_[np.full(4, 3000/UNIT), np.full(18, np.inf), n/FS*B]
        def obj(z):
            r, j = update(z)
            return .5*r@r, j.T@r
        def cj(z):
            v = np.zeros(23)
            v[20:22] = -2*z[20:22]
            return v
        opt = minimize(obj, closure(x, n), jac=True, method='SLSQP', bounds=list(zip(lo, hi)),
                       constraints=[dict(type='ineq', fun=lambda z: 1-z[20:22]@z[20:22], jac=cj)],
                       options=dict(maxiter=300, ftol=1e-12))
        nit = int(opt.nit)
    rr, jj = evaluate(opt.x, f, y, path)
    grad = jj.T@rr
    mapping = np.linalg.norm(opt.x-closure(opt.x-grad, n), ord=np.inf)
    return opt.x, dict(cost=float(.5*rr@rr), status=int(opt.status), success=bool(opt.success),
                       message=str(opt.message), iterations=nit, seconds=time.perf_counter()-start,
                       projected_gradient=float(mapping), stationary=bool(mapping <= 1e-7),
                       p=opt.x[:4].reshape(2, 2)*UNIT,
                       reflection_power=None if path is None else float(opt.x[20:22]@opt.x[20:22]),
                       normalized_delay=None if path is None else float(opt.x[22]), x=opt.x)


def search(x, f, y, n, use, sigma2):
    start = time.perf_counter()
    r, j, (h, inv, s) = evaluate(x, f, y, mean_tangent=True)
    un, sn, _ = np.linalg.svd(j[:, 4:], full_matrices=False)
    if sn[-1] <= sn[0]*1e-10:
        raise ValueError('LOS gain tangent singular')
    a = j[:, :4]-un@(un.T@j[:, :4])
    l0 = np.linalg.pinv(a, rcond=1e-10)
    uq, sq, _ = np.linalg.svd(j, full_matrices=False)
    if sq[-1] <= sq[0]*1e-10:
        raise ValueError('Full LOS tangent singular')
    rp = r-uq@(uq.T@r)
    vectors = np.column_stack([uq, l0.T, rp])
    count = len(f)*5
    vc = (vectors[:count]+1j*vectors[count:]).reshape(len(f), 5, -1)
    all_rows, numerical = [], {}
    candidates = 10*(n-1)
    threshold = float(chi2.isf(.05/candidates, 2))
    ph = np.einsum('fmq,fqm->fm', h, inv).real
    for m in range(5):
        for q in range(2):
            signal = h[:, m, q]*s[:, q]
            padded = np.zeros((n, vectors.shape[1]), complex)
            padded[use] = vc[:, m].conj()*signal[:, None]
            dot = np.sqrt(2)*np.fft.fft(padded, axis=0)
            cross = np.stack([dot.real, -dot.imag], axis=-1)
            norm2 = float(2*np.sum(abs(signal)**2*(1-ph[:, m])))
            residual_gram = norm2*np.eye(2)[None]-np.einsum('nki,nkj->nij', cross[:, :20], cross[:, :20])
            eigen = np.linalg.eigvalsh(residual_gram)
            valid = (eigen[:, 0] > norm2*1e-10)
            valid[0] = False
            ca = np.zeros_like(residual_gram)
            ca[valid] = np.linalg.inv(residual_gram[valid])
            btr = cross[:, 24]
            amplitude = np.einsum('nij,nj->ni', ca, btr)
            d = cross[:, 20:24]
            shift = np.einsum('nki,ni->nk', d, amplitude)*UNIT
            # j position coordinates are C/B; convert both risk terms to m^2.
            penalty = sigma2*np.einsum('nki,nij,nkj->n', d, ca, d)*UNIT**2
            evidence = np.einsum('ni,ni->n', amplitude, btr)/sigma2
            gain = np.sum(shift**2, axis=1)-2*penalty
            eligible = valid & (evidence > threshold) & (gain > 0)
            for delay_index in (1, n//3, n-1):
                b = np.zeros((len(f), 5, 2), complex)
                b[:, m, 0] = signal*np.exp(-2j*np.pi*f*delay_index/FS)
                b[:, m, 1] = 1j*b[:, m, 0]
                b -= np.einsum('fmq,fqk->fmk', h, np.einsum('fqm,fmk->fqk', inv, b))
                direct = cols(b)
                numerical[f'fft_{m}_{q}_{delay_index}'] = bool(np.allclose(vectors.T@direct, cross[delay_index], rtol=2e-7, atol=1e-8*np.linalg.norm(direct)))
            # Every candidate retained, not only favorable ones.
            for k in range(1, n):
                all_rows.append([m, q, k, int(valid[k]), evidence[k], gain[k], penalty[k],
                                 amplitude[k, 0], amplitude[k, 1], int(eligible[k])])
    bank = np.asarray(all_rows)
    picks = {}
    for name, col in [('RISK', 5), ('FIT', 4)]:
        allowed = (bank[:, 3] == 1) & (bank[:, 4] > threshold)
        if name == 'RISK':
            allowed &= bank[:, 5] > 0
        ids = np.flatnonzero(allowed)
        idx = None if not len(ids) else int(ids[np.argmax(bank[ids, col])])
        if idx is None:
            picks[name] = None
        else:
            row = bank[idx]
            picks[name] = dict(station=int(row[0]), source=int(row[1]), sample_delay=int(row[2]),
                               normalized_delay=float(row[2]/FS*B), evidence=float(row[4]),
                               predicted_gain_m2=float(row[5]), penalty_m2=float(row[6]),
                               rho_real=float(row[7]), rho_imag=float(row[8]))
    return picks, bank, dict(seconds=time.perf_counter()-start, threshold=threshold,
                            candidates=candidates, regular_count=int(np.sum(bank[:, 3])), checks=numerical,
                            scope='Estimated-design plug-in; nominal multiple-test threshold is not an exact nonlinear guarantee')


def error(p, truth):
    dist = np.linalg.norm(np.asarray(p)[:, None]-truth[None], axis=2)
    ii, jj = linear_sum_assignment(dist**2)
    e = np.zeros(2)
    e[jj] = dist[ii, jj]
    return e


def cases(short):
    for folder, inputs, layouts, reps in [
        ('consensus_development_20261029_223235_334946', 'independent_backend_20261029_222111_026721', range(20, 24), range(200, 203)),
        ('consensus_validation_20261029_223520_495993', 'consensus_validation_20261029_223520_495993', range(40, 44), range(300, 303))]:
        for li in layouts:
            for rep in reps:
                if short and (li not in (20, 40) or rep not in (200, 300)):
                    continue
                for mode in ('LOS', 'MP'):
                    tag = f'L{li}_R{rep}_{mode}'
                    yield tag, BASE/folder/(tag+'.json'), BASE/inputs/f'L{li}_R{rep}_input.npz', mode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    selected = list(cases(not args.full))
    if args.full:
        pointer = json.loads((HERE/'short_latest.json').read_text(encoding='utf-8'))
        short = json.loads((Path(pointer['path'])/'result.json').read_text(encoding='utf-8'))
        assert short['all_checks_pass'], 'Short correctness gate must pass'
    old_protocol = BASE/'contribution_check_20261030/task2_full_20261030_154144_272124/protocol.json'
    known = json.loads(old_protocol.read_text(encoding='utf-8'))['hashes']
    paths = [Path(__file__), HERE/'执行协议.md', BASE/'structural_probe.py', old_protocol]
    paths += [p for _, old, inp, _ in selected for p in (old, inp)]
    hashes = {str(p.resolve(strict=True)): sha(p) for p in paths}
    assert all(Path(p).is_relative_to(BASE) for p in hashes)
    assert all(h == known[p] for p, h in hashes.items() if p in known)
    out = HERE/(('full_' if args.full else 'short_')+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    (out/'pilot.py').write_bytes(Path(__file__).read_bytes())
    write(out/'preflight.json', dict(hashes=hashes, no_truth_in_solver=True, full=args.full))
    print('输出', out, flush=True)
    checks, rows = {}, []
    begin = time.perf_counter()
    for tag, old, inp, mode in selected:
        saved = json.loads(old.read_text(encoding='utf-8'))
        with np.load(inp, allow_pickle=False) as data:
            iq = (data['los']+(data['reflected'] if mode == 'MP' else 0)).sum(axis=0)+data['noise']
        identity = hashlib.sha256(np.ascontiguousarray(iq).tobytes()).hexdigest()
        checks[tag+'/iq_identity'] = identity == saved['iq_sha']
        n = iq.shape[1]
        freq = np.fft.fftfreq(n, 1/FS)
        use = abs(freq) <= B/2
        fft = np.fft.fft(iq, axis=1, norm='ortho')
        sigma2 = float(np.mean(abs(fft[:, ~use])**2))
        f, y = freq[use], fft[:, use].T
        x0 = np.r_[np.asarray(saved['initial']).ravel()/UNIT, np.asarray(saved['initial_v'])[:16]]
        x, los = fit(x0, f, y, n)
        picks, bank, meta = search(x, f, y, n, use, sigma2)
        np.savez_compressed(out/(tag+'_candidates.npz'), bank=bank)
        checks.update({tag+'/'+k: v for k, v in meta['checks'].items()})
        result = dict(tag=tag, mode=mode, sigma2_estimated=sigma2, los=los, search=meta, picks=picks, methods={})
        for method, pick in picks.items():
            if pick is None:
                result['methods'][method] = dict(added=False, result=los)
                continue
            initial = np.r_[x, pick['rho_real'], pick['rho_imag'], pick['normalized_delay']]
            initial = closure(initial, n)
            path = (pick['station'], pick['source'])
            if not rows:
                rr, jj = evaluate(initial, f, y, path)
                eps = 1e-5
                numeric = np.column_stack([(evaluate(initial+eps*np.eye(23)[k], f, y, path)[0]-evaluate(initial-eps*np.eye(23)[k], f, y, path)[0])/(2*eps) for k in range(23)])
                checks[method+'/exact_vp_jacobian'] = np.linalg.norm(numeric-jj)/np.linalg.norm(numeric) < 1e-6
            final, fitmeta = fit(initial, f, y, n, path)
            checks[tag+'/'+method+'/finite'] = bool(np.all(np.isfinite(final)))
            checks[tag+'/'+method+'/closure'] = bool(np.linalg.norm(final-closure(final, n)) < 1e-7)
            # Nonlinear improvement over LOS is not guaranteed; keep worse fits as failures, not re-tune.
            result['methods'][method] = dict(added=True, result=fitmeta,
                 cost_better_than_los=bool(fitmeta['cost'] <= los['cost']+1e-10),
                 strict_dominance=bool(fitmeta['reflection_power'] < 1),
                 strict_positive_delay=bool(fitmeta['normalized_delay'] > 0),
                 alias_half_period=bool(fitmeta['normalized_delay'] > n/FS*B/2))
        # Truth is accessed AFTER all estimator decisions, evaluation only.
        with np.load(inp, allow_pickle=False) as data:
            truth = data['p']
            noise = np.fft.fft(data['noise'], axis=1, norm='ortho')
            result['noise_estimate_ratio_audit'] = sigma2/float(np.mean(abs(noise)**2))
        result['los_errors_m'] = error(los['p'], truth)
        for item in result['methods'].values():
            item['errors_m'] = error(item['result']['p'], truth)
        rows.append(result)
        write(out/(tag+'.json'), result)
        print(tag, 'LOS', np.round(result['los_errors_m'], 3),
              'RISK', np.round(result['methods']['RISK']['errors_m'], 3), 'FIT', np.round(result['methods']['FIT']['errors_m'], 3), flush=True)
        if not all(checks.values()):
            break
    checks['frozen_identity'] = all(sha(p) == h for p, h in hashes.items())
    write(out/'result.json', dict(all_checks_pass=all(checks.values()), checks=checks,
                                rows=rows, seconds=time.perf_counter()-begin, new_observations=False))
    write(HERE/('full_latest.json' if args.full else 'short_latest.json'), dict(path=str(out)))
    print('完成',len(rows), '检查', all(checks.values()), flush=True)
    assert all(checks.values())


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
