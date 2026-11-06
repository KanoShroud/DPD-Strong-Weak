"""Local position-update elimination audit, not a blind localization solver."""
from pathlib import Path
from datetime import datetime
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
from structural_probe import ROOT, STATIONS, LAYOUTS, C, B, response, sha, write
from alternative_fit import SOURCE
from solver_reliability import datasets, physical, evaluate


def channel(v):
    los = np.ones((5, 2), complex)
    los[1:] = v[:8].reshape(4, 2) + 1j*v[8:16].reshape(4, 2)
    ratio = v[16:36].reshape(5, 2, 2) + 1j*v[36:56].reshape(5, 2, 2)
    gain = np.concatenate([los[:, :, None], los[:, :, None]*ratio], axis=2)
    delay = np.concatenate([np.zeros((5, 2, 1)), np.cumsum(v[56:].reshape(5, 2, 2), axis=2)], axis=2)
    return gain, delay


def position_jac(v, p, f, y):
    """Exact profiled residual derivative, coordinates measured in C/B metres."""
    gain, delay = channel(v)
    h, _ = response(p, delay, gain, f)
    inverse = np.linalg.pinv(h, rcond=1e-12)
    coef = np.einsum('fqm,fm->fq', inverse, y)
    residual = y-np.einsum('fmq,fq->fm', h, coef)
    unit = p[None]-STATIONS[:, None]
    unit /= np.linalg.norm(unit, axis=2)[:, :, None]
    dh = np.zeros((len(f), 5, 2, 4), complex)
    for q in range(2):
        for xy in range(2):
            dh[:, :, q, 2*q+xy] = h[:, :, q]*(-2j*np.pi*f[:, None]/B)*unit[None, :, q, xy]
    ds = np.einsum('fmqv,fq->fmv', dh, coef)
    j = -ds+np.einsum('fmq,fqv->fmv', h, np.einsum('fqm,fmv->fqv', inverse, ds))
    j -= np.einsum('fqm,fqv->fmv', inverse.conj(), np.einsum('fmqv,fm->fqv', dh.conj(), residual))
    scale = np.sqrt(2)/np.linalg.norm(y)
    return scale*np.concatenate([j.real.reshape(-1, 4), j.imag.reshape(-1, 4)])


def valid(v):
    gain, delay = channel(v)
    return bool(np.all(np.sum(abs(gain[:, :, 1:])**2, axis=2) < abs(gain[:, :, 0])**2)
                and np.all(abs(gain[:, :, 0]) > 0)
                and np.min(np.diff(delay, axis=2)) >= 3
                and np.max(np.diff(delay, axis=2)) <= 20)


def reduced_step(r, jp, jn, tol):
    # Column scaling preserves the nuisance column space. No weak directions
    # are removed except the declared relative SVD numerical-rank threshold.
    scale = np.linalg.norm(jn, axis=0)
    assert np.all(scale > 0)
    n = jn/scale
    u, s, vh = np.linalg.svd(n, full_matrices=False)
    rank = int(np.sum(s > s[0]*tol))
    basis = u[:, :rank]
    a = jp-basis@(basis.T@jp)
    b = r-basis@(basis.T@r)
    dp = np.linalg.lstsq(a, -b, rcond=tol)[0]
    dn = -(vh[:rank].T@((basis.T@(r+jp@dp))/s[:rank]))/scale
    return dp, dn, a, rank


def main():
    start = time.perf_counter()
    out = ROOT/('reduction_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    frozen = SOURCE/'inputs.npz'
    assert frozen.resolve().is_relative_to(ROOT.resolve())
    assert sha(frozen) == json.loads((SOURCE/'result.json').read_text(encoding='utf-8'))['input_sha']
    paths = [frozen, Path(__file__), ROOT/'solver_reliability.py', ROOT/'structural_probe.py', ROOT/'alternative_fit.py']
    hashes = {str(p): sha(p) for p in paths}
    write(out/'protocol.json', dict(kind='LOCAL_PROFILED_GN_ELIMINATION_NOT_BLIND_LOCALIZATION',
        hashes=hashes, layouts=[0, 1], position_offsets_m=[.1, 1., 5., 20.],
        nuisance_perturbation_levels=[0., .01, .05], seed=20261027092,
        position_direction=[1., -.5, -.7, .4], position_unit_m=C/B,
        nuisance_coordinates='Actual reflection/LOS complex ratios; reference LOS gain1; same gap[3,20] and strict DRR>0.',
        rank_tolerances=[1e-8, 1e-10, 1e-12],
        steps=[1., .5, .25, .125, .0625],
        rule='One GN step only. Accept first strictly feasible nonincreasing objective; no truth-based selection.',
        oracle='Starts use true channel plus predetermined perturbations, and positions near truth. No blind success or convergence claim.',
        stopping='24 prescribed local tests only; no optimizer retries or new assumptions.'))
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    cache = {k: (x, p, f, y) for k, x, p, f, y in datasets()}
    checks = {}
    rows = []
    rng = np.random.default_rng(20261027092)
    for layout in [0, 1]:
        x0, _, f, y = cache[f'L{layout}_q0_d20_i0']
        v0 = physical(x0)
        p0 = LAYOUTS[layout].copy()
        perturb = rng.standard_normal(76)
        # Physical-coordinate perturbations, not new channel realizations.
        perturb[:16] *= np.maximum(abs(v0[:16]), .1)
        for level in [0., .01, .05]:
            v = v0+level*perturb
            if not valid(v):
                # No clipping or performance-conditioned repair; retain a failure.
                rows.append(dict(layout=layout, level=level, invalid_start=True))
                continue
            for offset in [.1, 1., 5., 20.]:
                tag = f'L{layout}_e{level}_d{offset}'
                displacement = np.array([1., -.5, -.7, .4]).reshape(2, 2)*offset
                p = p0+displacement
                r, jn, metrics = evaluate(v, p, f, y, 'physical')
                jp = position_jac(v, p, f, y)
                eps = 1e-5
                fd = []
                for j in range(4):
                    delta = np.zeros((2, 2))
                    delta.flat[j] = eps*C/B
                    fd.append((evaluate(v, p+delta, f, y, 'physical')[0]
                               - evaluate(v, p-delta, f, y, 'physical')[0])/(2*eps))
                fd = np.column_stack(fd)
                checks[tag+'/position_jacobian'] = np.linalg.norm(fd-jp)/np.linalg.norm(fd) < 1e-6
                steps = [reduced_step(r, jp, jn, tol) for tol in [1e-8, 1e-10, 1e-12]]
                dp, dn, a, rank = steps[1]
                whole = np.column_stack([jp, jn])
                scales = np.linalg.norm(whole, axis=0)
                full = np.linalg.lstsq(whole/scales, -r, rcond=1e-10)[0]/scales
                difference = np.linalg.norm(dp-full[:4])/max(1., np.linalg.norm(dp))
                checks[tag+'/full_position_step'] = difference < 1e-8
                checks[tag+'/linear_prediction'] = np.linalg.norm(whole@np.r_[dp, dn]-whole@full) < 1e-8
                checks[tag+'/rank_stability'] = len({s[3] for s in steps}) == 1
                checks[tag+'/step_stability'] = max(np.linalg.norm(s[0]-dp) for s in steps) < 1e-8*max(1., np.linalg.norm(dp))
                before = .5*float(r@r)
                trials = []
                chosen = 0.
                after = before
                for step in [1., .5, .25, .125, .0625]:
                    vv = v+step*dn
                    pp = p+step*dp.reshape(2, 2)*C/B
                    feasible = valid(vv)
                    value = None
                    if feasible:
                        rr = evaluate(vv, pp, f, y, 'physical')[0]
                        value = .5*float(rr@rr)
                    trials.append(dict(step=step, feasible=feasible, value=value))
                    if feasible and value <= before:
                        chosen = step
                        after = value
                        break
                updated = p+chosen*dp.reshape(2, 2)*C/B
                checks[tag+'/accepted_objective'] = after <= before
                row = dict(layout=layout, level=level, offset_m=offset,
                    nuisance_rank=rank, position_rank=int(np.linalg.matrix_rank(a, tol=np.linalg.svd(a, compute_uv=False)[0]*1e-10)),
                    step_equivalence_error=difference, jacobian_error=float(np.linalg.norm(fd-jp)/np.linalg.norm(fd)),
                    before_position_error_m=float(np.linalg.norm(p-p0)),
                    after_position_error_m=float(np.linalg.norm(updated-p0)),
                    before_objective=before, after_objective=after, accepted_step=chosen,
                    position_update_m=dp*C/B, nuisance_update=dn, trials=trials,
                    min_channel_rank_ratio=metrics['min_rank_ratio'])
                rows.append(row)
        # Unconstrained waveform LS is invariant to each source's common
        # complex scale. This is a gauge, not known gains or known waveform.
        gain, delay = channel(v0)
        h, _ = response(p0, delay, gain, f)
        a = np.array([2*np.exp(.4j), .7*np.exp(-.8j)])
        h2 = h*a[None, None]
        fit1 = np.einsum('fmq,fq->fm', h, np.einsum('fqm,fm->fq', np.linalg.pinv(h), y))
        fit2 = np.einsum('fmq,fq->fm', h2, np.einsum('fqm,fm->fq', np.linalg.pinv(h2), y))
        checks[f'L{layout}/scale_gauge'] = np.linalg.norm(fit1-fit2)/np.linalg.norm(y) < 1e-12
    checks.update({'hash/'+p: sha(p) == h for p, h in hashes.items()})
    write(out/'result.json', dict(rows=rows, checks={k: bool(v) for k, v in checks.items()},
                                 passed=bool(all(checks.values())), seconds=time.perf_counter()-start))
    print('checks', len(checks), 'pass', all(checks.values()), 'rows', len(rows))
    print('REPORT', out)
    assert all(checks.values()), [k for k, v in checks.items() if not v]


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
