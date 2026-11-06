"""G11-P1 bounded attribution on saved IQ. Oracle branches are NOT algorithms.

No frozen solver, data, scene, success radius, or paper is modified. The local
refiner deliberately audits accuracy more tightly; it is not a timed contender.
"""
import argparse
import hashlib
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy.optimize import minimize
from scipy.signal.windows import hann

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.g11.power_gap_review import raw, read, write, csv_write, resolve, PROVENANCE
from 统一模型代码.gates.g10.r2.band.physics import Model
from 统一模型代码.gates.g10.r4.paper_checks.joint_profile_math import derivatives, check
from 统一模型代码.gates.g10.r4.paper_checks.assignment_metrics import evaluate, self_test, REVISION
from 统一模型代码.gates.g8.physics import C

P0 = ROOT/'outputs_e2e/unified/e2e_g11/p0_power_gap_20261022_190417_148954'
DIFFICULT = (13, 14, 15)
CONTROLS = (0, 4, 8, 12)  # first geometry in each category; NOT selected by errors
TASKS = [(l, r, g) for l in DIFFICULT+CONTROLS for r in range(1000, 1005) for g in (0, 15)]
VARIANTS = ('original_saved', 'original_truth', 'no_noise_truth',
            'correct_A_truth', 'correct_B_truth', 'correct_AB_truth')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def set_x(m, x):
    m.x = torch.as_tensor(x, device=m.device, dtype=torch.complex128)
    m.energy = m.x.abs().square().sum()


def matched(truth, positions):
    metric = evaluate(truth, positions)
    p = np.asarray(positions)[[j for _, j in sorted(metric['assignment'])]]
    return p, metric


def local(m, start):
    """Same reduced LS, no global search. Tighter audit stop independent of truth.

    L-BFGS-B followed by at most 12 undamped/damped Newton polish steps. Keep
    all outcomes including numerical stagnation; certify residual Newton step.
    """
    t = time.perf_counter()
    p = np.asarray(start).copy().reshape(2, 2)
    before = m.joint(p)
    opt = minimize(lambda x: m.joint(x.reshape(2, 2), True), p.ravel(), jac=True,
                   method='L-BFGS-B', bounds=[(-1000., 1000.)]*4,
                   options=dict(maxiter=200, ftol=0., gtol=1e-12, maxls=40))
    if np.isfinite(opt.fun) and opt.fun <= before:
        p = opt.x.reshape(2, 2)
    polished = []
    for _ in range(12):
        value, gradient, h = derivatives(m, p)
        h = (h+h.T)/2
        eig = np.linalg.eigvalsh(h)
        direction = -np.linalg.solve(h+np.eye(4)*max(0., 1e-16-eig.min()), gradient)
        step = float(np.linalg.norm(direction))
        if eig.min() > 0 and step <= 1e-5:
            break
        direction *= min(1., 20/max(step, 1e-30))
        accepted = False
        for k in range(30):
            trial = np.clip(p.ravel()+direction*2.**(-k), -1000, 1000).reshape(2, 2)
            new = m.joint(trial)
            if new < value:
                accepted = True
                p = trial
                break
        polished.append(dict(step_m=step, accepted=accepted))
        if not accepted:
            break
    value, gradient, h = derivatives(m, p)
    eig = np.linalg.eigvalsh((h+h.T)/2)
    assert np.isfinite(p).all() and np.isfinite(h).all() and value <= before+1e-14
    correction = float(np.linalg.norm(np.linalg.solve(h, gradient)))
    return dict(positions=p.tolist(), objective=value, initial_objective=before,
                gradient_max=float(abs(gradient).max()), hessian_min=float(eig.min()),
                newton_correction_m=correction, local_certificate=bool(eig.min()>0 and correction<1e-3),
                optimizer_message=str(opt.message), iterations=int(opt.nit), polish=polished,
                elapsed_seconds=time.perf_counter()-t)


def components(m, signal, noise, truth, window=None):
    n = signal.shape[-1]
    window = np.ones(n) if window is None else window
    keep = m.frequency_indices.cpu().numpy()
    z = (np.fft.fft(signal*window, axis=-1)/n)[..., keep].transpose(0, 2, 1)
    w = (np.fft.fft(noise*window, axis=-1)/n)[..., keep].T
    a = m.response(truth).cpu().numpy()  # unit-norm columns; coefficients absorb gain
    coef = np.einsum('flk,kfl->fk', a.conj(), z)
    exact = np.einsum('flk,fk->kfl', a, coef)
    residual = z-exact
    np.testing.assert_allclose(np.einsum('flk,kfl->fk', a.conj(), residual), 0, atol=1e-13)
    return z, w, a, coef, exact, residual


def sensitivity(m, truth, a, coef, residual, w, sigma2=1.):
    """First-order deterministic perturbation map; NOT full CRB/Monte Carlo.

    G=Re(J^H J), dtheta=G^-1 Re[J^H delta], J=P_A_perp dA*s.
    For circular noise FFT/N: cov(dtheta)=sigma2/(2*N) G^-1.
    Column normalization derivative lies in span(A), hence drops after Pperp.
    """
    stations = m.stations.cpu().numpy()
    dvec = truth[:, None]-stations
    d = np.linalg.norm(dvec, axis=-1)
    unit = (dvec/d[..., None]).transpose(1, 0, 2)
    factor = -1/d.T[None]-2j*np.pi*m.f.cpu().numpy()[:, None, None]/C
    q = (a[..., None]*factor[..., None]*unit[None]*coef[:, None, :, None]).reshape(len(a), 4, 4)
    inv = np.linalg.pinv(a, rcond=1e-10)
    j = q-a@(inv@q)
    gram = np.einsum('fli,flj->ij', j.conj(), j).real
    gi = np.linalg.inv(gram)
    noises = np.sqrt(np.trace(gi[2:, 2:])*sigma2/(2*m.sample_count))
    parts = {}
    for name, delta in (('A', residual[0]), ('B', residual[1]), ('noise', w)):
        g = np.einsum('fli,fl->i', j.conj(), delta).real
        parts[name] = (gi@g).reshape(2, 2).tolist()
    predicted = sum(np.array(v) for v in parts.values())
    eigen = np.linalg.eigvalsh(gram)
    # Positive, rank-full local sensitivity is required, not regularized silently.
    assert eigen.min() > 0
    return dict(predicted_shift_m=predicted.tolist(), components=parts,
                weak_noise_linear_rms_m=float(noises), gram_eigenvalues=eigen.tolist(),
                gram_condition=float(eigen.max()/eigen.min()),
                weak_gain_from_A_residual_m=float(np.linalg.norm(parts['A'][1])))


def load_case(folder, expected, task):
    l, r, g = task
    tag = f'{l}_{r}_{g}'
    pair_path = folder/'pairs'/f'geometry_{tag}.json'
    pair = read(pair_path, expected[str(pair_path)])
    meta_path = folder/'inputs'/f'new_{tag}.json'
    meta = read(meta_path, expected[str(meta_path)])
    payload = raw(folder/meta['path'], meta['sha256'])
    with np.load(io.BytesIO(payload), allow_pickle=False) as z:
        signal, noise, truth = z['signal'], z['noise'], z['positions']
    iq = signal.sum(0)+noise
    assert hashlib.sha256(np.ascontiguousarray(iq).tobytes()).hexdigest() == pair['iq_sha']
    np.testing.assert_array_equal(truth, pair['truth'])
    assert meta['sample_rate_hz'] == 50e6 and meta['N'] == 8192 and meta['snr_inband_db'] == 6.
    result = next(row['result'] for row in pair['rows'] if row['method'] == 'AP_1_JOINT')
    return tag, signal, noise, truth, result


def numerics(m, signal, noise, truth):
    checks = dict(assignment=self_test(), exact_derivatives=check(m))
    z, w, a, coef, exact, residual = components(m, signal, noise, truth)
    np.testing.assert_allclose(z.sum(0)+w, m.x.cpu().numpy(), atol=1e-13)
    original = m.x.clone()
    set_x(m, exact.sum(0))
    value, g, h = derivatives(m, truth)
    assert value < 1e-24 and abs(g).max() < 1e-13
    p = local(m, truth+np.array([[.4, -.3], [-.2, .5]]))
    assert np.max(np.linalg.norm(np.array(p['positions'])-truth, axis=1)) < 1e-3
    checks['exact_model_recovery_error_m'] = float(np.linalg.norm(np.array(p['positions'])-truth))
    set_x(m, original)
    # Linear perturbation agrees with a small controlled coefficient-space displacement.
    ss = sensitivity(m, truth, a, coef, residual, w)
    assert np.isfinite(ss['weak_noise_linear_rms_m'])
    # Normalized and physical columns give the same LS projection.
    d = np.linalg.norm(truth[None]-m.stations.cpu().numpy()[:, None], axis=-1)
    raw_a = np.exp(-2j*np.pi*m.f.cpu().numpy()[:, None, None]*d[None]/C)/d[None]
    raw_r = z.sum(0)-(raw_a@(np.linalg.pinv(raw_a)@z.sum(0)[..., None])).squeeze(-1)
    norm_r = z.sum(0)-(a@(np.linalg.pinv(a)@z.sum(0)[..., None])).squeeze(-1)
    np.testing.assert_allclose(raw_r, norm_r, atol=1e-13)
    checks['normalized_physical_projection_equal'] = True
    return checks


def diagnose(folder, expected, task):
    tag, signal, noise, truth, result = load_case(folder, expected, task)
    l, r, gap = task
    m = Model(signal.sum(0)+noise)
    z, w, a, coef, exact, residual = components(m, signal, noise, truth)
    baseline, metric = matched(truth, result['positions'])
    value, _ = m.joint(np.array(result['positions']), True)
    assert abs(value-result['native_objective']) < 1e-12
    details = {}
    for mode in VARIANTS:
        data = z.sum(0)+w
        if mode == 'no_noise_truth':
            data = z.sum(0)
        elif mode == 'correct_A_truth':
            data = exact[0]+z[1]+w
        elif mode == 'correct_B_truth':
            data = z[0]+exact[1]+w
        elif mode == 'correct_AB_truth':
            data = exact.sum(0)+w
        set_x(m, data)
        out = local(m, baseline if mode == 'original_saved' else truth)
        p, ev = matched(truth, out['positions'])
        out.update(positions=p.tolist(), errors_m=ev['errors_m'], both20=ev['both20'])
        details[mode] = out
    sen = sensitivity(m, truth, a, coef, residual, w)
    # Model residual under smooth taper, no localization or covariance-independence claim.
    hz, _, _, _, _, hr = components(m, signal, noise, truth, hann(8192, sym=False))
    power = np.sum(abs(z)**2, axis=(1, 2))
    rp = np.sum(abs(residual)**2, axis=(1, 2))
    hp = np.sum(abs(hr)**2, axis=(1, 2))/np.sum(abs(hz)**2, axis=(1, 2))
    delta = (np.array(details['original_truth']['positions'])-truth)
    projection = lambda x: x-(a@(np.linalg.pinv(a)@x[..., None])).squeeze(-1)
    pr = projection(residual[0])
    expected_noise = (4-2)*len(a)/8192
    rinfo = dict(source_residual_fractions=(rp/power).tolist(),
                 source_retained_energy=(1-rp/power).tolist(),
                 A_residual_over_weak_power=float(rp[0]/power[1]),
                 A_joint_projected_residual_over_projected_noise=float(np.sum(abs(pr)**2)/expected_noise),
                 hann_residual_fractions=hp.tolist(),
                 hann_to_rect_relative_fraction=(hp/(rp/power)).tolist(),
                 receiver_A_B_power_db=(10*np.log10(np.mean(abs(signal[0])**2, axis=1)
                                                  /np.mean(abs(signal[1])**2, axis=1))).tolist())
    return dict(case=tag, layout=l, repeat=r, gap=gap, group='difficult' if l in DIFFICULT else 'control',
                truth=truth.tolist(), saved_errors_m=metric['errors_m'], saved_positions=baseline.tolist(),
                saved_native_objective=result['native_objective'], saved_gradient=result['projected_gradient'],
                variants=details, model=rinfo, sensitivity=sen, actual_shift_m=delta.tolist(),
                notes='Oracle diagnostic; never a deployable estimate or formal method comparison.')


def summary(rows):
    table = []
    for group in ('all', 'difficult', 'control'):
        for gap in (0, 15):
            part = [r for r in rows if r['gap']==gap and (group=='all' or r['group']==group)]
            if not part:
                continue
            for mode in ('saved',)+VARIANTS:
                errors = np.array([r['saved_errors_m'] if mode=='saved' else r['variants'][mode]['errors_m'] for r in part])
                table.append(dict(group=group, gap=gap, variant=mode, n=len(part),
                    strong_rmse_m=float(np.sqrt(np.mean(errors[:, 0]**2))),
                    weak_rmse_m=float(np.sqrt(np.mean(errors[:, 1]**2))),
                    weak_median_m=float(np.median(errors[:, 1])), weak_p95_m=float(np.quantile(errors[:, 1], .95)),
                    both20=int(np.all(errors<=20, axis=1).sum())))
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    tasks = [(13, 1000, 0), (13, 1000, 15)] if args.pilot else TASKS
    started = time.perf_counter()
    expected = read(P0/'source_manifest.json')
    pointer = ROOT/'outputs_e2e/unified/e2e_g10/r4/review_supplement_release.json'
    folder = resolve(read(pointer, expected[str(pointer)])['out'])
    out = ROOT/'outputs_e2e/unified/e2e_g11'/('p1_precision_'+('pilot_' if args.pilot else '')+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    assert not out.exists() and out.resolve().is_relative_to(ROOT/'outputs_e2e')
    out.mkdir()
    (out/'cases').mkdir()
    protected = {str(p): sha(p) for root in ('paper', 'paper_short_vp')
                 for ext in ('*.tex', '*.pdf') for p in (ROOT/root).glob(ext)}
    code = {str(p): sha(p) for p in (ROOT/'统一模型代码/gates').rglob('*.py')}
    write(out/'protocol.json', dict(tasks=tasks, difficult=DIFFICULT, controls=CONTROLS,
          control_selection='first geometry of each category, predeclared not error-selected',
          variants=VARIANTS, scope='bounded oracle attribution; no new scenes, no frozen algorithm change',
          refinement='L-BFGS-B 200/ftol0/gtol1e-12 + <=12 Newton polish; local audit only',
          interpretation='not population inference, not new method ranking, not new global optimum guarantee',
          evaluation=REVISION, precision='complex128/float64', python=sys.version,
          numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__,
          source_code=code, protected=protected))
    tag, sig, noise, truth, saved = load_case(folder, expected, tasks[0])
    m = Model(sig.sum(0)+noise)
    checks = numerics(m, sig, noise, truth)
    write(out/'checks.json', checks)
    print('身份与投影/导数/精确模型恢复检查通过。', flush=True)
    rows = []
    for i, task in enumerate(tasks):
        row = diagnose(folder, expected, task)
        write(out/'cases'/(row['case']+'.json'), row)
        rows.append(row)
        if (i+1)%10==0 or i+1==len(tasks):
            print(f'有限诊断 {i+1}/{len(tasks)}，累计 {time.perf_counter()-started:.1f}s', flush=True)
    table = summary(rows)
    csv_write(out/'summary.csv', table)
    by_layout = []
    for l in sorted({r['layout'] for r in rows}):
        by_layout += [dict(layout=l, **r) for r in summary([r for r in rows if r['layout']==l]) if r['group']=='all']
    csv_write(out/'by_layout.csv', by_layout)
    assert all(sha(p)==h for p, h in {**protected, **code, **PROVENANCE}.items())
    certificates = [v['local_certificate'] for r in rows for v in r['variants'].values()]
    audit = dict(status='COMPLETED_BOUNDED_DIAGNOSTIC', observations=len(rows), local_solves=len(certificates),
                 local_certificates=sum(certificates), identities_checked=len(PROVENANCE),
                 frozen_sources_unchanged=True, papers_unchanged=True, source_identity_passed=True,
                 elapsed_seconds=time.perf_counter()-started)
    write(out/'report.json', dict(audit=audit, summary=table, rows=rows))
    write(out/'source_manifest.json', PROVENANCE)
    write(out/'audit.json', audit)
    print(str(out), flush=True)
    print(json.dumps(audit, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
