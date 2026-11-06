"""Architecture-admission mathematics, NOT an estimator or blind benchmark."""
import sys
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE/'numerical_repair_20261104'))
import stable  # noqa: E402

cfg, ad = stable.cfg, stable.ad
INPUT = BASE/'inside_pilot_20261104/batch_v2'


def spectral(h, s):
    return np.einsum('fmq,fq->fm', h, s)


def waveform_test():
    """Exact auxiliary identity and an independently iterated proximal check."""
    rng = np.random.default_rng(2026110517)
    h = rng.normal(size=(32, 4, 2))+1j*rng.normal(size=(32, 4, 2))
    y = rng.normal(size=(32, 4))+1j*rng.normal(size=(32, 4))
    c = np.array([1.2, .8]); lam = .1
    gram = h.conj().transpose(0, 2, 1)@h
    rhs = np.einsum('fmq,fm->fq', h.conj(), y)
    ls = np.linalg.solve(gram, rhs[..., None])[..., 0]

    def obj(s):
        return float(.5*np.linalg.norm(y-spectral(h, s))**2+lam*c@np.linalg.norm(s, axis=0))

    a = ls.copy(); losses = [obj(a)]
    for it in range(1000):
        weights = lam*c/np.linalg.norm(a, axis=0)
        a = np.linalg.solve(gram+np.diag(weights), rhs[..., None])[..., 0]
        losses.append(obj(a))
        if abs(losses[-1]-losses[-2]) < 1e-13:
            break
    p = ls.copy(); lipschitz = np.linalg.eigvalsh(gram).max()
    for pit in range(20000):
        grad = np.einsum('fij,fj->fi', gram, p)-rhs
        z = p-grad/lipschitz
        pnext = z*np.maximum(0., 1-lam*c/(lipschitz*np.linalg.norm(z, axis=0)))
        if np.linalg.norm(pnext-p) < 1e-12:
            p = pnext
            break
        p = pnext
    z = np.linalg.norm(a, axis=0)
    w = lam*c/z
    variational = .5*np.sum(w*z*z+(lam*c)**2/w)
    kkt = np.einsum('fij,fj->fi', gram, a)-rhs+a*(lam*c/z)
    error = np.linalg.norm(a-p)/np.linalg.norm(a)
    checks = dict(aux_identity=abs(variational-lam*c@z) < 1e-12,
                  descent=bool(np.max(np.diff(losses)) < 1e-10),
                  independent_solver_match=bool(error < 1e-8),
                  stationarity=bool(np.max(abs(kkt)) < 1e-7),
                  ordinary_ls_not_regularized_solution=bool(obj(ls)-obj(a) > 1e-5))
    return dict(checks=checks, passed=all(checks.values()), iterations=it+1,
                proximal_iterations=pit+1, relative_difference=error,
                objective_ls=obj(ls), objective_regularized=obj(a),
                diagnostic_lambda=lam, localization_lambda_selected=False)


def information_check(f):
    """A single-link local delay direction; not the 4-station location CRB."""
    omega = 2*np.pi*f/cfg.B
    a0 = np.ones((len(f), 1), complex)
    j = -1j*omega
    residual0 = j-a0@(np.linalg.lstsq(a0, j, rcond=None)[0])
    rows = []
    for d in [1., .1, .01, .001, .0001]:
        basis = np.column_stack([a0[:, 0], np.exp(-1j*omega*d)])
        remaining = j-basis@np.linalg.lstsq(basis, j, rcond=None)[0]
        rows.append(dict(normalized_delay=d, retained_local_information=float(
            np.vdot(remaining, remaining).real/np.vdot(residual0, residual0).real)))
    return rows


def main():
    out = HERE/'run_v2'
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    hashes = ad.source_hashes()
    hashes.update({str(p): cfg.sha(p) for p in [Path(__file__).resolve(), HERE/'protocol.md']})
    for i in range(100, 106):
        path = INPUT/'banks'/f'{i}_00.npz'
        assert cfg.sha(path) == cfg.read(path.with_suffix('.json'))['sha']
        hashes[str(path)] = cfg.sha(path)
    cfg.write(out/'identity.json', hashes)
    f = np.fft.fftfreq(cfg.N, 1/cfg.FS)
    keep = abs(f) <= cfg.B/2
    f = f[keep]
    t = np.arange(cfg.N)/cfg.FS
    rows = []
    for i in range(100, 106):
        with np.load(INPUT/'banks'/f'{i}_00.npz', allow_pickle=False) as z:
            bank = {k: z[k] for k in z.files}
        p = bank['p']; wrong = p@np.array([[0., 1.], [-1., 0.]])
        assert cfg.geometry_ok(wrong) and np.max(abs(wrong)) <= 3000
        tau = np.linalg.norm(cfg.STATIONS[:, None]-p[None], axis=-1)/cfg.C
        other = np.linalg.norm(cfg.STATIONS[:, None]-wrong[None], axis=-1)/cfg.C
        shift = np.maximum(0., np.max(other-tau, axis=0))+10/cfg.B
        delta = shift[None]+tau-other
        g = bank['gains']
        def signal(q, times):
            return ad.wave.synthesize(bank['bits'][q], int(bank['first']), bank['offsets'][q], times)/np.sqrt(bank['pulse_norm2'])
        source = np.column_stack([signal(q, t) for q in range(2)])
        s = np.fft.fft(source, axis=0, norm='ortho')[keep]
        noise = np.fft.fft(bank['noise'], axis=-1, norm='ortho')[:, keep].T
        directphase = np.exp(-2j*np.pi*f[:, None, None]*tau)
        wrongphase = np.exp(-2j*np.pi*f[:, None, None]*other)
        shifted_s = s*np.exp(2j*np.pi*f[:, None]*shift)
        for name in ['L2_P075', 'L3_P150']:
            con = next(c for c in cfg.CONDITIONS if c['name'] == name)
            _, powers = cfg.observation(bank, con)
            rho = np.sqrt(powers)*bank['phases']
            delays = bank['delays']/cfg.B
            original = np.ones((len(f), 4, 2), complex)
            for ell in con['ids']:
                original += rho[None, :, :, ell]*np.exp(-2j*np.pi*f[:, None, None]*delays[None, :, :, ell])
            h = directphase*g[None]*original
            original_y = spectral(h, s)
            free_channel = h/wrongphase
            free_error = np.linalg.norm(spectral(wrongphase*free_channel, s)-original_y)/np.linalg.norm(original_y)
            original_time = bank['los'].sum(axis=0)+np.sum(bank['reflected']*np.sqrt(powers.transpose(1, 0, 2))[..., None], axis=(0, 2))
            normalized_tv = np.sum(abs(rho), axis=-1)
            weighted_original = float(np.sum(normalized_tv*abs(g)*np.linalg.norm(s, axis=0)[None]))
            for amplitude in [1., 16.]:
                for u in [.1, .01, .001, .0001]:
                    eps = u/cfg.B
                    assert 4*eps < delta.min()
                    reflected = (-.5*amplitude*np.exp(-2j*np.pi*f[:, None, None]*eps)
                                 -.5*amplitude*np.exp(-2j*np.pi*f[:, None, None]*2*eps))
                    reflected = np.broadcast_to(reflected, h.shape).copy()
                    reflected += .5*np.exp(-2j*np.pi*f[:, None, None]*(delta[None]-eps))
                    reflected += .5*np.exp(-2j*np.pi*f[:, None, None]*(delta[None]+eps))
                    for ell in con['ids']:
                        reflected += rho[None, :, :, ell]*np.exp(-2j*np.pi*f[:, None, None]*(delta+delays[:, :, ell])[None])
                    new_h = wrongphase*g[None]*(amplitude+reflected)
                    observed_error = np.linalg.norm(spectral(new_h, shifted_s)-original_y)
                    beta = 1.5*np.pi*cfg.B*eps+.5*(np.pi*cfg.B*eps)**2
                    bound = beta*np.sum(abs(g).sum(axis=0)*np.linalg.norm(s, axis=0))*max(1., amplitude)
                    # Separate nonperiodic check, using the original saved RRC symbols.
                    new_time = original_time.copy()
                    time_ratio = None
                    time_status = 'verified_saved_symbols'
                    try:
                        for q in range(2):
                            for m in range(4):
                                false_times = t+shift[q]-other[m, q]
                                defect = amplitude*(signal(q, false_times)-.5*signal(q, false_times-eps)-.5*signal(q, false_times-2*eps))
                                defect += .5*signal(q, t-tau[m, q]+eps)+.5*signal(q, t-tau[m, q]-eps)-signal(q, t-tau[m, q])
                                new_time[m] += g[m, q]*defect
                    except AssertionError:
                        # The frozen synthesizer's sole assertion checks symbol support.
                        # No extension, padding, time-window trimming, or periodic fallback.
                        time_status = 'unavailable_saved_symbol_support'
                    if time_status == 'verified_saved_symbols':
                        time_difference = new_time-original_time
                        time_band_difference = np.fft.fft(time_difference, axis=-1, norm='ortho')[:, keep]
                        time_ratio = float(np.linalg.norm(time_band_difference)/np.linalg.norm(noise))
                    new_tv = 1+(1+normalized_tv)/amplitude
                    weighted_new = float(np.sum((amplitude+1+normalized_tv)*abs(g)*np.linalg.norm(s, axis=0)[None]))
                    actual_delays = [np.full_like(delta, eps), np.full_like(delta, 2*eps), delta-eps, delta+eps]
                    actual_delays += [delta+delays[:, :, ell] for ell in con['ids']]
                    packed_delays = np.stack(actual_delays)
                    unique_gap = float(np.min(np.diff(np.sort(packed_delays, axis=0), axis=0)))
                    row = dict(layout=i, condition=name, amplitude_multiplier=amplitude, epsilon_B=u,
                        wrong_positions=wrong, source_shift_s=shift,
                        spectral_relative_error=float(observed_error/np.linalg.norm(original_y)),
                        spectral_error_over_noise=float(observed_error/np.linalg.norm(noise)),
                        time_band_error_over_noise=time_ratio, time_check_status=time_status,
                        analytic_bound_valid=bool(observed_error <= bound*(1+1e-10)),
                        free_channel_identity_error=float(free_error),
                        reflections=8*(con['L']+4), min_delay_s=float(packed_delays.min()),
                        max_delay_s=float(packed_delays.max()), min_separation_s=unique_gap,
                        max_reflection_ratio=float(max(.5, .5/amplitude, np.max(abs(rho))/amplitude)),
                        relative_penalty_original=float(normalized_tv.sum()), relative_penalty_wrong=float(new_tv.sum()),
                        received_penalty_original=weighted_original, received_penalty_wrong=weighted_new,
                        constraints_valid=bool(unique_gap>0 and packed_delays.min()>0 and packed_delays.max()<cfg.N/cfg.FS),
                        uses_truth=True, estimator_run=False)
                    rows.append(row)
            print('DONE', i, name, flush=True)
            cfg.write(out/'counterexamples.json', rows)
        if time.perf_counter()-started > 600:
            raise RuntimeError('Administrative diagnostic budget exceeded; not convergence')
    info = information_check(f)
    wave = waveform_test()
    # Received-component penalty is invariant to source/channel complex scaling.
    scales = np.array([3*np.exp(.4j), .2*np.exp(-.3j)])
    basepen = np.sum(normalized_tv*abs(g)*np.linalg.norm(s, axis=0)[None])
    scaledpen = np.sum(normalized_tv*abs(g*scales)*np.linalg.norm(s/scales, axis=0)[None])
    checks = dict(all_constraints=all(r['constraints_valid'] and r['reflections']<=64 and r['max_reflection_ratio']<1 for r in rows),
        analytic_bound=all(r['analytic_bound_valid'] for r in rows),
        arbitrary_channel_erases_geometry=max(r['free_channel_identity_error'] for r in rows)<1e-12,
        local_information_collapses=info[-1]['retained_local_information']<1e-6,
        relative_penalty_counterexample=all(r['relative_penalty_wrong']<r['relative_penalty_original'] for r in rows if r['amplitude_multiplier']==16),
        received_penalty_rejects_this_family=all(r['received_penalty_wrong']>r['received_penalty_original'] for r in rows),
        received_penalty_scale_invariant=bool(abs(basepen-scaledpen)<1e-12*basepen),
        conditional_waveform_correct=wave['passed'])
    stable.inside.rr.check_hashes(hashes)
    cfg.write(out/'summary.json', dict(checks=checks, mathematical_gate=all(checks.values()),
        rows=len(rows), local_information=info, waveform=wave,
        identity_unchanged=True, elapsed_s=time.perf_counter()-started,
        localization_performance_tested=False, solver_ready=False,
        caveat='Counterexamples belong to the wider estimator class, not the separated L2/L3 physical generator class'))
    print(checks, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
