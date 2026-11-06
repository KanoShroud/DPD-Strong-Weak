"""Five saved-state cone corrections; no new candidate or full-model stopping."""
import time
import numpy as np
from threadpoolctl import threadpool_limits
from local_cone import HERE, PREVIOUS, cfg, ad, stable, data, INPUT, score, selftest, tangent_constraints


def main():
    oldroot = PREVIOUS/'single_full_v2'
    stable.inside.rr.check_hashes(cfg.read(oldroot/'identity.json'))
    out = HERE/'local_v1'
    out.mkdir(exist_ok=False)
    hashes = ad.source_hashes()
    hashes.update({str(p.resolve()): cfg.sha(p) for p in [HERE/'local_cone.py', HERE/'run_local.py', HERE/'protocol.md']})
    cfg.write(out/'identity.json', hashes)
    cfg.write(out/'correctness.json', selftest())
    tags = [r['tag'] for r in cfg.read(oldroot/'rows.json') if not r['normal']]+['IN_100_R00_L2_P075']
    rows = []
    for tag in tags:
        start = time.perf_counter()
        old = cfg.read(oldroot/(tag+'.json'))['result']
        which = [i for i, h in enumerate(old['history']) if 'fit' in h][-1]
        event = old['history'][which]
        previous = old['los'] if which == 0 else old['history'][which-1]['fit']
        x = np.asarray(previous['x']); k = event['K']-1
        paths = old['paths'][:k]
        iq = data(tag); e = stable.Engine()
        f, y, n, sigma = ad.b.fc.cal.spectral(iq)
        use = abs(np.fft.fftfreq(n, 1/cfg.FS)) <= cfg.B/2
        pick, _, search_meta = e.search(x, paths, f, y, n, use, sigma, which+1)
        assert all(search_meta['checks'].values())
        pp = paths+[(pick['m'], pick['q'])]
        zero = np.r_[x, 0., 0., pick['delay']]
        r, a, _ = e.evaluate(x, f, y, paths, True)
        rr, aa, _ = e.evaluate(zero, f, y, pp, True)
        assert np.allclose(r, rr) and np.allclose(a, aa[:, :len(x)])
        b = aa[:, -3:-1]
        c, labels = tangent_constraints(e, x, n)
        stable_active = all(tangent_constraints(e, x, n, tol)[1] == labels for tol in [1e-10, 1e-6])
        d0, d1, scored = score(a, b, r, c)
        joint = np.r_[d1, 0.]
        value, gradient = e.objective(zero, f, y, pp)
        slope = float(gradient@joint)
        eps = 1e-6/max(1., np.linalg.norm(joint))
        fd = (e.objective(zero+eps*joint, f, y, pp)[0]-e.objective(zero-eps*joint, f, y, pp)[0])/(2*eps)
        derivative_ok = bool(np.isclose(fd, slope, rtol=1e-3, atol=1e-8))
        accepted = False; probes = []; fit = None
        if scored['certified'] and stable_active and derivative_ok and slope < 0:
            for i in range(20):
                alpha = .5**i
                trial = e.ref.project(zero+alpha*joint, n)
                v, _ = e.objective(trial, f, y, pp)
                actual_slope = float(gradient@(trial-zero))
                probes.append(dict(alpha=alpha, cost=v, projected_slope=actual_slope))
                if actual_slope < 0 and v < value and v <= value+1e-4*actual_slope and np.linalg.norm(trial[-3:-1]) > 0:
                    accepted = True
                    break
        result = dict(tag=tag, input_sha=cfg.digest(iq), selected=pick, old_threshold=search_meta['threshold'],
            constraints=labels, stable_active=stable_active, cone=scored,
            cone_score=scored['gain']/sigma, direction=joint, slope=slope, finite_difference=fd,
            derivative_ok=derivative_ok, accepted=accepted, probes=probes,
            initial_cost=value, previous=previous, paths=pp, null_direction=d0)
        cfg.write(out/(tag+'_proposal.json'), result)
        print('PROPOSAL', tag, result['cone_score'], 'old', pick['evidence'], 'certificate', scored['certified'], 'accepted', accepted, flush=True)
        if accepted:
            xx, fit = e.fit(trial, f, y, pp, n, 'CONE_LOCAL_DIAGNOSTIC')
            result['fit'] = fit
            result['seconds'] = time.perf_counter()-start
            cfg.write(out/(tag+'_decision.json'), result)
            rv, jac = e.evaluate(xx, f, y, pp)
            cv, gg = e.objective(xx, f, y, pp)
            assert abs(cv-fit['cost']) < 1e-9 and np.allclose(gg, jac.T@rv, rtol=1e-5, atol=1e-9)
        row = dict(tag=tag, score=scored['gain']/sigma, old_score=pick['evidence'],
            cone_certified=scored['certified'], stable_active=stable_active, derivative_ok=derivative_ok,
            accepted=accepted, stationary=False, seconds=time.perf_counter()-start)
        if fit is not None:
            with np.load(INPUT/'banks'/f'{tag.split("_")[1]}_00.npz', allow_pickle=False) as z:
                truth = z['p']
            errors = stable.inside.rr.evaluation(fit['p'], truth)
            row.update(stationary=fit['stationary'], pg=fit['projected_gradient'],
                feasible=fit['closure_error'] < 1e-7, nonincrease=fit['cost'] <= previous['cost']+1e-10,
                rmse=float(np.sqrt(np.mean(np.square(errors)))), iterations=fit['iterations'])
        rows.append(row); cfg.write(out/'rows.json', rows)
        print('END', row, flush=True)
    stable.inside.rr.check_hashes(hashes)
    cfg.write(out/'summary.json', dict(rows=rows, local_gate=all(r['accepted'] and r['stationary'] and r['feasible'] and r['nonincrease'] for r in rows),
        complete_detector_calibrated=False, identity_unchanged=True))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
