"""Common-covariance multi-frequency SBL + exact native-evidence off-grid updates.

Nannuru et al. (2017), eqs. (38)-(40), b=1/2; EM retained as reference.
Known-noise/response specialization; NOT verbatim unknown-gain R002 PDL.
"""
import numpy as np
import torch
from scipy.optimize import minimize
from ..physics import atom, grid, shortlist, sbl_posterior


def statistics(m, positions, gamma, chunk, want_cov=False):
    second, power, numerator, denominator = [torch.zeros_like(gamma) for _ in range(4)]
    evidence, covs = 0., []
    for lo in range(0, m.n, chunk):
        m.check()
        a = m.response(positions, m.f[lo:lo+chunk])
        mu, diagonal, cov, value = sbl_posterior(a, m.x[lo:lo+chunk], gamma, 1/m.n)
        second += (abs(mu)**2+diagonal).sum(0)
        power += (abs(mu)**2).sum(0)
        numerator += (abs(mu/gamma[None])**2).sum(0)
        denominator += ((gamma[None]-diagonal)/gamma[None]**2).sum(0)
        evidence += float(value.sum().cpu())/m.n
        m.counts['covariance_decompositions'] += len(a)
        m.counts['sbl_dictionary_columns'] = m.counts.get('sbl_dictionary_columns', 0)+len(a)*len(gamma)
        if want_cov:
            covs.append(cov.detach())
    return dict(em=second/m.n, power=power/m.n, numerator=numerator,
                denominator=denominator, evidence=evidence, covs=covs)


def coordinate_update(m, positions, gamma, ids, covs, chunk, step, scale=1.):
    old_points = positions[ids].copy()
    fixed = []
    active = gamma[ids]
    for j, lo in enumerate(range(0, m.n, chunk)):
        a = m.response(old_points, m.f[lo:lo+chunk])
        fixed.append(covs[j]-(a*active[None, None])@a.mH)

    def objective(flat):
        value, grad = 0., np.zeros_like(flat)
        for j, lo in enumerate(range(0, m.n, chunk)):
            m.check()
            p = torch.tensor((flat*scale).reshape(-1, 2), dtype=torch.float64, device=m.device, requires_grad=True)
            a = atom(p, m.f[lo:lo+chunk], m.stations)
            m.counts['frequency_response_points'] += len(p)*len(a)
            cov = fixed[j]+(a*active[None, None])@a.mH
            chol = torch.linalg.cholesky(cov)
            x = m.x[lo:lo+chunk]
            invx = torch.cholesky_solve(x[..., None], chol).squeeze(-1)
            v = (2*torch.log(chol.diagonal(dim1=1, dim2=2).real).sum(-1)
                 +(x.conj()*invx).sum(-1).real).sum()/m.n
            grad += torch.autograd.grad(v, p)[0].detach().cpu().numpy().ravel()*scale
            value += float(v.detach().cpu())
        return value, grad
    before, initial_gradient = objective(old_points.ravel()/scale)
    bounds = [(max(-1000., v-step)/scale, min(1000., v+step)/scale) for v in old_points.ravel()]
    res = minimize(objective, old_points.ravel()/scale, jac=True, method='L-BFGS-B', bounds=bounds,
                   options=dict(maxiter=10, ftol=1e-10, gtol=1e-7, maxls=15))
    if np.isfinite(res.fun) and res.fun <= before:
        positions[ids] = (res.x*scale).reshape(-1, 2)
    if not hasattr(m, 'local_optimizers'):
        m.local_optimizers = []
    m.local_optimizers.append(dict(scale_m=scale, iterations=int(res.nit), evaluations=int(res.nfev),
                                  success=bool(res.success), message=str(res.message), before=before,
                                  after=float(res.fun), initial_gradient_norm=float(np.linalg.norm(initial_gradient)),
                                  maximum_step_m=float(np.linalg.norm(positions[ids]-old_points,axis=1).max())))
    return objective(positions[ids].ravel()/scale)[0]


def solve(m, t, count, cfg):
    pp = grid(cfg['sbl_grid'])
    gamma = torch.full((len(pp),), float(m.energy.cpu())/m.n/len(pp),
                       dtype=torch.float64, device=m.device)
    chunk = cfg['fft_chunk']
    if cfg.get('dictionary_cache', True):
        with t.stage('exact_dictionary_cache_build'):
            m.enable_response_cache(pp, chunk)
    previous, previous_positions, stable = np.inf, None, 0
    with t.stage('sbl_posterior_and_variance'):
        stats = statistics(m, pp, gamma, chunk)
    for iteration in range(cfg['sbl_iterations']):
        t.check()
        old_gamma, old_value = gamma, stats['evidence']
        proposal = (stats['em'] if cfg['sbl_update'] == 'em' else
                    gamma*torch.sqrt(stats['numerator']/stats['denominator'].clamp_min(1e-30)))
        proposal = proposal.clamp_min(1e-20)
        refine_now = (iteration+1) % cfg['sbl_refine_every'] == 0
        with t.stage('sbl_posterior_and_variance'):
            for backtrack in range(9):
                fraction = .5**backtrack
                gamma = torch.exp((1-fraction)*torch.log(old_gamma)+fraction*torch.log(proposal))
                trial = statistics(m, pp, gamma, chunk, refine_now)
                if trial['evidence'] <= old_value+1e-9:
                    break
            else:
                gamma, trial = old_gamma, stats
                refine_now = False
            stats = trial
        ids = shortlist(pp, stats['power'].cpu().numpy(), count)
        if refine_now:
            with t.stage('sbl_native_coordinate_refinement'):
                coordinate_update(m, pp, gamma, ids, stats['covs'], chunk, cfg['sbl_grid'],
                                  cfg.get('sbl_coordinate_scale',1.))
                m.refresh_response_cache(pp, chunk)
            with t.stage('sbl_posterior_and_variance'):
                stats = statistics(m, pp, gamma, chunk)
            ids = shortlist(pp, stats['power'].cpu().numpy(), count)
        value, positions = stats['evidence'], pp[ids].copy()
        shift = (float(np.linalg.norm(positions-previous_positions, axis=1).max())
                 if previous_positions is not None else None)
        gamma_change = float((abs(gamma-old_gamma).sum()/old_gamma.sum()).cpu())
        t.save(positions, value)
        t.iteration(positions, value, iteration=iteration+1, gamma_relative_change=gamma_change,
                    max_shift_m=shift, refined=refine_now)
        m.counts['iterations'] += 1
        if refine_now:
            stable = stable+1 if ((previous-value)/max(1., abs(previous)) < 1e-5 and gamma_change < 1e-2
                                  and shift is not None and shift < cfg['position_tolerance']) else 0
        if stable >= 2:
            t.reason = 'NATIVE_STABLE'
            break
        previous, previous_positions = value, positions
    else:
        t.reason = 'ITERATION_LIMIT'
    m.native_map = dict(points=pp.tolist(), power=stats['power'].cpu().tolist(),
                        gamma=gamma.cpu().tolist(),
                        meaning='SBL posterior mean power; adaptive dictionary')
