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


def active_indices(gamma, limit):
    """Dictionary coordinates, not decoded physical sources: no spatial NMS."""
    return torch.argsort(gamma, descending=True)[:min(limit, len(gamma))].cpu().numpy()


def evidence_problem(m, positions, gamma, ids, covariance, scale=1000.):
    """Exact native evidence for a small working set, vectorized over frequencies."""
    old = np.asarray(positions)[ids].copy()
    active = gamma[ids]
    a = m.response(old)
    fixed = covariance-(a*active[None, None])@a.mH

    def objective(flat):
        m.check()
        p = torch.tensor((np.asarray(flat)*scale).reshape(-1, 2), dtype=torch.float64,
                         device=m.device, requires_grad=True)
        a = atom(p, m.f, m.stations)
        m.counts['frequency_response_points'] += len(p)*m.n
        cov = fixed+(a*active[None, None])@a.mH
        chol = torch.linalg.cholesky(cov)
        invx = torch.cholesky_solve(m.x[..., None], chol).squeeze(-1)
        value = (2*torch.log(chol.diagonal(dim1=1, dim2=2).real).sum(-1)
                 +(m.x.conj()*invx).sum(-1).real).mean()
        grad = torch.autograd.grad(value, p)[0]*scale
        return float(value.detach().cpu()), grad.detach().cpu().numpy().ravel()
    return old, objective


def coordinate_update(m, positions, gamma, ids, covs, chunk, step, scale=1., maxiter=10):
    old_points = positions[ids].copy()
    _, objective = evidence_problem(m, positions, gamma, ids, torch.cat(covs), scale)
    before, initial_gradient = objective(old_points.ravel()/scale)
    bounds = [(max(-1000., v-step)/scale, min(1000., v+step)/scale) for v in old_points.ravel()]
    res = minimize(objective, old_points.ravel()/scale, jac=True, method='L-BFGS-B', bounds=bounds,
                   options=dict(maxiter=maxiter, ftol=1e-10, gtol=1e-7, maxls=15))
    if np.isfinite(res.fun) and res.fun <= before:
        positions[ids] = (res.x*scale).reshape(-1, 2)
    if not hasattr(m, 'local_optimizers'):
        m.local_optimizers = []
    m.local_optimizers.append(dict(active_atoms=len(ids), scale_m=scale, iterations=int(res.nit), evaluations=int(res.nfev),
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
        active_shift = None
        if refine_now:
            with t.stage('sbl_native_coordinate_refinement'):
                # K is only the final output count. Multiple neighboring atoms
                # can represent one off-grid source during evidence optimization.
                move = active_indices(gamma, cfg['sbl_active_atoms']) if cfg.get('sbl_active_atoms', 0) else ids
                coordinate_update(m, pp, gamma, move, stats['covs'], chunk, cfg['sbl_grid'],
                                  cfg.get('sbl_coordinate_scale', 1.), cfg.get('sbl_coordinate_iterations', 10))
                active_shift = m.local_optimizers[-1]['maximum_step_m']
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
                    max_shift_m=shift, active_max_shift_m=active_shift, refined=refine_now)
        m.counts['iterations'] += 1
        if refine_now:
            stable = stable+1 if ((previous-value)/max(1., abs(previous)) < 1e-5 and gamma_change < 1e-2
                                  and shift is not None and shift < cfg['position_tolerance']
                                  and active_shift < cfg['position_tolerance']) else 0
        if stable >= 2:
            t.reason = 'NATIVE_STABLE'
            break
        previous, previous_positions = value, positions
    else:
        t.reason = 'ITERATION_LIMIT'
    m.native_map = dict(points=pp.tolist(), power=stats['power'].cpu().tolist(),
                        gamma=gamma.cpu().tolist(),
                        meaning='SBL posterior mean power; adaptive dictionary')
