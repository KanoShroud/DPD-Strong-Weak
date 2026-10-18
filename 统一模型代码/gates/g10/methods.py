"""Eight native solvers, timed from IQ preprocessing; no truth enters this module."""
import time
import traceback
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.ndimage import maximum_filter
from scipy.stats import qmc
from .physics import Model, Spectrum, grid, shortlist, atom, sbl_posterior
from .config import DEFAULT, METHODS


class Deadline(Exception):
    pass


class Trace:
    def __init__(self, budget, guard):
        torch.cuda.synchronize()
        self.start, self.budget, self.guard = time.perf_counter(), budget, guard
        self.events, self.best = [], np.inf

    def elapsed(self):
        torch.cuda.synchronize()
        return time.perf_counter()-self.start

    def check(self):
        self.guard()
        if self.elapsed() >= self.budget:
            raise Deadline()

    def save(self, positions, native_objective, monotone=True):
        elapsed = self.elapsed()
        p = np.asarray(positions).reshape(-1, 2)
        if elapsed > self.budget or not np.isfinite(p).all() or not np.isfinite(native_objective):
            return
        if monotone and native_objective >= self.best:
            return
        self.best = float(native_objective)
        self.events.append(dict(seconds=elapsed, positions=p.tolist(), native_objective=self.best))


def peaks(points, scores, top):
    side = int(round(np.sqrt(len(points))))
    if side*side == len(points):
        mask = scores.reshape(side, side) == maximum_filter(scores.reshape(side, side), 3,
            mode='constant', cval=-np.inf)
        ids = np.flatnonzero(mask.ravel())
    else:
        ids = np.arange(len(points))
    return ids[shortlist(points[ids], scores[ids], top)]


def refine_joint(model, positions, maxiter=30):
    positions = np.asarray(positions).reshape(-1, 2)
    initial = model.joint(positions)
    result = minimize(lambda p: model.joint(p, True), positions.ravel(), jac=True,
        method='L-BFGS-B', bounds=[(-1000., 1000.)]*positions.size,
        options=dict(maxiter=maxiter, ftol=1e-10, gtol=1e-7, maxls=12))
    value = model.joint(result.x)
    return (result.x.reshape(-1, 2), value) if value <= initial else (positions, initial)


def spectrum_solver(m, trace, method, count, cfg):
    spectrum = None if method == 'LS' else Spectrum(m, method, count, cfg['segments'], cfg['loading'])
    evaluate = m.conditional if spectrum is None else spectrum.evaluate
    points = grid(cfg['grid'])
    values = evaluate(points)
    ids = peaks(points, values, max(cfg['top'], count))
    options = [(points[i].copy(), float(values[i])) for i in ids]

    def publish():
        pp = np.asarray([x[0] for x in options])
        vv = np.asarray([x[1] for x in options])
        ii = shortlist(pp, vv, count)
        if len(ii) == count:
            trace.save(pp[ii], -float(vv[ii].sum()))
    publish()
    for i in range(len(options)):
        p, value = options[i]
        scale = max(abs(value), 1e-30)
        result = minimize(lambda v: -evaluate(np.asarray(v)[None])[0]/scale, p,
            method='Powell', bounds=[(max(-1000., v-cfg['grid']), min(1000., v+cfg['grid'])) for v in p],
            options=dict(maxiter=20, xtol=.05, ftol=1e-8))
        new = float(evaluate(result.x[None])[0])
        if new >= value:
            options[i] = (result.x, new)
        publish()


def coordinate(m, positions, slot):
    initial = np.asarray(positions).copy()
    old = m.joint(initial)
    def objective(p):
        candidate = initial.copy()
        candidate[slot] = p
        v, g = m.joint(candidate, True)
        return v, g.reshape(-1, 2)[slot]
    result = minimize(objective, initial[slot], jac=True, method='L-BFGS-B',
        bounds=[(-1000., 1000.)]*2,
        options=dict(maxiter=30, ftol=1e-10, gtol=1e-7, maxls=12))
    initial[slot] = result.x
    return initial if m.joint(initial) <= old else np.asarray(positions).copy()


def ap_solver(m, trace, count, cfg):
    points = grid(cfg['grid'])
    values = m.conditional(points)
    starts = points[peaks(points, values, cfg['top'])]
    best_positions, best_value = None, np.inf
    for first in starts:
        positions, _ = refine_joint(m, [first])
        if count == 2:
            scores = m.conditional(points, positions)
            ids = peaks(points, scores, cfg['top'])
            best_second, value = None, np.inf
            for i in ids:
                p = np.vstack([positions, points[i]])
                initial_value = m.joint(p)
                trace.save(p, initial_value)
                p = coordinate(m, p, 1)
                v = m.joint(p)
                trace.save(p, v)
                if v < value:
                    best_second, value = p, v
            positions = best_second
        v = m.joint(positions)
        trace.save(positions, v)
        for iteration in range(cfg['cycles']):
            before = v
            for k in range(count):
                others = np.delete(positions, k, axis=0)
                if iteration in (2, 4):
                    scores = m.conditional(points, others)
                    for i in peaks(points, scores, cfg['top']):
                        p = positions.copy()
                        p[k] = points[i]
                        if m.joint(p) < m.joint(positions):
                            positions = p
                positions = coordinate(m, positions, k)
                v = m.joint(positions)
                trace.save(positions, v)
            m.counts['iterations'] += 1
            # Global replacement at iterations 2/4 is still performed after local convergence.
            if iteration >= 4 and before-v < 1e-9:
                break
        if v < best_value:
            best_positions, best_value = positions.copy(), v
    # Sufficient-coverage continuation, identical objective, not an extra AP-only time budget.
    positions = best_positions
    for step in (10., 5.):
        dense = grid(step)
        for k in range(count):
            others = np.delete(positions, k, axis=0)
            scores = m.conditional(dense, others)
            for i in peaks(dense, scores, cfg['top']):
                p = positions.copy()
                p[k] = dense[i]
                p = coordinate(m, p, k)
                v = m.joint(p)
                if v < best_value:
                    positions, best_value = p, v
                    trace.save(positions, v)


def newton_single(m, initial, residual, steps=3):
    p = np.asarray(initial).copy()
    for _ in range(steps):
        m.check()
        z = torch.tensor(p, dtype=torch.float64, device=m.device, requires_grad=True)
        fun = lambda q: m.residual_score(q, residual)
        value = fun(z)
        grad = torch.autograd.grad(value, z)[0].detach().cpu().numpy()
        hessian = torch.autograd.functional.hessian(fun, z).detach().cpu().numpy()
        m.counts['candidate_points'] += 1
        minimum = np.linalg.eigvalsh(hessian).min()
        hessian += np.eye(2)*max(0., 1e-9-minimum)
        direction = -np.linalg.solve(hessian, grad)
        direction *= min(1., 50./max(np.linalg.norm(direction), 1e-30))
        old = float(value.detach().cpu())
        improved = False
        for scale in (1., .5, .25, .125, .0625):
            m.check()
            proposal = np.clip(p+scale*direction, -1000, 1000)
            v = float(fun(torch.as_tensor(proposal, device=m.device)).detach().cpu())
            if v <= old:
                p, improved = proposal, True
                break
        if not improved or np.linalg.norm(scale*direction) < .01:
            break
    return p


def nomp_solver(m, trace, count, cfg):
    positions = []
    residual = m.x
    for _ in range(count):
        points = grid(cfg['grid'])
        scores = m.conditional(points, residual=residual)
        first = points[int(np.argmax(scores))]
        positions.append(newton_single(m, first, residual))
        pp = np.asarray(positions)
        if len(pp) == count:
            trace.save(pp, m.joint(pp))
        for _ in range(cfg['cycles']):
            for k in range(len(pp)):
                a, s, r = m.fitted(pp)
                residual_k = r+a[:, :, k]*s[:, k, None]
                proposal = pp.copy()
                proposal[k] = newton_single(m, pp[k], residual_k)
                if m.joint(proposal) <= m.joint(pp):
                    pp = proposal
                    if len(pp) == count:
                        trace.save(pp, m.joint(pp))
            m.counts['iterations'] += 1
        positions = list(pp)
        _, _, residual = m.fitted(pp)


def initial_pairs(m, count, cfg, rng):
    points = grid(cfg['grid'])
    values = m.conditional(points)
    choices = points[peaks(points, values, max(8, count))]
    seed = choices[:count]
    cover = qmc.Sobol(d=2*count, scramble=True, seed=int(rng.integers(2**31))).random_base2(6)
    starts = [-1000+2000*x.reshape(count, 2) for x in cover]
    starts[0] = seed
    if count == 2:
        for i in range(1, min(len(choices), 8)):
            starts[i] = np.asarray([choices[0], choices[i]])
    return starts


def multistart_solver(m, trace, count, cfg, rng):
    for p in initial_pairs(m, count, cfg, rng)[:cfg['starts']]:
        trace.save(p, m.joint(p))
        pp, value = refine_joint(m, p, maxiter=60)
        trace.save(pp, value)
        m.counts['iterations'] += 1


def sbx_mutate(parents, rng):
    """Deb/Agrawal real-coded SBX eta=15 and polynomial mutation eta=20."""
    children = parents.copy()
    for i in range(0, len(children)-1, 2):
        if rng.random() < .9:
            u = rng.random(children.shape[1])
            beta = np.where(u <= .5, (2*u)**(1/16), (1/(2*(1-u)))**(1/16))
            p, q = parents[i], parents[i+1]
            children[i] = .5*((1+beta)*p+(1-beta)*q)
            children[i+1] = .5*((1-beta)*p+(1+beta)*q)
    children = np.clip(children, -1000, 1000)
    mask = rng.random(children.shape) < .25
    u = rng.random(children.shape)
    d1, d2 = (children+1000)/2000, (1000-children)/2000
    delta = np.where(u < .5, (2*u+(1-2*u)*(1-d1)**21)**(1/21)-1,
                     1-(2*(1-u)+2*(u-.5)*(1-d2)**21)**(1/21))
    return np.clip(children+mask*delta*2000, -1000, 1000)


def ga_solver(m, trace, count, cfg, rng):
    population = np.asarray(initial_pairs(m, count, cfg, rng)[:cfg['population']]).reshape(-1, count*2)
    best, stale = np.inf, 0
    for _ in range(200):
        values = []
        for lo in range(0, len(population), 4):
            batch = population[lo:lo+4].reshape(-1, count, 2)
            vv = m.joint_batch(batch)
            values.extend(vv.tolist())
            for p, value in zip(batch, vv):
                trace.save(p, value)
        values = np.asarray(values)
        order = np.argsort(values)
        stale = stale+1 if best-values[order[0]] < 1e-8 else 0
        best = min(best, values[order[0]])
        m.counts['iterations'] += 1
        if stale >= 15:
            break
        tournament = rng.integers(0, len(values), (len(values)-4, 3))
        winners = tournament[np.arange(len(tournament)), np.argmin(values[tournament], axis=1)]
        population = np.vstack([population[order[:4]], sbx_mutate(population[winners], rng)])
    # Same local solver, cost charged to GA.
    p, value = refine_joint(m, np.asarray(trace.events[-1]['positions']))
    trace.save(p, value)


def sbl_solver(m, trace, count, cfg):
    positions = grid(cfg['sbl_grid'])
    g = len(positions)
    gamma = torch.full((g,), float(m.energy.cpu())/m.n/g,
                       dtype=torch.float64, device=m.device)
    noise = 1./m.n  # approved white-noise variance with FFT/N
    chunk = cfg['fft_chunk']
    old_evidence = np.inf

    def posterior(points, gam, want_cov=False):
        second = torch.zeros_like(gam)
        energy = torch.zeros_like(gam)
        evidence, covs = 0., []
        for lo in range(0, m.n, chunk):
            m.check()
            a = m.response(points, m.f[lo:lo+chunk])
            mu, diag, cov, nll = sbl_posterior(a, m.x[lo:lo+chunk], gam, noise)
            second += (mu.abs().square()+diag).sum(0)
            energy += mu.abs().square().sum(0)
            evidence += float(nll.sum().cpu())
            if want_cov:
                covs.append(cov.detach())
            m.counts['covariance_decompositions'] += len(a)
        return second/m.n, energy/m.n, evidence/m.n, covs

    for iteration in range(cfg['sbl_iterations']):
        updated, power, value, _ = posterior(positions, gamma)
        ids = shortlist(positions, power.cpu().numpy(), count)
        trace.save(positions[ids], value)
        if trace.elapsed() <= trace.budget:
            m.native_map = dict(points=positions.tolist(), power=power.cpu().tolist())
        if value > old_evidence+1e-7:
            raise ArithmeticError('SBL evidence increased; implementation requires review')
        improvement = old_evidence-value
        old_evidence = value
        gamma = updated.clamp_min(1e-20)
        # Exact evidence line search for active continuous positions. This replaces
        # the paper's linearized free-gain/offset update, not its verbatim Algorithm 1.
        _, power, current, covs = posterior(positions, gamma, True)
        ids = shortlist(positions, power.cpu().numpy(), count)
        old_points = positions[ids].copy()
        active_gamma = gamma[ids]
        fixed = []
        for j, lo in enumerate(range(0, m.n, chunk)):
            a0 = m.response(old_points, m.f[lo:lo+chunk])
            fixed.append(covs[j]-(a0*active_gamma[None, None])@a0.mH)

        def evidence_grad(flat):
            total, gradient = 0., np.zeros_like(flat)
            for j, lo in enumerate(range(0, m.n, chunk)):
                m.check()
                p = torch.tensor(flat.reshape(-1, 2), device=m.device, requires_grad=True)
                a = atom(p, m.f[lo:lo+chunk], m.stations)
                m.counts['frequency_response_points'] += len(p)*len(a)
                cov = fixed[j]+(a*active_gamma[None, None])@a.mH
                chol = torch.linalg.cholesky(cov)
                x = m.x[lo:lo+chunk]
                invx = torch.cholesky_solve(x[..., None], chol).squeeze(-1)
                nll = (2*torch.log(chol.diagonal(dim1=1, dim2=2).real).sum(-1)
                       +(x.conj()*invx).sum(-1).real).sum()/m.n
                gradient += torch.autograd.grad(nll, p)[0].detach().cpu().numpy().ravel()
                total += float(nll.detach().cpu())
            return total, gradient
        bounds = [(max(-1000., v-cfg['sbl_grid']/2), min(1000., v+cfg['sbl_grid']/2))
                  for v in old_points.ravel()]
        result = minimize(evidence_grad, old_points.ravel(), jac=True, method='L-BFGS-B',
            bounds=bounds, options=dict(maxiter=3, ftol=1e-8, gtol=1e-6, maxls=8))
        if result.fun <= current+1e-9:
            positions[ids] = result.x.reshape(-1, 2)
        m.counts['iterations'] += 1
        if iteration >= 2 and 0 <= improvement < 1e-6:
            break
    _, power, value, _ = posterior(positions, gamma)
    ids = shortlist(positions, power.cpu().numpy(), count)
    trace.save(positions[ids], value)
    if trace.elapsed() <= trace.budget:
        m.native_map = dict(points=positions.tolist(), power=power.cpu().tolist())


def run(iq, method, count=2, budget=60., config=None, seed=0, guard=lambda: None):
    if method not in METHODS or count not in (1, 2):
        raise ValueError('Unknown method/count')
    cfg = dict(DEFAULT, **(config or {}))
    trace = Trace(budget, guard)
    model, status, error = None, 'FINISHED', None
    try:
        model = Model(iq, trace.check)
        if method in ('LS', 'MUSIC', 'MVDR'):
            spectrum_solver(model, trace, method, count, cfg)
        elif method == 'AP':
            ap_solver(model, trace, count, cfg)
        elif method == 'BNOMP':
            nomp_solver(model, trace, count, cfg)
        elif method == 'SBL':
            sbl_solver(model, trace, count, cfg)
        else:
            function = multistart_solver if method == 'MULTISTART' else ga_solver
            function(model, trace, count, cfg, np.random.default_rng(seed))
    except Deadline:
        status = 'BUDGET_EXHAUSTED'
    except (ArithmeticError, ValueError, torch.linalg.LinAlgError) as exc:
        status, error = 'NUMERICAL_FAILURE', str(exc)+'\n'+traceback.format_exc()
    elapsed = trace.elapsed()
    if status == 'FINISHED' and elapsed > budget:
        status = 'BUDGET_EXHAUSTED'
    return dict(method=method, count=count, budget=budget, seconds=elapsed, status=status,
                error=error, events=trace.events, seed=seed, config=cfg,
                counts={} if model is None else model.counts,
                native_map=getattr(model, 'native_map', None))


def at_time(result, limit):
    events = [e for e in result['events'] if e['seconds'] <= min(limit, result['budget'])]
    return events[-1]['positions'] if events else []
