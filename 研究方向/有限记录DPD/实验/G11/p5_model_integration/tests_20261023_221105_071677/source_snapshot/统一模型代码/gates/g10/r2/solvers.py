"""Observation-only solvers. Diagnostics never affect selection or termination."""
import copy
import time
import numpy as np
import torch
from scipy.optimize import minimize
from ..physics import grid
from ..methods import peaks, newton_single
from ..r1.physics import Model, small_fit
from ..r1.methods import Trace, nomp_solver
from .config import CONFIG, METHODS


class CountModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counts.update(gradient_calls=0, grid_scans=0, local_calls=0, residual_scalar_calls=0)

    def joint(self, positions, gradient=False):
        self.counts['gradient_calls'] += int(gradient)
        return super().joint(positions, gradient)

    def conditional(self, points, *args, **kwargs):
        self.counts['grid_scans'] += 1
        return super().conditional(points, *args, **kwargs)

    def residual_score(self, p, residual):
        self.counts['residual_scalar_calls'] += 1
        return super().residual_score(p, residual)


def local(m, positions, cfg, trace=None, slot=None):
    """Same scalar objective/tolerances for joint and exact-coordinate refinement."""
    initial = np.asarray(positions, float).reshape(-1, 2).copy()
    before = m.joint(initial)
    m.counts['local_calls'] += 1

    def full(x):
        if slot is None:
            return np.asarray(x).reshape(-1, 2)
        candidate = initial.copy()
        candidate[slot] = x
        return candidate

    def objective(x):
        value, gradient = m.joint(full(x), True)
        return value, gradient if slot is None else gradient.reshape(-1, 2)[slot]

    def callback(x):
        if trace is not None:
            candidate = full(x)
            value = m.joint(candidate)
            trace.save(candidate, value)
            trace.iteration(candidate, value, phase='local', slot=slot)

    x0 = initial.ravel() if slot is None else initial[slot]
    opt = minimize(objective, x0, jac=True, method='L-BFGS-B',
                   bounds=[(-1000., 1000.)]*len(x0), callback=callback,
                   options={k: cfg[k] for k in ('ftol', 'gtol', 'maxls')}
                   | dict(maxiter=cfg['local_maxiter']))
    candidate = full(opt.x)
    value = m.joint(candidate)
    if not np.isfinite(value) or not np.isfinite(candidate).all():
        raise ArithmeticError('Nonfinite local solution')
    accepted = value <= before
    result, value = (candidate, value) if accepted else (initial, before)
    if trace is not None:
        trace.save(result, value)
    info = dict(slot=slot, iterations=int(opt.nit), evaluations=int(opt.nfev),
                optimizer_success=bool(opt.success), status=int(opt.status),
                message=str(opt.message), accepted=accepted, before=float(before), after=float(value))
    return result, value, info


def alternate(m, positions, cfg, trace):
    p = np.asarray(positions, float).copy()
    value, stable, info = m.joint(p), 0, []
    for sweep in range(cfg['alternating_sweeps']):
        before, old = value, p.copy()
        for slot in range(len(p)):
            p, value, item = local(m, p, cfg, trace, slot)
            info.append(item)
        shift = float(np.linalg.norm(p-old, axis=1).max())
        stable = stable+1 if before-value < cfg['objective_tolerance'] and shift < cfg['position_tolerance'] else 0
        m.counts['iterations'] += 1
        trace.iteration(p, value, phase='alternating', sweep=sweep+1, shift_m=shift)
        if stable >= 2:
            return p, value, info, 'NATIVE_STABLE'
    return p, value, info, 'ITERATION_LIMIT'


def ap(m, t, method, cfg):
    points = grid(cfg['grid'])
    with t.stage('first_source'):
        power = m.conditional(points)
        first = points[peaks(points, power, cfg['top'])[0]]
        first, _, first_info = local(m, first[None], cfg)
    with t.stage('conditional_candidates'):
        power = m.conditional(points, first)
        ids = peaks(points, power, cfg['top'])
        ids = ids[:1] if method.startswith('AP_1_') else ids
        raw = [np.vstack([first, points[i]]) for i in ids]
    branches, local_info = [], [first_info]
    with t.stage('branch_refinement'):
        for rank, start in enumerate(raw, 1):
            before = m.joint(start)
            t.save(start, before)
            if method.endswith('_JOINT'):
                p, value, detail = local(m, start, cfg, t)
                details = [detail]
                reason = 'OPTIMIZER_CONVERGED' if detail['optimizer_success'] else 'OPTIMIZER_EXIT'
            else:
                p, value, details, reason = alternate(m, start, cfg, t)
            branches.append(dict(initial_rank=rank, initial_positions=start.tolist(),
                                 initial_objective=before, final_positions=p.tolist(),
                                 final_objective=value, improvement=before-value, stop_reason=reason))
            local_info.extend(details)
    selected = min(branches, key=lambda x: (x['final_objective'], x['initial_rank']))
    t.reason = 'COMPLETED_BRANCHES'
    return dict(first_position=first.tolist(), branches=branches,
                selected_rank=selected['initial_rank'], local_optimizers=local_info)


def extra_cycles(m, t, start, cfg):
    p, stable = np.asarray(start, float).copy(), 0
    for iteration in range(cfg['extra_cycles']):
        before, old = m.joint(p), p.copy()
        for slot in range(len(p)):
            a, s, r = m.fitted(p)
            residual = r+a[:, :, slot]*s[:, slot, None]
            proposal = p.copy()
            proposal[slot] = newton_single(m, p[slot], residual, steps=5)
            if m.joint(proposal) <= m.joint(p):
                p = proposal
        value = m.joint(p)
        shift = float(np.linalg.norm(p-old, axis=1).max())
        stable = stable+1 if before-value < cfg['objective_tolerance'] and shift < cfg['position_tolerance'] else 0
        t.save(p, value)
        t.iteration(p, value, phase='extra_cycles', cycle=iteration+1, shift_m=shift)
        m.counts['iterations'] += 1
        if stable >= 2:
            return 'NATIVE_STABLE'
    return 'ITERATION_LIMIT'


def curvature(m, positions):
    """Post-solve diagnostic: exact profiled Hessian, not a solver controller."""
    p = torch.tensor(np.asarray(positions).ravel(), dtype=torch.float64,
                     device=m.device, requires_grad=True)
    def smooth_objective(z):
        # abs(z)**2 has an intermediate nondifferentiable abs at exact zero;
        # the equivalent real quadratic retains the correct second derivative.
        residual = small_fit(m.response(z.reshape(-1, 2)), m.x)[1]
        return (residual.real.square()+residual.imag.square()).sum()/m.energy
    h = torch.autograd.functional.hessian(smooth_objective, p).detach().cpu().numpy()
    h = (h+h.T)/2
    if not np.isfinite(h).all():
        return dict(valid=False, reason='NONFINITE_HESSIAN')
    ev = np.linalg.eigvalsh(h)
    info = dict(valid=True, hessian=h.tolist(), eigenvalues=ev.tolist(),
                positive_definite=bool(ev[0] > 1e-10*max(abs(ev[-1]), 1e-30)),
                ideal_block_gs_rho=None)
    if info['positive_definite']:
        transition = np.linalg.solve(h[2:, 2:], h[2:, :2]) @ np.linalg.solve(h[:2, :2], h[:2, 2:])
        info['ideal_block_gs_rho'] = float(max(abs(np.linalg.eigvals(transition))))
    return info


def run(iq, method, cfg=None, guard=lambda: None, diagnostics=True):
    """No truth, labels or expected success in the algorithm interface."""
    if method not in METHODS:
        raise ValueError(method)
    cfg = copy.deepcopy(CONFIG if cfg is None else cfg)
    t = Trace(guard)
    with t.stage('input_fft_transfer'):
        m = CountModel(iq, t.check, sample_rate_hz=cfg['sample_rate_hz'])
    if method.startswith('AP_'):
        details = ap(m, t, method, cfg)
    else:
        nomp_solver(m, t, 2, cfg['original_nomp'])
        start = np.array(t.events[-1]['positions'])
        details = dict(prefix_positions=start.tolist(), prefix_seconds=t.elapsed(),
                       prefix_objective=t.best, prefix_stop_reason=t.reason)
        if method == 'NOMP_EXTRA12':
            with t.stage('extra_cycles'):
                t.reason = extra_cycles(m, t, start, cfg)
        elif method == 'NOMP_JOINT':
            with t.stage('joint_refinement'):
                _, _, detail = local(m, start, cfg, t)
            details['local_optimizers'] = [detail]
            t.reason = 'OPTIMIZER_CONVERGED' if detail['optimizer_success'] else 'OPTIMIZER_EXIT'
    torch.cuda.synchronize()
    result = dict(method=method, count=2, config=cfg, seconds=t.elapsed(),
                  status='FINISHED', stop_reason=t.reason, events=t.events,
                  history=t.history, stages=t.stages, counts=copy.deepcopy(m.counts),
                  positions=t.events[-1]['positions'], mechanism=details)
    if diagnostics and method in ('NOMP', 'NOMP_JOINT'):
        begin = time.perf_counter()
        result['curvature'] = curvature(m, details['prefix_positions'])
        torch.cuda.synchronize()
        result['diagnostics_seconds'] = time.perf_counter()-begin
    else:
        result['diagnostics_seconds'] = 0.
    return result
