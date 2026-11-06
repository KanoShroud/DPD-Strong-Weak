"""Matched full-gradient stopping; native AP initialization remains unchanged."""
import numpy as np
from scipy.optimize import minimize
from ..physics import grid
from ..methods import peaks
from ..r2.solvers import local as native_local
from .solvers import projected_gradient


class PrecisionReached(Exception):
    def __init__(self, positions, value, gradient):
        self.positions = np.asarray(positions).copy()
        self.value = float(value)
        self.gradient = np.asarray(gradient).copy()


def checked_point(model, trace, positions, cfg, phase, slot=None):
    """Only call on initialization or an accepted iterate, never line-search trials."""
    p = np.asarray(positions, dtype=float).reshape(2, 2)
    value, gradient = model.joint(p, True)
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise ArithmeticError('Nonfinite common-stop state')
    trace.save(p, value)
    trace.iteration(p, value, phase=phase, slot=slot)
    pg = projected_gradient(p, gradient)
    if pg <= cfg['gtol']:
        raise PrecisionReached(p, value, gradient)
    return value, pg


def common_local(model, positions, cfg, trace, slot=None):
    """Same L-BFGS-B update/bounds, ftol=0; full-gradient callback ends both APs."""
    initial = np.asarray(positions, float).reshape(2, 2).copy()
    before = model.joint(initial)
    model.counts['local_calls'] += 1

    def full(x):
        if slot is None:
            return np.asarray(x).reshape(2, 2)
        result = initial.copy()
        result[slot] = x
        return result

    def objective(x):
        value, gradient = model.joint(full(x), True)
        return value, gradient if slot is None else gradient.reshape(2, 2)[slot]

    accepted_steps = 0

    def callback(x):
        nonlocal accepted_steps
        accepted_steps += 1
        checked_point(model, trace, full(x), cfg, 'common_local', slot)

    x0 = initial.ravel() if slot is None else initial[slot]
    opt = minimize(objective, x0, jac=True, method='L-BFGS-B',
                   bounds=[(-1000., 1000.)]*len(x0), callback=callback,
                   options=dict(ftol=cfg['common_ftol'], gtol=cfg['gtol'],
                                maxls=cfg['maxls'], maxiter=cfg['local_maxiter']))
    p = full(opt.x)
    value = model.joint(p)
    if not np.isfinite(value) or not np.isfinite(p).all():
        raise ArithmeticError('Nonfinite common local solution')
    accepted = value <= before
    p, value = (p, value) if accepted else (initial, before)
    # Include a terminal point if the optimizer returned without a callback.
    checked_point(model, trace, p, cfg, 'common_local_return', slot)
    return p, value, dict(slot=slot, iterations=int(opt.nit), evaluations=int(opt.nfev),
        optimizer_success=bool(opt.success), status=int(opt.status), message=str(opt.message),
        accepted=accepted, before=float(before), after=float(value), accepted_steps=accepted_steps)


def refine_pair(model, trace, start, method, cfg):
    p = np.asarray(start, float).copy()
    local_info, completed_sweeps = [], 0
    reason = 'ITERATION_LIMIT'
    try:
        checked_point(model, trace, p, cfg, 'common_initial')
        if method == 'AP_1_JOINT':
            p, _, info = common_local(model, p, cfg, trace)
            local_info.append(info)
            reason = 'ITERATION_LIMIT' if info['status'] == 1 else 'NUMERICAL_STAGNATION'
        elif method == 'AP_1_ALT':
            for _ in range(cfg['alternating_sweeps']):
                old = p.copy()
                for slot in range(2):
                    p, _, info = common_local(model, p, cfg, trace, slot)
                    local_info.append(info)
                completed_sweeps += 1
                model.counts['iterations'] += 1
                # Exact no-motion is a numerical exit, not relaxed convergence.
                if np.array_equal(p, old):
                    reason = 'NUMERICAL_STAGNATION'
                    break
        else:
            raise ValueError('Common stop is only for the two single-candidate APs')
    except PrecisionReached as hit:
        p = hit.positions
        reason = 'COMMON_GRADIENT_REACHED'
    value, gradient = model.joint(p, True)
    trace.reason = reason
    return p, value, dict(local_optimizers=local_info, completed_sweeps=completed_sweeps,
                         stop_reason=reason, projected_gradient=projected_gradient(p, gradient))


def ap_common(model, trace, method, cfg):
    if cfg['common_ftol'] != 0. or cfg['gtol'] != 1e-8:
        raise ValueError('Approved common stop contract changed')
    points = grid(cfg['grid'])
    # Preserve the original single-source local ftol/gtol and starting point.
    with trace.stage('first_source'):
        power = model.conditional(points)
        first = points[peaks(points, power, cfg['top'])[0]]
        first, _, first_info = native_local(model, first[None], cfg)
    with trace.stage('conditional_candidates'):
        power = model.conditional(points, first)
        second = points[peaks(points, power, cfg['top'])[0]]
        start = np.vstack([first, second])
    with trace.stage('branch_refinement'):
        before = model.joint(start)
        trace.save(start, before)
        p, value, details = refine_pair(model, trace, start, method, cfg)
    return dict(first_position=first.tolist(), selected_rank=1, output_positions=p.tolist(),
        common_stop=details, local_optimizers=[first_info]+details['local_optimizers'],
        branches=[dict(initial_rank=1, initial_positions=start.tolist(), initial_objective=before,
                       final_positions=p.tolist(), final_objective=value, improvement=before-value,
                       stop_reason=details['stop_reason'])])
