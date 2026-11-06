"""Two distinct native pipelines; no shared finite-record candidate frontend.

AP5: conditional grid discovery -> five independent joint refinements.
VP-BNOMP: residual grid discovery -> profiled Newton -> cyclic source updates
          -> joint coefficients after each completed discovery stage.
"""
import time

import numpy as np
from scipy.optimize import minimize

from ...g10.methods import peaks


CONFIG = dict(top=5, local_maxiter=100, ftol=0., gtol=1e-8, maxls=25,
              cycles=10000, energy_tolerance=1e-8, newton_steps=5,
              step_cap_m=50., min_step_m=.01, hessian_difference_m=.001)


def projected_gradient(p, g):
    return float(np.max(abs(np.asarray(p).ravel()-np.clip(np.asarray(p).ravel()-g, -1000, 1000))))


def local(model, start, cfg):
    start = np.asarray(start, float).reshape(-1, 2)
    history = []

    def objective(x):
        fit = model.fit(x.reshape(-1, 2), gradient=True)
        return fit['value'], fit['gradient']

    def callback(x):
        history.append(x.reshape(-1, 2).tolist())

    initial = model.fit(start)['value']
    opt = minimize(objective, start.ravel(), jac=True, method='L-BFGS-B', callback=callback,
                   bounds=[(-1000., 1000.)]*start.size,
                   options=dict(maxiter=cfg['local_maxiter'], ftol=cfg['ftol'],
                                gtol=cfg['gtol'], maxls=cfg['maxls']))
    p = opt.x.reshape(-1, 2)
    fit = model.fit(p, gradient=True)
    if fit['value'] > initial:
        p, fit = start, model.fit(start, gradient=True)
    pg = projected_gradient(p, fit['gradient'])
    return p, dict(objective=fit['value'], initial_objective=initial,
        projected_gradient=pg, iterations=int(opt.nit), evaluations=int(opt.nfev),
        stop_reason='GRADIENT_REACHED' if pg <= cfg['gtol'] else
                    ('ITERATION_LIMIT' if opt.status == 1 else 'NUMERICAL_EXIT'),
        message=str(opt.message), history=history)


def ap(model, points, cfg, progress=None):
    scores = model.scan(points, progress=progress)
    first_grid = points[peaks(points, scores, 1)[0]]
    first, first_detail = local(model, first_grid[None], cfg)
    scores = model.scan(points, others=first, progress=progress)
    ids = peaks(points, scores, cfg['top'])
    branches = []
    for rank, i in enumerate(ids, 1):
        start = np.vstack([first, points[i]])
        p, info = local(model, start, cfg)
        branches.append(dict(rank=rank, start=start.tolist(), positions=p.tolist(), **info))
    selected = min(branches, key=lambda x: (x['objective'], x['rank']))
    return dict(positions=selected['positions'], objective=selected['objective'],
        stop_reason=selected['stop_reason'], projected_gradient=selected['projected_gradient'],
        first_grid=first_grid.tolist(), first_detail=first_detail, branches=branches,
        selected_rank=selected['rank'], candidates=points[ids].tolist())


def profile_hessian(model, p, data, step=.001):
    """Reference reduced Hessian via centered differences of analytic gradients.

    Every displaced point re-eliminates its waveform. This is not a fixed-q
    Hessian or Gauss-Newton substitution. Agreement at step/2 is tested.
    """
    center = model.fit(np.asarray(p)[None], data, gradient=True)
    columns = []
    for delta in np.eye(2)*step:
        gp = model.fit((p+delta)[None], data, gradient=True)['gradient']
        gm = model.fit((p-delta)[None], data, gradient=True)['gradient']
        columns.append((gp-gm)/(2*step))
    h = np.column_stack(columns)
    return center, (h+h.T)/2


def newton(model, start, data, cfg):
    p, info = np.asarray(start, float).copy(), []
    for _ in range(cfg['newton_steps']):
        fit, h = model.hessian(p, data)
        before = fit['value']
        g = fit['gradient']
        if not np.isfinite(h).all() or not np.isfinite(g).all():
            raise ArithmeticError('Nonfinite VP derivatives')
        h += np.eye(2)*max(0., 1e-9-np.linalg.eigvalsh(h).min())
        direction = -np.linalg.solve(h, g)
        direction *= min(1., cfg['step_cap_m']/max(np.linalg.norm(direction), 1e-30))
        accepted, shift = False, 0.
        after = fit['value']
        for scale in (1., .5, .25, .125, .0625):
            proposal = np.clip(p+scale*direction, -1000, 1000)
            candidate = model.fit(proposal[None], data)
            after = candidate['value']
            if after <= fit['value']:
                shift = float(np.linalg.norm(proposal-p))
                p, fit, accepted = proposal, candidate, True
                break
        info.append(dict(before=before,
                         after=after, accepted=accepted, step_m=shift))
        if not accepted or shift < cfg['min_step_m']:
            break
    return p, model.fit(p[None], data), info


def vp(model, points, cfg, progress=None):
    residual = model.z.copy()
    positions, contributions, penalties = [], [], []
    discovery, cycles, history = [], [], []
    for source in range(2):
        initial = points[int(np.argmax(model.scan(points, data=residual, progress=progress)))]
        p, fit, details = newton(model, initial, residual, cfg)
        positions.append(p)
        contributions.append(fit['contributions'][0])
        penalties.append(model.ridge*float(np.vdot(fit['coefficients'], fit['coefficients']).real))
        residual -= contributions[-1]
        discovery.append(dict(initial=initial.tolist(), refined=p.tolist(), updates=details))
        reason = 'ITERATION_LIMIT'
        for cycle in range(cfg['cycles']):
            before = (float(np.vdot(residual, residual).real)+sum(penalties))/model.energy
            for slot in range(len(positions)):
                data = residual+contributions[slot]
                p, fit, _ = newton(model, positions[slot], data, cfg)
                positions[slot] = p
                contributions[slot] = fit['contributions'][0]
                penalties[slot] = model.ridge*float(np.vdot(fit['coefficients'], fit['coefficients']).real)
                residual = fit['residual']
            after = (float(np.vdot(residual, residual).real)+sum(penalties))/model.energy
            if after > before+1e-10:
                raise ArithmeticError('Cyclic updates increased their own regularized objective')
            history.append(dict(discovered=source+1, cycle=cycle+1, before=before, after=after,
                                positions=np.asarray(positions).tolist()))
            if abs(before-after) <= cfg['energy_tolerance']:
                reason = 'NATIVE_ENERGY_STABLE'
                break
        # Preserve PROFILE's joint-refit schedule, not the old more-frequent NOMP refits.
        joint = model.fit(positions)
        residual = joint['residual']
        contributions = list(joint['contributions'])
        c = joint['coefficients']
        if c.ndim == 2:  # per-frequency periodic coefficients
            penalties = [0.]*len(positions)
        else:
            penalties = [model.ridge*float(np.vdot(q, q).real) for q in c.reshape(len(positions), -1)]
        cycles.append(dict(discovered=source+1, cycles=cycle+1, stop_reason=reason,
                           before_joint=after, after_joint=joint['value']))
    final = model.fit(positions, gradient=True)
    return dict(positions=np.asarray(positions).tolist(), objective=final['value'],
        projected_gradient=projected_gradient(positions, final['gradient']),
        stop_reason='ITERATION_LIMIT' if any(x['stop_reason'] == 'ITERATION_LIMIT' for x in cycles)
                    else 'NATIVE_ENERGY_STABLE', discovery=discovery, cycles=cycles, history=history)


def run(model, algorithm, points, config=None, progress=None):
    if algorithm not in ('AP5', 'VP'):
        raise ValueError(algorithm)
    cfg = dict(CONFIG, **(config or {}))
    tick = time.perf_counter()
    result = (ap if algorithm == 'AP5' else vp)(model, np.asarray(points), cfg, progress)
    result.update(algorithm=algorithm, config=cfg, counts=dict(model.counts),
                  solve_seconds=time.perf_counter()-tick, preparation_seconds=model.preparation_seconds,
                  model=model.metadata)
    result['seconds'] = result['solve_seconds']+result['preparation_seconds']
    return result
