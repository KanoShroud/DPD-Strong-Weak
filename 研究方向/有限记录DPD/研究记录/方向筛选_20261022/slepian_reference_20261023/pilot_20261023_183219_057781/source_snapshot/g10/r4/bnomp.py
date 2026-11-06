"""R117 §4.2 (13)-(18), Appendix II: fixed-coefficient Newton, cyclic then LS.

RF/baseband geometry adaptation; NOT an author-code reproduction. Safeguarded
Newton is explicit: positive Hessian, 50 m cap, descent backtracking, box bounds.
Raw 1/d responses are necessary to retain the paper's fixed-q parameterization.
"""
import numpy as np
import torch
from ...g8.physics import C
from ..physics import grid, atom
from ..r1.physics import small_fit


def raw_response(model, positions):
    p = torch.as_tensor(np.asarray(positions).reshape(-1, 2), dtype=torch.float64,
                        device=model.device)
    model.counts['frequency_response_points'] += len(p)*model.n
    return atom(p, model.f, model.stations, normalized=False)


def coefficient(a, data):
    return (a.conj()*data).sum(1)/a.abs().square().sum(1)


def fixed_derivatives(model, position, data, q):
    """Value/gradient/Hessian of ||data-a(p)q||²/E, with q held fixed.

    Independent analytic evaluation of Appendix II; finite-difference/autograd
    checks cover both amplitude and delay derivatives, including cross terms.
    """
    p = torch.as_tensor(position, dtype=torch.float64, device=model.device)
    delta = p[None]-model.stations
    distance = torch.linalg.vector_norm(delta, dim=1)
    d = distance.clamp_min(1.)
    u = delta/d[:, None]
    u = torch.where((distance > 1.)[:, None], u, torch.zeros_like(u))
    kappa = 2*torch.pi*model.f[:, None]/C
    a = torch.exp(-1j*kappa*d[None])/d[None]
    first = -1/d[None]-1j*kappa
    aa = a[:, :, None]*first[:, :, None]*u[None]
    uu = u[:, :, None]*u[:, None, :]
    du = (torch.eye(2, device=model.device)[None]-uu)/d[:, None, None]
    du = torch.where((distance > 1.)[:, None, None], du, torch.zeros_like(du))
    aaa = a[:, :, None, None]*((first**2+1/d[None]**2)[:, :, None, None]*uu[None]
                                 +first[:, :, None, None]*du[None])
    residual = data-a*q[:, None]
    jac = aa*q[:, None, None]
    second = aaa*q[:, None, None, None]
    gradient = -2*(residual.conj()[:, :, None]*jac).sum((0, 1)).real/model.energy
    hessian = 2*(torch.einsum('fli,flj->ij', jac.conj(), jac)
                 -(residual.conj()[:, :, None, None]*second).sum((0, 1))).real/model.energy
    value = residual.abs().square().sum()/model.energy
    model.counts['frequency_response_points'] += model.n
    model.counts['fixed_q_derivative_calls'] = model.counts.get('fixed_q_derivative_calls', 0)+1
    return float(value.cpu()), gradient.cpu().numpy(), hessian.cpu().numpy()


def refine(model, initial, data, cfg):
    p = np.asarray(initial, float).copy()
    a = raw_response(model, p).squeeze(-1)
    q = coefficient(a, data)
    details = []
    for _ in range(cfg['bnomp_newton_steps']):
        model.check()
        value, gradient, hessian = fixed_derivatives(model, p, data, q)
        if not np.isfinite(value) or not np.isfinite(gradient).all() or not np.isfinite(hessian).all():
            raise ArithmeticError('Nonfinite fixed-q Newton state')
        hessian = (hessian+hessian.T)/2
        hessian += np.eye(2)*max(0., 1e-9-np.linalg.eigvalsh(hessian).min())
        direction = -np.linalg.solve(hessian, gradient)
        direction *= min(1., cfg['bnomp_step_cap_m']/max(np.linalg.norm(direction), 1e-30))
        accepted, shift = False, 0.
        for scale in (1., .5, .25, .125, .0625):
            proposal = np.clip(p+scale*direction, -1000., 1000.)
            candidate = raw_response(model, proposal).squeeze(-1)
            # Accept against the SAME fixed-q objective, not a different joint LS.
            after = float((data-candidate*q[:, None]).abs().square().sum().cpu()/model.energy.cpu())
            if after <= value:
                shift = float(np.linalg.norm(proposal-p))
                p, a, accepted = proposal, candidate, True
                q = coefficient(a, data)  # Eq. (15), after position acceptance.
                break
        details.append(dict(before=value, fixed_q_after=after, accepted=accepted, step_m=shift))
        if not accepted or shift < cfg['bnomp_min_step_m']:
            break
    return p, a, q, details


def solve(model, trace, count, cfg):
    residual, positions, atoms, coeffs = model.x, [], [], []
    discovery, cycles, updates = [], [], []
    reasons = []
    with trace.stage('discovery_fixed_q_newton_cyclic_then_joint_ls'):
        for source in range(count):
            points = grid(cfg['grid'])
            scores = model.conditional(points, residual=residual)
            initial = points[int(np.argmax(scores))]
            p, a, q, info = refine(model, initial, residual, cfg)
            positions.append(p); atoms.append(a); coeffs.append(q)
            residual = residual-a*q[:, None]
            discovery.append(dict(source=source, grid_position=initial.tolist(), position=p.tolist()))
            updates.extend(info)
            reason = 'ITERATION_LIMIT'
            for cycle in range(cfg['bnomp_cycles']):
                before = float(residual.abs().square().sum().cpu()/model.energy.cpu())
                for slot in range(len(positions)):
                    data = residual+atoms[slot]*coeffs[slot][:, None]
                    p, a, q, info = refine(model, positions[slot], data, cfg)
                    positions[slot], atoms[slot], coeffs[slot] = p, a, q
                    residual = data-a*q[:, None]
                    updates.extend(info)
                after = float(residual.abs().square().sum().cpu()/model.energy.cpu())
                trace.iteration(positions, after, phase='fixed_q_cyclic', discovered=source+1,
                                cycle=cycle+1, energy_before=before)
                model.counts['iterations'] += 1
                if abs(before-after) <= cfg['bnomp_energy_tolerance']:
                    reason = 'NATIVE_ENERGY_STABLE'
                    break
            # Paper step (d): joint coefficients ONLY after cyclic convergence/cap.
            matrix = torch.stack(atoms, -1)
            scale = torch.linalg.vector_norm(matrix, dim=1)
            normalized = matrix/scale[:, None, :]
            fitted, residual = small_fit(normalized, model.x)
            fitted = fitted/scale
            coeffs = [fitted[:, i] for i in range(len(positions))]
            model.counts['coefficient_joint_refits'] = model.counts.get('coefficient_joint_refits', 0)+1
            cycles.append(dict(discovered=source+1, cycles=cycle+1, stop_reason=reason,
                               residual_energy=float(residual.abs().square().sum().cpu()/model.energy.cpu())))
            reasons.append(reason)
            if source+1 == count:
                trace.save(positions, model.joint(positions))
    trace.reason = 'ITERATION_LIMIT' if 'ITERATION_LIMIT' in reasons else 'NATIVE_ENERGY_STABLE'
    return dict(output_positions=np.asarray(positions).tolist(), discovery=discovery,
                native_cycles=cycles, fixed_q_updates=updates,
                joint_refit_schedule='once after cyclic feedback for each new source',
                response_parameterization='unnormalized_1_over_d_fixed_coefficients')
