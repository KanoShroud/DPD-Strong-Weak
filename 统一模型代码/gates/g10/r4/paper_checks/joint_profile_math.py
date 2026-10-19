"""Exact joint variable-projection curvature for two unknown wideband sources.

For real position theta and complex frequency coefficients s, let
Q_i=(dA/dtheta_i)s, B_i=(dA/dtheta_i)^H r-A^H Q_i, G=A^H A.
The reduced Hessian is H_fixed - 2 Re sum_f B_f^H G_f^-1 B_f / E.
No large frequency-by-frequency coefficient Hessian is assembled.
"""
import numpy as np
import torch
from ....g8.physics import C
from ...r3.solvers import projected_gradient


def derivatives(model, positions):
    p = torch.as_tensor(np.asarray(positions).reshape(2, 2), device=model.device, dtype=torch.float64)
    delta = p[:, None]-model.stations
    distance = torch.linalg.vector_norm(delta, dim=-1)
    d = distance.clamp_min(1.)
    u = torch.where((distance > 1.)[..., None], delta/d[..., None], torch.zeros_like(delta))
    d, u = d.T, u.permute(1, 0, 2)  # receiver, source, [coordinate]
    kappa = 2*torch.pi*model.f[:, None, None]/C
    a = torch.exp(-1j*kappa*d[None])/d[None]
    first = -1/d[None]-1j*kappa
    da = a[..., None]*first[..., None]*u[None]
    uu = u[..., :, None]*u[..., None, :]
    du = (torch.eye(2, device=model.device)-uu)/d[..., None, None]
    du = torch.where((distance.T > 1.)[..., None, None], du, torch.zeros_like(du))
    daa = a[..., None, None]*((first**2+1/d[None]**2)[..., None, None]*uu[None]
                             +first[..., None, None]*du[None])
    inv = torch.linalg.pinv(a, rtol=1e-10)
    s = (inv@model.x[..., None]).squeeze(-1)
    r = model.x-(a@s[..., None]).squeeze(-1)
    q = (da*s[:, None, :, None]).reshape(len(model.f), 4, 4)
    value = r.abs().square().sum()/model.energy
    gradient = -2*(r.conj()[..., None]*q).sum((0, 1)).real/model.energy
    fixed = 2*torch.einsum('fli,flj->ij', q.conj(), q).real
    second = 2*torch.einsum('fl,flkij,fk->kij', r.conj(), daa, s).real
    for k in range(2):
        fixed[2*k:2*k+2, 2*k:2*k+2] -= second[k]
    b = -(a.mH@q)
    extra = torch.einsum('flki,fl->fki', da.conj(), r)
    for k in range(2):
        b[:, k, 2*k:2*k+2] += extra[:, k]
    # G^-1 = A^dagger (A^dagger)^H at full column rank.
    gram_inverse = inv@inv.mH
    correction = 2*torch.einsum('fki,fkj->ij', b.conj(), gram_inverse@b).real
    hessian = (fixed-correction)/model.energy
    model.counts['joint_profile_hessians'] = model.counts.get('joint_profile_hessians', 0)+1
    model.counts['frequency_response_points'] += 2*len(model.f)
    model.counts['coefficient_frequency_solves'] = model.counts.get('coefficient_frequency_solves', 0)+len(model.f)
    value_cpu = float(value.cpu())
    gradient_cpu = gradient.cpu().numpy()
    # Share the same exact-position memoization used by L-BFGS-B certificates.
    if getattr(model, 'memo_enabled', False):
        key_positions = np.ascontiguousarray(np.asarray(positions, dtype=np.float64).reshape(2, 2))
        model.memo[(key_positions.shape, key_positions.tobytes())] = (value_cpu, gradient_cpu.copy())
    return value_cpu, gradient_cpu, hessian.cpu().numpy()


def refine(model, start, cfg, block=False):
    """Identical objective/gradient and safeguards; only off-diagonal source blocks differ."""
    p = np.asarray(start, float).reshape(2, 2).copy()
    history = []
    reason = 'ITERATION_LIMIT'
    for iteration in range(cfg['local_maxiter']):
        model.check()
        value, g, h = derivatives(model, p)
        pg = projected_gradient(p, g)
        if not np.isfinite(h).all() or not np.isfinite(g).all():
            raise ArithmeticError('Nonfinite joint profile derivatives')
        if pg <= cfg['gtol']:
            reason = 'COMMON_GRADIENT_REACHED'
            break
        if block:
            h[:2, 2:] = 0.
            h[2:, :2] = 0.
        h = (h+h.T)/2
        eigen_min = float(np.linalg.eigvalsh(h).min())
        damping = max(0., 1e-9-eigen_min)
        direction = -np.linalg.solve(h+np.eye(4)*damping, g)
        direction *= min(1., 50./max(np.linalg.norm(direction), 1e-30))
        accepted = False
        for ls in range(cfg['maxls']):
            trial = np.clip(p.ravel()+2.**(-ls)*direction, -1000., 1000.).reshape(2, 2)
            step = trial-p
            new = model.joint(trial)
            if new <= value+1e-4*float(g@step.ravel()) and float(g@step.ravel()) < 0:
                accepted = True
                break
        history.append(dict(iteration=iteration+1, value=value, projected_gradient=pg,
                            step_m=float(np.linalg.norm(step)), accepted=accepted,
                            line_search_trials=ls+1, damping=damping))
        if not accepted:
            reason = 'NUMERICAL_STAGNATION'
            break
        if np.array_equal(p, trial):
            reason = 'NUMERICAL_STAGNATION'
            break
        p = trial
    value, gradient = model.joint(p, True)
    pg = projected_gradient(p, gradient)
    if pg <= cfg['gtol']:
        reason = 'COMMON_GRADIENT_REACHED'
    return p, value, dict(stop_reason=reason, projected_gradient=pg, iterations=len(history), history=history)


def check(model):
    """Independent autograd and gradient finite differences at off-grid positions."""
    probes = [np.array([[-210., 131.], [670., -460.]]),
              np.array([[130., -250.], [-310., 220.]])]
    rows = []
    for p in probes:
        value, gradient, hessian = derivatives(model, p)
        x = torch.tensor(p.ravel(), device=model.device, dtype=torch.float64, requires_grad=True)
        ref_value = model.joint_tensor(x)
        ref_g = torch.autograd.grad(ref_value, x)[0].detach().cpu().numpy()
        ref_h = torch.autograd.functional.hessian(model.joint_tensor, x).cpu().numpy()
        np.testing.assert_allclose(value, float(ref_value.detach().cpu()), rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(gradient, ref_g, rtol=1e-7, atol=1e-10)
        np.testing.assert_allclose(hessian, ref_h, rtol=1e-6, atol=1e-10)
        finite = []
        for d in np.eye(4)*.001:
            g1 = model.joint((p.ravel()+d).reshape(2, 2), True)[1]
            g0 = model.joint((p.ravel()-d).reshape(2, 2), True)[1]
            finite.append((g1-g0)/.002)
        np.testing.assert_allclose(hessian, np.array(finite).T, rtol=2e-4, atol=1e-9)
        rows.append(dict(value_error=abs(value-float(ref_value.detach().cpu())),
                         gradient_max_error=float(np.max(abs(gradient-ref_g))),
                         hessian_max_error=float(np.max(abs(hessian-ref_h)))))
    return rows
