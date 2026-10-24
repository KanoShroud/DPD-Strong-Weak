"""Full-waveform elimination with two-dimensional position-block Newton steps.

For fixed other-source positions, project data AND response derivatives onto
their orthogonal complement. Eliminating the current coefficient there is
equivalent to eliminating all waveform coefficients in the joint LS problem.
This is a separately labelled candidate, never a frozen-baseline replacement.
"""
import copy
import numpy as np
import torch

from ....g8.physics import C
from ...physics import grid
from ...methods import peaks
from ...r3.solvers import Model, Trace, projected_gradient
from ..adaptation_review import refine as native_step
from ..bnomp import raw_response
from .vp_route_diagnostic import fit_all, block_refine, now


def projected(z, unit):
    if unit is None:
        return z
    u = unit.reshape(*unit.shape, *([1]*(z.ndim-2)))
    return z-u*(u.conj()*z).sum(1, keepdim=True)


class ConditionalBlock:
    def __init__(self, model, others):
        self.m = model
        if len(others):
            a = raw_response(model, others).squeeze(-1)
            self.unit = a/torch.linalg.vector_norm(a, dim=1)[:, None]
        else:
            self.unit = None
        self.data = projected(model.x, self.unit)

    def response_derivatives(self, position, second=True):
        m = self.m
        p = torch.as_tensor(position, dtype=torch.float64, device=m.device)
        delta = p[None]-m.stations
        distance = torch.linalg.vector_norm(delta, dim=1)
        d = distance.clamp_min(1.)
        u = torch.where((distance > 1.)[:, None], delta/d[:, None], torch.zeros_like(delta))
        k = 2*torch.pi*m.f[:, None]/C
        a = torch.exp(-1j*k*d[None])/d[None]
        b = -1/d[None]-1j*k
        da = a[:, :, None]*b[:, :, None]*u[None]
        m.counts['frequency_response_points'] += m.n
        if not second:
            return projected(a, self.unit), projected(da, self.unit)
        uu = u[:, :, None]*u[:, None, :]
        du = (torch.eye(2, device=m.device)[None]-uu)/d[:, None, None]
        du = torch.where((distance > 1.)[:, None, None], du, torch.zeros_like(du))
        daa = a[:, :, None, None]*((b*b+1/d[None]**2)[:, :, None, None]*uu[None]
                                  +b[:, :, None, None]*du[None])
        return projected(a, self.unit), projected(da, self.unit), projected(daa, self.unit)

    def evaluate(self, position, derivatives=False):
        m = self.m
        if derivatives:
            a, da, daa = self.response_derivatives(position)
        else:
            a = projected(raw_response(m, position).squeeze(-1), self.unit)
        norm = a.abs().square().sum(1)
        if bool((norm < 1e-24).any()):
            raise ArithmeticError('Rank-deficient conditional response')
        q = (a.conj()*self.data).sum(1)/norm
        r = self.data-a*q[:, None]
        value = r.abs().square().sum()/m.energy
        if not derivatives:
            return float(value.cpu())
        jac = da*q[:, None, None]
        g = -2*(r.conj()[:, :, None]*jac).sum((0, 1)).real/m.energy
        fixed = 2*(torch.einsum('fli,flj->ij', jac.conj(), jac)
                   -(r.conj()[:, :, None, None]*daa*q[:, None, None, None]).sum((0, 1))).real/m.energy
        cross = (a.conj()[:, :, None]*da).sum(1)*q[:, None]
        res = (r.conj()[:, :, None]*da).sum(1)
        re, im = 2*(cross-res).real, 2*(cross+res).imag
        weight = 1/(2*norm*m.energy)
        correction = (torch.einsum('fi,fj,f->ij', re, re, weight)
                      +torch.einsum('fi,fj,f->ij', im, im, weight))
        m.counts['conditional_hessians'] = m.counts.get('conditional_hessians', 0)+1
        return float(value.cpu()), g.cpu().numpy(), (fixed-correction).cpu().numpy()


def local_block(m, start, others, cfg):
    block = ConditionalBlock(m, others)
    p = np.asarray(start, float).copy()
    detail = []
    for _ in range(cfg['bnomp_newton_steps']):
        value, g, h = block.evaluate(p, True)
        h = (h+h.T)/2
        if not np.isfinite(h).all() or not np.isfinite(g).all():
            raise ArithmeticError('Nonfinite conditional curvature')
        eig = float(np.linalg.eigvalsh(h).min())
        direction = -np.linalg.solve(h+np.eye(2)*max(0., 1e-9-eig), g)
        norm = np.linalg.norm(direction)
        direction *= min(1., cfg['bnomp_step_cap_m']/max(norm, 1e-30))
        accepted, shift, after = False, 0., value
        for scale in (1., .5, .25, .125, .0625):
            trial = np.clip(p+scale*direction, -1000., 1000.)
            try:
                after = block.evaluate(trial)
            except ArithmeticError:
                continue
            if after <= value:
                shift = float(np.linalg.norm(trial-p))
                p, accepted = trial, True
                break
        detail.append(dict(before=value, after=after,
                           accepted=accepted, shift=shift, floor_active=eig < 1e-9))
        if not accepted or shift < cfg['bnomp_min_step_m']:
            break
    return p, detail


def first_source(m, cfg):
    points = grid(cfg['grid'])
    first = points[int(np.argmax(m.conditional(points)))]
    p, a, q, _ = native_step(m, first, m.x, cfg, True)
    residual = m.x-a*q[:, None]
    for _ in range(cfg['bnomp_cycles']):
        before = float(residual.abs().square().sum().cpu()/m.energy.cpu())
        p, a, q, _ = native_step(m, p, m.x, cfg, True)
        residual = m.x-a*q[:, None]
        after = float(residual.abs().square().sum().cpu()/m.energy.cpu())
        if abs(before-after) <= cfg['bnomp_energy_tolerance']:
            break
    return p, points


def conditional_refine(m, initial, cfg, common=True):
    p = np.asarray(initial, float).copy()
    previous = m.joint(p)
    history, attempts = [], 0
    checkpoint, reason = None, 'ITERATION_LIMIT'
    for cycle in range(cfg['bnomp_cycles']):
        old = p.copy()
        for slot in range(2):
            p[slot], details = local_block(m, p[slot], p[1-slot][None], cfg)
            attempts += len(details)
        value, gradient = m.joint(p, True)
        pg = projected_gradient(p, gradient)
        if value > previous+1e-12:
            raise ArithmeticError('Conditional cycle increased joint objective')
        record = dict(cycle=cycle+1, value=value, projected_gradient=pg, positions=p.tolist())
        if abs(previous-value) <= cfg['bnomp_energy_tolerance'] and checkpoint is None:
            checkpoint = dict(record)
        if cycle < 3 or (cycle+1) % 25 == 0:
            history.append(record)
        if common and pg <= 1e-8:
            reason = 'COMMON_GRADIENT_REACHED'; break
        if not common and checkpoint is not None:
            reason = 'NATIVE_ENERGY_STABLE'; break
        if np.array_equal(old, p):
            reason = 'NUMERICAL_STAGNATION'; break
        previous = value
    return p, dict(cycles=cycle+1, attempts=attempts, history=history,
                   native_checkpoint=checkpoint, stop_reason=reason)


def solve(iq, method, cfg, ap_cfg):
    """No truth argument. Fixed variants isolate discovery and all-coefficient updates."""
    started = now()
    t = Trace(lambda: None)
    m = Model(iq, t.check, sample_rate_hz=cfg['sample_rate_hz'], half_band_hz=cfg['half_band_hz'])
    t.model = m
    first, points = first_source(m, cfg)
    if method in ('SEARCH_ONLY', 'FULL_CONDITIONAL'):
        scores = m.conditional(points, first[None])
    elif method == 'UPDATE_ONLY':
        _, _, residual = fit_all(m, first[None])
        scores = m.conditional(points, residual=residual)
    else:
        raise ValueError(method)
    second = points[peaks(points, scores, cfg['top'])[0]]
    initial = np.vstack([first, second])
    initialized = now()
    if method == 'SEARCH_ONLY':
        p, detail = block_refine(m, initial, cfg)
    else:
        p, detail = conditional_refine(m, initial, cfg, common=True)
    value, g = m.joint(p, True)
    done = now()
    t.model = None
    return {**detail, 'positions': p.tolist(), 'initial_positions': initial.tolist(), 'objective': value,
            'projected_gradient': projected_gradient(p, g), 'complete_seconds': done-started,
            'initialization_seconds': initialized-started, 'refinement_seconds': done-initialized,
            'counts': copy.deepcopy(m.counts)}


def checks(iq, cfg):
    """Autograd of the full joint objective independently checks each 2-D block."""
    m = Model(iq, sample_rate_hz=cfg['sample_rate_hz'], half_band_hz=cfg['half_band_hz'])
    probes = [np.array([[-210., 131.], [670., -460.]]),
              np.array([[130., -250.], [-310., 220.]])]
    records = []
    for p in probes:
        x = torch.tensor(p.ravel(), device=m.device, dtype=torch.float64, requires_grad=True)
        value = m.joint_tensor(x)
        gradient = torch.autograd.grad(value, x)[0].detach().cpu().numpy()
        hessian = torch.autograd.functional.hessian(m.joint_tensor, x).cpu().numpy()
        for slot in range(2):
            b = ConditionalBlock(m, p[1-slot][None])
            v, g, h = b.evaluate(p[slot], True)
            ss = slice(2*slot, 2*slot+2)
            np.testing.assert_allclose(v, float(value.detach().cpu()), rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(g, gradient[ss], rtol=1e-6, atol=1e-10)
            np.testing.assert_allclose(h, hessian[ss, ss], rtol=1e-5, atol=1e-10)
            fd = np.column_stack([(b.evaluate(p[slot]+d, True)[1]-b.evaluate(p[slot]-d, True)[1])/.002
                                  for d in np.eye(2)*.001])
            np.testing.assert_allclose(h, fd, rtol=2e-4, atol=1e-9)
            records.append(dict(slot=slot, objective_error=abs(v-float(value.detach().cpu())),
                                gradient_error=float(np.max(abs(g-gradient[ss]))),
                                hessian_error=float(np.max(abs(h-hessian[ss, ss])))))
        candidates = np.array([[70., 140.], [510., -330.], [-430., -270.]])
        scores = m.conditional(candidates, p[0][None])
        base = m.joint(p[0][None])
        for candidate, score in zip(candidates, scores):
            np.testing.assert_allclose(base-score, m.joint(np.vstack([p[0], candidate])),
                                       rtol=1e-10, atol=1e-12)
    return dict(passed=True, records=records,
                scope='Full joint objective with other positions fixed; all coefficients eliminated')
