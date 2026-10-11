"""相同投影目标、连续精修器和预算；A/B/C只改变候选搜索。"""
import time

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats import rankdata

from ..g8.physics import FS, C, RECEIVERS, Spectrum


class Deadline(Exception):
    """保存截止前已完成的最终位置，不把未完成更新用于评价。"""


class Score:
    def __init__(self, iq, check=lambda: None):
        self.check = check
        self.x = torch.as_tensor(np.fft.fftshift(np.fft.fft(iq, axis=-1), axes=-1).T/iq.shape[-1], device='cuda')
        self.f = torch.as_tensor(np.fft.fftshift(np.fft.fftfreq(iq.shape[-1], 1/FS)), device='cuda')
        self.stations = torch.as_tensor(RECEIVERS, dtype=torch.float64, device='cuda')
        self.energy = float(self.x.abs().square().sum().cpu())
        self.points = self.response_points = 0
        self.set_sources([])

    def steering(self, points, gradient=False):
        p = torch.as_tensor(np.asarray(points).reshape(-1, 2), dtype=torch.float64, device='cuda')
        difference = p[:, None]-self.stations
        distance = torch.linalg.vector_norm(difference, dim=-1).clamp_min(1.)
        a = torch.exp(-2j*np.pi*self.f[None, :, None]*distance[:, None, :]/C)/distance[:, None, :]
        if not gradient:
            return a
        dd = difference/distance[..., None]
        da = a[..., None]*(-1/distance[:, None, :, None]
            -2j*np.pi*self.f[None, :, None, None]/C)*dd[:, None]
        return a, da

    def set_sources(self, positions):
        self.check()
        if len(positions):
            a = self.steering(positions).permute(1, 2, 0)
            # 标准化列不改变投影，改善伪逆条件。
            a = a/torch.linalg.vector_norm(a, dim=1, keepdim=True)
            self.p = torch.eye(4, dtype=torch.complex128, device='cuda')[None]-a@torch.linalg.pinv(a, rtol=1e-10)
            self.p = (self.p+self.p.mH)/2
            self.residual = torch.einsum('fmn,fn->fm', self.p, self.x)
        else:
            self.p, self.residual = None, self.x

    def evaluate(self, points, variation=False):
        scores, changes = [], []
        points = np.asarray(points).reshape(-1, 2)
        for lo in range(0, len(points), 16):
            self.check()
            pts = points[lo:lo+16]
            a, da = self.steering(pts, True) if variation else (self.steering(pts), None)
            if self.p is not None:
                a = torch.einsum('fmn,pfn->pfm', self.p, a)
                if variation:
                    da = torch.einsum('fmn,pfnd->pfmd', self.p, da)
            den = a.abs().square().sum(-1)
            valid = den > 1e-20
            safe = den.clamp_min(1e-20)
            dot = (a.conj()*self.residual[None]).sum(-1)
            value = torch.where(valid, dot.abs().square()/safe, 0.).sum(-1)
            scores.extend((value/self.energy).cpu().tolist())
            self.points += len(pts)
            if variation:
                # 投影后单位响应的相位不变空间变化：去掉纯整体相位/幅度变化。
                inner = (a.conj()[..., None]*da).sum(-2)
                metric = (da.abs().square().sum(-2)/safe[..., None]
                          -inner.abs().square()/safe[..., None]**2).clamp_min(0).sum(-1)
                metric = torch.where(valid, metric, 0.).mean(-1)
                changes.extend(metric.cpu().tolist())
                self.response_points += len(pts)
        return (np.asarray(scores), np.asarray(changes)) if variation else np.asarray(scores)

    def value_gradient(self, point):
        self.check()
        a, da = self.steering([point], True)
        if self.p is not None:
            a = torch.einsum('fmn,pfn->pfm', self.p, a)
            da = torch.einsum('fmn,pfnd->pfmd', self.p, da)
        den = a.abs().square().sum(-1)
        safe = den.clamp_min(1e-20)
        dot = (a.conj()*self.residual[None]).sum(-1)
        ddot = (da.conj()*self.residual[None, :, :, None]).sum(-2)
        dden = 2*(a.conj()[..., None]*da).sum(-2).real
        valid = den > 1e-20
        value = torch.where(valid, dot.abs().square()/safe, 0.).sum()/self.energy
        deriv = (2*(dot.conj()[..., None]*ddot).real/safe[..., None]
                 -dot.abs().square()[..., None]*dden/safe[..., None]**2)
        deriv = torch.where(valid[..., None], deriv, 0.).sum((0, 1))/self.energy
        self.points += 1
        return -float(value.cpu()), -deriv.cpu().numpy()


def coarse_grid():
    axis = np.arange(-975., 1000., 50.)
    x, y = np.meshgrid(axis, axis)
    return np.stack([x.ravel(), y.ravel()], -1)


def far(points, others):
    return np.ones(len(points), bool) if not len(others) else (
        np.linalg.norm(points[:, None]-np.asarray(others)[None], axis=-1) >= 30).all(-1)


def local(calc, position, others):
    result = minimize(calc.value_gradient, np.asarray(position), jac=True,
                      method='L-BFGS-B', bounds=[(-1000, 1000)]*2,
                      options=dict(maxiter=30, ftol=1e-10, gtol=1e-7, maxls=12))
    if not far(result.x[None], others)[0]:
        return np.asarray(position)
    return result.x


def candidates(calc, others, method, count=5):
    pts = coarse_grid()
    valid = far(pts, others)
    pts = pts[valid]
    if method == 'C':
        scores, variation = calc.evaluate(pts, True)
    else:
        scores, variation = calc.evaluate(pts), np.zeros(len(pts))
    radii = np.full(len(pts), 25.)
    active = np.ones(len(pts), bool)
    if method in ('B', 'C'):
        for step in range(48):
            available = np.flatnonzero(active & (radii > 1.5))
            if not len(available):
                break
            # 每四次有一次全局探索；B/C完全一致。
            if step % 4 == 3:
                chosen = available[np.argmax(radii[available])]
            else:
                priority = rankdata(scores[available])/len(available)
                if method == 'C':
                    priority += .25*rankdata(variation[available]*radii[available]**2)/len(available)
                chosen = available[np.argmax(priority)]
            active[chosen] = False
            radius = radii[chosen]/2
            children = pts[chosen]+radius*np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
            children = children[far(children, others)]
            if not len(children):
                continue
            if method == 'C':
                values, change = calc.evaluate(children, True)
            else:
                values, change = calc.evaluate(children), np.zeros(len(children))
            pts = np.concatenate([pts, children])
            scores = np.r_[scores, values]
            variation = np.r_[variation, change]
            radii = np.r_[radii, np.full(len(children), radius)]
            active = np.r_[active, np.ones(len(children), bool)]
    chosen = []
    for index in np.argsort(-scores, kind='stable'):
        if far(pts[index:index+1], chosen)[0]:
            chosen.append(pts[index])
        if len(chosen) == count:
            break
    return chosen


def run(iq, method, budget, count=2, guard=lambda: None, hr_config=(2, 1e-4)):
    torch.cuda.synchronize()
    started = time.perf_counter()
    events, updates, calc = [], 0, None
    native_points = 0

    def check():
        guard()
        if time.perf_counter()-started >= budget:
            raise Deadline()

    def save(positions):
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        if elapsed <= budget:
            events.append(dict(seconds=elapsed, positions=np.asarray(positions).tolist()))

    try:
        if method in ('DPD', 'HR'):
            calc = Spectrum(iq, method='dpd' if method == 'DPD' else 'hr',
                            segments=hr_config[0], loading=hr_config[1], guard=check)
            points = coarse_grid()
            def native_evaluate(points):
                nonlocal native_points
                value = calc.evaluate(points)
                native_points += len(points)
                return value
            values = native_evaluate(points)
            candidates_native = []
            for index in np.argsort(-values, kind='stable'):
                if far(points[index:index+1], candidates_native)[0]:
                    candidates_native.append(points[index])
                if len(candidates_native) == count:
                    break
            save(candidates_native)
            for k in range(count):
                others = np.delete(np.asarray(candidates_native), k, axis=0)
                def objective(p):
                    check()
                    return -float(native_evaluate(np.asarray(p)[None])[0])/max(float(values.max()), 1e-30)
                result = minimize(objective, candidates_native[k], method='Powell',
                                  bounds=[(-1000, 1000)]*2,
                                  options=dict(maxiter=15, xtol=.1, ftol=1e-7))
                if far(result.x[None], others)[0] and objective(result.x) <= objective(candidates_native[k]):
                    candidates_native[k] = result.x
                save(candidates_native)
            return dict(events=events, seconds=time.perf_counter()-started,
                        score_points=native_points, response_points=0, updates=0, status='FINISHED')
        calc = Score(iq, check)
        starts = candidates(calc, [], method)
        best = -np.inf
        for start in starts:
            calc.set_sources([])
            first = local(calc, start, [])
            positions = [first]
            if count == 2:
                calc.set_sources(positions)
                second = candidates(calc, positions, method, 1)[0]
                positions.append(local(calc, second, positions))
            positions = np.asarray(positions)
            for iteration in range(5):
                for k in range(count):
                    others = np.delete(positions, k, axis=0)
                    calc.set_sources(others)
                    proposal = positions[k]
                    # 初轮只精修；后续同样允许所有方法回访整个区域。
                    if iteration in (2, 4):
                        found = candidates(calc, others, method, 1)[0]
                        if calc.evaluate([found])[0] > calc.evaluate([proposal])[0]:
                            proposal = found
                    refined = local(calc, proposal, others)
                    if calc.evaluate([refined])[0] >= calc.evaluate([positions[k]])[0]:
                        positions[k] = refined
                    updates += 1
                calc.set_sources(positions)
                score = 1-float(calc.residual.abs().square().sum().cpu())/calc.energy
                if score > best:
                    best = score
                    save(positions)
                if count == 1:
                    break
    except Deadline:
        pass
    torch.cuda.synchronize()
    return dict(events=events, seconds=time.perf_counter()-started,
                score_points=getattr(calc, 'points', native_points),
                response_points=getattr(calc, 'response_points', 0), updates=updates,
                status='BUDGET_EXHAUSTED' if time.perf_counter()-started >= budget else 'FINISHED')
