"""Cell score upper bound from a response perturbation cone; frozen score unchanged."""
import time

import numpy as np
import torch

from ...g8.physics import C, peaks
from ..solver import far, local
from .coverage_search import BatchedScore


class BoundScore(BatchedScore):
    def bound(self, points, halfwidth):
        points = np.asarray(points).reshape(-1, 2)
        halfwidth = np.broadcast_to(halfwidth, (len(points),))
        values, bounds = [], []
        residual_energy = self.residual.abs().square().sum(-1)
        for lo in range(0, len(points), self.chunk):
            self.check()
            pts = torch.as_tensor(points[lo:lo+self.chunk], device='cuda')
            rho = torch.as_tensor(halfwidth[lo:lo+self.chunk]*np.sqrt(2), device='cuda')
            distance = torch.linalg.vector_norm(pts[:, None]-self.stations, dim=-1).clamp_min(1.)
            low = (distance-rho[:, None]).clamp_min(1.)
            phase = (2*np.pi*self.f.abs()[None, :]*rho[:, None]/C).clamp_max(np.pi)
            # |1/d'-1/d| + |exp(j phi')-exp(j phi)|/d, with |d'-d|<=rho.
            eps_per_station = (rho[:, None, None]/(distance*low)[:, None, :]
                               +2*torch.sin(phase/2)[..., None]/distance[:, None, :])
            eps = torch.linalg.vector_norm(eps_per_station, dim=-1)
            a = self.steering(points[lo:lo+self.chunk])
            if self.p is not None:
                a = torch.einsum('fmn,pfn->pfm', self.p, a)
            den = a.abs().square().sum(-1)
            dot = (a.conj()*self.residual[None]).sum(-1)
            q = torch.where(den>1e-20, dot.abs().square()/den.clamp_min(1e-20), 0.)
            t = (eps/den.sqrt().clamp_min(1e-30)).clamp(0, 1)
            u = (q/residual_energy.clamp_min(1e-30)).clamp(0, 1).sqrt()
            complement = (1-u.square()).clamp_min(0).sqrt()
            cap = (u*(1-t.square()).clamp_min(0).sqrt()+complement*t).square()
            cap = torch.where(t<complement, cap, 1.)
            cap = torch.where(den>1e-20, cap, 1.)
            values.extend((q.sum(-1)/self.energy).cpu().tolist())
            bounds.extend(((cap*residual_energy).sum(-1)/self.energy+1e-12).cpu().tolist())
            self.points += len(a)
        return np.asarray(values), np.asarray(bounds)


def update(calc, other, initial, method, guard=lambda: None):
    torch.cuda.synchronize()
    started = time.perf_counter()
    calc.set_sources([other])
    initial_score = float(calc.evaluate([initial])[0])
    before_points = calc.points
    if method in ('D', 'shift10'):
        axis = np.arange(-1000.,1000.01,5.) if method=='D' else np.arange(-995.,1000.,10.)
        xx, yy = np.meshgrid(axis,axis)
        pts = np.stack([xx.ravel(),yy.ravel()],-1)
        pts = pts[far(pts,[other])]
        values = calc.evaluate(pts)
        upper = np.zeros(len(pts))
        pruned = 0
    else:
        axis = np.arange(-975.,1000.,50.)
        xx, yy = np.meshgrid(axis,axis)
        pts = np.stack([xx.ravel(),yy.ravel()],-1)
        widths = np.full(len(pts),25.)
        if method=='E':
            values,upper = calc.bound(pts,widths)
        else:
            values=calc.evaluate(pts);upper=values.copy()
        active=np.ones(len(pts),bool)
        best=initial_score
        pruned=0
        while len(pts)+4<=16384:
            guard()
            valid=far(pts,[other])
            if valid.any():best=max(best,float(np.max(values[valid])))
            available=active & (widths>1.5625)
            if method=='E':
                remove=available & (upper<best)
                pruned+=int(remove.sum());active[remove]=False
                available &= ~remove
            indices=np.flatnonzero(available)
            if not len(indices):break
            count=min(32,len(indices),(16384-len(pts))//4)
            # Upper score first; score itself breaks saturated-upper-bound ties.
            order=np.lexsort((indices,-values[indices],-upper[indices]))
            parents=indices[order[:count]]
            active[parents]=False
            childwidth=np.repeat(widths[parents]/2,4)
            children=(pts[parents,None]+widths[parents,None,None]/2
                      *np.array([[-1,-1],[-1,1],[1,-1],[1,1]])).reshape(-1,2)
            if method=='E':v,b=calc.bound(children,childwidth)
            else:v=calc.evaluate(children);b=v.copy()
            pts=np.concatenate([pts,children]);values=np.r_[values,v];upper=np.r_[upper,b]
            widths=np.r_[widths,childwidth];active=np.r_[active,np.ones(len(children),bool)]
        valid=far(pts,[other]);pts,values,upper=pts[valid],values[valid],upper[valid]
    sampled=calc.points-before_points
    starts=pts[peaks(pts,values,5)]
    options=[np.asarray(initial)]
    for p in starts:
        options.extend([p,local(calc,p,[other])])
    scores=calc.evaluate(options)
    index=int(np.argmax(scores))
    torch.cuda.synchronize()
    return dict(position=np.asarray(options[index]).tolist(),score=float(scores[index]),
                seconds=time.perf_counter()-started,grid_points=sampled,pruned_cells=pruned,
                bound_max=float(upper.max()),initial_score=initial_score)
