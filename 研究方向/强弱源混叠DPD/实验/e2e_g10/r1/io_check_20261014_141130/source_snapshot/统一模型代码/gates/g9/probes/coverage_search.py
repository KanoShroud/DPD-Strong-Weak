"""Same FP64 projected score; batch-only optimization and dense search helpers."""
import time

import numpy as np
import torch

from ...g8.physics import peaks
from ..solver import Score, far, local


class BatchedScore(Score):
    def __init__(self, iq, check=lambda: None, chunk=64):
        self.chunk = chunk
        super().__init__(iq, check)

    def evaluate(self, points, variation=False):
        if variation:
            return super().evaluate(points, variation)
        outputs = []
        points = np.asarray(points).reshape(-1, 2)
        for lo in range(0, len(points), self.chunk):
            self.check()
            a = self.steering(points[lo:lo+self.chunk])
            if self.p is not None:
                a = torch.einsum('fmn,pfn->pfm', self.p, a)
            den = a.abs().square().sum(-1)
            dot = (a.conj()*self.residual[None]).sum(-1)
            value = torch.where(den > 1e-20, dot.abs().square()/den.clamp_min(1e-20), 0.).sum(-1)
            outputs.extend((value/self.energy).cpu().tolist())
            self.points += len(a)
        return np.asarray(outputs)


def dense(calc, other, initial, step, guard=lambda: None):
    """Truth-independent single-coordinate update; retain input and score ascent."""
    torch.cuda.synchronize()
    started = time.perf_counter()
    calc.set_sources([other])
    axis = np.arange(-1000., 1000.01, step)
    x, y = np.meshgrid(axis, axis)
    points = np.stack([x.ravel(), y.ravel()], -1)
    points = points[far(points, [other])]
    scores = calc.evaluate(points)
    starts = points[peaks(points, scores, 5)]
    candidates = [np.asarray(initial)]
    for p in starts:
        guard()
        candidates.extend([p, local(calc, p, [other])])
    values = calc.evaluate(candidates)
    index = int(np.argmax(values))
    torch.cuda.synchronize()
    return dict(position=np.asarray(candidates[index]).tolist(), score=float(values[index]),
                grid_points=len(points), seconds=time.perf_counter()-started)
