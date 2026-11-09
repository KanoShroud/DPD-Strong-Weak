# ruff: noqa: F401
# Preserve imported globals for legacy function cloning and reexports.
"""Eq9--16 with the approved <= convention and exact histogram endpoints.

No clipping of a genuinely excessive threshold, no hidden fallback or ridge.
The optional raw decoder is a distinct diagnostic, never the reference result.
"""
import sys
import time
from pathlib import Path
import numpy as np

pass  # Explicit package imports replace historical search-path mutation.
from 研究方向.多径DPD.代码.mp_dpd.baselines.xia_reference import score  # noqa: E402


def threshold(values):
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite spatial score')
    lo, hi = float(values.min()), float(values.max())
    if lo == hi:
        return {'status': 'DEGENERATE_RANGE', 'keep': np.zeros(len(values), bool)}
    count, edges = np.histogram(values, bins=10)
    b = int(count.argmax()) + 1  # one-based modal bin
    # T=lo+2*b*(hi-lo)/10. At b=5 its exact value is hi.
    # This is an algebraic endpoint identity, not a tolerance or a cap.
    t = hi if b == 5 else lo + (b/5.)*(hi-lo)
    return {'status': 'OK', 'keep': values >= t, 'threshold': t,
            'modal_bin': b, 'histogram': count.tolist(), 'minimum': lo, 'maximum': hi}


def decode(points, values, k, delta_h, use_threshold=True):
    filt = threshold(values)
    keep = filt['keep'] if use_threshold else np.ones(len(values), bool)
    meta = {key: value for key, value in filt.items() if key != 'keep'}
    meta['retained'] = int(keep.sum())
    if not keep.any():
        return {**meta, 'status': 'NO_RETAINED_PARTICLE', 'positions': None}
    p, w = points[keep], values[keep]
    first = p[w.argmax()]
    if k == 1:
        return {**meta, 'status': 'SINGLE_SOURCE_EXTENSION', 'positions': [first.tolist()]}
    d = np.linalg.norm(p-first, axis=1)
    previous = 0
    trace = []
    for i in range(int(np.ceil(d.max()/delta_h))+2):
        h = (i+1)*delta_h
        count = int((d <= h).sum())
        trace.append([h, count, count-previous])
        if count == previous:
            outside = d > h
            if not outside.any():
                return {**meta, 'status': 'EMPTY_SECOND_CLUSTER', 'positions': None,
                        'first_position': first.tolist(), 'trace': trace}
            second = p[outside][w[outside].argmax()]
            return {**meta, 'status': 'OK', 'positions': [first.tolist(), second.tolist()],
                    'trace': trace}
        previous = count
    raise AssertionError('Finite cumulative count must stop')


def scores(points, stations, y, frequency, n):
    return np.concatenate([score(points[i:i+64], stations, y, frequency, n)[0]
                           for i in range(0, len(points), 64)])


def solve(y, frequency, stations, k, n, bound, seed=2701, particles=1000,
          delta_h=None, std=None):
    start = time.perf_counter()
    delta_h = 400*bound/3000 if delta_h is None else delta_h
    std = 10*bound/3000 if std is None else std
    rng = np.random.default_rng(seed)
    points = rng.uniform(-bound, bound, (particles, 2))
    steps = []
    for iteration in range(3):
        values = scores(points, stations, y, frequency, n)
        out = decode(points, values, k, delta_h)
        steps.append({'round': iteration+1, 'points': points.tolist(), 'scores': values.tolist(),
                      'output': out})
        if out['positions'] is None:
            break
        if iteration < 2:
            center = np.asarray(out['positions'])
            owner = np.arange(particles) % k
            points = center[owner]+rng.normal(size=(particles, 2))*std
            # Eq16 imposes no truncation of the Gaussian. All generated points
            # are retained here; public-domain excursions are logged, not clipped.
    return {'version': 'XIA_LE0_EXACT_ENDPOINT', 'positions': out['positions'],
            'status': out['status'], 'steps': steps, 'seconds': time.perf_counter()-start,
            'delta_h_m': delta_h, 'resampling_std_m': std,
            'outside_domain_count': int((abs(points)>bound).any(axis=1).sum())}
