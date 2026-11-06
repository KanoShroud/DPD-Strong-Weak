"""User-approved <= zero interpretation; no alteration of literal audit."""
import time
import numpy as np
from xia_reference import score, sparse_threshold


def decode(points, values, k, delta_h):
    filtered = sparse_threshold(values)
    keep = filtered['retained']
    if not keep.any():
        return {'status': 'NO_RETAINED_PARTICLE', 'positions': None, 'retained': 0}
    p, w = points[keep], values[keep]
    first = p[np.argmax(w)]
    if k == 1:
        return {'status': 'SINGLE_SOURCE_EXTENSION', 'positions': [first.tolist()],
                'retained': len(p)}
    distance = np.linalg.norm(p-first, axis=1)
    previous = 0
    trace = []
    for i in range(int(np.ceil(distance.max()/delta_h))+2):
        radius = (i+1)*delta_h
        count = int(np.sum(distance <= radius))
        trace.append([radius, count, count-previous])
        if count-previous <= 0:  # approved <= epsilon=0 convention
            outside = distance > radius
            if not outside.any():
                return {'status': 'EMPTY_SECOND_CLUSTER', 'positions': None,
                        'first_position': first.tolist(), 'retained': len(p), 'trace': trace}
            second = p[outside][np.argmax(w[outside])]
            return {'status': 'OK', 'positions': [first.tolist(), second.tolist()],
                    'retained': len(p), 'trace': trace}
        previous = count
    raise AssertionError('Finite cumulative counts must reach zero increment')


def scores(points, stations, y, frequency, n):
    return np.concatenate([score(points[i:i+64], stations, y, frequency, n)[0]
                           for i in range(0, len(points), 64)])


def solve(y, frequency, stations, k, n, bound=1000., seed=2701, particles=1000):
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    points = rng.uniform(-bound, bound, (particles, 2))
    steps = []
    scale = bound/3000.
    for iteration in range(3):
        values = scores(points, stations, y, frequency, n)
        output = decode(points, values, k, 400*scale)
        steps.append({'round': iteration+1, 'output': output,
                      'points': points.tolist(), 'scores': values.tolist()})
        if output['positions'] is None:
            break
        if iteration < 2:
            centers = np.array(output['positions'])
            # Eq16 covariance=100 m^2 scaled by squared domain-length ratio.
            points = centers[np.arange(particles) % k] + rng.normal(size=(particles, 2))*10*scale
            for _ in range(20):
                invalid = (abs(points) > bound).any(axis=1)
                if not invalid.any():
                    break
                points[invalid] = (centers[np.arange(particles)[invalid] % k]
                                   + rng.normal(size=(invalid.sum(), 2))*10*scale)
            if (abs(points) > bound).any():
                raise RuntimeError('Public-bound sampling failed')
    return {'positions': output['positions'], 'status': output['status'], 'steps': steps,
            'seconds': time.perf_counter()-started, 'score_evaluations': len(steps)*particles,
            'version': 'XIA_LE0_3ROUNDS_NO_RIDGE', 'delta_h_m': 400*scale,
            'resampling_std_m': 10*scale}
