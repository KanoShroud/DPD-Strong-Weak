# ruff: noqa: F401
# Preserve imported globals for legacy function cloning and reexports.
"""Xia 2021 equations 9--15: literal score and guarded literal decoder.

No regularizer, Top-K fallback, truth input, or unpublished clustering repair.
"""
import numpy as np

C = 299792458.0


def relative_delays(points, stations):
    d = np.linalg.norm(np.asarray(points)[:, None] - stations[None], axis=-1)
    return (d - d[:, :1]) / C


def cross_spectrum(y):
    denominator = abs(y[0]) ** 2
    if np.any(denominator == 0):
        raise ValueError('REFERENCE_SPECTRAL_ZERO: eq9 undefined; no silent ridge')
    return y[0].conj()[None] * y / denominator[None]


def score(points, stations, y, frequencies, fft_length):
    """Eq9/10; selected band is explicit adaptation; 1/K stays original N."""
    cross = cross_spectrum(y)
    delay = relative_delays(np.atleast_2d(points), stations)
    phase = np.exp(2j * np.pi * delay[:, :, None] * frequencies[None, None])
    components = np.sum(cross[None] * phase, axis=-1) / fft_length
    return np.prod(abs(components), axis=1), components


def scalar_score(point, stations, y, frequencies, fft_length):
    """Independent explicit station/frequency loop for formula verification."""
    distance = [np.linalg.norm(point - station) for station in stations]
    result = 1.0
    for m in range(len(stations)):
        z = 0j
        for j, f in enumerate(frequencies):
            z += (y[0, j].conjugate() * y[m, j] / abs(y[0, j])**2
                  * np.exp(2j * np.pi * f * (distance[m] - distance[0]) / C))
        result *= abs(z / fft_length)
    return result


def sparse_threshold(values):
    """Eq11--13, 10 equal bins, deterministic first mode for unspecified ties."""
    width = np.ptp(values) / 10
    if width == 0:
        return {'status': 'DEGENERATE_RANGE', 'retained': np.zeros(len(values), bool)}
    counts, edges = np.histogram(values, bins=10)
    index = int(np.argmax(counts))
    threshold = 2 * edges[index + 1] - values.min()
    return {'status': 'OK', 'threshold': float(threshold),
            'retained': values >= threshold, 'modal_bin': index}


def literal_bandwidth(points, center, delta_h=400., epsilon=0.):
    """Eq15/step iii. Stop safely after all points included, not indefinitely.

    Exhaustion is not a successful stop, and never emits a location estimate.
    """
    radius = np.linalg.norm(points - center, axis=1)
    previous = 0
    history = []
    for i in range(int(np.ceil(radius.max() / delta_h)) + 2):
        h = (i + 1) * delta_h
        count = int(np.sum(radius <= h))
        increment = (count - previous) / len(points)
        history.append({'radius_m': h, 'cumulative_count': count,
                        'increment': increment, 'stop': bool(increment < epsilon)})
        if increment < epsilon:
            return {'status': 'STOP_RULE_MET', 'bandwidth_m': h, 'trace': history}
        previous = count
    return {'status': 'LITERAL_RULE_CANNOT_STOP', 'bandwidth_m': None, 'trace': history}


def decode_literal(points, values, k=2, delta_h=400., epsilon=0.):
    filtered = sparse_threshold(values)
    mask = filtered['retained']
    if not np.any(mask):
        return {'status': 'NO_RETAINED_PARTICLE', 'positions': None}
    particles, weights = points[mask], values[mask]
    first = particles[np.argmax(weights)]
    if k == 1:
        return {'status': 'SINGLE_SOURCE_EXTENSION', 'positions': first[None].tolist()}
    gate = literal_bandwidth(particles, first, delta_h, epsilon)
    if gate['status'] != 'STOP_RULE_MET':
        return {**gate, 'positions': None}
    outside = np.linalg.norm(particles - first, axis=1) > gate['bandwidth_m']
    if not np.any(outside):
        return {**gate, 'status': 'EMPTY_SECOND_CLUSTER', 'positions': None}
    second = particles[outside][np.argmax(weights[outside])]
    return {**gate, 'positions': np.array([first, second]).tolist()}
