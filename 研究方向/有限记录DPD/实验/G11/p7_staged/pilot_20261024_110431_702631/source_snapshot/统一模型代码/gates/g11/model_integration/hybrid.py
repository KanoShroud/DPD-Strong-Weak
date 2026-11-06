"""G11-P6 fixed-stage hybrids. IQ/public settings only, no truth argument.

Hann AP discovery is shared explicitly; it does NOT use finite-model residuals.
The finite refinement always reopens the unwindowed selected-band observation.
"""
import hashlib
import time

import numpy as np

from .models import build
from .solvers import CONFIG, local, run
from ...g10.physics import grid
from ...g10.methods import peaks


AP_KINDS = ('hann', 'periodic', 'finite_fourier', 'slepian')
METHODS = tuple('AP_'+k for k in AP_KINDS)+('VP_hann', 'VP_finite_fourier', 'VP_slepian')


def array_hash(x):
    a = np.ascontiguousarray(x)
    return hashlib.sha256(str((a.shape, a.dtype)).encode()+a.tobytes()).hexdigest()


def ap_prefix(iq):
    tick = time.perf_counter()
    model = build(iq, 'hann')
    points = grid(20.)
    first_scores = model.scan(points)
    first_grid = points[peaks(points, first_scores, 1)[0]]
    first, detail = local(model, first_grid[None], CONFIG)
    scores = model.scan(points, others=first)
    ids = peaks(points, scores, CONFIG['top'])
    starts = np.stack([np.vstack([first, points[i]]) for i in ids])
    if len(starts) != CONFIG['top']:
        raise RuntimeError('Native Hann prefix did not produce all five candidates')
    data = dict(starts=starts.tolist(), starts_sha=array_hash(starts), input_sha=array_hash(iq),
        first_grid=first_grid.tolist(), first=first.tolist(), first_detail=detail,
        seconds=time.perf_counter()-tick, counts=dict(model.counts),
        definition='native Hann single-source and conditional whole-grid searches; before branch refinement')
    return data, model


def refine_candidates(iq, kind, prefix, prefix_model=None):
    if array_hash(iq) != prefix['input_sha']:
        raise RuntimeError('Refinement input differs from the registered prefix')
    starts = np.asarray(prefix['starts'], float)
    if array_hash(starts) != prefix['starts_sha']:
        raise RuntimeError('Candidate identity changed')
    tick = time.perf_counter()
    model = prefix_model if kind == 'hann' and prefix_model is not None else build(iq, kind)
    # No selected branch or true positions may warm-start another representation.
    before = dict(model.counts)
    branches = []
    for rank, start in enumerate(starts, 1):
        positions, detail = local(model, start.copy(), CONFIG)
        branches.append(dict(rank=rank, start=start.tolist(), positions=positions.tolist(), **detail))
    best = min(branches, key=lambda row: (row['objective'], row['rank']))
    own = time.perf_counter()-tick
    counts = {k: value-before.get(k, 0) for k, value in model.counts.items()}
    return dict(method='AP_'+kind, kind=kind, positions=best['positions'], objective=best['objective'],
        stop_reason=best['stop_reason'], projected_gradient=best['projected_gradient'],
        selected_rank=best['rank'], branches=branches, starts_sha=prefix['starts_sha'],
        seconds=prefix['seconds']+own, prefix_seconds=prefix['seconds'], refinement_seconds=own,
        representation_preparation_seconds=0. if model is prefix_model else model.preparation_seconds,
        resume_rebuild_charged=bool(kind == 'hann' and prefix_model is None),
        counts=counts, prefix_counts=prefix['counts'], model=model.metadata,
        data_sha=array_hash(model.z), data_source='unwindowed retained DFT' if kind != 'hann' else 'Hann approximate criterion')


def vp_prefix(iq):
    model = build(iq, 'hann')
    result = run(model, 'VP', grid(20.))
    result['input_sha'] = array_hash(iq)
    result['positions_sha'] = array_hash(np.asarray(result['positions']))
    return result


def vp_refine(iq, kind, prefix):
    if array_hash(iq) != prefix['input_sha']:
        raise RuntimeError('VP refinement input identity changed')
    start = np.asarray(prefix['positions'])
    if array_hash(start) != prefix['positions_sha']:
        raise RuntimeError('VP prefix positions changed')
    tick = time.perf_counter()
    model = build(iq, kind)
    positions, detail = local(model, start.copy(), CONFIG)
    own = time.perf_counter()-tick
    return dict(method='VP_'+kind, kind=kind, positions=positions.tolist(), **detail,
        seconds=prefix['seconds']+own, prefix_seconds=prefix['seconds'], refinement_seconds=own,
        representation_preparation_seconds=model.preparation_seconds,
        prefix_positions_sha=prefix['positions_sha'], start=start.tolist(),
        prefix_counts=prefix['counts'], counts=dict(model.counts), model=model.metadata,
        data_sha=array_hash(model.z), data_source='unwindowed retained DFT')


def checks(iq):
    prefix, model = ap_prefix(iq)
    split = refine_candidates(iq, 'hann', prefix, model)
    original = run(build(iq, 'hann'), 'AP5', grid(20.))
    np.testing.assert_allclose(split['positions'], original['positions'], atol=1e-8, rtol=0)
    np.testing.assert_allclose([x['start'] for x in split['branches']],
                               [x['start'] for x in original['branches']], atol=0, rtol=0)
    assert split['selected_rank'] == original['selected_rank']
    for a, b in zip(split['branches'], original['branches']):
        np.testing.assert_allclose(a['objective'], b['objective'], atol=1e-12, rtol=0)
    expected = build(iq, 'periodic').z
    raw_hash = array_hash(expected)
    z_hann = build(iq, 'hann').z
    assert array_hash(z_hann) != raw_hash
    for kind in ('finite_fourier', 'slepian'):
        finite = build(iq, kind)
        np.testing.assert_array_equal(finite.z, expected)
    changed = dict(prefix, input_sha='invalid')
    try:
        refine_candidates(iq, 'hann', changed)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Input identity check not enforced')
    return dict(passed=True, native_hann_split_equivalence=True, raw_band_readback=True,
        exactly_five_original_candidates=True, selected_branch_by_own_final_objective=True,
        input_identity_guard=True, timing_prefix_included=True, starts_sha=prefix['starts_sha'],
        raw_band_sha=raw_hash, hann_sha=array_hash(z_hann), unicode='中文核验通过')
