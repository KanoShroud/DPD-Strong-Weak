"""One declared post-hoc control: keep FE position, use periodic residual.

Same eight development observations; no new candidates, tuning or truth input.
Run only after the primary diagnosis is complete, separately from its evidence.
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .models import build
from .solvers import CONFIG, local
from .hybrid import array_hash
from .hybrid_runner import read, write, sha, read_row, save_row, key, get_input, ROOT
from .early_stage import finish, taper_retained, verify
from ...g10.physics import grid
from ...g10.methods import peaks
from ...g10.r4.paper_checks.assignment_metrics import evaluate

VARIANT = 'finite_position_periodic_residual'


def solve(iq, first):
    if first['input_sha'] != array_hash(iq):
        raise RuntimeError('Input identity mismatch')
    tick = time.perf_counter()
    finite = build(iq, 'finite_fourier')
    original = np.asarray(first['position'])
    p, first_detail = local(finite, original.copy(), CONFIG)
    periodic, hann = build(iq, 'periodic'), build(iq, 'hann')
    residual = periodic.fit(p)['residual']
    points = grid(20.)
    scores = hann.scan(points, data=taper_retained(residual, periodic))
    ids = peaks(points, scores, CONFIG['top'])
    starts = np.stack([np.vstack([p, points[i]]) for i in ids])
    details = dict(variant=VARIANT, first_original=original.tolist(), first=p.tolist(),
                   first_detail=first_detail, starts=starts.tolist(), starts_sha=array_hash(starts),
                   grid_indices=ids.tolist(), scores_at_candidates=scores[ids].tolist(),
                   prefix_seconds=first['seconds'], discovery_seconds=time.perf_counter()-tick,
                   representation_preparation_seconds=finite.preparation_seconds,
                   data_sha=array_hash(finite.z))
    result = finish(finite, starts, details)
    result['seconds'] = result['prefix_seconds']+result['discovery_seconds']+result['refinement_seconds']
    return result, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    folder = parser.parse_args().folder.resolve(strict=True)
    contract = read(folder/'contract.json')
    assert read(folder/'audit.json')['completed']
    verify(folder, contract)
    out = folder/('residual_ablation_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir()
    (out/'rows').mkdir()
    (out/'spectra').mkdir()
    source = Path(__file__)
    write(out/'contract.json', dict(variant=VARIANT, primary=str(folder), source=str(source),
        source_sha=sha(source), config=CONFIG, tasks=contract['tasks'],
        status='POST_HOC_MECHANISM_CONTROL_NOT_INDEPENDENT_VALIDATION'))
    (out/source.name).write_bytes(source.read_bytes())
    tick = time.perf_counter()
    with threadpool_limits(limits=2):
        for task in contract['tasks']:
            iq, truth = get_input(folder, task)
            ident = key(task)
            first = read_row(folder/'first'/(ident+'.json'))
            result, scores = solve(iq, first)
            baseline = read_row(folder/'rows'/(ident+'_finite_both.json'))['result']
            np.testing.assert_allclose(result['first'], baseline['first'], atol=1e-8, rtol=0)
            np.savez_compressed(out/'spectra'/(ident+'_'+VARIANT+'.npz'), scores=scores)
            save_row(out/'rows'/(ident+'_'+VARIANT+'.json'), dict(task=task, variant=VARIANT,
                result=result, truth=truth.tolist(), input_sha=array_hash(iq),
                metrics=evaluate(truth, result['positions']), post_hoc=True))
    verify(folder, contract)
    assert sha(source) == read(out/'contract.json')['source_sha']
    write(out/'audit.json', dict(completed=True, source_identity=True, input_identity=True,
        identical_finite_first=True, post_hoc=True, rows=8, seconds=time.perf_counter()-tick,
        primary_sha=sha(folder/'contract.json'), source_relative=str(source.relative_to(ROOT))))
    print(out, flush=True)


if __name__ == '__main__':
    main()
