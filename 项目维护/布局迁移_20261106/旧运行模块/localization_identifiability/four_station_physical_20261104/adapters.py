# ruff: noqa: F401
# Preserve imported globals for legacy function cloning and reexports.
"""Isolated four-station/4500m instantiations of frozen estimator code."""
import functools
import importlib.util
import inspect
import math
import sys
import time
from pathlib import Path
import numpy as np
import scene as cfg


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


b = load('physical_station_backend', cfg.BASE/'station_count_20261104/backend.py')
# The frozen import graph inserts source directories. Restore this run's directory first.
sys.path.insert(0, str(cfg.HERE))
kernel_module = load('physical_xia_kernel', cfg.BASE/'xia_scale_20261103/kernel.py')
xia = load('physical_xia_checked', cfg.BASE.parent/'xia_pairing/recheck/xia_checked.py')
import flood_band_pair as flood  # noqa: E402
import finite_iq_clues as wave  # noqa: E402,F401 -- public generator dependency


def clone(fn, edits=(), bindings=None):
    source = getattr(fn, 'generated_source', None) or inspect.getsource(fn)
    for a, z in edits:
        assert a in source, (fn.__name__, a)
        source = source.replace(a, z)
    scope = {**fn.__globals__, 'BOUND': cfg.BOUND, **(bindings or {})}
    exec(compile(source, str(cfg.HERE/('instantiated_'+fn.__name__)), 'exec'), scope)
    result = scope[fn.__name__]
    result.__defaults__ = fn.__defaults__
    result.generated_source = source
    return result


class Engine(b.Backend):
    def __init__(self):
        super().__init__([1, 2, 3, 4], 0)
        assert np.array_equal(self.stations, cfg.STATIONS)
        self.ref.project = clone(self.ref.project, [('3000/UNIT', 'BOUND/UNIT')])
        self.base_fit = clone(self.base_fit, [('3000/ref.UNIT', 'BOUND/ref.UNIT')])
        self.base_fit.__globals__['fit_base'] = self.base_fit
        self.fc.fit_base = self.base_fit
        self.h.scope = self.base_fit.__globals__
        self.finish.__globals__['ref'].project = self.ref.project
        self.h.fit.__globals__['base_fit'] = self.base_fit
        self.los_fit = clone(self.los_fit, [('3000/UNIT', 'BOUND/UNIT')], {'closure': self.ref.project})
        self.estimate_function.__globals__['fast_los'] = self.los_fit
        self.geometry_fit = clone(self.geometry_fit, [('-3000,3000', '-BOUND,BOUND')])
        self.triples = clone(self.triples, [('> SCALE+1e-6', '> BOUND+1e-6')])
        self.predicted = self.geometry_fit.__globals__['quantities'].__globals__['predicted']
        lp = functools.partial(flood.pruned_lp, capacity=len(self.pairs))
        re = functools.partial(flood.reestimate, capacity=len(self.pairs))
        self.flood = clone(flood.estimate, [('bounds=(-3000,3000)', 'bounds=(-BOUND,BOUND)')],
                           dict(predicted=self.predicted, pruned_lp=lp, reestimate=re))
        self.last_geometry = None
        self.last_geometry_seconds = None
        self.front_snapshot = None
        self.estimate_function = clone(self.estimate_function,
            [('front_seconds = time.perf_counter()-start',
              'front_seconds = time.perf_counter()-start\n    capture_front(los, front_seconds)')],
            {'capture_front': self.capture_front})

    def capture_front(self, los, seconds):
        self.front_snapshot = dict(los=los, frontend_seconds=seconds, geo=self.geo_snapshot)

    def frontend(self, iq):
        start = time.perf_counter()
        result = super().frontend(iq)
        ids = self.engine.joint_select(result['reward'])[0]
        self.last_geometry = result
        self.last_geometry_seconds = time.perf_counter()-start
        self.geo_snapshot = dict(p=result['refined'][ids], seconds=self.last_geometry_seconds,
            candidates=len(result['pool']), selected=ids, refinement_status=result['status'][ids])
        return result

    def mp(self, iq):
        self.last_geometry = None
        self.front_snapshot = None
        self.geo_snapshot = None
        solved = self.estimate(iq)
        solved['geo'] = self.geo_snapshot
        return solved

    def flood_solve(self, iq):
        start = time.perf_counter()
        extracted, _, _ = self.extract(iq)
        peaks = extracted['CC_ABS']
        lags = np.array([sorted(np.asarray(peaks[f'{m}_{n}']['lag_s'][:6])*cfg.C) for m, n in self.pairs])
        pool = self.triples(lags)
        if len(pool) < 2:
            return dict(p=None, normal=False, status='insufficient_candidates', seconds=time.perf_counter()-start)
        p, meta = self.flood(pool, lags, known_k=2)
        compact = {k: v for k, v in meta.items() if k not in ('cost', 'M', 'trash', 'q', 'association', 'association_trash')}
        normal = all(isinstance(v['status'], int) and v['status'] > 0 for v in meta['refinement'])
        return dict(p=p, normal=normal, status='returned' if normal else 'refinement_incomplete', seconds=time.perf_counter()-start,
                    meta=compact, pool_count=len(pool), lags=lags,
                    association_sha=cfg.digest(meta['association']), pool_sha=cfg.digest(pool))


def xia_design():
    radius = cfg.C/(4*cfg.B)
    probability = np.pi*radius**2/(4*(2*cfg.BOUND)**2)
    count = math.ceil(math.log(.05/2)/math.log1p(-probability))
    return dict(radius_m=radius, particles=count, miss_bound=2*np.exp(count*np.log1p(-probability)),
                bound=cfg.BOUND, rounds=3, delta_h_m=400., std_m=10.)


def xia_solve(iq, seed, particles=None):
    start = time.perf_counter()
    f, y, n, _ = b.fc.cal.spectral(iq)
    kernel = kernel_module.CrossSpectrumKernel(y.T, f, n, cfg.FS)
    config = xia_design()
    count = config['particles'] if particles is None else particles
    rng = np.random.default_rng(seed)
    rng_state = rng.bit_generator.state
    points = rng.uniform(-cfg.BOUND, cfg.BOUND, (count, 2))
    steps = []
    for iteration in range(3):
        values = kernel.scores(points, cfg.STATIONS)
        assert np.isfinite(values).all() and np.all(values >= 0)
        indices = np.unique(np.r_[np.linspace(0, count-1, min(count, 64), dtype=int),
                                  np.argpartition(values, -min(count, 32))[-min(count, 32):]])
        direct = xia.scores(points[indices], cfg.STATIONS, y.T, f, n)
        error = float(np.max(abs(direct-values[indices]))/max(values.max(), np.finfo(float).tiny))
        assert error < 5e-11, error
        answer = xia.decode(points, values, 2, config['delta_h_m'])
        steps.append(dict(round=iteration+1, output=answer, points_sha=cfg.digest(points),
            scores_sha=cfg.digest(values), rng_before_generation=rng_state,
            particle_count=count, all_candidates_scored=True, direct_error=error,
            outside_domain_count=int((abs(points)>cfg.BOUND).any(axis=1).sum())))
        if answer['positions'] is None:
            break
        if iteration < 2:
            rng_state = rng.bit_generator.state
            centers = np.asarray(answer['positions'])
            points = centers[np.arange(count) % 2]+rng.normal(size=(count, 2))*config['std_m']
    full = [s['output']['positions'] for s in steps if s['output']['positions'] is not None]
    return dict(p=steps[-1]['output']['positions'], lastcomplete=full[-1] if full else None,
                normal=steps[-1]['output']['positions'] is not None, status=steps[-1]['output']['status'],
                seconds=time.perf_counter()-start, config=config, seed=seed, steps=steps,
                kernel=kernel.metadata(), evidence_storage='RNG states and hashes; full pools reconstructible')


def source_hashes():
    files = set(cfg.HERE.glob('*.py')) | {cfg.HERE/'protocol.md', cfg.HERE/'contract.json'}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            p = Path(name).resolve()
            if p.suffix == '.py' and p.is_relative_to(cfg.ROOT):
                files.add(p)
    return {str(p.resolve(strict=True)): cfg.sha(p) for p in sorted(files)}
