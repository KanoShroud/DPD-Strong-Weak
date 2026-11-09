"""Current four-station blind localization API, independent of result folders.

The runtime preserves the reviewed numerical implementation and its known
failure modes. It does not claim a newly validated estimator or launch a batch.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np

HOME = Path(__file__).resolve().parent
PROJECT = next(p for p in HOME.parents if (p/'AGENTS.md').is_file())
OUTPUT = PROJECT/'研究方向/多径DPD/实验/单条运行'


@lru_cache(maxsize=1)
def runtime():
    from .solvers import orthogonal_vp as module
    return module


def create_engine():
    """Return the unchanged ORTH_SAFE four-station, all-inside engine."""
    return runtime().Engine()


def locate(iq, method='MP', seed=2026110426):
    """Localize IQ only. No true path count, position, or waveform is accepted.

    XIA retains the registered coverage rule; it is not silently capped for
    speed. FLOOD retains its adapted association implementation.
    """
    module = runtime()
    iq = np.asarray(iq)
    if iq.shape != (4, module.cfg.N) or not np.isfinite(iq).all():
        raise ValueError(f'Expected finite IQ of shape (4, {module.cfg.N})')
    if method == 'MP':
        return create_engine().mp(iq)
    if method == 'FLOOD':
        return create_engine().flood_solve(iq)
    if method == 'XIA':
        return module.inside.xia_solve(iq, seed)
    raise ValueError(f'Unknown method: {method}')
