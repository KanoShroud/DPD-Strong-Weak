"""Approved common contract; algorithm candidate settings are selected on dev only."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'outputs_e2e/unified/e2e_g10'
METHODS = ('LS', 'MUSIC', 'MVDR', 'AP', 'BNOMP', 'SBL', 'MULTISTART', 'GA')
SEED = 2026101301
GAPS = (0, 3, 6, 9, 12, 15)
N = 16384
CHECKPOINTS = (5., 15., 30., 60.)
DEFAULT = dict(grid=20., top=5, segments=4, loading=1e-3, cycles=5,
               sbl_grid=50., sbl_iterations=50, starts=32, population=64,
               fft_chunk=256, point_chunk=64)


def candidates(method):
    """Finite, sequential dev comparisons, not a Cartesian search."""
    changes = {
        'LS': ({}, {'grid': 10.}, {'top': 10}),
        'MUSIC': ({}, {'segments': 8}, {'segments': 16}, {'grid': 10.}),
        'MVDR': ({}, {'segments': 8}, {'segments': 16},
                 {'loading': 1e-4}, {'loading': 1e-2}, {'grid': 10.}),
        'AP': ({}, {'grid': 10.}, {'top': 10}),
        'BNOMP': ({'cycles': 3}, {}, {'grid': 10.}),
        'SBL': ({}, {'sbl_grid': 25.}),
        'MULTISTART': ({},), 'GA': ({},),
    }
    return [dict(DEFAULT, **x) for x in changes[method]]
