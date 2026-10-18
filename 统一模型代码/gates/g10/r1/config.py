"""Approved geometry, paired data and common-grid contract."""
from ..config import ROOT as ROOT, BASE as G10_BASE, METHODS as METHODS

HISTORY_BASE = G10_BASE / 'r1'
BASE = HISTORY_BASE / 'fs50m_n8192'
SEED = 2026101401
FORMAL_REPEATS = 5
LAYOUTS = (
    ((173., 117.), (-137., -213.)),
    ((813., 213.), (173., 117.)),
    ((173., 117.), (813., 213.)),
    ((813., 213.), (-727., -383.)),
)
LAYOUT_NAMES = ('both_inside', 'strong_out_weak_in', 'strong_in_weak_out', 'both_outside')
GAPS = (0, 3, 6, 9, 12, 15)
TIMES = (.5, 1., 2., 3., 5., 15., 30., 60.)
FS = 50e6
N = 8192
SNR_DB = 6.
GRID = 20.
DEFAULT = dict(grid=GRID, top=5, segments=8, loading=1e-3,
               cycles=12, position_tolerance=.05, objective_tolerance=1e-8,
               starts=16, population=64, generations=200, stagnation=15,
               sample_rate_hz=FS,
               sbl_grid=GRID, sbl_iterations=300, sbl_update='fp_half',
               sbl_active_atoms=8, sbl_coordinate_scale=1000., sbl_coordinate_iterations=20,
               sbl_output='gamma', sbl_spectral_model='flat',
               sbl_refine_every=5, fft_chunk=128, point_chunk=64, dictionary_cache=False,
               music_weight='uniform')


def method_configs():
    """Only the verified SBL cache is enabled; other methods retain pilot settings."""
    return {method: dict(DEFAULT, dictionary_cache=(method == 'SBL')) for method in METHODS}
