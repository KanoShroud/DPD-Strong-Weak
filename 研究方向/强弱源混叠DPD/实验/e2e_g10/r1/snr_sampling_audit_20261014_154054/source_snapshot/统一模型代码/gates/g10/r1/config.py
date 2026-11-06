"""Approved geometry, paired data and common-grid contract."""
from ..config import ROOT as ROOT, BASE as G10_BASE, METHODS as METHODS

BASE = G10_BASE / 'r1'
SEED = 2026101401
LAYOUTS = (
    ((173., 117.), (-137., -213.)),
    ((813., 213.), (173., 117.)),
    ((173., 117.), (813., 213.)),
    ((813., 213.), (-727., -383.)),
)
LAYOUT_NAMES = ('both_inside', 'strong_out_weak_in', 'strong_in_weak_out', 'both_outside')
GAPS = (0, 3, 6, 9, 12, 15)
TIMES = (.5, 1., 2., 3., 5., 15., 30., 60.)
N = 16384
SNR_DB = 6.
GRID = 20.
DEFAULT = dict(grid=GRID, top=5, segments=8, loading=1e-3,
               cycles=12, position_tolerance=.05, objective_tolerance=1e-8,
               starts=16, population=64, generations=200, stagnation=15,
               sbl_grid=GRID, sbl_iterations=300, sbl_update='em',
               sbl_refine_every=5, fft_chunk=256, point_chunk=64,
               music_weight='uniform')
