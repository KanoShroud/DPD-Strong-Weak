"""Approved R2 scope. Independent validation remains locked until candidate freeze."""
from ..config import ROOT
from ..r1.config import BASE as R1_BASE, DEFAULT as R1_DEFAULT, FS, N, GAPS

__all__ = ['ROOT', 'R1_BASE', 'BASE', 'METHODS', 'TIMES', 'REVISION', 'CONFIG', 'tasks']

BASE = ROOT / 'outputs_e2e/unified/e2e_g10/r2'
METHODS = ('AP_1_ALT', 'AP_1_JOINT', 'AP_5_ALT', 'AP_5_JOINT',
           'NOMP', 'NOMP_EXTRA12', 'NOMP_JOINT')
TIMES = (.5, 1., 2., 3., 5., 15., 30., 60.)
REVISION = 'G10_R2_DUAL_V1'
CONFIG = dict(revision=REVISION, original_nomp=dict(R1_DEFAULT),
              sample_rate_hz=FS, N=N, grid=20., top=5,
              local_maxiter=100, ftol=1e-12, gtol=1e-8, maxls=25,
              alternating_sweeps=12, extra_cycles=12,
              position_tolerance=.05, objective_tolerance=1e-8,
              methods=list(METHODS), times=list(TIMES), gaps=list(GAPS),
              protocol='EVAL-20261011-v2', hard_solver_deadline=False,
              cumulative_hours=20., independent_validation='LOCKED_PENDING_REVIEW')


def tasks(stage):
    realizations = (31,) if stage == 'prepare' else range(100, 105)
    gaps = (0, 15) if stage == 'prepare' else GAPS
    return [(g, r, gap) for g in range(4) for r in realizations for gap in gaps]
