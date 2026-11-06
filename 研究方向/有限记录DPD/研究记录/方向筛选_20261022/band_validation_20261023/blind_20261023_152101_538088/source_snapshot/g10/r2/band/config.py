"""Approved finite calibration, separate innovation factors and independent data."""
from ...r1.config import DEFAULT, FS, N, GAPS, LAYOUTS
from ..config import ROOT, CONFIG as R2_CONFIG

BASE = ROOT/'outputs_e2e/unified/e2e_g10/r2/band'
REVISION = 'G10_R2_BAND_V1'
HALF_BAND_HZ = 7e6
DEVELOPMENT_REPEATS = (40, 41)
INDEPENDENT_REPEATS = (200, 201, 202, 203, 204)
GEOMETRY_SEED = 2026101607
WAVEFORM_SEED = 2026101608
MAIN = ('LS', 'MUSIC', 'MVDR', 'AP_FULL', 'AP_1_ALT', 'AP_1_JOINT',
        'AP_5_ALT', 'AP_5_JOINT', 'NOMP', 'NOMP_EXTRA12', 'NOMP_JOINT', 'MULTISTART', 'GA')
METHODS = MAIN+('SBL',)
SINGLE = ('LS', 'MUSIC', 'MVDR', 'AP_FULL', 'NOMP', 'MULTISTART', 'GA', 'SBL')
TIMES = (.5, 1., 2., 3., 5., 15., 30., 60., 120., 300.)
DEFAULT_CONFIG = dict(DEFAULT, **{k: R2_CONFIG[k] for k in
    ('local_maxiter', 'ftol', 'gtol', 'maxls', 'alternating_sweeps', 'extra_cycles')},
    revision=REVISION, half_band_hz=HALF_BAND_HZ, sbl_stop_rule='candidate',
    sbl_position_tolerance=1., sbl_stop_patience=3, sbl_stop_minimum=25)


def configurations(method):
    if method == 'MUSIC':
        return [dict(segments=s, music_weight=w) for s in (4, 8) for w in ('uniform', 'excess_power')]
    if method == 'MVDR':
        return [dict(segments=s, loading=a) for s in (4, 8) for a in (1e-3, 1e-2)]
    if method == 'MULTISTART':
        return [dict(starts=s) for s in (8, 16, 32)]
    if method == 'GA':
        return [dict(population=p, stagnation=s) for p in (64, 128) for s in (15, 30)]
    if method == 'AP_FULL':
        return [dict(cycles=c) for c in (6, 12)]
    if method == 'SBL':
        return [dict(sbl_stop_rule='observe', dictionary_cache=True)]
    return [{}]


def observations(formal=False):
    repeats = INDEPENDENT_REPEATS if formal else DEVELOPMENT_REPEATS
    gaps = GAPS if formal else (0, 15)
    for group in range(4):
        for repeat in repeats:
            yield group, repeat, 0, 'weak'
            for gap in gaps:
                if formal or gap == 15:
                    yield group, repeat, gap, 'strong'
                yield group, repeat, gap, 'mixed'


def jobs(formal=False):
    for task in observations(formal):
        methods = METHODS if task[-1] == 'mixed' else SINGLE
        for method in methods:
            if not formal and method == 'SBL' and task[1] != DEVELOPMENT_REPEATS[0]:
                continue
            for variant, changes in enumerate([{}] if formal else configurations(method)):
                yield task, method, variant, changes


__all__ = ['BASE', 'ROOT', 'REVISION', 'HALF_BAND_HZ', 'DEVELOPMENT_REPEATS',
           'INDEPENDENT_REPEATS', 'GEOMETRY_SEED', 'WAVEFORM_SEED', 'METHODS',
           'MAIN', 'SINGLE', 'TIMES', 'DEFAULT_CONFIG', 'FS', 'N', 'GAPS', 'LAYOUTS',
           'configurations', 'observations', 'jobs']
