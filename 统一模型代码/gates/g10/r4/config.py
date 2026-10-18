"""Approved finite preparation; formal data remain ungenerated and locked."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT/'outputs_e2e/unified/e2e_g10/r4'
REVISION = 'G10_R4_SOURCE_MATCH_V2_FAST_PROCESS_CHECK'
METHODS = ('BNOMP_REF', 'NOMP', 'NOMP_JOINT', 'AP_1_ALT', 'AP_1_JOINT',
           'AP_2_JOINT', 'AP_5_JOINT')
TASKS = tuple((g, r, gap, 'mixed') for g in range(4)
              for r in (200, 201) for gap in (0, 15))
FORMAL_CASES = 20*20*6
FORMAL_SOLVES = FORMAL_CASES*len(METHODS)


def configurations(reference):
    # Preserve the original NOMP dictionary/newton/cyclic/stopping configuration.
    nomp = dict(reference['configurations']['NOMP'])
    ap = dict(reference['configurations']['AP_1_JOINT'])
    ap.update(common_stop=True, common_ftol=0.)
    result = {m: dict(nomp if m in ('BNOMP_REF', 'NOMP', 'NOMP_JOINT') else ap)
              for m in METHODS}
    result['NOMP_JOINT'].update(common_stop=True, common_ftol=0.)
    # A dimensionless energy tolerance, not the paper's acoustic-unit 1e-4.
    # Preserve the paper's single adjacent-cycle test and post-cycle joint LS.
    result['BNOMP_REF'].update(bnomp_cycles=100, bnomp_energy_tolerance=1e-8,
        bnomp_newton_steps=5, bnomp_step_cap_m=50., bnomp_min_step_m=.01)
    return result


def jobs():
    output = []
    for i, task in enumerate(TASKS):
        ordered = METHODS[i % len(METHODS):]+METHODS[:i % len(METHODS)]
        output.extend(dict(task=list(task), method=m) for m in ordered)
    return output
