"""Approved R3 scope, with no performance-selected geometry or new model."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT/'outputs_e2e/unified/e2e_g10/r3'
R2_BASE = ROOT/'outputs_e2e/unified/e2e_g10/r2/band'
ENTRY = ROOT/'运行入口/E2E/G10/G10R3轻量AP验证一键运行.py'
REVISION = 'G10_R3_AP_COMMON_STOP_V2'
PREP_METHODS = ('AP_1_ALT', 'AP_1_JOINT')
CORE = ('AP_1_ALT', 'AP_1_JOINT', 'NOMP')
METHODS = CORE+('MULTISTART_FIXED',)
GAPS = (0, 3, 6, 9, 12, 15)
GEOMETRY_SEED, WAVEFORM_SEED = 2026101617, 2026101618
SHORT_TASKS = tuple((g, r, gap, 'mixed') for g in range(4)
                   for r in (200, 201) for gap in (0, 6, 15))
FORMAL_TASKS = tuple((g, r, gap, 'mixed') for g in range(4)
                    for r in range(300, 305) for gap in GAPS)
TIMING_REPEATS = 5


def jobs(stage):
    """Rotate execution order; repetitions are timing repeats, not new IQ."""
    tasks = SHORT_TASKS if stage == 'prepare' else FORMAL_TASKS
    result = []
    for i, task in enumerate(tasks):
        repeats = range(TIMING_REPEATS) if stage == 'prepare' else range(1)
        for repeat in repeats:
            methods = PREP_METHODS if stage == 'prepare' else METHODS
            offset = (i+repeat) % len(methods)
            for method in methods[offset:]+methods[:offset]:
                result.append(dict(task=list(task), method=method, repetition=repeat))
    return result
