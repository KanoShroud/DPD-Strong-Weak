"""Resume the finite-noise calibration pilot, never the 640-case experiment."""
from pathlib import Path
import json
import runpy
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').is_file())
TASK = ROOT/'outputs_e2e/research_admission_20261027/localization_identifiability/risk_bootstrap_20261103'
sys.path.insert(0, str(TASK))
if len(sys.argv) == 1:
    pointer = TASK/'latest.json'
    if pointer.exists():
        out = Path(json.loads(pointer.read_text(encoding='utf-8'))['path']).resolve(strict=True)
        sys.argv += ['--resume', str(out)]
    # User-run continuation may exceed the Codex one-hour handoff limit.
    # Numerical and scientific gates remain unchanged.
    sys.argv += ['--continue-long']
runpy.run_path(str(TASK/'runner.py'), run_name='__main__')
