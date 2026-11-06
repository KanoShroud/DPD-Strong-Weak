"""Rerun fixed-input migration checks, not localization performance research."""
import runpy
import sys
from datetime import datetime
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').is_file())

if __name__ == '__main__':
    output = ROOT/'研究方向/多径DPD/实验/工程检查'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')/'probe.npz'
    script = ROOT/'研究方向/多径DPD/验证/migration_probe.py'
    sys.argv = [str(script), 'api', str(output)]
    runpy.run_path(str(script), run_name='__main__')
