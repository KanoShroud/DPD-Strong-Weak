"""一次点击运行已批准的32条保存轨迹风险回放，自动恢复。"""
from pathlib import Path
import runpy
import sys

if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    root = Path(__file__).resolve().parents[3]
    target = root/'outputs_e2e/research_admission_20261027/localization_identifiability/risk_replay_20261102/runner.py'
    runpy.run_path(str(target), run_name='__main__')
