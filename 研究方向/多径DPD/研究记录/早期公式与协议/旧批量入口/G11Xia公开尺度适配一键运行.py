"""已批准Xia尺度适配；恢复原批，保持短测科学门禁。"""
from pathlib import Path
import runpy
import sys

if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    root = Path(__file__).resolve().parents[3]
    folder = root/'outputs_e2e/research_admission_20261027/localization_identifiability/xia_scale_20261103'
    sys.path.insert(0, str(folder))
    if '--check' not in sys.argv and '--continue-long' not in sys.argv:
        sys.argv.append('--continue-long')
    runpy.run_path(str(folder/'runner.py'), run_name='__main__')
