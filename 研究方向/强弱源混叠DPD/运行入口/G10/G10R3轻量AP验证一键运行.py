"""PyCharm：SBL结束后首击短测；短测放行后再次点击独立比较。"""
import os
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').is_file())
os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8:replace'
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


if __name__ == '__main__':
    os.chdir(ROOT)
    from 研究方向.强弱源混叠DPD.代码.gates.g10.r3.runner import main
    main()
