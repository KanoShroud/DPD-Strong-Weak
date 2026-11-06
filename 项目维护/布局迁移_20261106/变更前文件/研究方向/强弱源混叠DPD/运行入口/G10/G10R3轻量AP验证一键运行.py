"""PyCharm：SBL结束后首击短测；短测放行后再次点击独立比较。"""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8:replace'
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


if __name__ == '__main__':
    os.chdir(ROOT)
    from 统一模型代码.gates.g10.r3.runner import main
    main()
