"""PyCharm default: non-SBL main batch; slow/all batches require explicit flags."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8:replace'
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')


if __name__ == '__main__':
    from 统一模型代码.gates.g10.r2.band.runner import main
    options = sys.argv[1:]
    mode = 'check' if '--check-only' in options else 'replot' if '--replot' in options else 'run'
    if mode == 'run' and not any(x == '--batch' or x.startswith('--batch=') for x in options):
        options += ['--batch', 'main']
        print('本次默认仅运行非SBL主批，生成报告后结束；不会自动补跑SBL。\n'
              '主批预计约50分钟，保守约1小时40分钟；下方完整预算仍包含未启动的SBL。', flush=True)
    sys.argv = [sys.argv[0], mode]+[x for x in options if x not in ('--check-only', '--replot')]
    main()
