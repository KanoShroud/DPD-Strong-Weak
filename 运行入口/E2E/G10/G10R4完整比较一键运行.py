"""PyCharm：一次点击连续完成核心主批、补充比较和SBL；支持中断恢复。"""
from pathlib import Path
import os
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
os.environ['PYTHONUTF8']='1'
os.environ['PYTHONIOENCODING']='utf-8:replace'
for stream in (sys.stdout,sys.stderr):
    if hasattr(stream,'reconfigure'):
        stream.reconfigure(encoding='utf-8',errors='replace')

if __name__=='__main__':
    from 统一模型代码.gates.g10.r4.formal_runner import main
    main()
