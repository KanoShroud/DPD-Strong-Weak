"""Continue the frozen 8192-point single-observation cost/accuracy probe.

This is not the 20-layout formal comparison. Saved whole-method outputs resume;
an interrupted, unsaved optimization has to be computed again.
"""
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from 统一模型代码.gates.g11.model_integration.hybrid_runner import BASE, create, execute, read, verify, write
from 统一模型代码.gates.g11.model_integration.hybrid_report import make_report


if __name__ == '__main__':
    selected = None
    for folder in sorted(BASE.glob('length_*'), reverse=True):
        contract = read(folder/'contract.json')
        if [r['n'] for r in contract['tasks']] != [8192]:
            continue
        try:
            verify(folder, contract)
        except (RuntimeError, FileNotFoundError):
            continue
        selected = folder, contract
        break
    if '--check-only' in sys.argv:
        print('入口检查通过：仅8192点预定单例、七种组合；不启动正式大批。')
        print('恢复目录：', selected[0] if selected else '首次运行时新建')
        raise SystemExit(0)
    if selected is None:
        selected = create('length', [8192])
    folder, contract = selected
    print('运行范围：8192点预定开发单例；Hann候选搜索＋各表示精修、VP末端精修。', flush=True)
    print('长记录Slepian仍可能较慢；完成一种方法立即保存，不重复已保存的方法。', flush=True)
    print('结果目录：', folder, flush=True)
    verify(folder, contract)
    stopped = threading.Event()
    started = time.perf_counter()
    monitor_errors = []

    def heartbeat():
        try:
            while not stopped.wait(30):
                completed = len(list((folder/'rows').glob('*.json')))
                print(f'\r长记录核验 | 已保存 {completed}/7 方法 | 本次累计 {(time.perf_counter()-started)/60:.1f} min    ', end='', flush=True)
        except Exception as error:
            monitor_errors.append(repr(error))

    monitor = threading.Thread(target=heartbeat, daemon=True)
    monitor.start()
    try:
        execute(folder, contract)
    finally:
        stopped.set()
        monitor.join(timeout=1)
        if monitor_errors:
            write(folder/'monitor_failure.json', dict(errors=monitor_errors))
    if monitor_errors:
        raise RuntimeError('进度显示失败，结果已保留：'+monitor_errors[0])
    make_report([folder])
