"""PyCharm: complete the one-observation native pilot, NOT the formal cohort.

The current dense Slepian all-grid reference is expensive. Finish this finite
pilot first; no 20-layout batch is implicitly approved or launched here.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from 统一模型代码.gates.g11.model_integration.runner import BASE, check, main, read


if __name__ == '__main__':
    candidates = sorted(BASE.glob('pilot_*'), reverse=True) if BASE.exists() else []
    chosen = None
    for folder in candidates:
        contract = read(folder/'contract.json')
        if contract['lengths'] != [1024]:
            continue
        try:
            check(folder, contract)
        except (RuntimeError, FileNotFoundError):
            continue
        chosen = folder
        break
    print('本入口只完成1条开发观测的4表示×2算法原生核验，不启动正式大批。', flush=True)
    print('完整20 m网格；Slepian参照可能需要数十分钟。已保存的完整方法结果不重跑。', flush=True)
    if '--check-only' in sys.argv:
        print('只读入口检查通过；续跑目录：', chosen or '尚无兼容目录，将新建', flush=True)
        raise SystemExit(0)
    args = ['pilot', '--lengths', '1024']
    if chosen:
        args += ['--resume', str(chosen)]
    main(args)
    from 统一模型代码.gates.g11.model_integration.report import main as report
    report()
