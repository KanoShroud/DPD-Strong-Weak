"""冻结投稿补证：240条曲率配对 → 16新布局×5实现×6功率差，支持恢复。

运行此入口采用预留1小时累计任务预算；单个完整配对不在中途截断。
若预算不足，完成当前配对后暂停，不缩减样本或改变算法。
"""
import os
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').is_file())
sys.path.insert(0, str(ROOT))
os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8:replace'
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')


if __name__ == '__main__':
    from 研究方向.强弱源混叠DPD.代码.gates.g10.r4.paper_checks import review_supplement as study
    from 研究方向.强弱源混叠DPD.代码.gates.g10.runtime import read, sha
    pointer = read(study.BASE/'review_supplement_release.json')
    out = Path(pointer['out']).resolve(strict=True)
    assert out.is_relative_to(study.BASE.resolve())
    assert sha(out/'release.json') == pointer['release_sha']
    release = read(out/'release.json')
    assert release['ready'] and sha(out/'contract.json') == release['formal_contract_sha']
    summary = out/'summary.json'
    if summary.exists() and read(summary)['status'] == 'COMPLETED':
        print('正式补证已完成，无需重复运行：'+str(out))
    else:
        print('本次预算1小时：先曲率机制配对，再新几何验证；不重跑八类方法。')
        sys.argv = [sys.argv[0], '--resume', str(out)]
        study.main()
