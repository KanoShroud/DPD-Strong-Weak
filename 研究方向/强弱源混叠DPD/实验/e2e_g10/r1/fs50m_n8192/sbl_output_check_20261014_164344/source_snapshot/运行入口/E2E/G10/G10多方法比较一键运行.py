"""PyCharm click-to-run after approved short-test release; never auto-run preparation."""
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

# Switch to 'replot' only to regenerate figures from saved results.
MODE = 'run'


def main():
    from 统一模型代码.gates.g10.config import BASE
    from 统一模型代码.gates.g10.runtime import read, write, sha
    from 统一模型代码.gates.g10.runner import formal, fresh
    release = BASE/'release.json'
    if not release.exists():
        raise RuntimeError('G10尚未放行：需先完成全部短测、配置冻结及12小时预算预检。不要运行旧G9入口代替。')
    info = read(release)
    preparation = Path(info['preparation']).resolve(strict=True)
    if sha(preparation/'readiness.json') != info['readiness_sha']:
        raise RuntimeError('Readiness identity changed')
    active = BASE/'active_run.json'
    if active.exists():
        out = Path(read(active)['out']).resolve()
        resume = (out/'contract.json').exists()
    else:
        out, resume = fresh('formal'), False
        write(active, dict(out=str(out), preparation=str(preparation)))
    print('G10结果目录：'+str(out), flush=True)
    if MODE == 'replot':
        from 统一模型代码.gates.g10.report import generate
        generate(out, label='FORMAL')
    elif (out/'status.json').exists() and read(out/'status.json')['status'] == 'COMPLETED':
        print('正式比较已完成，不重复运行。回读运行摘要.md、comparison_report.json、final_audit_report.json及figures/。')
    else:
        formal(out, preparation, resume=resume)
        print('G10完成。请回读运行摘要.md、comparison_report.json、final_audit_report.json及figures/。')


if __name__ == '__main__':
    main()
