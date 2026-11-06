"""User-only complete G10-R1 comparison, after qualified short-test release."""
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

MODE = 'run'  # 'replot' regenerates figures from saved rows only.


def main():
    from 统一模型代码.gates.g10.r1.config import BASE
    from 统一模型代码.gates.g10.r1.runtime import read, write, sha, fresh
    from 统一模型代码.gates.g10.r1.runner import formal
    release = BASE/'release.json'
    if not release.exists():
        raise RuntimeError('G10-R1尚未通过全部适配/短测核验，禁止启动完整比较。请查看当前任务.md；不要用旧G10入口替代。')
    record = read(release)
    preparation = Path(record['preparation']).resolve(strict=True)
    if sha(preparation/'readiness.json') != record['readiness_sha']:
        raise RuntimeError('R1 readiness identity differs')
    ready = read(preparation/'readiness.json')
    print(f"六档比较保守估算：{ready['estimated_hours']:.1f}小时；每例按原生条件结束，不按秒截断。", flush=True)
    active = BASE/'active_run.json'
    if active.exists():
        info = read(active)
        if info['preparation'] != str(preparation):
            raise RuntimeError('Active run belongs to a different preparation')
        out = Path(info['out']).resolve()
    else:
        out = fresh('formal')
        write(active, dict(out=str(out), preparation=str(preparation)))
    print('结果目录：'+str(out), flush=True)
    if MODE == 'replot':
        from 统一模型代码.gates.g10.r1.report import generate
        generate(out)
    elif (out/'status.json').exists() and read(out/'status.json')['status'] == 'COMPLETED':
        print('已经完成，不重复运行。', flush=True)
    else:
        formal(out, preparation, resume=(out/'contract.json').exists())
    print('请回读：运行摘要.md、comparison_report.json、final_audit_report.json及figures/。', flush=True)


if __name__ == '__main__':
    main()
