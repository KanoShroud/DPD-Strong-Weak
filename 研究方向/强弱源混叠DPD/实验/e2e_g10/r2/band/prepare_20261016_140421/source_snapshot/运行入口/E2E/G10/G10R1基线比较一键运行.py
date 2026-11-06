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

MODE = 'replot' if '--replot' in sys.argv else 'run'


def main():
    from 统一模型代码.gates.g10.r1.config import BASE
    from 统一模型代码.gates.g10.r1.runtime import read, write, sha, fresh
    from 统一模型代码.gates.g10.r1.runner import formal, prepare, approved_hours
    if MODE == 'replot':
        active = BASE/'active_run.json'
        pointer = active if active.exists() else BASE/'preparation_pointer.json'
        if not pointer.exists():
            raise RuntimeError('没有已保存运行可重绘')
        out = Path(read(pointer)['out']).resolve(strict=True)
        if not active.exists() and (out/'contract.json').exists():
            comparison = read(out/'contract.json').get('comparison_evidence_dir')
            if comparison:
                out = Path(comparison).resolve(strict=True)
        from 统一模型代码.gates.g10.r1.report import generate
        generate(out)
        print('已仅重绘：'+str(out),flush=True)
        return
    release = BASE/'release.json'
    if not release.exists():
        pointer = BASE/'preparation_pointer.json'
        if not pointer.exists():
            raise RuntimeError('尚未登记共同状态准备；请查看当前任务.md，不要用旧G10入口替代。')
        preparation = Path(read(pointer)['out']).resolve(strict=True)
        print('继续已登记的38次准备；完成并确认预算后才能正式运行。',flush=True)
        prepare(preparation,resume=True,limit_minutes=None)
        if not release.exists():
            return
    record = read(release)
    preparation = Path(record['preparation']).resolve(strict=True)
    if sha(preparation/'readiness.json') != record['readiness_sha']:
        raise RuntimeError('R1 readiness identity differs')
    ready = read(preparation/'readiness.json')
    hours = approved_hours(ready['config_sha'])
    if not ready.get('engineering_ready') or ready['estimated_hours'] > hours:
        raise RuntimeError(f"工程就绪={ready.get('engineering_ready')}；完整估算{ready['estimated_hours']:.1f}小时，已确认{hours:g}小时。请先确认预算；不会自动缩减规模。")
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
