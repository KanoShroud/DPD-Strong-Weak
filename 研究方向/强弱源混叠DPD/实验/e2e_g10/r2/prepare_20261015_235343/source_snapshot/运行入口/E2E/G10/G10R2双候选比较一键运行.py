"""PyCharm: run development only; repeated clicks resume, never start validation."""
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


def main():
    from 统一模型代码.gates.g10.r2.config import BASE
    from 统一模型代码.gates.g10.r2.runtime import read, write, fresh
    from 统一模型代码.gates.g10.r2.runner import release_check, develop
    if '--independent' in sys.argv:
        raise RuntimeError('独立复核尚未放行：先回读开发结果、冻结候选，再登记新布局。')
    if '--replot' in sys.argv:
        pointer = BASE/'active_run.json'
        if not pointer.exists():
            pointer = BASE/'preparation_pointer.json'
        out = Path(read(pointer)['out']).resolve(strict=True)
        from 统一模型代码.gates.g10.r2.report import generate
        generate(out)
        print('仅重绘已保存结果：'+str(out), flush=True)
        return
    preparation, ready = release_check()
    print(f"准备已通过；开发比较保守估算{ready['estimated_hours']:.2f}小时。", flush=True)
    if '--check-only' in sys.argv:
        print('只读预检通过；没有创建正式目录或启动定位。', flush=True)
        return
    active = BASE/'active_run.json'
    if active.exists():
        info = read(active)
        if info['preparation'] != str(preparation):
            raise RuntimeError('现有运行与准备版本不同，请先回读当前任务。')
        out = Path(info['out']).resolve()
    else:
        out = fresh('develop')
        write(active, dict(out=str(out), preparation=str(preparation)))
    if not out.is_relative_to(BASE.resolve()) or out == BASE.resolve():
        raise RuntimeError('输出目录不属于G10-R2')
    print('结果目录：'+str(out), flush=True)
    develop(out, preparation, resume=(out/'contract.json').exists())
    print('请回读：运行摘要.md、comparison_report.json、mechanism_report.json、final_audit_report.json。', flush=True)
    print('独立复核未启动，等待开发结果回读与候选冻结。', flush=True)


if __name__ == '__main__':
    main()
