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


def completed_run(base):
    """Check completed evidence against its saved snapshot, without rerunning."""
    from 统一模型代码.gates.g10.r2.runtime import read, sha
    pointer = base/'active_run.json'
    if not pointer.exists():
        return None
    out = Path(read(pointer)['out']).resolve(strict=True)
    if not out.is_relative_to(base.resolve()) or out == base.resolve():
        raise RuntimeError('输出目录不属于G10-R2')
    if not (out/'status.json').exists() or read(out/'status.json')['status'] != 'COMPLETED':
        return None
    final = read(out/'final_audit_report.json')
    contract = read(out/'contract.json')
    if not all(final.get(k) for k in ('complete', 'identity_passed', 'paired_checks_passed')):
        raise RuntimeError('已完成结果缺少通过的完整审计')
    if final['n_rows'] != 840 or len(final['rows']) != 840 or read(out/'status.json')['rows'] != 840:
        raise RuntimeError('已完成结果数量不完整')
    if final['sources'] != contract['sources'] | contract['r2_sources']:
        raise RuntimeError('审计与运行源码登记不一致')
    for rel, identity in final['sources'].items():
        if sha(out/'source_snapshot'/rel) != identity:
            raise RuntimeError('运行时源码快照变化：'+rel)
    if {p.name for p in (out/'rows').glob('*.json')} != set(final['rows']):
        raise RuntimeError('已完成结果清单变化')
    for name, identity in final['rows'].items():
        if sha(out/'rows'/name) != identity:
            raise RuntimeError('已完成结果身份变化：'+name)
    for entry in read(out/'input_manifest.json')['entries'].values():
        if sha(entry['metadata']) != entry['metadata_sha'] or sha(entry['payload']) != entry['payload_sha']:
            raise RuntimeError('已完成运行的输入身份变化')
    return out


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
    finished = completed_run(BASE)
    if finished is not None:
        print('开发比较已完成；840条结果、冻结输入和运行时源码快照核对通过。', flush=True)
        print('结果目录：'+str(finished), flush=True)
        print('无需重新运行定位；下一步回读分析，独立复核尚未启动。', flush=True)
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
