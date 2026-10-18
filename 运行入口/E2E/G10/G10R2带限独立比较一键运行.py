"""Two-click workflow: finish main batch, then resume SBL on the next launch."""
import os
from pathlib import Path
import sys
import shutil

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8:replace'
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')


def active_directory(base):
    from 统一模型代码.gates.g10.r2.band.runtime import read
    pointer = base/'active_run.json'
    if not pointer.exists():
        return None
    out = Path(read(pointer)['out']).resolve(strict=True)
    if not out.is_relative_to(base.resolve()):
        raise ValueError('Active run outside result root')
    return out


def next_batch(base):
    from 统一模型代码.gates.g10.r2.band.runtime import read, sha
    out = active_directory(base)
    if out is None or not (out/'main_batch_ready.json').exists():
        return 'main'
    marker = read(out/'main_batch_ready.json')
    for name, expected in marker['main_rows'].items():
        if sha(out/'rows'/name) != expected:
            raise RuntimeError('Completed main result changed: '+name)
    return 'slow'


def finish_notice(base, batch):
    from 统一模型代码.gates.g10.r2.band.runtime import read, write, sha, tag
    out = active_directory(base)
    if out is None:
        return
    state = read(out/'status.json')
    if state['status'] == 'COMPLETED':
        print('\n全部方法已完成，结果目录：'+str(out), flush=True)
        return
    if batch != 'main' or state['status'] != 'BATCH_COMPLETED':
        print('\n本批尚未完成；再次点击将恢复未完成部分。', flush=True)
        return
    audit = read(out/'final_audit_report.json')
    jobs = [j for j in read(out/'contract.json')['jobs'] if j['method'] != 'SBL']
    main_rows = {}
    for job in jobs:
        name = tag(job['task'], job['method'], job['variant'])+'.json'
        expected = audit['rows'][name]
        if sha(out/'rows'/name) != expected:
            raise RuntimeError('Main result changed before notification')
        main_rows[name] = expected
    # Keep a stable report for Codex to read while the slow batch updates root reports.
    marker = out/'main_batch_ready.json'
    folder = out/'main_batch_report'
    if not marker.exists():
        folder.mkdir(exist_ok=True)
        for name in ('运行摘要.md', 'comparison_report.json', 'paired_comparisons.json',
                     'final_audit_report.json', 'status.json'):
            shutil.copyfile(out/name, folder/name)
        for pattern in ('*.csv', 'summary_table.*'):
            for path in out.glob(pattern):
                shutil.copyfile(path, folder/path.name)
        shutil.copytree(out/'figures', folder/'figures', dirs_exist_ok=True)
        write(marker, dict(main_rows=main_rows, report_directory=str(folder),
                           next_batch='slow', scientific_configuration_unchanged=True))
    print('\n========== 非SBL主批已完成，已暂停接续 ==========\n'
          '可将以下主批报告目录交给Codex回读：\n'+str(folder)+'\n'
          '本次程序正常结束。再次点击同一脚本即可继续SBL，无需改参数。', flush=True)


if __name__ == '__main__':
    # Saved relative pointers are project-root-relative, not IDE-cwd-relative.
    os.chdir(ROOT)
    from 统一模型代码.gates.g10.r2.band.runner import main
    from 统一模型代码.gates.g10.r2.band.config import BASE
    options = sys.argv[1:]
    mode = 'check' if '--check-only' in options else 'replot' if '--replot' in options else 'run'
    automatic = mode == 'run' and not any(x == '--batch' or x.startswith('--batch=') for x in options)
    batch = next_batch(BASE) if automatic else None
    if automatic:
        options += ['--batch', batch]
        print('本次运行非SBL主批，结束后提示并退出。' if batch == 'main'
              else '主批已完成，本次继续SBL；可同时让Codex回读main_batch_report。', flush=True)
    sys.argv = [sys.argv[0], mode]+[x for x in options if x not in ('--check-only', '--replot')]
    main()
    if mode == 'run':
        if batch is None:
            batch = (options[options.index('--batch')+1] if '--batch' in options
                     else next(x.split('=', 1)[1] for x in options if x.startswith('--batch=')))
        finish_notice(BASE, batch)
