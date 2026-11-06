"""Audit approved task artifacts without launching localization or task three."""
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote

HERE = Path(__file__).resolve().parent
BASE = HERE.parent


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    checks, runs = {}, {}
    for key in ['task1_short', 'task1_full', 'task2_short', 'task2_full']:
        folder = Path(read(HERE/(key+'_latest.json'))['path'])
        assert folder.is_relative_to(HERE)
        data, protocol = read(folder/'result.json'), read(folder/'protocol.json')
        runs[key] = dict(path=str(folder), checks=len(data['checks']), seconds=data['seconds'])
        checks[key+'/all_runtime_checks'] = all(data['checks'].values())
        checks[key+'/task3_not_authorized'] = protocol['task3_authorized'] is False
        for filename, expected in protocol['hashes'].items():
            original = Path(filename)
            # Driver may be revised after a completed run; its captured version is authoritative.
            snapshot = folder/'source'/original.name
            target = snapshot if original.parent == HERE and snapshot.exists() else original
            checks[key+'/sha/'+filename] = sha(target) == expected
    failed = HERE/'task2_short_20261030_153702_743492'
    failed_result = read(failed/'result.json')
    failures = [k for k, v in failed_result['checks'].items() if not v]
    checks['failed_attempt_preserved'] = len(failures) == 4 and all(k.endswith('/MP/input') for k in failures)
    checks['failed_attempt_excluded'] = all(Path(v['path']) != failed for v in runs.values())
    metrics = read(HERE/'analysis/metrics.json')
    checks['task3_not_run'] = metrics['task3_started'] is False
    checks['formal_48_pairs'] = len(read(Path(runs['task2_full']['path'])/'result.json')['rows']) == 96
    for filename in ['执行协议.md', '理论口径与近邻对照.md', '实验与贡献判断.md', 'analysis/图注与来源.md']:
        p = HERE/filename
        text = p.read_text(encoding='utf-8')
        for target in re.findall(r'\]\(([^)]+)\)', text):
            if '://' in target or target.startswith('#'):
                continue
            q = unquote(target.split('#')[0])
            if (p.parent/q).resolve() == (HERE/'audit.json').resolve():
                continue  # This audit's own output is created after all other checks.
            checks['link/'+filename+'/'+q] = (p.parent/q).exists()
    figures = ['task1_mechanism', 'task2_backend']
    for figure in figures:
        for suffix in ['png', 'pdf']:
            p = HERE/'analysis'/(figure+'.'+suffix)
            checks['figure/'+p.name] = p.exists() and p.stat().st_size > 1000
    result = dict(checks=checks, passed=all(checks.values()), nchecks=len(checks), runs=runs,
                  failed_attempt=dict(path=str(failed), failures=failures,
                    seconds=failed_result['seconds'], cause='Floating-point addition order before FFT; not reused'),
                  task3_started=False, outputs_sha256={str(p.relative_to(HERE)):sha(p)
                    for p in [*HERE.glob('*.py'), *HERE.glob('*.md'), *(HERE/'analysis').glob('*')]
                    if p.is_file()})
    (HERE/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'],nchecks=len(checks),
                         failures=[k for k,v in checks.items() if not v]),ensure_ascii=False))
    assert result['passed']


if __name__ == '__main__':
    main()
