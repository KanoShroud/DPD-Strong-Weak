"""Offline v2/v3 replay over all frozen R4 evidence; never writes old results."""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .assignment_metrics import evaluate, self_test, REVISION
from ..formal_protocol import BASE, ROOT, key
from ..formal_runtime import fresh, safe, read, write, sha, seal


def run():
    sources = {}
    for name in ('formal_run', 'profile_full_run', 'paper_pair_run'):
        pointer = read(BASE/(name+'.json'))
        folder = safe(pointer['out']).resolve(strict=True)
        assert sha(folder/'contract.json') == pointer['contract_sha']
        audit = read(folder/'final_audit_report.json')
        assert audit['passed']
        sources[name] = dict(folder=str(folder), contract_sha=pointer['contract_sha'],
                             audit_sha=sha(folder/'final_audit_report.json'), audit=audit)
    out = safe(fresh('assignment_audit'))
    assert not out.exists()
    out.mkdir()
    tests = self_test()
    entries, changes, groups, hashes = [], [], defaultdict(list), {}
    inputs = sources['formal_run']['audit']['inputs']
    for name, source in sources.items():
        folder = Path(source['folder'])
        paired = name == 'paper_pair_run'
        files = source['audit']['pair_hashes' if paired else 'rows']
        for filename, expected in files.items():
            p = folder/('pairs' if paired else 'rows')/filename
            assert sha(p) == expected, str(p)
            row = read(p)
            assert row['record_sha'] == seal(row)
            task = row['task'] if paired else row['job']['task']
            identity = inputs[key(task)]
            assert row['input_identity'] == identity['identity']
            assert row['truth'] == identity['truth']
            results = row['results'] if paired else {row['job']['method']: row['result']}
            for method, result in results.items():
                old = row['metrics'][method] if paired else row['metrics']
                new = evaluate(row['truth'], result['positions'])
                errors_same = np.allclose(old['errors_m'], new['errors_m'], rtol=0, atol=1e-10)
                item = dict(source=name, case=key(task), layout=task[0], repeat=task[1],
                            gap=task[2], method=method, strong_error_m=new['errors_m'][0],
                            weak_error_m=new['errors_m'][1], old_strong=old['success']['20'][0],
                            old_weak=old['success']['20'][1], strong20=new['success20'][0],
                            weak20=new['success20'][1], both20=new['both20'],
                            old_both=all(old['success']['20']), errors_unchanged=bool(errors_same),
                            matched=new['matched'], valid_output=new['valid_output'])
                entries.append(item)
                groups[(name, method)].append(item)
                if old['success']['20'] != new['success20'] or not errors_same:
                    changes.append(dict(**item, truth=row['truth'], positions=result['positions']))
            hashes[str(p)] = expected
        print(f'离线核验完成：{name}，累计{len(entries)}份方法输出', flush=True)
    summaries = []
    for (name, method), rows in groups.items():
        summaries.append(dict(source=name, method=method, n=len(rows),
            strong_changed=sum(r['old_strong'] != r['strong20'] for r in rows),
            weak_changed=sum(r['old_weak'] != r['weak20'] for r in rows),
            both_changed=sum(r['old_both'] != r['both20'] for r in rows),
            error_changed=sum(not r['errors_unchanged'] for r in rows),
            old_success=sum(r['old_both'] for r in rows), new_success=sum(r['both20'] for r in rows)))
    for name, rows in [('per_case', entries), ('summary', summaries)]:
        with (out/(name+'.csv')).open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    for source in sources.values():
        folder = Path(source['folder'])
        assert sha(folder/'contract.json') == source['contract_sha']
        assert sha(folder/'final_audit_report.json') == source['audit_sha']
    report = dict(passed=True, protocol=REVISION, tests=tests, output_count=len(entries),
                  changes=changes, summaries=summaries, input_and_row_identity=True,
                  sources={n:{k:v for k,v in s.items() if k!='audit'} for n,s in sources.items()},
                  checked_rows=hashes, code={str(p.relative_to(ROOT)):sha(p) for p in
                      (Path(__file__), Path(__file__).with_name('assignment_metrics.py'))})
    write(out/'audit.json', report)
    write(BASE/'assignment_audit_run.json', dict(out=str(out), report_sha=sha(out/'audit.json')))
    (out/'回读分析.md').write_text(
        f'# 统一匹配口径离线核验\n\n版本：{REVISION}。\n\n'
        f'核验{len(entries)}份方法输出；存在指标差异的输出{len(changes)}份。'
        '样本规模按来源分别统计，不将重叠观测累加为独立样本。\n\n'
        'CDF/RMSE和成功率统一使用平方距离最优匹配；历史结果及冻结评价器未修改。'
        '新结果、旧结果和输入身份逐条配对；详细差异见audit.json及per_case.csv。\n', encoding='utf-8')
    print(str(out), flush=True)
    print(summaries, flush=True)


if __name__ == '__main__':
    run()
