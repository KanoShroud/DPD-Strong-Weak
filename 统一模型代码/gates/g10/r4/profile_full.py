"""Frozen PROFILE on the existing R4 main cohort; no algorithm changes."""
import argparse
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from ..runtime import read, write, sha, verify
from ..metrics import evaluate
from ..r2.runtime import seal
from ..r2.report import csv_write
from .paper_pair_runner import build_contract, check_contract, consume, fresh, safe
from .formal_runtime import compact_result, snapshot
from .formal_protocol import BASE, ROOT, key, solver_seed
from .formal_adapter import run
from .formal_report import flat, summary, table
from .runtime import require_exclusive, resource_check, lock
from ..r1.runtime import Ledger

ACTIVE = BASE/'profile_full_run.json'


def validate(row, task, c):
    assert row['record_sha'] == seal(row)
    assert row['job'] == dict(task=list(task), method='PROFILE_DIAG')
    expected = c['inputs'][key(task)]
    assert row['truth'] == expected['truth'] and row['input_identity'] == expected['identity']
    assert row['result']['config'] == c['profile_configuration']
    assert row['metrics'] == evaluate(row['truth'], row['result']['positions'])
    assert np.isfinite(row['result']['positions']).all()


def create():
    c = build_contract()
    original = check_contract(c)
    c['profile_configuration'] = original['configurations']['PROFILE_DIAG']
    c['sources'][str(Path(__file__).relative_to(ROOT))] = sha(__file__)
    c['purpose'] = 'PROFILE full 2400 validation; old methods read-only; no retuning'
    out = safe(fresh('profile_full'))
    assert not out.exists()
    out.mkdir()
    write(out/'contract.json', c)
    snapshot(out, c['sources'])
    write(ACTIVE, dict(out=str(out), contract_sha=sha(out/'contract.json')))
    return out


def report(out, c):
    source = Path(c['source'])
    audit = read(source/'final_audit_report.json')
    rows, hashes, recovered, harmed, replayed = [], {}, [], [], 0
    for task in c['tasks']:
        p = out/'rows'/(key(task)+'.json')
        q = read(p); validate(q, task, c); hashes[p.name] = sha(p)
        rows.append(flat(q, c))
        for method in ('NOMP', 'AP_5_JOINT'):
            old_path = source/'rows'/(key(task)+'_'+method+'.json')
            assert sha(old_path) == audit['rows'][old_path.name]
            old = read(old_path)
            assert old['input_identity'] == q['input_identity']
            assert old['metrics'] == evaluate(old['truth'], old['result']['positions'])
            rows.append(flat(old, c))
            if method == 'NOMP':
                a = all(old['metrics']['success']['20']); b = all(q['metrics']['success']['20'])
                if b and not a: recovered.append(key(task))
                if a and not b: harmed.append(key(task))
        old_path = source/'rows'/(key(task)+'_PROFILE_DIAG.json')
        if old_path.exists():
            assert sha(old_path) == audit['rows'][old_path.name]
            old = read(old_path)
            assert np.allclose(old['result']['positions'], q['result']['positions'], rtol=0, atol=1e-5)
            assert old['result']['stop_reason'] == q['result']['stop_reason']
            replayed += 1
    folder = out/'reports'; folder.mkdir(exist_ok=True)
    csv_write(folder/'per_case_metrics.csv', rows)
    summaries = []
    for method in ('PROFILE_DIAG', 'NOMP', 'AP_5_JOINT'):
        part = [r for r in rows if r['method'] == method]
        summaries.append(dict(method=method, scope='all', **summary(part)))
        for layout in range(20):
            summaries.append(dict(method=method, scope='layout_'+str(layout),
                                  **summary([r for r in part if r['layout'] == layout])))
    table(folder, 'summary_table', summaries)
    write(folder/'comparison_report.json', dict(summaries=summaries,
          recovered_vs_nomp=recovered, harmed_vs_nomp=harmed,
          old_subset_reproduced=replayed,
          timing_note='PROFILE freshly timed; NOMP/AP historical same hardware, not interleaved timing.'))
    write(out/'final_audit_report.json', dict(passed=True, rows=hashes,
          metric_replay=True, old_subset_reproduced=replayed, source_identity=True))
    text = '# PROFILE完整主批完成\n\n'
    for s in summaries:
        if s['scope'] == 'all':
            text += f"- {s['method']}：{s['both_count']}/2400；弱源RMSE {s['weak_rmse_m']:.4f}m；平均{s['mean_seconds']:.4f}s。\n"
    text += '\nPROFILE为本轮计时，其他方法沿用原R4计时，不作为同批严格配对加速比例。\n'
    (out/'运行摘要.md').write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, help='Finite integration prefix, resume without this option')
    args = parser.parse_args()
    require_exclusive(); resource_check()
    if ACTIVE.exists():
        pointer = read(ACTIVE); out = safe(pointer['out'])
        assert sha(out/'contract.json') == pointer['contract_sha']
    else:
        out = create()
    c = read(out/'contract.json'); original = check_contract(c)
    if (out/'status.json').exists() and read(out/'status.json')['status'] == 'COMPLETED':
        print('已经完成：'+str(out)); return
    print('结果目录：'+str(out), flush=True)
    with lock(out):
        clock = Ledger(out)
        try:
            t = time.perf_counter()
            torch.zeros(1, device='cuda'); torch.cuda.synchronize()
            print(f'公共预热{time.perf_counter()-t:.3f}s，单列', flush=True)
            last = 0.; times = []
            for i, task in enumerate(c['tasks'], 1):
                iq, truth, identity, _ = consume(c, original, task)
                path = out/'rows'/(key(task)+'.json')
                if path.exists():
                    q = read(path); validate(q, task, c)
                else:
                    clock.flush(dict(task=task)); mark = [0.]
                    def guard():
                        if time.perf_counter()-mark[0] >= 1:
                            clock.check(); mark[0] = time.perf_counter()
                    result = run(iq, 'PROFILE_DIAG', c['profile_configuration'], solver_seed(task), guard)
                    q = dict(job=dict(task=task, method='PROFILE_DIAG'), truth=truth.tolist(),
                             input_identity=identity, result=compact_result(result),
                             metrics=evaluate(truth, result['positions']))
                    q['record_sha'] = seal(q); validate(q, task, c); write(path, q); clock.flush()
                times.append(q['result']['complete_seconds'])
                if time.perf_counter()-last > 20 or i == len(c['tasks']):
                    print(f'PROFILE {i}/2400 | 累计{clock.total()/60:.1f}min | ETA {np.mean(times)*(2400-i)/60:.1f}min', flush=True)
                    last = time.perf_counter()
                if args.limit and i >= args.limit:
                    clock.flush(); write(out/'status.json', dict(status='PREFIX_COMPLETE', count=i)); return
            verify(c['sources']); check_contract(c)
            for task in c['tasks']:
                consume(c, original, task)
            report(out, c)
            clock.flush(); write(out/'status.json', dict(status='COMPLETED', seconds=clock.total()))
            print('完成：'+str(out), flush=True)
        except BaseException:
            clock.flush(keep=True)
            write(out/'status.json', dict(status='STOPPED', error=traceback.format_exc()))
            raise


if __name__ == '__main__':
    main()
