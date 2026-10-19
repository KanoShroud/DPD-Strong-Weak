"""Paired full-cost AP M=1..5 versus native PROFILE on frozen common 240 IQs.

Only the two method-name allowlists are extended in private function globals.
The frozen adapter and solver bytecode, physics, candidate order, stopping and
timing boundaries are unchanged. No global module state is patched.
"""
import argparse
import copy
import gc
import time
import traceback
from pathlib import Path
from types import FunctionType

import numpy as np

from .assignment_metrics import evaluate, REVISION, self_test
from .. import formal_adapter, solvers
from ..formal_protocol import BASE, ROOT, key, solver_seed
from ..formal_runtime import read, write, sha, seal, compact_result, snapshot
from ..paper_pair_runner import build_contract, check_contract, consume, fresh, safe
from ..runtime import require_exclusive, resource_check, lock
from ...runtime import verify
from ...r1.runtime import Ledger

METHODS = tuple(f'AP_{n}_JOINT' for n in range(1, 6))+('PROFILE_DIAG',)
ACTIVE = BASE/'candidate_cost_run.json'


def extended_adapter():
    g = dict(solvers.run.__globals__)
    g['METHODS'] = solvers.METHODS+('AP_3_JOINT', 'AP_4_JOINT')
    core = FunctionType(solvers.run.__code__, g, 'allowlisted_core', solvers.run.__defaults__)
    g = dict(formal_adapter.run.__globals__)
    g['CORE'] = formal_adapter.CORE+('AP_3_JOINT', 'AP_4_JOINT')
    g['core_run'] = core
    return FunctionType(formal_adapter.run.__code__, g, 'allowlisted_adapter', formal_adapter.run.__defaults__)


RUN = extended_adapter()


def plan(tasks):
    # Frozen representative prefix includes all four classes and known layout 3.
    common = [t for t in tasks if t[1] in (400, 410)]
    prefix = [t for t in common if t[0] in (0, 3, 5, 10, 15) and t[1] == 400 and t[2] in (0, 15)]
    return prefix+[t for t in common if t not in prefix]


def create():
    c = build_contract()
    original = check_contract(c)
    c.update(tasks=plan(c['tasks']), methods=list(METHODS), metric_protocol=REVISION,
             purpose='Interleaved AP candidate-count and native PROFILE complete cost',
             approved_expected_solves=1440,
             timing='Unchanged formal_adapter complete_seconds; per-call input FFT/transfer included; '
                    'warmup, input disk IO, audit and JSON IO recorded outside algorithm timing.')
    assert len(c['tasks']) == 240
    c['cost_configurations'] = {m: copy.deepcopy(original['configurations'][
        'AP_5_JOINT' if m.startswith('AP_') else m]) for m in METHODS}
    for m in ('AP_1_JOINT', 'AP_2_JOINT', 'AP_5_JOINT'):
        assert c['cost_configurations'][m] == original['configurations'][m]
    for p in (Path(__file__), Path(__file__).with_name('assignment_metrics.py')):
        c['sources'][str(p.relative_to(ROOT))] = sha(p)
    out = safe(fresh('candidate_cost'))
    assert not out.exists() and not out.is_relative_to(Path(c['source']))
    out.mkdir()
    write(out/'contract.json', c)
    snapshot(out, c['sources'])
    write(ACTIVE, dict(out=str(out), contract_sha=sha(out/'contract.json')))
    write(out/'checks.json', self_test())
    return out


def validate(row, task, method, c, old):
    assert row['record_sha'] == seal(row)
    assert row['task'] == task and row['method'] == method
    assert row['input_identity'] == c['inputs'][key(task)]['identity']
    assert row['truth'] == c['inputs'][key(task)]['truth']
    result = row['result']
    assert result['config'] == c['cost_configurations'][method]
    assert row['metrics'] == evaluate(row['truth'], result['positions'])
    assert result['complete_seconds'] > 0 and np.isfinite(result['joint_objective'])
    if method.startswith('AP_'):
        n = int(method.split('_')[1])
        ref = min(old['result']['mechanism']['branches'][:n], key=lambda b:(b['objective'], b['rank']))
        np.testing.assert_allclose(result['positions'], ref['positions'], rtol=0, atol=1e-5)
        np.testing.assert_allclose(result['mechanism']['candidate_list'],
                                   old['result']['mechanism']['candidate_list'], rtol=0, atol=1e-8)
        assert result['mechanism']['selected_rank'] == ref['rank']
        assert len(result['mechanism']['branches']) == n
    else:
        path = Path(c['source'])/'rows'/(key(task)+'_PROFILE_DIAG.json')
        audit = read(Path(c['source'])/'final_audit_report.json')
        assert sha(path) == audit['rows'][path.name]
        ref = read(path)
        np.testing.assert_allclose(result['positions'], ref['result']['positions'], rtol=0, atol=1e-5)
        assert result['counts'] == ref['result']['counts']
        assert result['stop_reason'] == ref['result']['stop_reason']


def progress(out, c):
    rows = [read(p) for p in sorted((out/'rows').glob('*.json'))]
    by_method = {}
    for m in METHODS:
        rr = [r for r in rows if r['method'] == m]
        if not rr:
            continue
        v = [r['result']['complete_seconds'] for r in rr]
        by_method[m] = dict(n=len(rr), mean_seconds=float(np.mean(v)),
                           p95_seconds=float(np.quantile(v, .95)),
                           both20=sum(r['metrics']['both20'] for r in rr))
    estimate = 240*sum(v['mean_seconds'] for v in by_method.values())*1.25+300
    write(out/'progress.json', dict(solves=len(rows), expected=1440, summaries=by_method,
          estimate_total_seconds=estimate, estimate_basis='representative prefix, 25% margin + 300s audit/report allowance'))
    return rows, estimate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, help='Complete observation groups; resume without limit')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    require_exclusive(); resource_check()
    if ACTIVE.exists():
        pointer = read(ACTIVE); out = safe(pointer['out']).resolve(strict=True)
        assert sha(out/'contract.json') == pointer['contract_sha']
    else:
        out = create()
    c = read(out/'contract.json'); original = check_contract(c)
    if (out/'status.json').exists() and read(out/'status.json')['status'] == 'COMPLETED' and not args.audit_only:
        print('已经完成：'+str(out), flush=True); return
    print('结果目录：'+str(out), flush=True)
    with lock(out):
        clock = Ledger(out)
        try:
            if not args.audit_only:
                start = time.perf_counter()
                iq, _, _, old = consume(c, original, c['tasks'][0])
                for m in METHODS:
                    gc.collect()
                    result = RUN(iq, m, c['cost_configurations'][m], solver_seed(c['tasks'][0]))
                    if m.startswith('AP_'):
                        n = int(m.split('_')[1])
                        b = min(old['result']['mechanism']['branches'][:n], key=lambda x:(x['objective'],x['rank']))
                        np.testing.assert_allclose(result['positions'], b['positions'], rtol=0, atol=1e-5)
                warm = dict(seconds=time.perf_counter()-start, methods=list(METHODS), separate=True)
                write(out/('warmup_'+str(time.time_ns())+'.json'), warm)
                print(f"统一预热六版本：{warm['seconds']:.2f}s，单列", flush=True)
            last = 0.
            for i, task in enumerate(c['tasks']):
                iq, truth, identity, old = consume(c, original, task)
                order = METHODS[i % 6:]+METHODS[:i % 6]
                for m in order:
                    p = out/'rows'/(key(task)+'_'+m+'.json')
                    if p.exists():
                        validate(read(p), task, m, c, old)
                        continue
                    if args.audit_only:
                        raise RuntimeError('Missing completed row: '+str(p))
                    require_exclusive()
                    gc.collect()  # symmetric between-trial cleanup, outside timers
                    clock.flush(dict(task=task, method=m, ledger_start_seconds=clock.total()))
                    result = RUN(iq, m, c['cost_configurations'][m], solver_seed(task))
                    row = dict(task=task, method=m, order=list(order), input_identity=identity,
                               truth=truth.tolist(), result=compact_result(result),
                               metrics=evaluate(truth, result['positions']))
                    row['record_sha'] = seal(row)
                    validate(row, task, m, c, old)
                    write(p, row); clock.flush()
                if time.perf_counter()-last > 20 or i == 239:
                    print(f'共同计时 {i+1}/240组 | 累计{clock.total()/60:.1f}min | {key(task)}', flush=True)
                    last = time.perf_counter()
                if args.limit and i+1 >= args.limit:
                    rows, estimate = progress(out, c)
                    clock.flush()
                    write(out/'status.json', dict(status='PREFIX_COMPLETE', solves=len(rows)))
                    print(f'前缀完成；全批保守估算{estimate/60:.1f}min', flush=True)
                    return
            verify(c['sources']); check_contract(c)
            for task in c['tasks']:
                consume(c, original, task)
            rows, _ = progress(out, c)
            assert len(rows) == 1440
            hashes = {p.name:sha(p) for p in (out/'rows').glob('*.json')}
            write(out/'final_audit_report.json', dict(passed=True, rows=hashes, input_identity=True,
                  metric_replay=True, original_output_reproduced=True, count=1440,
                  stopping='native frozen per-method rules, not common-gradient stopping',
                  source_identity=True, protocol=REVISION))
            clock.flush()
            write(out/'status.json', dict(status='COMPLETED', seconds=clock.total(), solves=1440))
            print('完成：'+str(out), flush=True)
        except BaseException:
            clock.flush(keep=True)
            write(out/'status.json', dict(status='STOPPED', error=traceback.format_exc()))
            raise


if __name__ == '__main__':
    main()
