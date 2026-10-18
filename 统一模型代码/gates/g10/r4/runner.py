"""Preparation only. Completed runs exit; formal validation is never auto-started."""
import argparse
import time
import traceback
from pathlib import Path
from unittest.mock import patch
import sys
import numpy as np
import torch
from ..metrics import evaluate
from ..runtime import save_scene, load_scene
from ..r1.scenes import make as make_scene
from ..r1.runtime import Ledger
from . import solvers
from .config import BASE, METHODS
from .runtime import (read, write, seal, sha, lock, create, check_contract, job_key, load,
                      validate, audit, require_exclusive)
from .checks import cpu, gpu
from .report import generate


def record(out, job, contract, clock):
    _, iq, truth, identity = load(contract, job['task'])
    cfg = contract['configurations'][job['method']]
    path = out/'rows'/(job_key(job)+'.json')
    if path.exists():
        row = read(path); validate(row, job, identity, cfg, truth)
        return row, False
    require_exclusive()
    clock.flush(dict(tag=path.stem, ledger_start_seconds=clock.total()))
    result = solvers.run(iq, job['method'], cfg, clock.check)
    row = dict(job=job, truth=truth.tolist(), input_sha=identity, result=result,
               metrics=evaluate(truth, result['positions']))
    row['record_sha'] = seal(row)
    write(path, row)
    validate(read(path), job, identity, cfg, truth)
    clock.flush()
    return row, True


def resume_check(out, c, clock):
    job = c['jobs'][0]
    toy = out/'recovery_fixture'
    with patch.object(solvers, 'run', side_effect=KeyboardInterrupt('intentional test')):
        try:
            record(toy, job, c, clock)
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError('Interruption swallowed')
    assert not (toy/'rows'/(job_key(job)+'.json')).exists()
    existing = read(out/'rows'/(job_key(job)+'.json'))['result']
    with patch.object(solvers, 'run', return_value=existing) as mock:
        _, computed = record(toy, job, c, clock)
        assert computed and mock.call_count == 1
    with patch.object(solvers, 'run', side_effect=AssertionError('Unexpected rerun')):
        assert not record(toy, job, c, clock)[1]
        assert not record(out, job, c, clock)[1]
        generate(out)
    write(out/'recovery_checks.json', dict(passed=True, interrupted_row_not_committed=True,
        completed_row_reused=True, redraw_does_not_invoke_solver=True))


def execute(out, checks_only=False):
    require_exclusive()
    c = check_contract(out)
    with lock(out):
        clock = Ledger(out)
        try:
            if not (out/'cpu_checks.json').exists():
                print('阶段1/3：原文公式、梯度、输入与评价口径检查。', flush=True)
                cpu(out, c)
            if not (out/'gpu_checks.json').exists():
                print('阶段2/3：冻结前缀等价和无噪声离网检查。', flush=True)
                gpu(out, c)
            if checks_only:
                clock.flush()
                write(out/'status.json', dict(status='CHECKS_COMPLETE', formal_started=False))
                print('正确性检查完成；尚未运行16条成本短测。', flush=True)
                return
            if not (out/'io_cost.json').exists():
                start = time.perf_counter()
                scene = make_scene(0, 200, 0)
                save_scene(out/'io_probe', 'development', scene)
                recovered = load_scene(out/'io_probe', 'development')
                np.testing.assert_array_equal(recovered['signal'], scene['signal'])
                write(out/'io_cost.json', dict(seconds=time.perf_counter()-start,
                    scope='one already-used development configuration; no formal samples'))
            start = time.perf_counter()
            _, iq, _, _ = load(c, c['jobs'][0]['task'])
            for method in METHODS:
                solvers.run(iq, method, c['configurations'][method], clock.check)
            torch.cuda.synchronize()
            warmups = read(out/'warmups.json') if (out/'warmups.json').exists() else []
            env = dict(python=sys.version, numpy=np.__version__, torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0), threads=torch.get_num_threads())
            if warmups and warmups[-1]['environment'] != env:
                raise RuntimeError('Resume environment differs')
            warmups.append(dict(seconds=time.perf_counter()-start, environment=env))
            write(out/'warmups.json', warmups)
            print(f"阶段3/3：16条×7版本成本短测；公共预热{warmups[-1]['seconds']:.2f}s单列。", flush=True)
            start, last, computed = time.perf_counter(), 0., 0
            for i, job in enumerate(c['jobs'], 1):
                if clock.total() >= 90*60:
                    write(out/'status.json', dict(status='PREPARATION_SCHEDULE_PAUSED', completed=i-1,
                        formal_started=False, note='90分钟准备排期；未截断已开始的求解。'))
                    print('\n准备排期已到，在单例之间暂停。结果已保存，可由同一入口继续。')
                    clock.flush(); return
                _, new = record(out, job, c, clock)
                computed += int(new)
                now = time.perf_counter()
                if now-last >= 15 or i == len(c['jobs']):
                    filled = int(24*i/len(c['jobs']))
                    eta = (now-start)/max(computed, 1)*(len(c['jobs'])-i)
                    print(f"\rG10-R4 [{'#'*filled}{'-'*(24-filled)}] {i}/{len(c['jobs'])}"
                          f" | {(now-start)/60:.1f}min | ETA {eta/60:.1f}min | {job_key(job)}".ljust(145),
                          end='\n' if i == len(c['jobs']) else '', flush=True)
                    last = now
            write(out/'batch_timing.json', dict(seconds=time.perf_counter()-start, newly_computed=computed))
            audit(out)
            report, forecast = generate(out)
            resume_check(out, c, clock)
            clock.flush()
            write(out/'status.json', dict(status='COMPLETED_PREPARATION', completed=len(c['jobs']),
                formal_started=False, seconds=clock.total(), budget_pending_confirmation=True,
                formal_release='HOLD_BNOMP_NATIVE_STOP_REVIEW'))
            print(f"短测完成：{out}", flush=True)
            print(f"正式均值外推约{forecast['estimate_hours']:.2f}小时；P95情景约{forecast['p95_scenario_hours']:.2f}小时。")
            print('正式比较未启动，等待回读与预算确认。')
            return report, forecast
        except BaseException as exc:
            clock.flush(keep=True)
            write(out/'status.json', dict(status='STOPPED', error=str(exc), traceback=traceback.format_exc(),
                formal_started=False, seconds=clock.total()))
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description='G10-R4短测；不会启动正式2400条比较。')
    parser.add_argument('--new', action='store_true')
    parser.add_argument('--checks-only', action='store_true')
    parser.add_argument('--replot', type=Path)
    args = parser.parse_args(argv)
    if args.replot:
        generate(args.replot.resolve(strict=True)); return
    require_exclusive()
    pointer = BASE/'preparation.json'
    if args.new or not pointer.exists():
        out = create()
    else:
        item = read(pointer)
        out = Path(item['out']).resolve(strict=True)
        if sha(out/'contract.json') != item['contract_sha']:
            raise RuntimeError('Preparation contract identity changed')
        check_contract(out)
        if read(out/'status.json')['status'] == 'COMPLETED_PREPARATION':
            print('短测已经完成，不重复运行。结果目录：', out)
            print('正式比较需另行确认预算，本入口不会自动启动。'); return
    print('R4结果目录：', out, flush=True)
    execute(out, args.checks_only)


if __name__ == '__main__':
    main()
