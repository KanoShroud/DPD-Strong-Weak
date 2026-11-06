"""User-run AP5 paired comparison. Atomic complete-pair commits and safe resume."""
import argparse
import copy
from datetime import datetime
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
import torch
from ..runtime import read, write, sha, verify
from ..metrics import evaluate
from ..r1.runtime import Ledger
from ..r3.solvers import Model, Trace, projected_gradient
from .config import ROOT, BASE
from .formal_protocol import tasks, key
from .formal_runtime import check as check_original, ensure_input
from .runtime import require_exclusive, resource_check, seal, lock
from .solvers import ap

ENTRY = ROOT/'运行入口/E2E/G10/G10R4五候选配对一键运行.py'
RELEASE = BASE/'paper_pair_release.json'
ACTIVE = BASE/'paper_pair_run.json'
METHODS = ('AP_5_ALT', 'AP_5_JOINT')
REVISION = 'G10_R4_AP5_PAIRED_V1'


def fresh(prefix):
    return BASE/(prefix+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))


def safe(out):
    p = Path(out).resolve()
    if not p.is_relative_to(BASE.resolve()) or p == BASE.resolve():
        raise ValueError('Expected isolated R4 output directory')
    return p


def build_contract():
    pointer = read(BASE/'formal_run.json')
    source = safe(pointer['out']).resolve(strict=True)
    if sha(source/'contract.json') != pointer['contract_sha']:
        raise RuntimeError('Original contract changed')
    original = check_original(source)
    audit = read(source/'final_audit_report.json')
    if not audit['passed'] or read(source/'status.json')['status'] != 'COMPLETED':
        raise RuntimeError('Original R4 has not completed')
    cfg = read(source/'rows/0_400_0_AP_5_JOINT.json')['result']['config']
    extra = [Path(__file__), Path(__file__).with_name('paper_pair_report.py'), ENTRY]
    sources = dict(original['sources'])
    sources.update({str(p.relative_to(ROOT)): sha(p) for p in extra})
    plan = [list(t) for t in tasks()]
    return dict(revision=REVISION, source=str(source), source_contract_sha=pointer['contract_sha'],
        original_audit_sha=sha(source/'final_audit_report.json'), sources=sources,
        configuration=cfg, tasks=plan, layouts=original['layouts'],
        inputs=audit['inputs'], references={key(t): audit['rows'][key(t)+'_AP_5_JOINT.json'] for t in plan},
        forecast_hours=1.48, methods=list(METHODS),
        pairing='Alternate method order per observation; commit only complete pairs; no sample rejection by performance.')


def check_contract(c):
    if c['revision'] != REVISION:
        raise RuntimeError('Unknown paired contract')
    verify(c['sources'])
    source = safe(c['source']).resolve(strict=True)
    if sha(source/'contract.json') != c['source_contract_sha'] or sha(source/'final_audit_report.json') != c['original_audit_sha']:
        raise RuntimeError('Frozen original evidence changed')
    original = check_original(source)
    for task in c['tasks']:
        name = key(task)
        meta = source/'inputs'/(name+'.json')
        if sha(meta) != c['inputs'][name]['identity']['metadata_sha']:
            raise RuntimeError('Input metadata changed: '+name)
        payload = (source/read(meta)['path']).resolve(strict=True)
        if not payload.is_relative_to(source/'inputs'):
            raise RuntimeError('Input path escapes original input root')
        old = source/'rows'/(name+'_AP_5_JOINT.json')
        if sha(old) != c['references'][name]:
            raise RuntimeError('Original joint result changed: '+name)
    return original


def consume(c, original, task):
    source = Path(c['source'])
    name = key(task)
    # Never enter ensure_input's generation branch: both frozen files must exist.
    meta = source/'inputs'/(name+'.json')
    if sha(meta) != c['inputs'][name]['identity']['metadata_sha']:
        raise RuntimeError('Input metadata differs')
    if not (source/read(meta)['path']).is_file():
        raise FileNotFoundError('Frozen observation missing: '+name)
    iq, truth, identity = ensure_input(source, tuple(task), original)
    expected = c['inputs'][name]
    if identity != expected['identity'] or truth.tolist() != expected['truth']:
        raise RuntimeError('Consumed input identity differs')
    old_path = source/'rows'/(name+'_AP_5_JOINT.json')
    if sha(old_path) != c['references'][name]:
        raise RuntimeError('Original result differs')
    old = read(old_path)
    if old['result']['config'] != c['configuration']:
        raise RuntimeError('Configuration differs between original samples')
    return iq, truth, identity, old


def solve(iq, method, cfg, clock):
    """Same AP and common-stop implementation as the validated pilot."""
    torch.cuda.synchronize()
    started = time.perf_counter()
    last = [started]

    def guard():
        now = time.perf_counter()
        if now-last[0] >= 1.:
            clock.check()
            last[0] = now

    trace = Trace(guard)
    model = Model(iq, guard, sample_rate_hz=cfg['sample_rate_hz'], half_band_hz=cfg['half_band_hz'])
    trace.model = model
    try:
        detail = ap(model, trace, method, cfg)
        p = np.asarray(detail['output_positions'])
        value, grad = model.joint(p, True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        return dict(method=method, positions=p.tolist(), seconds=elapsed,
            joint_objective=value, projected_gradient=projected_gradient(p, grad),
            counts=copy.deepcopy(model.counts), mechanism=detail)
    finally:
        trace.model = None


def validate(row, task, c, old):
    if row.get('record_sha') != seal(row) or row['task'] != list(task):
        raise RuntimeError('Pair identity/sequence mismatch')
    expected = c['inputs'][key(task)]
    if row['input_identity'] != expected['identity'] or row['truth'] != expected['truth']:
        raise RuntimeError('Pair/input mismatch')
    for method in METHODS:
        r = row['results'][method]
        if r['method'] != method or row['metrics'][method] != evaluate(row['truth'], r['positions']):
            raise RuntimeError('Method/metric replay mismatch')
        if not np.isfinite(r['joint_objective']) or not np.isfinite(r['projected_gradient']):
            raise RuntimeError('Nonfinite result')
        if len(r['mechanism']['branches']) != 5:
            raise RuntimeError('Expected exactly five branches')
    a, b = (row['results'][m] for m in METHODS)
    if not np.allclose(a['mechanism']['candidate_list'], b['mechanism']['candidate_list'], rtol=0, atol=1e-8):
        raise RuntimeError('Five initial candidates not paired')
    if not np.allclose(b['mechanism']['candidate_list'], old['result']['mechanism']['candidate_list'], rtol=0, atol=1e-8):
        raise RuntimeError('Original initial candidates not reproduced')
    if not np.allclose(b['positions'], old['result']['positions'], rtol=0, atol=1e-5):
        raise RuntimeError('Original joint final output not reproduced')
    # Native numerical stops are reported, not filtered to manufacture a success.


def create(c, prefix):
    out = safe(fresh(prefix))
    if out.exists() or out.is_relative_to(Path(c['source'])):
        raise RuntimeError('Fresh separate output required')
    out.mkdir()
    write(out/'contract.json', c)
    for rel, h in c['sources'].items():
        p = out/'source_snapshot'/rel
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/rel, p)
        if sha(p) != h:
            raise RuntimeError('Source snapshot mismatch')
    write(out/'status.json', dict(status='CREATED', completed_pairs=0))
    return out


def execute(out, solver=solve, stop_after=None):
    out = safe(out).resolve(strict=True)
    c = read(out/'contract.json')
    original = check_contract(c)
    require_exclusive(); resource_check()
    if read(out/'status.json')['status'] == 'COMPLETED':
        print('完整配对已完成，不重复求解。结果：'+str(out), flush=True)
        return
    with lock(out):
        clock = Ledger(out)
        done, computed, last_log, sum_pair_seconds = 0, 0, 0., 0.
        try:
            warmup_done = False
            pair_hashes = {}
            for index, task in enumerate(c['tasks']):
                name = key(task)
                iq, truth, identity, old = consume(c, original, task)
                path = out/'pairs'/(name+'.json')
                if path.exists():
                    saved = read(path)
                    validate(saved, task, c, old)
                    sum_pair_seconds += sum(saved['results'][m]['seconds'] for m in METHODS)
                    done += 1
                    pair_hashes[path.name] = sha(path)
                    continue
                if not warmup_done:
                    print(f'已有{done}/{len(c["tasks"])}对完整结果；开始本次预热（单列）。', flush=True)
                    clock.flush(dict(tag='warmup', ledger_start_seconds=clock.total()))
                    t = time.perf_counter()
                    solver(iq, 'AP_5_JOINT', c['configuration'], clock)
                    write(out/('warmup_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json'), dict(seconds=time.perf_counter()-t))
                    clock.flush(); warmup_done = True
                order = METHODS if index % 2 == 0 else METHODS[::-1]
                clock.flush(dict(tag=name, ledger_start_seconds=clock.total()))
                results = {m: solver(iq, m, c['configuration'], clock) for m in order}
                row = dict(task=list(task), input_identity=identity, truth=truth.tolist(), order=list(order),
                    results=results, metrics={m: evaluate(truth, results[m]['positions']) for m in METHODS})
                row['record_sha'] = seal(row)
                validate(row, task, c, old)
                write(path, row)
                validate(read(path), task, c, old)
                pair_hashes[path.name] = sha(path)
                done += 1; computed += 1
                sum_pair_seconds += sum(results[m]['seconds'] for m in METHODS)
                clock.flush()
                now = time.perf_counter()
                if now-last_log >= 20 or done == len(c['tasks']):
                    width = int(24*done/len(c['tasks']))
                    mean = sum_pair_seconds/done
                    eta = (len(c['tasks'])-done)*mean/60
                    print((f'\r五候选配对 [{"#"*width}{"-"*(24-width)}] {done}/{len(c["tasks"])}对'
                           f' | 累计{clock.total()/60:.1f}min | ETA约{eta:.1f}min | {name}').ljust(150),
                          end='\n' if done == len(c['tasks']) else '', flush=True)
                    last_log = now
                    write(out/'status.json', dict(status='RUNNING', completed_pairs=done, seconds=clock.total()))
                if stop_after is not None and computed >= stop_after:
                    write(out/'status.json', dict(status='PREPARATION_PAUSE', completed_pairs=done, seconds=clock.total()))
                    return
            print('\n求解完成，正在审计并生成统计图表。', flush=True)
            check_contract(c)
            for task in c['tasks']:
                name = key(task)
                p = Path(c['source'])/'inputs'/(name+'.json')
                meta = read(p)
                if sha(Path(c['source'])/meta['path']) != c['inputs'][name]['identity']['payload_sha']:
                    raise RuntimeError('Final input payload identity changed')
            write(out/'final_audit_report.json', dict(passed=True, pairs=done, solves=done*2,
                pair_hashes=pair_hashes, common_candidates=True, original_joint_reproduced=True,
                metric_replay=True, input_and_source_identity=True))
            from .paper_pair_report import generate
            generate(out)
            clock.flush()
            write(out/'status.json', dict(status='COMPLETED', completed_pairs=done, seconds=clock.total()))
            print('五候选完整配对完成：'+str(out), flush=True)
        except BaseException:
            clock.flush(keep=True)
            write(out/'status.json', dict(status='STOPPED', completed_pairs=done,
                seconds=clock.total(), error=traceback.format_exc()))
            raise


def release():
    pointer = read(RELEASE)
    p = safe(pointer['contract']).resolve(strict=True)
    if sha(p) != pointer['contract_sha']:
        raise RuntimeError('Release contract identity differs')
    ready = safe(pointer['readiness']).resolve(strict=True)
    if sha(ready) != pointer['readiness_sha'] or not read(ready)['ready']:
        raise RuntimeError('Preparation not released')
    c = read(p)
    check_contract(c)
    return c


def main(argv=None):
    parser = argparse.ArgumentParser(description='五候选AP交替/联合完整配对：一键运行与恢复')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--replot', type=Path)
    args = parser.parse_args(argv)
    if args.replot:
        from .paper_pair_report import generate
        generate(safe(args.replot).resolve(strict=True))
        print('仅重绘完成，没有调用定位器。', flush=True)
        return
    c = release()
    if args.check_only:
        print('放行检查通过：2400条×2版本；预计约1.5小时。未创建正式任务、未启动定位。', flush=True)
        return
    require_exclusive(); resource_check()
    if ACTIVE.exists():
        p = read(ACTIVE)
        out = safe(p['out']).resolve(strict=True)
        if sha(out/'contract.json') != p['contract_sha'] or read(out/'contract.json') != c:
            raise RuntimeError('Active pair contract differs')
    else:
        out = create(c, 'paper_pair_formal')
        write(ACTIVE, dict(out=str(out), contract_sha=sha(out/'contract.json')))
    print('结果目录：'+str(out), flush=True)
    execute(out)


if __name__ == '__main__':
    main()
