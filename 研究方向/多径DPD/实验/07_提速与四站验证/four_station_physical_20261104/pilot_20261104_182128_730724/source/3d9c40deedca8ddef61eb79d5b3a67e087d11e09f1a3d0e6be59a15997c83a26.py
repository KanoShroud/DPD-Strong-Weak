"""Serial frozen paired runner. Formal mode requires a separately issued release."""
import argparse
from contextlib import contextmanager
from datetime import datetime
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
from threadpoolctl import threadpool_limits
import scene as cfg
import adapters as ad
# Frozen imports include another report/runner; resolve ours explicitly.
report = ad.load('physical_report', cfg.HERE/'report.py')


def atomic_npz(path, **arrays):
    tmp = path.with_name(path.name+'.pending.npz')
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


@contextmanager
def lock(out):
    path = out/'run.lock'
    if path.exists():
        pid = cfg.read(path)['pid']
        import psutil
        if psutil.pid_exists(pid):
            raise RuntimeError(f'Run already locked by PID {pid}')
        path.rename(out/f'stale_lock_{time.time_ns()}.json')
    with path.open('x', encoding='utf-8') as f:
        f.write('{"pid":'+str(os.getpid())+'}')
    try:
        yield
    finally:
        path.unlink()


def check_hashes(identity):
    for p, h in identity.items():
        if cfg.sha(p) != h:
            raise RuntimeError('Frozen source changed: '+p)


def evaluation(p, truth):
    if p is None:
        return None
    p = np.asarray(p)
    if p.shape != (2, 2) or not np.isfinite(p).all():
        return None
    a = np.linalg.norm(p-truth, axis=1)
    z = np.linalg.norm(p[::-1]-truth, axis=1)
    return (a if a@a <= z@z else z).tolist()


def saved_decision(dst, method, iq, engine, seed):
    path = dst/(method+'_decision.json')
    h = cfg.digest(iq)
    if path.exists():
        value = cfg.read(path)
        assert value['iq_sha'] == h
        return value
    start = time.perf_counter()
    try:
        result = engine.mp(iq) if method == 'MP' else engine.flood_solve(iq) if method == 'FLOOD' else ad.xia_solve(iq, seed)
        value = dict(method=method, status='returned', result=result, iq_sha=h)
    except (ValueError, np.linalg.LinAlgError) as exc:
        # Preserve genuine estimator failure. Programming assertions abort the batch.
        value = dict(method=method, status='failed', exception=repr(exc), traceback=traceback.format_exc(),
                     result=dict(p=None, normal=False, status='estimator_failure', seconds=time.perf_counter()-start), iq_sha=h)
    assert cfg.digest(iq) == h
    cfg.write(path, value)  # Save decisions before evaluation opens truth.
    return value


def run_case(out, phase, layout, repeat, condition, bank, engine, ordinal, generation_seconds):
    tag = f"{layout['kind']}_{layout['index']:02d}_R{repeat:02d}_{condition['name']}"
    dst = out/'cases'/tag
    dst.mkdir(parents=True, exist_ok=True)
    if (dst/'evaluation.json').exists():
        return
    iq, powers = cfg.observation(bank, condition)
    if (dst/'iq.npz').exists():
        with np.load(dst/'iq.npz', allow_pickle=False) as z:
            assert np.array_equal(z['iq'], iq)
    else:
        atomic_npz(dst/'iq.npz', iq=iq)
        cfg.write(dst/'input.json', dict(iq_sha=cfg.digest(iq), power_ratios=powers,
            bank_seed=bank['seed'], condition=condition, layout=layout['index'], repeat=repeat,
            physical_iq_unit='sqrt(W)', noise_variance=cfg.NOISE_VARIANCE))
    methods = ['MP', 'FLOOD', 'XIA']
    methods = methods[ordinal % 3:]+methods[:ordinal % 3]
    results = {}
    seed = int(np.random.SeedSequence([2026110426, 0 if phase == 'pilot' else 1, layout['index'], repeat]).generate_state(1)[0])
    for method in methods:
        print('START', tag, method, flush=True)
        results[method] = saved_decision(dst, method, iq, engine, seed)
        rr = results[method]['result']
        print('END', tag, method, round(rr['seconds'], 3), rr.get('stop', rr.get('status')), flush=True)
    truth = np.asarray(bank['p'])  # No truth is passed to any estimator above.
    mm = {}
    for method in ('MP', 'FLOOD', 'XIA'):
        r = results[method]['result']
        p = r['final']['p'] if method == 'MP' and results[method]['status'] == 'returned' else r.get('p')
        mm[method] = dict(p=p, errors_m=evaluation(p, truth), normal=r['normal'],
                          status=r.get('stop', r.get('status')), seconds=r['seconds'])
    mp = results['MP']['result']
    if results['MP']['status'] == 'returned':
        for m, key, seconds in [('GEO', 'geo', mp['geo']['seconds']), ('LOS', 'los', mp['frontend_seconds'])]:
            r = mp[key]
            mm[m] = dict(p=r['p'], errors_m=evaluation(r['p'], truth), normal=r.get('stationary', True),
                         status=r.get('reason', 'geometry_output'), seconds=seconds)
        mm['MP'].update(K=len(mp['paths']), PG=mp['final']['projected_gradient'])
    else:
        for m in ('GEO', 'LOS'):
            mm[m] = dict(p=None, errors_m=None, normal=False, status='pipeline_failure', seconds=mp['seconds'])
    row = dict(tag=tag, layout=layout['index'], kind=layout['kind'], repeat=repeat,
               condition=condition['name'], methods=mm, iq_sha=cfg.digest(iq),
               generation_seconds=generation_seconds, observed_snr_db=bank['realized_los_snr_db'])
    cfg.write(dst/'evaluation.json', row)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['pilot', 'formal'], required=True)
    ap.add_argument('--limit', type=int)
    args = ap.parse_args(argv)
    if not (cfg.HERE/'contract.json').exists():
        raise RuntimeError('Run correctness.py to create and verify the approved contract first')
    contract = cfg.read(cfg.HERE/'contract.json')
    qa = cfg.read(cfg.HERE/'correctness.json')
    assert qa['passed']
    check_hashes(qa['hashes'])
    if args.phase == 'formal':
        release = cfg.read(cfg.HERE/'release.json')
        if not release['formal_allowed']:
            raise RuntimeError('Formal release held: '+str(release['reasons']))
        check_hashes(release['hashes'])
    engine = ad.Engine()
    identity = ad.source_hashes()
    pointer = cfg.HERE/(args.phase+'_latest.json')
    if pointer.exists():
        out = Path(cfg.read(pointer)['path']).resolve(strict=True)
        old = cfg.read(out/'identity.json')
        assert identity == old['hashes'], 'Source set changed; refuse implicit resume'
    else:
        if shutil.disk_usage(cfg.HERE).free < (1 if args.phase == 'pilot' else 12)*1024**3:
            raise RuntimeError('Insufficient free space for evidence; no run started')
        out = cfg.HERE/(args.phase+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        out.mkdir(exist_ok=False)
        cfg.write(pointer, dict(path=str(out)))
        cfg.write(out/'identity.json', dict(hashes=identity, command=sys.argv, python=sys.version, threads=2))
        (out/'source').mkdir()
        for p, h in identity.items():
            shutil.copy2(p, out/'source'/(h+Path(p).suffix))
    print('结果目录：', out, flush=True)
    expected = 52 if args.phase == 'pilot' else 3120
    start = time.perf_counter(); count = 0
    with lock(out):
        try:
            for layout in contract[args.phase+'_layouts']:
                for rep in range(contract[args.phase+'_repeats']):
                    bankpath = out/'banks'/f"{layout['index']:02d}_{rep:02d}.npz"
                    t = time.perf_counter()
                    if bankpath.exists():
                        with np.load(bankpath, allow_pickle=False) as z:
                            bank = dict(z)
                        generation = cfg.read(bankpath.with_suffix('.json'))['generation_seconds']
                    else:
                        bank = cfg.make_bank(layout, args.phase, rep, ad.wave.synthesize, ad.wave.rrc)
                        generation = time.perf_counter()-t
                        bankpath.parent.mkdir(exist_ok=True)
                        atomic_npz(bankpath, **bank)
                        cfg.write(bankpath.with_suffix('.json'), dict(generation_seconds=generation, sha=cfg.sha(bankpath)))
                    assert cfg.sha(bankpath) == cfg.read(bankpath.with_suffix('.json'))['sha']
                    for cond in cfg.CONDITIONS:
                        if args.limit is not None and count >= args.limit:
                            break
                        run_case(out, args.phase, layout, rep, cond, bank, engine, count, generation/13)
                        count += 1
                        summary = report.summarize(out, args.phase)
                        elapsed = time.perf_counter()-start
                        print(f"完成 {summary['completed']}/{expected} | 本次累计 {elapsed/60:.2f} min", flush=True)
                    if args.limit is not None and count >= args.limit:
                        break
                if args.limit is not None and count >= args.limit:
                    break
            check_hashes(identity)
            summary = report.summarize(out, args.phase, finished=count == expected)
            cfg.write(out/'postflight.json', dict(unchanged=True, completed=summary['completed']))
        except Exception:
            cfg.write(out/f'failure_{time.time_ns()}.json', dict(traceback=traceback.format_exc(), count=count))
            raise
    print('结束。报告：', out/'结果报告.html', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
