# ruff: noqa: F401
# Preserve imported globals for legacy function cloning and reexports.
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
        receipt = path.with_suffix('.sha.json')
        assert receipt.exists() and cfg.sha(path) == cfg.read(receipt)['sha'], 'Decision identity mismatch'
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
        if method == 'MP':
            value['result']['front_snapshot'] = engine.front_snapshot
            value['result']['geo_snapshot'] = engine.geo_snapshot
    assert cfg.digest(iq) == h
    cfg.write(path, value)  # Save decisions before evaluation opens truth.
    cfg.write(path.with_suffix('.sha.json'), dict(sha=cfg.sha(path)))
    return value


def run_case(out, phase, layout, repeat, condition, bank, engine, ordinal, generation_seconds):
    tag = f"{layout['kind']}_{layout['index']:02d}_R{repeat:02d}_{condition['name']}"
    dst = out/'cases'/tag
    dst.mkdir(parents=True, exist_ok=True)
    if (dst/'evaluation.json').exists():
        receipt = cfg.read(dst/'receipt.json')
        assert all(cfg.sha(dst/name) == h for name, h in receipt.items()), 'Saved case identity mismatch'
        iq, _ = cfg.observation(bank, condition)
        assert cfg.digest(iq) == cfg.read(dst/'evaluation.json')['iq_sha']
        for method in ('MP', 'FLOOD', 'XIA'):
            assert cfg.read(dst/(method+'_decision.json'))['iq_sha'] == cfg.digest(iq)
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
        snap = mp.get('front_snapshot')
        if snap:
            state = snap['los']
            mm['LOS'] = dict(p=state['p'], errors_m=evaluation(state['p'], truth), normal=state['stationary'],
                             status=state.get('reason', 'LOS_output'), seconds=snap['frontend_seconds'])
        geom = mp.get('geo_snapshot')
        if geom:
            mm['GEO'] = dict(p=geom['p'], errors_m=evaluation(geom['p'], truth), normal=True,
                             status='geometry_output', seconds=geom['seconds'])
    row = dict(tag=tag, layout=layout['index'], kind=layout['kind'], repeat=repeat,
               condition=condition['name'], methods=mm, iq_sha=cfg.digest(iq),
               generation_seconds=generation_seconds, observed_snr_db=bank['realized_los_snr_db'])
    cfg.write(dst/'evaluation.json', row)
    cfg.write(dst/'receipt.json', {p.name: cfg.sha(p) for p in dst.iterdir() if p.is_file() and p.name != 'receipt.json'})


# Historical main retained only in the evidence archive.


# Historical batch entry disabled; use the public mp_dpd API.
