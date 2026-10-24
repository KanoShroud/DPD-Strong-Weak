"""G11-P6 finite paired development test, with per-method saved recovery.

No formal cohort or automatic parameter search. Long-length probe is separate.
"""
import argparse
import copy
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from 统一模型代码.gates.g11.model_integration.runner import sha, write, read, layouts
from 统一模型代码.gates.g11.model_integration.models import FS, BAND, build
from 统一模型代码.gates.g11.model_integration.solvers import CONFIG, run
from 统一模型代码.gates.g11.model_integration.hybrid import (
    METHODS, array_hash, ap_prefix, refine_candidates, vp_prefix, vp_refine, checks,
)
from 统一模型代码.gates.g10.r1.scenes import make
from 统一模型代码.gates.g10.physics import grid
from 统一模型代码.gates.g10.r4.paper_checks.assignment_metrics import evaluate, self_test

BASE = ROOT/'outputs_e2e/unified/e2e_g11/p6_hybrid'
P5 = ROOT/'outputs_e2e/unified/e2e_g11/p5_model_integration'
REVISION = 'G11_P6_FIXED_HANN_DISCOVERY_FINITE_REFINEMENT_V1'


def tasks(mode, lengths):
    if mode == 'pilot':
        rows = [dict(group=g, realization=28001, gap=gap, n=1024, role='cohort')
                for g in range(4) for gap in (0, 15)]
        rows.append(dict(group=0, realization=28000, gap=15, n=1024, role='native_reference'))
        return rows
    if mode == 'length':
        return [dict(group=0, realization=28000, gap=15, n=n, role='length_cost_probe') for n in lengths]
    return [dict(group=0, realization=28000, gap=15, n=1024, role='engineering_test')]


def key(task):
    return f"{task['group']}_{task['realization']}_{task['gap']}_{task['n']}"


def create(mode, lengths):
    out = BASE/(mode+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    assert not out.exists() and out.resolve().is_relative_to(ROOT/'outputs_e2e')
    out.mkdir(parents=True)
    (out/'inputs').mkdir()
    (out/'rows').mkdir()
    (out/'prefixes').mkdir()
    manifest = {}
    rows = tasks(mode, lengths)
    coords = layouts()
    source_paths = sorted((ROOT/'统一模型代码/gates').rglob('*.py'))
    sources = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    for p in source_paths:
        dst = out/'source_snapshot'/p.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(p.read_bytes())
    # Existing native reference is selected by recorded identity, not its outcome.
    old = sorted(P5.glob('pilot_*/cases/1024_finite_fourier_AP5.json'))
    if not old:
        raise FileNotFoundError('Required P5 native reference is missing')
    source_case = old[-1]
    source_root = source_case.parent.parent
    old_contract = read(source_root/'contract.json')
    for name in ('models.py', 'solvers.py'):
        relative = str((Path('统一模型代码/gates/g11/model_integration')/name))
        assert old_contract['sources'][relative] == sha(ROOT/relative)
    raw = source_root/'input.npz'
    assert sha(raw) == read(source_root/'input_identity.json')['sha256']
    for g, rep in sorted({(r['group'], r['realization']) for r in rows}):
        p = out/'inputs'/f'{g}_{rep}.npz'
        if rep == 28000:
            p.write_bytes(raw.read_bytes())
        else:
            scene = make(g, rep, gap=0., n=8192, fs=FS, snr=6, positions=coords[g], seed=20261024053)
            np.savez_compressed(p, signal=scene['signal'], noise=scene['noise'], positions=scene['positions'])
        with np.load(p, allow_pickle=False) as z:
            np.testing.assert_array_equal(z['positions'], coords[g])
        manifest[p.name] = sha(p)
    contract = dict(revision=REVISION, mode=mode, tasks=rows, sources=sources, inputs=manifest,
        fs=FS, band=BAND, generator_n=8192, snr_inband_db=6, waveform_seed=20261024053,
        layouts=coords, config=CONFIG, python=sys.version, platform=platform.platform(), threads=2,
        methods=list(METHODS), window_order='Hann for discovery; finite final fit uses original retained unwindowed DFT',
        AP='same five raw Hann conditional pairs, all branches refined under own objective',
        VP='unchanged native Hann PROFILE prefix, then one joint-refinement call',
        scope='bounded development; no performance-based tuning or case removal; no formal release',
        statistics='descriptive paired RMSE on 4 fixed geometries and 1 realization each; gaps correlated',
        native_reference=dict(case=str(source_case), case_sha=sha(source_case), raw=str(raw), raw_sha=sha(raw)),
        timing='cold per-representation preparation plus shared prefix fully charged to every derivative algorithm; offline metrics excluded')
    write(out/'contract.json', contract)
    write(out/'ledger.json', dict(attempts=[], completed_jobs=0))
    return out, contract


def verify(out, contract):
    assert out.resolve().is_relative_to(BASE)
    for name, identity in contract['sources'].items():
        if sha(ROOT/name) != identity:
            raise RuntimeError('Source changed: '+name)
    for name, identity in contract['inputs'].items():
        if sha(out/'inputs'/name) != identity:
            raise RuntimeError('Input changed: '+name)


def get_input(out, task):
    with np.load(out/'inputs'/f"{task['group']}_{task['realization']}.npz", allow_pickle=False) as z:
        x = (z['signal'][0]*10**(task['gap']/20)+z['signal'][1]+z['noise'])[:, :task['n']]
        return x, z['positions'].copy()


def seal(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save_row(path, row):
    write(path, dict(payload=row, sha256=seal(row)))


def read_row(path):
    row = read(path)
    if seal(row['payload']) != row['sha256']:
        raise RuntimeError('Saved result integrity mismatch: '+str(path))
    return row['payload']


def execute(out, contract):
    ledger = read(out/'ledger.json')
    for a in ledger['attempts']:
        if a['status'] == 'RUNNING':
            a['status'] = 'INTERRUPTED_UNKNOWN_TAIL'
    attempt = dict(start_utc=datetime.now(timezone.utc).isoformat(), seconds_observed=0., status='RUNNING')
    ledger['attempts'].append(attempt)
    start = time.perf_counter()

    def checkpoint():
        attempt['seconds_observed'] = time.perf_counter()-start
        ledger['completed_jobs'] = len(list((out/'rows').glob('*.json')))
        write(out/'ledger.json', ledger)

    checkpoint()
    try:
        with threadpool_limits(limits=2):
            warm = time.perf_counter()
            dummy = np.random.default_rng(20261024060).normal(size=(4, 128)).astype(complex)
            for kind in ('hann', 'periodic', 'finite_fourier', 'slepian'):
                build(dummy, kind).fit([[170., 120.], [-130., -210.]], gradient=True)
            write(out/f'warmup_{len(ledger["attempts"])}.json', dict(seconds=time.perf_counter()-warm))
            if contract['mode'] == 'tests':
                iq, _ = get_input(out, contract['tasks'][0])
                result = checks(iq)
                result['metrics'] = self_test()
                path = out/'roundtrip_中文.json'
                sample = dict(text='中文路径、结果保存与恢复检查', input_sha=array_hash(iq))
                save_row(path, sample)
                assert read_row(path) == sample
                result['json_resume_integrity'] = True
                write(out/'checks.json', result)
            else:
                total = len(contract['tasks'])*len(METHODS)+(1 if contract['mode'] == 'pilot' else 0)
                for index, task in enumerate(contract['tasks']):
                    iq, truth = get_input(out, task)
                    common_path = out/'prefixes'/(key(task)+'_ap.json')
                    vp_path = out/'prefixes'/(key(task)+'_vp.json')
                    prefix_model = None
                    if common_path.exists():
                        prefix = read_row(common_path)
                    else:
                        prefix, prefix_model = ap_prefix(iq)
                        save_row(common_path, prefix)
                    assert prefix['input_sha'] == array_hash(iq)
                    checkpoint()
                    shift = index % len(METHODS)
                    ordered = METHODS[shift:]+METHODS[:shift]
                    for method in ordered:
                        path = out/'rows'/(key(task)+'_'+method+'.json')
                        if path.exists():
                            old = read_row(path)
                            assert old['input_sha'] == array_hash(iq) and old['task'] == task and old['method'] == method
                            assert old['metrics'] == evaluate(truth, old['result']['positions'])
                            continue
                        print(f"\rP6 {ledger['completed_jobs']}/{total} | {(time.perf_counter()-start)/60:.1f} min | {key(task)} {method}   ", end='', flush=True)
                        if method.startswith('AP_'):
                            result = refine_candidates(iq, method[3:], prefix, prefix_model)
                        else:
                            if not vp_path.exists():
                                save_row(vp_path, vp_prefix(iq))
                            vp_saved = read_row(vp_path)
                            if method == 'VP_hann':
                                result = copy.deepcopy(vp_saved)
                                result.update(method=method, kind='hann', prefix_seconds=result['seconds'], refinement_seconds=0.)
                            else:
                                result = vp_refine(iq, method[3:], vp_saved)
                        row = dict(revision=REVISION, task=task, method=method, input_sha=array_hash(iq),
                            result=result, truth=truth.tolist(), metrics=evaluate(truth, result['positions']))
                        save_row(path, row)
                        checkpoint()
                    if task['role'] == 'native_reference':
                        path = out/'rows'/(key(task)+'_AP_NATIVE_FE.json')
                        if not path.exists():
                            print('\n固定参照子集：重跑同观测原生有限Fourier AP（预计数分钟）', flush=True)
                            result = run(build(iq, 'finite_fourier'), 'AP5', grid(20.))
                            save_row(path, dict(revision=REVISION, task=task, method='AP_NATIVE_FE', input_sha=array_hash(iq),
                                result=result, truth=truth.tolist(), metrics=evaluate(truth, result['positions'])))
                            checkpoint()
        verify(out, contract)
        attempt['status'] = 'COMPLETED'
        checkpoint()
        write(out/'audit.json', dict(completed=True, source_identity=True, input_identity=True,
            rows=ledger['completed_jobs'], formal_release=False, task_count=len(contract['tasks'])))
        print('\n阶段完成：', out, flush=True)
    except BaseException as exc:
        attempt['status'] = 'FAILED_OR_INTERRUPTED'
        checkpoint()
        write(out/f'failure_{len(ledger["attempts"])}.json', dict(error=repr(exc), traceback=traceback.format_exc()))
        raise


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('tests', 'pilot', 'length'))
    p.add_argument('--lengths', type=int, nargs='+', default=[4096])
    p.add_argument('--resume', type=Path)
    args = p.parse_args(argv)
    if args.resume:
        out = args.resume.resolve(strict=True)
        contract = read(out/'contract.json')
        assert contract['mode'] == args.mode
        if args.mode == 'length':
            assert [t['n'] for t in contract['tasks']] == args.lengths
    else:
        out, contract = create(args.mode, args.lengths)
    verify(out, contract)
    print('结果目录：', out, flush=True)
    execute(out, contract)


if __name__ == '__main__':
    main()
