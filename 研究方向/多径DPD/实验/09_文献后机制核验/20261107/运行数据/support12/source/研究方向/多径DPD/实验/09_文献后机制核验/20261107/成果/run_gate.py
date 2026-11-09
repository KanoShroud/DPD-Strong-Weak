"""Bounded post-literature recheck; no estimator or scenario changes."""
import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists())
sys.path.insert(0, str(ROOT))
from 研究方向.多径DPD.代码.mp_dpd import api  # noqa: E402
from 研究方向.多径DPD.代码.mp_dpd.baselines import xia_decoder as xd  # noqa: E402
from 研究方向.多径DPD.代码.mp_dpd.baselines.xia_kernel import CrossSpectrumKernel  # noqa: E402
from 研究方向.多径DPD.代码.mp_dpd.models import scene as cfg  # noqa: E402

BASE = ROOT/'研究方向/多径DPD'
OLD_X = BASE/'实验/06_近邻复现与比较/xia_pairing/reconstructed/pilot_20261027_170421_805496'
OLD_C = BASE/'实验/07_提速与四站验证/inside_pilot_20261104/batch_v2'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def copy_checked(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    before = sha(src)
    with open(src, 'rb', buffering=0) as a, open(dst, 'xb', buffering=0) as b:
        shutil.copyfileobj(a, b, 1024*1024)
    assert before == sha(dst) == sha(src)
    return before


def errors(p, truth):
    if p is None:
        return None
    p = np.asarray(p)
    assert p.shape == (2, 2) and np.isfinite(p).all()
    a = np.linalg.norm(p-truth, axis=1)
    b = np.linalg.norm(p[::-1]-truth, axis=1)
    return a if a@a <= b@b else b


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    files = list((BASE/'代码/mp_dpd').rglob('*.py')) + [Path(__file__), Path(__file__).with_name('执行合同.md')]
    identity = {str(p): sha(p) for p in files}
    for p in files:
        copy_checked(p, out/'source'/p.relative_to(ROOT))
    cfg.write(out/'preflight.json', dict(source_hashes=identity, phase='identity_only', threads=2,
        argv=sys.argv, input_scope='saved_development_not_independent_validation'))
    return identity


def anchor(out, limit):
    rows = []
    # Exact same 12 existing realizations. No new seed/configuration search.
    for i, g in [(i, g) for i in range(12) for g in (1, 3)][:limit]:
        tag = f'r{i:02d}_snr5_sir3_G{g}'
        old = read(OLD_X/(tag+'.json'))
        src = OLD_X/f'input_{i:02d}.npz'
        assert sha(src) == old['input_sha']
        local = out/'inputs'/src.name
        if not local.exists():
            copy_checked(src, local)
        with np.load(local, allow_pickle=False) as z:
            y = (z['component'][:, :, :g].sum(2)*np.array([1., 10**(-3/20)])[:, None, None]).sum(0)
            y = y + np.sqrt(10**(-5/10))*z['noise_f']
            assert cfg.digest(y) == old['observation_sha']
            stations, f = z['stations'], z['frequency']
            # Full original Fourier sum remains used by the anchor solver.
            result = xd.solve(y, f, stations, 2, 2000, 3000, seed=old['particle_seed'])
            cfg.write(out/(tag+'_decision.json'), result)
            points = np.asarray(result['steps'][0]['points'])[::83]
            direct = xd.scores(points, stations, y, f, 2000)
            kernel = CrossSpectrumKernel(y, f, 2000, 1e6)
            fast = kernel.scores(points, stations)
            discrepancy = float(np.max(abs(fast-direct))/max(np.max(abs(direct)), np.finfo(float).tiny))
            assert discrepancy < 5e-11
            # Evaluation only after decision is durably written.
            err = errors(result['positions'], z['truth'])
            first = result['steps'][-1]['output'].get('first_position')
            if result['positions'] is not None:
                first = result['positions'][0]
            first_error = None if first is None else float(np.linalg.norm(np.asarray(first)-z['truth'][0]))
        old_terminal = old['output']['positions']
        same = (old_terminal is None and result['positions'] is None) or (
            old_terminal is not None and result['positions'] is not None and np.allclose(old_terminal, result['positions'], rtol=0, atol=1e-9))
        assert same and old['output']['status'] == result['status']
        rows.append(dict(tag=tag, G_total=g, errors_m=err, first_peak_target_error_m=first_error,
            status=result['status'], seconds=result['seconds'], kernel_error=discrepancy,
            old_terminal_reproduced=bool(same), input_sha=sha(local)))
        cfg.write(out/'rows.json', rows)
        print('ANCHOR', len(rows), tag, result['status'], flush=True)
    return rows


def common(out, limit):
    rows = []
    jobs = [(i, c) for i in range(100, 106) for c in ('L0', 'L1_P075', 'L2_P075')][:limit]
    for ordinal, (i, condition) in enumerate(jobs):
        tag = f'IN_{i}_R00_{condition}'
        old_dir = OLD_C/'cases'/tag
        meta = read(old_dir/'input.json')
        local = out/'inputs'/tag/'iq.npz'
        copy_checked(old_dir/'iq.npz', local)
        with np.load(local, allow_pickle=False) as z:
            iq = z['iq']
        assert cfg.digest(iq) == meta['iq_sha'] and iq.shape == (4, 8192)
        seed = int(np.random.SeedSequence([2026110426, 0, i, 0]).generate_state(1)[0])
        engine = api.create_engine()
        results = {}
        order = ['MP', 'FLOOD_LP_K2', 'XIA_ADAPTED']
        order = order[ordinal % 3:]+order[:ordinal % 3]
        for method in order:
            print('START', tag, method, flush=True)
            start = time.perf_counter()
            try:
                if method == 'MP':
                    result = engine.mp(iq)
                    p = result['final']['p']
                    normal = result['normal']
                elif method == 'FLOOD_LP_K2':
                    result = engine.flood_solve(iq)
                    p, normal = result['p'], result['normal']
                else:
                    result = api.runtime().inside.xia_solve(iq, seed)
                    p, normal = result['p'], result['normal']
                status = result.get('stop', result.get('status'))
            except (ValueError, np.linalg.LinAlgError) as exc:
                result, p, normal, status = dict(exception=repr(exc)), None, False, 'estimator_failure'
            seconds = time.perf_counter()-start
            assert cfg.digest(iq) == meta['iq_sha']
            cfg.write(out/f'{tag}_{method}_decision.json', result)
            results[method] = dict(p=p, normal=normal, status=status, seconds=seconds)
            if method == 'MP' and p is not None:
                for name, key, duration in [('GEO', 'geo', result['geo']['seconds']), ('LOS', 'los', result['frontend_seconds'])]:
                    results[name] = dict(p=result[key]['p'], normal=result[key].get('stationary', True),
                        status=result[key].get('reason', 'output'), seconds=duration)
                results[method]['paths'] = len(result['paths'])
                results[method]['projected_gradient'] = result['final'].get('projected_gradient')
            print('END', tag, method, round(seconds, 3), status, flush=True)
        # Truth remains outside each estimator API; read only after all decisions.
        bank_file = OLD_C/'banks'/f'{i}_00.npz'
        with np.load(bank_file, allow_pickle=False) as z:
            truth = z['p']
        for value in results.values():
            value['errors_m'] = errors(value['p'], truth)
        rows.append(dict(tag=tag, condition=condition, input_sha=sha(local), methods=results,
            truth_evaluation_file=str(bank_file), truth_file_sha=sha(bank_file)))
        cfg.write(out/'rows.json', rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['anchor', 'common'], required=True)
    parser.add_argument('--limit', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert 0 < args.limit <= (24 if args.stage == 'anchor' else 18)
    out = args.out.resolve()
    assert out.is_relative_to(BASE/'实验/09_文献后机制核验/20261107/运行数据')
    identity = prepare(out)
    start = time.perf_counter()
    with threadpool_limits(limits=2):
        rows = anchor(out, args.limit) if args.stage == 'anchor' else common(out, args.limit)
    assert all(sha(p) == h for p, h in identity.items())
    cfg.write(out/'completion.json', dict(stage=args.stage, count=len(rows), elapsed_seconds=time.perf_counter()-start,
        source_unchanged=True, complete=True, scope='development_diagnostic_not_paper_reproduction'))


if __name__ == '__main__':
    main()
