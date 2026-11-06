"""Xia native random-particle workflow on frozen current IQ; no shared frontend."""
from datetime import datetime
import importlib.util
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
CAL = BASE/'risk_calibration_20261102'
FORMAL = CAL/'formal_20261102_181736_622970'
sys.path.insert(0, str(CAL))
spec = importlib.util.spec_from_file_location('native_xia_frozen_cal', CAL/'runner.py')
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)
sys.path.insert(0, str(BASE.parent/'xia_pairing/recheck'))
import xia_checked as xia  # noqa: E402
from xia_reference import scalar_score  # noqa: E402
from geometry_association_gate import sha, STATIONS, B  # noqa: E402
from robust_geometry import evaluate  # noqa: E402
from storage import load, save, write, lock  # noqa: E402


def retained(result):
    full = [s['output']['positions'] for s in result['steps']
            if s['output'].get('positions') is not None]
    return full[-1] if full else None


def tests():
    rng = np.random.default_rng(971)
    y = rng.normal(size=(5, 29))+1j*rng.normal(size=(5, 29))
    f = np.linspace(-B/2, B/2, 29)
    p = np.array([[105., 230.], [-420., 178.]])
    v = xia.scores(p, STATIONS, y, f, 8192)
    permutation = [0, 3, 2, 4, 1]
    checks = {
        'independent_scalar': np.allclose(v, [scalar_score(q, STATIONS, y, f, 8192) for q in p], rtol=1e-10, atol=0),
        'translation': np.allclose(v, xia.scores(p+123, STATIONS+123, y, f, 8192)),
        'nonreference_station_permutation': np.allclose(v, xia.scores(p, STATIONS[permutation], y[permutation], f, 8192)),
        'common_complex_scale': np.allclose(v, xia.scores(p, STATIONS, y*(2+3j), f, 8192)),
        'fifth_bin_exact_endpoint': xia.threshold(np.r_[np.linspace(0, 1, 11), np.full(100, .45)])['threshold'] == 1.,
        'retained_output_is_previous_not_truth': retained({'steps':[{'output':{'positions':[[1,2],[3,4]]}}, {'output':{'positions':None}}]}) == [[1,2],[3,4]],
        'none_stays_none': retained({'steps':[{'output':{'positions':None}}]}) is None,
    }
    # Small formula-only native-workflow check; no geometry candidates supplied.
    seed = 718
    native = xia.solve(y, f, STATIONS, 2, 8192, 3000, seed=seed, particles=20)
    checks['native_uniform_first_pool'] = np.array_equal(native['steps'][0]['points'], np.random.default_rng(seed).uniform(-3000,3000,(20,2)))
    checks['native_no_more_than_three_rounds'] = 1 <= len(native['steps']) <= 3
    checks['native_finite_scores'] = all(np.isfinite(s['scores']).all() for s in native['steps'])
    checks = {k: bool(v) for k,v in checks.items()}
    assert all(checks.values()), checks
    return checks


def summarize(out, rows):
    metrics = {}
    for cond in cal.CONDITIONS:
        group = [r for r in rows if r['case']['condition'] == cond]
        if not group:
            continue
        metrics[cond] = {}
        for method in ('Xia_native_terminal', 'Xia_native_lastcomplete', 'GEO', 'LOS', 'WIDE'):
            all_eval = [r['evaluation'][method] for r in group]
            full = [r for r in all_eval if r['complete']]
            err = np.asarray([r['errors_m'] for r in full])
            rmse = float(np.sqrt(np.mean(err**2))) if len(full) else None
            metrics[cond][method] = dict(n=len(group), complete=len(full),
                success20=sum(max(r['errors_m']) <= 20 for r in full),
                conditional_rmse_m=rmse, full_rmse_m=rmse if len(full)==len(group) else None)
    write(out/'summary.json', dict(completed=len(rows), complete=len(rows)==32, metrics=metrics,
          total_seconds=sum(r['seconds'] for r in rows), new_independent_observations=False))


def main():
    checks = tests()
    contract = load(FORMAL/'contract.json')['contract']
    cal.verify(contract)
    frozen = BASE.parent/'xia_pairing/reconstructed/pilot_20261027_170421_805496'
    assert sha(Path(xia.__file__)) == sha(frozen/'xia_checked.py')
    import xia_reference
    assert sha(Path(xia_reference.__file__)) == sha(frozen/'xia_reference.py')
    files = [Path(__file__), HERE/'执行协议.md', FORMAL/'contract.json', Path(xia.__file__), Path(xia_reference.__file__)]
    files += sorted((FORMAL/'cases').glob('*/input.npz'))+sorted((FORMAL/'cases').glob('*/input.json'))+sorted((FORMAL/'cases').glob('*/solve.json'))
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name and Path(name).suffix == '.py' and Path(name).resolve().is_relative_to(BASE):
            files.append(Path(name))
    hashes = {str(p.resolve(strict=True)): sha(p) for p in files}
    out = HERE/('native_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    with lock(out):
        write(HERE/'latest_native.json', dict(path=str(out)))
        write(out/'preflight.json', dict(checks=checks, hashes=hashes, contract=contract,
            python=sys.version, numpy=np.__version__, threads=2, native_paper_frontend=True,
            seed_rule='20261029223+10000*20+repeat', common_geometry_candidates=False,
            truth_in_estimator=False, shared_growth=False, paper_parameter_replication=False))
        (out/'source').mkdir()
        for name, digest in hashes.items():
            if Path(name).suffix == '.py':
                (out/'source'/(digest+'.py')).write_bytes(Path(name).read_bytes())
        rows = []
        try:
            for i, case in enumerate(cal.cases()):
                start = time.perf_counter()
                folder = FORMAL/'cases'/case['id']
                dst = out/case['id']
                dst.mkdir(exist_ok=False)
                meta = load(folder/'input.json')
                with np.load(folder/'input.npz', allow_pickle=False) as z:
                    assert set(z.files)=={'iq'}
                    iq = z['iq']
                original = cal.digest(iq)
                assert original == meta['iq_sha'] and meta['case']==case
                f, y, n, _ = cal.spectral(iq)
                seed = 20261029223+10000*20+case['repeat']
                ans = xia.solve(y.T, f, STATIONS, 2, n, 3000, seed=seed,
                                particles=1000, delta_h=400., std=10.)
                assert np.array_equal(ans['steps'][0]['points'],np.random.default_rng(seed).uniform(-3000,3000,(1000,2)))
                assert all(np.isfinite(s['scores']).all() for s in ans['steps'])
                assert cal.digest(iq)==original
                last = retained(ans)
                save(dst/'decision_before_truth.json', dict(case=case, iq_sha=original,
                    seed=seed, output=ans, retained=last, seconds=time.perf_counter()-start))
                # True positions are opened only after estimator output is saved.
                with np.load(contract['conditions'][case['condition']]['bank'], allow_pickle=False) as z:
                    truth = z['p']
                old = load(folder/'solve.json')
                pred = dict(Xia_native_terminal=ans['positions'], Xia_native_lastcomplete=last,
                            GEO=old['initial'], LOS=old['los']['p'], WIDE=old['final']['p'])
                evaluation = {k:evaluate(np.asarray(v).reshape(-1,2) if v is not None else np.empty((0,2)), truth) for k,v in pred.items()}
                points0 = np.asarray(ans['steps'][0]['points'])
                coverage = np.linalg.norm(points0[:,None]-truth[None],axis=-1).min(axis=0)
                row = dict(case=case, iq_sha=original, evaluation=evaluation,
                           initial_uniform_coverage_m=coverage,
                           terminal_status=ans['status'], rounds=len(ans['steps']),
                           seconds=time.perf_counter()-start,
                           native_seconds=ans['seconds'])
                save(dst/'evaluation.json', row)
                rows.append(row)
                summarize(out, rows)
                print(case['id'],ans['status'],'seconds',round(row['seconds'],3),flush=True)
                if i==3:
                    estimate=sum(r['seconds'] for r in rows)*8
                    write(out/'short_gate.json',dict(correct=True, estimate_seconds=estimate,
                          continue_under_one_hour=estimate<3600, performance_not_correctness_gate=True))
                    if estimate>=3600:
                        return
            assert all(sha(Path(p))==h for p,h in hashes.items())
            cal.verify(contract)
            write(out/'postflight.json',dict(hashes_unchanged=True, completed=len(rows),passed=True))
        except Exception:
            write(out/'failure.json',dict(traceback=traceback.format_exc(),completed=len(rows)))
            raise
    print('COMPLETE',out,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
