"""Frozen Flood/Xia adapters on the existing 32 IQ records, no new backend."""
from datetime import datetime
import importlib.util
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import scipy
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
CAL = BASE / 'risk_calibration_20261102'
FORMAL = CAL / 'formal_20261102_181736_622970'
sys.path.insert(0, str(CAL))
spec = importlib.util.spec_from_file_location('paired_frozen_calibration', CAL/'runner.py')
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)
sys.path.insert(0, str(BASE))
import flood_band_pair as flood  # noqa: E402
import xia_common_candidates as xia  # noqa: E402
from geometry_association_gate import sha, signature, PAIRS, STATIONS, C, B  # noqa: E402
from robust_geometry import predicted, triple_candidates, evaluate  # noqa: E402
from xia_reference import scalar_score  # noqa: E402
from storage import load, save, write, lock  # noqa: E402


def independent_fixed_lp(c):
    n, j = c.shape
    width = n*j+n+j
    eq = np.zeros((n+1, width))
    ub = np.zeros((j+n*j, width))
    for i in range(n):
        eq[i, i*j:(i+1)*j] = 1
        eq[i, n*j+i] = 1
    eq[n, n*j+n:] = 1
    for k in range(j):
        ub[k, k:n*j:j] = 1
    for i in range(n):
        for k in range(j):
            ub[j+i*j+k, i*j+k] = 1
            ub[j+i*j+k, n*j+n+k] = -1
    ans = linprog(np.r_[c.ravel(), np.ones(n+j)], A_eq=eq,
                  b_eq=np.r_[np.ones(n), 2], A_ub=ub,
                  b_ub=np.r_[np.full(j, 10), np.zeros(n*j)], bounds=(0, 1),
                  method='highs')
    assert ans.success
    return float(ans.fun)


def selftest():
    checks = {}
    for seed in range(5):
        cost = np.random.default_rng(seed).uniform(0, 3, (8, 6))
        _, _, _, meta = flood.pruned_lp(cost, known_k=2)
        checks[f'fixed_k_dense_pruned_{seed}'] = abs(meta['objective']-independent_fixed_lp(cost)) < 1e-8
    p = np.array([[100., 200.]])
    lags = predicted(p)[0].T
    roots = triple_candidates(lags)
    checks['triple_roots_sign_units'] = float(np.linalg.norm(roots-p, axis=1).min()) < 1e-5
    rng = np.random.default_rng(12)
    y = rng.normal(size=(5, 31))+1j*rng.normal(size=(5, 31))
    f = np.linspace(-B/2, B/2, 31)
    a = scalar_score(p[0], STATIONS, y, f, 8192)
    b = xia.xia.scores(p, STATIONS, y, f, 8192)[0]
    checks['xia_original_scalar_vector'] = bool(np.isclose(a, b, rtol=1e-10, atol=0))
    checks['source_order_stations'] = np.array_equal(cal.ref.STATIONS, STATIONS)
    checks['source_order_pairs'] = cal.engine.PAIRS == PAIRS
    checks['band_sample_constants'] = cal.ref.FS == 50e6 and cal.ref.B == B and cal.engine.N == 8192
    assert all(checks.values()), checks
    return checks


def digest_paths(paths):
    return {str(p.resolve(strict=True)): sha(p) for p in paths}


def verify(hashes):
    for p, h in hashes.items():
        assert sha(Path(p)) == h, p


def run_case(out, case):
    folder = FORMAL/'cases'/case['id']
    dst = out/case['id']
    dst.mkdir(exist_ok=False)
    before = time.perf_counter()
    info = load(folder/'input.json')
    with np.load(folder/'input.npz', allow_pickle=False) as z:
        assert set(z.files) == {'iq'}
        iq = z['iq']
    assert cal.digest(iq) == info['iq_sha'] and info['case'] == case
    original = cal.digest(iq)
    started = time.perf_counter()
    extracted, _, _ = cal.engine.extract(iq)
    peaks = extracted['CC_ABS']
    lags = np.array([sorted(np.asarray(peaks[f'{m}_{n}']['lag_s'][:6])*C) for m, n in PAIRS])
    assert lags.shape == (10, 6) and np.isfinite(lags).all()
    pool = triple_candidates(lags)
    assert len(pool) >= 2 and np.isfinite(pool).all()
    front_seconds = time.perf_counter()-started
    started = time.perf_counter()
    f0 = np.fft.fftfreq(8192, 1/50e6)
    mask = abs(f0) <= B/2
    y = np.fft.fft(iq, axis=1, norm='ortho')[:, mask]
    f, yc, n, _ = cal.spectral(iq)
    assert np.array_equal(f, f0[mask]) and np.array_equal(y.T, yc) and n == 8192
    fft_seconds = time.perf_counter()-started
    before_inputs = signature(pool, lags, y, f)
    fp, fm = flood.estimate(pool, lags, known_k=2)
    seed = 20261029223 + 10000*20 + case['repeat']
    xr = xia.solve(pool, y, f, seed)
    assert signature(pool, lags, y, f) == before_inputs
    assert cal.digest(iq) == original
    assert all(np.isfinite(step['scores']).all() for step in xr['steps'])
    scalar = scalar_score(pool[0], STATIONS, y[:, :31], f[:31], n)
    vector = xia.xia.scores(pool[:1], STATIONS, y[:, :31], f[:31], n)[0]
    assert np.isclose(scalar, vector, rtol=1e-10, atol=0)
    np.savez_compressed(dst/'observed_frontend.npz', pool=pool, lags=lags, f=f,
                        **{k: fm[k] for k in ('cost', 'M', 'q', 'trash', 'association', 'association_trash', 'code_ids', 'paper_ids')})
    save(dst/'decisions_before_truth.json', dict(case=case, iq_sha=original, pool_count=len(pool),
         Flood=fp, Xia=xr, flood_metadata={k: v for k, v in fm.items() if k not in
         ('cost', 'M', 'q', 'trash', 'association', 'association_trash')}, peaks=peaks,
         seeds=dict(xia=seed), seconds=dict(common_frontend=front_seconds, fft=fft_seconds,
         Flood_complete=front_seconds+fm['seconds'], Xia_complete=front_seconds+fft_seconds+xr['seconds']),
         checks=dict(input_unchanged=True, fft_matches_frozen=True, xia_scalar=True)))
    # Estimator calls have finished and decisions are atomically saved above.
    contract = load(FORMAL/'contract.json')['contract']
    with np.load(contract['conditions'][case['condition']]['bank'], allow_pickle=False) as z:
        truth = z['p']
    solved = load(folder/'solve.json')
    points = dict(Flood=fp, Xia_terminal=xr['terminal'], Xia_lastcomplete=xr['retained'],
                  GEO=solved['initial'], LOS=solved['los']['p'], WIDE=solved['final']['p'])
    evals = {k: evaluate(np.asarray(p).reshape(-1, 2) if p is not None else np.empty((0, 2)), truth)
             for k, p in points.items()}
    from scipy.spatial.distance import cdist
    output = dict(case=case, iq_sha=original, evaluation=evals,
                  candidate_coverage_m=cdist(pool, truth).min(axis=0),
                  seconds=load(dst/'decisions_before_truth.json')['seconds'],
                  total_seconds=time.perf_counter()-before,
                  xia_terminal_status=xr['steps'][-1]['decoded']['status'],
                  original_wide_stop=solved['stop'],
                  original_reflection_count=len(solved['paths']))
    save(dst/'evaluation.json', output)
    print(case['id'], 'Flood', evals['Flood'], 'Xia', output['xia_terminal_status'],
          'seconds', round(output['total_seconds'], 3), flush=True)
    return output


def summarize(out, rows):
    metrics = {}
    for condition in cal.CONDITIONS:
        group = [r for r in rows if r['case']['condition'] == condition]
        if not group:
            continue
        metrics[condition] = {}
        for method in ('GEO', 'LOS', 'WIDE', 'Flood', 'Xia_terminal', 'Xia_lastcomplete'):
            data = [r['evaluation'][method] for r in group]
            full = [d for d in data if d['complete']]
            err = np.asarray([d['errors_m'] for d in full])
            metrics[condition][method] = dict(n=len(group), complete=len(full),
                success20=sum(max(d['errors_m']) <= 20 for d in full),
                conditional_rmse_m=float(np.sqrt(np.mean(err**2))) if len(full) else None,
                full_rmse_m=float(np.sqrt(np.mean(err**2))) if len(full) == len(group) else None)
    write(out/'summary.json', dict(completed=len(rows), complete=len(rows) == 32, metrics=metrics,
          total_seconds=sum(r['total_seconds'] for r in rows), independent_new_data=False))
    lines = ['# 当前32条：冻结近邻适配配对', '',
      '同一批已有IQ，不是新增独立数据。Flood是作者固定K运输LP的RF适配两步定位，非IQ多径DPD；Xia公共候选适配改变原随机首轮池，不是原论文参数下的严格复现。', '',
      '两新增方法均重新提峰、重建公共三站候选；不运行旧固定路径多径后端。GEO/LOS/WIDE仅复用当前原记录。未向估计器提供真位置、波形、路径数或时延。', '',
      '|条件|方法|完整/总数|20m成功/总数|全体RMSE(m)|完整子集条件RMSE(m)|',
      '|---|---|---:|---:|---:|---:|']
    for cond, methods in metrics.items():
        for method, d in methods.items():
            fmt = lambda v: '—' if v is None else f'{v:.6f}'
            lines.append(f"|{cond}|{method}|{d['complete']}/{d['n']}|{d['success20']}/{d['n']}|{fmt(d['full_rmse_m'])}|{fmt(d['conditional_rmse_m'])}|")
    lines += ['', '缺失输出不填零；严格终态和最近完整输出分列。计时为本批实际完整前端/评分处理；与原方法异批耗时不作速度排名。',
              f"本批{len(rows)}条，求解与保存合计{sum(r['total_seconds'] for r in rows):.3f}s。", '',
              '本轮只能补充已登记近邻适配的当前输入证据；不能据此宣称同信息强多径DPD基线已经齐备，也不将Xia适配的负面结果泛化为原论文无效。']
    (out/'近邻配对报告.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    tests = selftest()
    contract = load(FORMAL/'contract.json')['contract']
    cal.verify(contract)
    # Existing implementations must match their previously frozen run snapshots.
    snapshots = {
        BASE/'flood_band_pair.py': BASE/'flood_fixed_validation_20261029_235358_862467/source/flood_band_pair.py',
        BASE/'flood_lp_check.py': BASE/'flood_fixed_validation_20261029_235358_862467/source/flood_lp_check.py',
        BASE/'xia_common_candidates.py': BASE/'xia_common_validation_20261029_235513_195842/source/xia_common_candidates.py',
        BASE.parent/'xia_pairing/recheck/xia_checked.py': BASE/'xia_common_validation_20261029_235513_195842/source/xia_checked.py',
        BASE.parent/'xia_pairing/xia_reference.py': BASE/'xia_common_validation_20261029_235513_195842/source/xia_reference.py'}
    assert all(sha(a) == sha(b) for a, b in snapshots.items())
    files = [Path(__file__), HERE.parent/'执行协议.md', FORMAL/'contract.json']
    files += sorted((FORMAL/'cases').glob('*/input.npz'))
    files += sorted((FORMAL/'cases').glob('*/input.json')) + sorted((FORMAL/'cases').glob('*/solve.json'))
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename and Path(filename).suffix == '.py' and Path(filename).resolve().is_relative_to(BASE):
            files.append(Path(filename))
    hashes = digest_paths(files)
    out = HERE/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    with lock(out):
        write(HERE/'latest.json', dict(path=str(out)))
        write(out/'preflight.json', dict(hashes=hashes, frozen_contract=contract, tests=tests,
             python=sys.version, numpy=np.__version__, scipy=scipy.__version__, threads=2,
             prior_adapters_sha_match=True, seed_rule='20261029223+10000*20+repeat',
             truth_in_estimator=False, unchanged_parameters=True))
        (out/'source').mkdir()
        for filename, digest in hashes.items():
            if Path(filename).suffix == '.py':
                (out/'source'/(digest+'.py')).write_bytes(Path(filename).read_bytes())
        print('OUTPUT', out, flush=True)
        rows = []
        try:
            cases = cal.cases()
            for i, case in enumerate(cases):
                rows.append(run_case(out, case))
                summarize(out, rows)
                if i == 3:
                    estimate = sum(r['total_seconds'] for r in rows)*8
                    write(out/'four_case_short_test.json', dict(passed=True, estimate_seconds=estimate,
                         bound_seconds=3600, automatic_continuation=estimate < 3600))
                    print('FOUR CASE PASS estimated seconds', estimate, flush=True)
                    if estimate >= 3600:
                        return
            verify(hashes)
            cal.verify(contract)
            write(out/'postflight.json', dict(all_identity_unchanged=True, completed=len(rows), all_pass=True))
            print('COMPLETE', out, flush=True)
        except Exception:
            write(out/'failure.json', dict(traceback=traceback.format_exc(), completed=len(rows)))
            raise


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
