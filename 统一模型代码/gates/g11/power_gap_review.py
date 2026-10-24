"""G11-P0: saved-result power-gap pairs only. No localization or tuning."""
import csv
import hashlib
import io
import json
import sys
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.g10.r4.paper_checks.assignment_metrics import evaluate, REVISION

BASE = ROOT/'outputs_e2e/unified/e2e_g10/r4'
GAPS = (0, 3, 6, 9, 12, 15)
SEED, DRAWS = 20261022, 4000
PROVENANCE = {}


def resolve(p):
    p = Path(p)
    for old in ('F:/SourceCount_DPD_E2E', 'F:/DPD_WeakSource_Localization'):
        if p.is_relative_to(Path(old)):
            p = ROOT/p.relative_to(Path(old))
    if not p.is_absolute():
        p = ROOT/p
    p = p.resolve(strict=True)
    assert p.is_relative_to(ROOT)
    return p


def raw(p, expected=None):
    p = resolve(p)
    b = p.read_bytes(); h = hashlib.sha256(b).hexdigest()
    if expected is not None:
        assert h == expected, str(p)
    if str(p) in PROVENANCE:
        assert PROVENANCE[str(p)] == h
    PROVENANCE[str(p)] = h
    return b


def read(p, expected=None):
    return json.loads(raw(p, expected).decode('utf-8-sig'))


def write(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def csv_write(p, rows):
    with p.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def normalize(r, cohort):
    out = dict(r, cohort=cohort)
    for k in ('layout', 'repeat', 'gap'):
        out[k] = int(out[k])
    for k in ('both20', 'strong20', 'weak20'):
        out[k] = str(out[k]).lower() in ('true', '1')
    for k in ('strong_error_m', 'weak_error_m', 'seconds'):
        out[k] = float(out[k]); assert np.isfinite(out[k]) and out[k] >= 0
    return out


def verify_result(r, truth, result):
    q = evaluate(truth, result['positions'])
    assert q['valid_output'] and q['matched'] == 2 and q['protocol'] == REVISION
    np.testing.assert_allclose(q['errors_m'], [r['strong_error_m'], r['weak_error_m']], rtol=1e-12, atol=1e-10)
    assert q['both20'] == r['both20']
    assert q['success20'] == [r['strong20'], r['weak20']]
    time_key = 'review_complete_seconds' if r['cohort'] in ('new480', 'curvature240') else 'complete_seconds'
    assert abs(result[time_key]-r['seconds']) < 1e-10
    r['solver_seed'] = result.get('seed')
    r['config_sha'] = hashlib.sha256(json.dumps(result['config'], sort_keys=True).encode()).hexdigest()


def sources_and_rows():
    old = ROOT/'paper/data/migration_replay_20261019'
    supplement = ROOT/'paper/data/review_supplement_20261020'
    reports = [read(old/'report.json'), read(supplement/'audit.json')]
    expected = {str(resolve(k)): v for d in reports for k, v in d['source_files'].items()}
    formal = resolve(read(BASE/'formal_run.json')['out'])
    profile = resolve(read(BASE/'profile_full_run.json')['out'])
    new = resolve(read(BASE/'review_supplement_release.json')['out'])
    plans = [('common240', old/'common_per_case.csv'), ('core2400', old/'core_per_case.csv'),
             ('supplement', ROOT/'paper_short_vp/data/per_case.csv')]
    rows = []
    for name, p in plans:
        data = list(csv.DictReader(io.StringIO(raw(p).decode('utf-8-sig'))))
        for item in data:
            cohort = name if name != 'supplement' else ('new480' if item['stage'] == 'geometry' else 'curvature240')
            r = normalize(item, cohort); tag = r['case']; m = r['method']
            if name == 'supplement':
                p0 = new/'pairs'/(item['stage']+'_'+tag+'.json')
                pair = read(p0, expected[str(p0)])
                src = next(x for x in pair['rows'] if x['method'] == m)
                verify_result(r, pair['truth'], src['result'])
            else:
                p0 = profile/'rows'/(tag+'.json') if cohort == 'core2400' and m == 'PROFILE_DIAG' else formal/'rows'/(tag+'_'+m+'.json')
                q = read(p0, expected[str(p0)])
                verify_result(r, q['truth'], q['result'])
            r['source'] = str(p0)
            rows.append(r)
    assert len(rows) == 13920
    return rows, formal, new


def verify_power_pairs(folder, repeats, layouts, prefix=''):
    """Verify identical weak component/noise and scaled strong component in saved bytes."""
    groups = 0; max_scale_error = 0.
    original = read(folder/'final_audit_report.json')['inputs'] if not prefix else None
    for layout in range(layouts):
        for repeat in repeats:
            anchor = None; anchor_meta = None
            for gap in GAPS:
                tag = f'{layout}_{repeat}_{gap}'
                expected_meta = original[tag]['identity']['metadata_sha'] if original else None
                m = read(folder/'inputs'/f'{prefix}{tag}.json', expected_meta)
                b = raw(folder/m['path'], m['sha256'])
                with np.load(io.BytesIO(b), allow_pickle=False) as z:
                    signals, noise, positions = z['signal'], z['noise'], z['positions']
                if original:
                    expected_iq = original[tag]['identity']['iq_sha']
                    assert positions.tolist() == original[tag]['truth']
                else:
                    record = read(folder/'pairs'/f'geometry_{tag}.json')
                    expected_iq = record['iq_sha']
                    assert positions.tolist() == record['truth']
                assert hashlib.sha256(np.ascontiguousarray(signals.sum(0)+noise).tobytes()).hexdigest() == expected_iq
                assert signals.shape == (2, 4, 8192) and noise.shape == (4, 8192)
                assert m['sample_rate_hz'] == 50e6 and m['snr_inband_db'] == 6
                if anchor is None:
                    anchor = (signals.copy(), noise.copy(), positions.copy()); anchor_meta = m
                assert np.array_equal(noise, anchor[1]) and np.array_equal(positions, anchor[2])
                assert np.array_equal(signals[1], anchor[0][1])
                assert m['noise_sha'] == anchor_meta['noise_sha']
                assert m['seed'] == anchor_meta['seed']
                err = np.linalg.norm(signals[0]-anchor[0][0]*10**(gap/20))/np.linalg.norm(signals[0])
                assert err < 1e-12
                max_scale_error = max(max_scale_error, float(err))
            groups += 1
    return dict(groups=groups, observations=groups*6, weak_noise_exact=True,
                max_relative_strong_scale_error=max_scale_error)


def boot_means(array):
    """L x R x gap x statistic; fixed layouts, resample R with paired gaps."""
    rng = np.random.default_rng(SEED)
    total = np.zeros((DRAWS, array.shape[2], array.shape[3]))
    for values in array:
        index = rng.integers(len(values), size=(DRAWS, len(values)))
        total += values[index].mean(axis=1)/len(array)
    return total


def analyze(rows):
    results, per_layout, transitions, trajectories = [], [], [], []
    for cohort in sorted({r['cohort'] for r in rows}):
        for method in sorted({r['method'] for r in rows if r['cohort'] == cohort}):
            part = [r for r in rows if r['cohort'] == cohort and r['method'] == method]
            assert len({r['config_sha'] for r in part}) == 1
            groups = {}
            for r in part:
                k = (r['layout'], r['repeat'])
                assert r['gap'] not in groups.setdefault(k, {})
                groups[k][r['gap']] = r
            assert all(set(g) == set(GAPS) for g in groups.values())
            keys = sorted(groups); layouts = sorted({k[0] for k in keys})
            sizes = [sum(k[0] == layout for k in keys) for layout in layouts]
            assert len(set(sizes)) == 1
            ordered = [[groups[k][gap] for gap in GAPS] for k in keys]
            values = np.array([[[r['weak_error_m'], r['weak_error_m']**2, float(r['both20']), r['seconds']]
                                for r in scan] for scan in ordered])
            boot = boot_means(values.reshape(len(layouts), sizes[0], 6, 4))
            deltas = dict(weak_rmse_delta_m=np.sqrt(boot[:,:,1])-np.sqrt(boot[:,0:1,1]),
                          mean_weak_error_delta_m=boot[:,:,0]-boot[:,0:1,0],
                          both_success_delta_pp=100*(boot[:,:,2]-boot[:,0:1,2]),
                          mean_time_delta_s=boot[:,:,3]-boot[:,0:1,3],
                          time_change_percent=100*(boot[:,:,3]/boot[:,0:1,3]-1))
            for j, gap in enumerate(GAPS):
                now = values[:,j]; zero = values[:,0]
                weak = now[:,0]; a = zero[:,2].astype(bool); b = now[:,2].astype(bool)
                s = dict(cohort=cohort, method=method, gap=gap, layouts=len(layouts), n=len(keys),
                         both_count=int(b.sum()), both_percent=float(b.mean()*100),
                         strong_rmse_m=float(np.sqrt(np.mean([scan[j]['strong_error_m']**2 for scan in ordered]))),
                         weak_rmse_m=float(np.sqrt(now[:,1].mean())), weak_median_m=float(np.median(weak)),
                         weak_p95_m=float(np.quantile(weak,.95)), mean_seconds=float(now[:,3].mean()),
                         median_seconds=float(np.median(now[:,3])), p95_seconds=float(np.quantile(now[:,3],.95)),
                         harmed=int(np.sum(a & ~b)), rescued=int(np.sum(~a & b)),
                         persist_failed=int(np.sum(~a & ~b)), weak_error_increased=int(np.sum(weak>zero[:,0]+1e-9)),
                         weak_rmse_delta_m=float(np.sqrt(now[:,1].mean())-np.sqrt(zero[:,1].mean())),
                         mean_weak_error_delta_m=float(np.mean(weak-zero[:,0])),
                         both_success_delta_pp=float(100*np.mean(b.astype(float)-a)),
                         mean_time_delta_s=float(np.mean(now[:,3]-zero[:,3])),
                         time_change_percent=float(100*(now[:,3].mean()/zero[:,3].mean()-1)))
                for field, samples in deltas.items():
                    low, high = np.quantile(samples[:,j], [.025,.975])
                    s[field+'_ci_low'] = float(low); s[field+'_ci_high'] = float(high)
                results.append(s)
                for k, scan in zip(keys, ordered):
                    transitions.append(dict(cohort=cohort, method=method, layout=k[0], repeat=k[1], gap=gap,
                        zero_success=scan[0]['both20'], success=scan[j]['both20'],
                        weak_error_delta_m=scan[j]['weak_error_m']-scan[0]['weak_error_m'],
                        seconds_delta=scan[j]['seconds']-scan[0]['seconds']))
                for layout in layouts:
                    mask = np.array([k[0]==layout for k in keys]); v=now[mask]; z=zero[mask]
                    per_layout.append(dict(cohort=cohort, method=method, layout=layout, gap=gap, n=int(mask.sum()),
                        both_count=int(v[:,2].sum()), weak_rmse_m=float(np.sqrt(v[:,1].mean())),
                        weak_mse_delta=float(np.mean(v[:,1]-z[:,1])), mean_seconds=float(v[:,3].mean()),
                        mean_seconds_delta=float(np.mean(v[:,3]-z[:,3]))))
            trajectories.append(dict(cohort=cohort, method=method, groups=len(keys),
                configurations_constant=True,
                optimizer_randomness_confounded=method in ('GA','MULTISTART_FIXED'),
                seed_changed_groups=sum(scan[0]['solver_seed']!=scan[-1]['solver_seed'] for scan in ordered),
                all_gaps_success=int(np.sum(np.all(values[:,:,2]>0,axis=1))),
                all_gaps_failure=int(np.sum(np.all(values[:,:,2]==0,axis=1))),
                zero_success_later_failure=int(np.sum((values[:,0,2]>0)&np.any(values[:,1:,2]==0,axis=1))),
                zero_failure_later_success=int(np.sum((values[:,0,2]==0)&np.any(values[:,1:,2]>0,axis=1))),
                weak_error_monotonic_nondecreasing=int(np.sum(np.all(np.diff(values[:,:,0],axis=1)>=-1e-9,axis=1)))))
    return results, per_layout, transitions, trajectories


def main():
    out = ROOT/'outputs_e2e/unified/e2e_g11'/('p0_power_gap_20261022_'+datetime.now().strftime('%H%M%S_%f'))
    assert not out.exists(); out.mkdir(parents=True)
    protocol = dict(scope='Approved exploratory saved-result analysis; no new localization',
        evaluation=REVISION, gaps=GAPS, anchor_gap=0, bootstrap_draws=DRAWS, bootstrap_seed=SEED,
        interval='Pointwise paired percentile; fixed layouts, resample realizations within each; all six gaps and methods paired.',
        limits='Not independent new evidence; common240 subset of core2400; curvature uses same old240. No multiplicity-adjusted significance claims.',
        optimizer_seed='GA and MULTISTART_FIXED seeds vary with gap: descriptive performance, not isolated per-realization power effect. AP/spectral/SBL/PROFILE routes deterministic given input/config.',
        metrics='Success at existing 20m; all-case untruncated RMSE, median/P95 errors, complete time, transitions.',
        decision='No application utility threshold invented after results; no parameter/SNR/grid/scene changes.')
    write(out/'protocol.json', protocol)
    protected = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for directory in ('paper','paper_short_vp')
                 for p in (ROOT/directory).glob('*') if p.suffix in ('.tex','.pdf')}
    toy = np.zeros((2,3,6,4)); toy[:,:,0,3]=1; toy[:,:,1:,3]=2
    test = boot_means(toy); assert np.all(test[:,1:,3]-test[:,0:1,3] == 1)
    rows, formal, new = sources_and_rows()
    print('来源与13920项结果评价重放通过', flush=True)
    pairing = dict(old=verify_power_pairs(formal, range(400,420),20),
                   new=verify_power_pairs(new,range(1000,1005),16,'new_'))
    print('2880条输入的弱源/噪声配对及强源幅度缩放通过', flush=True)
    results, layouts, transitions, trajectories = analyze(rows)
    for name, data in [('power_summary',results), ('by_layout',layouts), ('paired_changes',transitions), ('trajectory_patterns',trajectories)]:
        csv_write(out/(name+'.csv'), data)
    write(out/'analysis.json', dict(summary=results, trajectories=trajectories, protocol=protocol))
    for p,h in PROVENANCE.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h, p
    for p,h in protected.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    write(out/'source_manifest.json', PROVENANCE)
    write(out/'audit.json', dict(passed=True, source_count=len(PROVENANCE), replay_rows=len(rows),
          input_pairing=pairing, bootstrap_sanity=True, manuscripts_unchanged=protected,
          script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), numpy=np.__version__))
    lines=['# G11-P0功率差配对核验','',
           '只分析已有结果；每组固定位置、弱源和噪声，仅增加源A功率。0dB时A/B等功率，B标签沿用以便配对。',
           '区间为固定构型内实现重采样的逐项95%描述区间；不是新构型总体区间，不以显著性代替实际价值。','',
           '|批次|方法|0→15dB成功数|弱源RMSE/m|平均耗时/s|新增失败/恢复|',
           '|---|---|---|---|---|---|']
    for s in results:
        if s['gap']!=15: continue
        z=next(v for v in results if (v['cohort'],v['method'],v['gap'])==(s['cohort'],s['method'],0))
        lines.append(f"|{s['cohort']}|{s['method']}|{z['both_count']}→{s['both_count']}/{s['n']}|{z['weak_rmse_m']:.3f}→{s['weak_rmse_m']:.3f}|{z['mean_seconds']:.3f}→{s['mean_seconds']:.3f}|{s['harmed']}/{s['rescued']}|")
    (out/'端点汇总.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(str(out), flush=True)


if __name__=='__main__':
    main()
