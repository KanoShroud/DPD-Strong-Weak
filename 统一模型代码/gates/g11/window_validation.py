"""G11-P2 fixed Hann versus rectangle; full blind search on saved observations.

Windowed LS is an explicitly approximate criterion: Hann-correlated FFT noise
is audited, never labelled independent or exact Gaussian ML. No oracle inputs
enter the locator. Frozen propagation/optimizer modules remain unchanged.
"""
import argparse
import copy
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy.signal.windows import hann

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.g11.precision_diagnostic import (
    TASKS, DIFFICULT, CONTROLS, P0, load_case, read, write, resolve, csv_write, PROVENANCE, sha,
)
from 统一模型代码.gates.g10.r4.solvers import run as core_run
from 统一模型代码.gates.g10.r3.solvers import Model, Trace
from 统一模型代码.gates.g10.r2.solvers import local
from 统一模型代码.gates.g10.physics import grid
from 统一模型代码.gates.g10.methods import peaks
from 统一模型代码.gates.g10.r4.paper_checks.assignment_metrics import evaluate, self_test, REVISION

WINDOWS = ('rect', 'hann')
PRIOR = ROOT/'outputs_e2e/unified/e2e_g11/p1_precision_20261022_193503_642436'


def window(name, n):
    if name == 'rect':
        return np.ones(n, dtype=np.float64)
    if name == 'hann':
        return hann(n, sym=False)
    raise ValueError(name)


def locate(iq, count, cfg, window_name):
    """No truth, clean source, old position, or outcome-dependent tuning input."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    w = window(window_name, iq.shape[1])
    processed = np.asarray(iq, np.complex128)*w[None]
    preprocessed = time.perf_counter()-start
    if count == 2:
        result = core_run(processed, 'AP_1_JOINT', copy.deepcopy(cfg))
    elif count == 1:
        # Same first-source AP primitive; no redundant second-source machinery.
        t = Trace(lambda: None)
        m = Model(processed, t.check, sample_rate_hz=cfg['sample_rate_hz'],
                  half_band_hz=cfg['half_band_hz'])
        t.model = m
        points = grid(cfg['grid'])
        scores = m.conditional(points)
        first = points[peaks(points, scores, cfg['top'])[0]]
        p, value, info = local(m, first[None], cfg)
        _, grad = m.joint(p, True)
        result = dict(method='AP_FIRST_SOURCE', positions=p.tolist(), native_objective=value,
                      projected_gradient=float(abs(grad).max()), optimizer=info,
                      stop_reason=str(info['message']), counts=m.counts)
    else:
        raise ValueError(count)
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    result.update(window=window_name, complete_seconds=elapsed, window_seconds=preprocessed,
                  samples=len(w), fft_scale_samples=len(w),
                  noise_bin_variance=float(np.sum(w*w)/len(w)**2),
                  criterion='approximate_windowed_profile_LS' if window_name=='hann' else 'rectangular_profile_LS')
    return result


def covariance_check():
    """Exact linear transform, not an empirical independence assumption."""
    n = 64
    w = window('hann', n)
    f = np.exp(-2j*np.pi*np.outer(np.arange(n), np.arange(n))/n)/n
    covariance = (f*w[None]**2)@f.conj().T
    difference = (np.arange(n)[:, None]-np.arange(n)[None]) % n
    kernel = np.fft.fft(w*w)/n**2
    np.testing.assert_allclose(covariance, kernel[difference], atol=1e-15)
    np.testing.assert_allclose(kernel[0].real, 3/(8*n), rtol=1e-14)
    np.testing.assert_allclose((kernel[[1, 2]]/kernel[0]).real, [-2/3, 1/6], atol=1e-14)
    assert np.max(abs(kernel[3:-2])) < 1e-15
    # Selected-bin covariance is the corresponding principal submatrix.
    index = np.flatnonzero(abs(np.fft.fftfreq(n, 1/50e6))<=7e6)
    np.testing.assert_allclose((f[index]*w[None])@(f[index]*w[None]).conj().T,
                               covariance[np.ix_(index,index)], atol=1e-15)
    rng = np.random.default_rng(2026102202)
    iq = rng.normal(size=(4, 8192))+1j*rng.normal(size=(4, 8192))
    checks = []
    for name in WINDOWS:
        ww = window(name, 8192)
        x = iq*ww
        m = Model(x)
        freq = np.fft.fftfreq(8192, 1/50e6)
        expected = (np.fft.fft(x)/8192)[:, abs(freq)<=7e6].T
        np.testing.assert_allclose(m.x.cpu().numpy(), expected, atol=1e-14)
        np.testing.assert_allclose(np.sum(abs(np.fft.fft(x)/8192)**2),
                                   np.sum(abs(x)**2)/8192, rtol=1e-13)
        assert m.sample_count == 8192
        checks.append(dict(window=name, retained_bins=len(expected), fft_normalization=8192,
                           variance_factor=float(np.mean(ww**2))))
    return dict(exact_covariance_passed=True, fft_checks=checks,
                hann_variance_per_bin=3/(8*8192), hann_adjacent_bin_correlation=-2/3,
                hann_distance_two_correlation=1/6,
                note='Correlated spectral noise; current LS is a surrogate, not exact ML.')


def statistics(rows):
    summary = []
    for kind in ('mixed', 'weak'):
        for group in ('all', 'difficult', 'control'):
            for gap in ((0, 15) if kind=='mixed' else (0,)):
                part = [r for r in rows if r['kind']==kind and r['gap']==gap
                        and (group=='all' or r['group']==group)]
                if not part:
                    continue
                for name in WINDOWS:
                    e = np.array([r['results'][name]['metrics']['errors_m'] for r in part])
                    sec = np.array([r['results'][name]['complete_seconds'] for r in part])
                    other = 'hann' if name=='rect' else 'rect'
                    ref = np.array([r['results'][other]['metrics']['errors_m'] for r in part])
                    summary.append(dict(kind=kind, group=group, gap=gap, window=name, n=len(part),
                        strong_rmse_m=float(np.sqrt(np.mean(e[:,0]**2))) if kind=='mixed' else None,
                        weak_rmse_m=float(np.sqrt(np.mean(e[:,-1]**2))),
                        weak_median_m=float(np.median(e[:,-1])), weak_p95_m=float(np.quantile(e[:,-1],.95)),
                        both20=int(np.all(e<=20,axis=1).sum()),
                        weak_improved_vs_other=int((e[:,-1]<ref[:,-1]).sum()),
                        mean_seconds=float(sec.mean()), median_seconds=float(np.median(sec)),
                        mean_window_seconds=float(np.mean([r['results'][name]['window_seconds'] for r in part]))))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    tasks = [(13,1000,0),(13,1000,15)] if args.pilot else TASKS
    start = time.perf_counter()
    expected = read(P0/'source_manifest.json')
    pointer = ROOT/'outputs_e2e/unified/e2e_g10/r4/review_supplement_release.json'
    folder = resolve(read(pointer, expected[str(pointer)])['out'])
    prior = read(PRIOR/'report.json')
    prior_cases = {r['case']:r for r in prior['rows']}
    out = ROOT/'outputs_e2e/unified/e2e_g11'/('p2_window_'+('pilot_' if args.pilot else '')
                                          +datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    assert not out.exists() and out.resolve().is_relative_to(ROOT/'outputs_e2e')
    out.mkdir(); (out/'pairs').mkdir()
    code = {str(p):sha(p) for p in (ROOT/'统一模型代码/gates').rglob('*.py')}
    protected = {str(p):sha(p) for root in ('paper','paper_short_vp')
                 for ext in ('*.tex','*.pdf') for p in (ROOT/root).glob(ext)}
    write(out/'protocol.json', dict(tasks=tasks, difficult=DIFFICULT, controls=CONTROLS,
          windows=['rectangular','periodic Hann, sym=False'], parameter_selection='none',
          scope='P1 same-observation finite diagnostic, not independent generalization',
          known_K=True, mixtures=2*len(tasks), isolated_weak=2*len(tasks)//2,
          original_solver='frozen AP_1_JOINT', initial_grid_m=20, sample_rate_hz=50e6,
          sample_count=8192, half_band_hz=7e6, evaluation=REVISION,
          exact_ML_claim=False, timing='including window multiplication, FFT, transfer, search, refinement, certificate; CUDA synchronized',
          source_code=code, protected=protected, numpy=np.__version__, scipy=scipy.__version__,
          torch=torch.__version__, python=sys.version))
    # Validate all task inputs before any localization.
    cached = [load_case(folder, expected, task) for task in tasks]
    config = cached[0][-1]['config']
    config_bytes = json.dumps(config, sort_keys=True)
    assert all(json.dumps(item[-1]['config'], sort_keys=True)==config_bytes for item in cached)
    assert all(tuple(item[3].shape)==(2,2) for item in cached)
    checks = dict(noise=covariance_check(), metrics=self_test())
    write(out/'config.json', config)
    # Each path is warmed once, on an already registered observation, outside timings.
    warm_start = time.perf_counter()
    _, signal, noise, _, _ = cached[0]
    for name in WINDOWS:
        locate(signal.sum(0)+noise, 2, config, name)
    warm = time.perf_counter()-warm_start
    write(out/'checks.json', dict(**checks, warmup_seconds=warm))
    print('频率轴、FFT归一化与噪声协方差检查通过，开始盲搜索配对。', flush=True)
    rows = []
    old_reproduction = []
    count=0
    for task, item in zip(tasks, cached):
        l, r, gap = task
        tag, signal, noise, truth, saved = item
        jobs = [('mixed',signal.sum(0)+noise,truth,2)]
        if gap==0:
            jobs.append(('weak',signal[1]+noise,truth[1:2],1))
        for kind, iq, gt, k in jobs:
            order = WINDOWS if count%2==0 else WINDOWS[::-1]
            results = {}
            for name in order:
                result = locate(iq, k, config, name)
                result['metrics'] = evaluate(gt, result['positions'])
                assert result['metrics']['valid_output'] and result['metrics']['matched']==k
                results[name] = result
            if kind=='mixed':
                err = evaluate(saved['positions'],results['rect']['positions'])['errors_m']
                assert max(err) < 1e-4, (tag,err)
                assert abs(results['rect']['native_objective']-saved['native_objective'])<1e-10
                old_reproduction.append(max(err))
                assert prior_cases[tag]['truth']==gt.tolist()
            row=dict(case=tag,layout=l,repeat=r,gap=gap,kind=kind,
                     group='difficult' if l in DIFFICULT else 'control',
                     truth=gt.tolist(),order=order,results=results,
                     iq_sha=hashlib.sha256(np.ascontiguousarray(iq).tobytes()).hexdigest())
            write(out/'pairs'/f'{kind}_{tag}.json',row)
            rows.append(row); count+=1
            if count%10==0:
                print(f'已完成{count}对，累计{time.perf_counter()-start:.1f}s',flush=True)
    summary=statistics(rows)
    csv_write(out/'summary.csv',summary)
    layout_rows=[]
    for l in sorted({r['layout'] for r in rows}):
        layout_rows += [dict(layout=l,**v) for v in statistics([r for r in rows if r['layout']==l]) if v['group']=='all']
    csv_write(out/'by_layout.csv',layout_rows)
    assert all(sha(p)==h for p,h in {**code,**protected,**PROVENANCE}.items())
    audit=dict(status='COMPLETED_BOUNDED_WINDOW_VALIDATION',pairs=len(rows),solves=2*len(rows),
               mixed_observations=len(tasks),weak_only_observations=len(rows)-len(tasks),
               max_rect_reproduction_shift_m=max(old_reproduction),frozen_sources_unchanged=True,
               papers_unchanged=True,identity_files=len(PROVENANCE),warmup_seconds=warm,
               elapsed_seconds=time.perf_counter()-start)
    write(out/'report.json',dict(audit=audit,summary=summary,rows=rows))
    write(out/'source_manifest.json',PROVENANCE)
    write(out/'audit.json',audit)
    print(str(out),flush=True)
    print(json.dumps(audit,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
