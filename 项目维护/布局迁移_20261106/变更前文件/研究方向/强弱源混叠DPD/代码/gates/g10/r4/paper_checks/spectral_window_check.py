"""Bounded same-record spectral applicability check; frozen solvers unchanged.

Actual-data window variants never receive truth. A separate oracle projection
removes out-of-model clean components, for attribution only, not method ranking.
"""
import argparse
import csv
import io
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.signal.windows import hann

from .segment_model_check import (
    ROOT, BASE, TASKS, atom, array_sha, checked_bytes, digest, read, resolve, write, project,
)
from ...physics import grid, shortlist
from ...methods import peaks
from ...metrics import evaluate
from ...r1.methods import Trace
from ...r2.band.physics import Model, Spectrum as FrozenSpectrum
from ...r2.band.solvers import run as frozen_run

VARIANTS = {
    'rect4': (2048, tuple(range(0, 8192, 2048)), 'rect'),
    'rect8': (1024, tuple(range(0, 8192, 1024)), 'rect'),
    'hann4': (2048, tuple(range(0, 8192, 2048)), 'hann'),
    'hann8': (1024, tuple(range(0, 8192, 1024)), 'hann'),
    'hann_long4': (4096, (0, 1365, 2730, 4096), 'hann'),
}


def transform(x, variant, fs):
    length, starts, name = VARIANTS[variant]
    window = hann(length, sym=False) if name == 'hann' else np.ones(length)
    frames = np.stack([x[..., start:start+length]*window for start in starts], axis=-2)
    f = np.fft.fftfreq(length, 1/fs)
    keep = abs(f) <= 7e6
    z = (np.fft.fft(frames, axis=-1)/length)[..., keep]
    # Leading axes preserved: (..., receiver, snapshot, freq) -> (..., freq, receiver, snapshot).
    z = np.moveaxis(z, -1, -3)
    return f[keep], z, float(np.sum(window**2)/length**2)


class Spectrum(FrozenSpectrum):
    def __init__(self, model, method, count, cfg, frequency, data, noise_bin):
        self.m, self.method, self.weight_mode = model, method, cfg['music_weight']
        self.freq = torch.as_tensor(frequency, device=model.device)
        z = torch.as_tensor(data, device=model.device)
        self.cov = z@z.mH/z.shape[-1]
        ev, vec = torch.linalg.eigh(self.cov)
        self.rank_min = int((ev > ev[:, -1:]*1e-10).sum(-1).min().cpu())
        model.counts['covariance_decompositions'] += len(frequency)
        if method == 'MUSIC':
            en = vec[:, :, :4-count]
            self.matrix = en@en.mH
        else:
            scale = self.cov.diagonal(dim1=1, dim2=2).real.mean(-1)
            self.matrix = torch.linalg.inv(self.cov+cfg['loading']*scale[:, None, None]
                                           *torch.eye(4, device=model.device))
            model.counts['covariance_decompositions'] += len(frequency)
        energy = self.cov.diagonal(dim1=1, dim2=2).real.sum(1)
        self.weight = ((energy-4*noise_bin).clamp_min(0) if self.weight_mode == 'excess_power'
                       else torch.ones_like(energy))
        if float(self.weight.sum().cpu()) <= 0:
            self.weight = torch.ones_like(energy)
        self.weight /= self.weight.sum()


def locate(iq, method, count, cfg, variant, spectral_override=None):
    """No truth parameter: only observed IQ, public configuration, optional diagnostic data."""
    t = Trace(lambda: None)
    m = Model(iq, t.check, sample_rate_hz=cfg['sample_rate_hz'], half_band_hz=cfg['half_band_hz'])
    f, z, noise_bin = transform(iq, variant, cfg['sample_rate_hz'])
    if spectral_override is not None:
        z = spectral_override
    spec = Spectrum(m, method, count, cfg, f, z, noise_bin)
    points = grid(cfg['grid'])
    values = spec.evaluate(points)
    ids = peaks(points, values, cfg['top'])
    options = [(float(values[i]), points[i].copy()) for i in ids]

    def publish():
        positions = np.asarray([p for _, p in options])
        scores = np.asarray([v for v, _ in options])
        take = shortlist(positions, scores, count)
        t.save(positions[take], -float(scores[take].sum()))

    publish()
    for i, (value, p) in enumerate(options):
        opt = minimize(lambda x: spec.value_grad(x, value), p, jac=True, method='L-BFGS-B',
                       bounds=[(-1000., 1000.)]*2,
                       options=dict(maxiter=cfg['local_maxiter'], ftol=cfg['ftol'],
                                    gtol=cfg['gtol'], maxls=cfg['maxls']))
        new = float(spec.evaluate(opt.x[None])[0])
        if new >= value:
            options[i] = new, opt.x
        publish()
    torch.cuda.synchronize()
    return dict(positions=t.events[-1]['positions'], seconds=t.elapsed(), counts=m.counts,
                native_objective=t.best)


def diagnostics(signal, noise, positions, fs, variant, sigma2):
    f, z, factor = transform(signal, variant, fs)
    _, w, _ = transform(noise, variant, fs)
    a = atom(positions, f)
    clean = z.sum(axis=0)
    residual = project(a, clean)
    projected = np.stack([z[j]-project(a[:, :, j:j+1], z[j]) for j in range(len(z))])
    oracle = projected.sum(axis=0)+w
    oracle_residual = project(a, projected.sum(axis=0))
    assert np.linalg.norm(oracle_residual)/np.linalg.norm(projected) < 1e-10
    expected = (4-len(positions))*len(f)*z.shape[-1]*sigma2*factor
    powers = np.sum(abs(z)**2, axis=(1, 2, 3))
    retained = np.sum(abs(projected)**2, axis=(1, 2, 3))/powers
    return oracle, dict(model_to_noise=float(np.sum(abs(residual)**2)/expected),
                        model_fraction=float(np.sum(abs(residual)**2)/np.sum(abs(clean)**2)),
                        retained_source_energy=retained.tolist(), frames=z.shape[-1],
                        retained_bins=len(f), noise_bin=sigma2*factor,
                        oracle_model_fraction=float(np.sum(abs(oracle_residual)**2)
                                                    /np.sum(abs(projected.sum(axis=0))**2)))


def load_inputs(limit):
    pointer = read(BASE/'preparation.json')
    prep = resolve(pointer['out'])
    contract = read(prep/'contract.json', pointer['contract_sha'])
    ref = contract['reference']
    source = resolve(ref['out'])
    frozen = read(source/'contract.json', ref['contract_sha'])
    manifest = read(source/'input_manifest.json', ref['input_manifest_sha'])
    identities = {str(prep/'contract.json'): pointer['contract_sha'],
                  str(source/'contract.json'): ref['contract_sha'],
                  str(source/'input_manifest.json'): ref['input_manifest_sha']}
    cache = []
    for g, repeat, gap in TASKS[:limit]:
        key = f'{g}_{repeat}_{gap}'
        entry = manifest[key]
        meta_path = source/'inputs'/f'{key}.json'
        meta = read(meta_path, entry['metadata_sha'])
        path = (source/meta['path']).resolve(strict=True)
        assert path.is_relative_to(source) and source.is_relative_to(ROOT/'outputs_e2e')
        assert entry['payload_sha'] == meta['sha256']
        payload = checked_bytes(path, entry['payload_sha'])
        with np.load(io.BytesIO(payload), allow_pickle=False) as z:
            signal, noise, positions = z['signal'], z['noise'], z['positions']
        assert signal.shape == (2, 4, 8192) and noise.shape == (4, 8192)
        assert meta['noise_variance'] == 1. and meta['sample_rate_hz'] == 50e6
        assert array_sha(signal) == meta['components_sha'] and array_sha(noise) == meta['noise_sha']
        assert array_sha(signal.sum(0)+noise) == contract['inputs'][key]['input_sha']
        np.testing.assert_array_equal(positions, contract['inputs'][key]['truth'])
        identities[str(path)] = entry['payload_sha']
        identities[str(meta_path)] = entry['metadata_sha']
        cache.append((key, signal, noise, positions, meta))
    configs = {method: next(j['config'] for j in frozen['jobs'] if j['method'] == method)
               for method in ('MUSIC', 'MVDR')}
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path and Path(path).suffix == '.py' and Path(path).resolve().is_relative_to(ROOT/'统一模型代码'):
            identities[str(Path(path).resolve())] = digest(Path(path).read_bytes())
    # Check the unchanged frozen spectral implementation against registered identity.
    for relative, expected in frozen['band_sources'].items():
        if Path(relative).name == 'physics.py':
            checked_bytes(ROOT/relative, expected)
    return cache, configs, identities


def checks(cache, configs):
    _, signal, noise, _, meta = cache[0]
    iq = signal.sum(0)+noise
    reports = []
    for method, cfg in configs.items():
        variant = f"rect{cfg['segments']}"
        m = Model(iq)
        f, z, factor = transform(iq, variant, meta['sample_rate_hz'])
        old = FrozenSpectrum(m, method, 2, cfg['segments'], cfg['loading'], cfg['music_weight'])
        new = Spectrum(m, method, 2, cfg, f, z, factor)
        torch.testing.assert_close(old.cov, new.cov, rtol=1e-12, atol=1e-14)
        torch.testing.assert_close(old.matrix, new.matrix, rtol=1e-10, atol=1e-12)
        p = np.array([[-230., 122.], [620., -360.]])
        np.testing.assert_allclose(old.evaluate(p), new.evaluate(p), rtol=1e-10)
        baseline = frozen_run(iq, method, config=cfg)
        updated = locate(iq, method, 2, cfg, variant)
        np.testing.assert_allclose(baseline['positions'], updated['positions'], atol=1e-5)
        reports.append(dict(method=method, frozen_output_equal=True))
    # Verify taper noise scaling algebra and Parseval for the exact sampled window.
    for length, _, name in VARIANTS.values():
        w = hann(length, sym=False) if name == 'hann' else np.ones(length)
        x = iq[:, :length]*w
        np.testing.assert_allclose(np.sum(abs(np.fft.fft(x)/length)**2),
                                   np.sum(abs(x)**2)/length, rtol=1e-12)
        if name == 'hann':
            np.testing.assert_allclose(np.sum(w*w)/length, 3/8, rtol=1e-12)
    f, z, noise_bin = transform(iq, 'hann_long4', meta['sample_rate_hz'])
    direct = np.stack([sum(np.outer(v, v.conj()) for v in band.T)/band.shape[-1] for band in z])
    np.testing.assert_allclose(direct, z@z.swapaxes(-1, -2).conj()/z.shape[-1], atol=1e-15)
    for method, cfg in configs.items():
        spec = Spectrum(m, method, 2, cfg, f, z, noise_bin)
        p = np.array([-230., 122.])
        scale = spec.evaluate(p)[0]
        _, gradient = spec.value_grad(p, scale)
        finite = []
        for direction in np.eye(2)*.001:
            finite.append((spec.value_grad(p+direction, scale)[0]
                           -spec.value_grad(p-direction, scale)[0])/.002)
        np.testing.assert_allclose(gradient, finite, rtol=1e-4, atol=1e-7)
    return reports


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.limit <= 8:
        raise ValueError('Bounded development check: 1 to 8 observations only')
    cache, configs, identities = load_inputs(args.limit)
    out = BASE/('spectral_window_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    assert not out.exists() and out.resolve().is_relative_to(BASE.resolve())
    out.mkdir()
    write(out/'contract.json', dict(inputs=identities, configurations=configs, variants=VARIANTS,
          cases=[c[0] for c in cache], oracle='truth-projected clean source components plus identical noise',
          scope='Development applicability diagnostic, not a new formal ranking',
          sources=['https://arlpy.readthedocs.io/en/latest/bf.html',
                   'https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.hann.html'],
          overlap_note='Long windows are correlated; four frames are not four independent snapshots.'))
    start = time.perf_counter()
    try:
        validation = checks(cache, configs)
        write(out/'checks.json', dict(passed=True, frozen_equivalence=validation))
        rows, diagnostic_rows = [], []
        for key, signal, noise, positions, meta in cache:
            iq = signal.sum(0)+noise
            for variant in VARIANTS:
                oracle, diag = diagnostics(signal, noise, positions, meta['sample_rate_hz'],
                                           variant, meta['noise_variance'])
                diagnostic_rows.append(dict(case=key, variant=variant, **diag))
                for method, cfg in configs.items():
                    # Oracle only for each method's frozen rectangular configuration.
                    modes = ('actual', 'oracle') if variant == f"rect{cfg['segments']}" else ('actual',)
                    for mode in modes:
                        result = locate(iq, method, 2, cfg, variant, oracle if mode == 'oracle' else None)
                        metric = evaluate(positions, result['positions'])
                        row = dict(case=key, gap_db=meta['gap_db'], method=method, variant=variant,
                                   mode=mode, both20=all(metric['success']['20']),
                                   strong20=metric['success']['20'][0], weak20=metric['success']['20'][1],
                                   strong_error_m=metric['errors_m'][0], weak_error_m=metric['errors_m'][1],
                                   gospa_m=metric['gospa_m'], **result)
                        row['kind'] = 'mixed'
                        rows.append(row)
                        write(out/f'{key}_{method}_{variant}_{mode}.json', row)
                    if meta['gap_db'] == 0:
                        # One unchanged weak-only observation per geometry, not repeated at 15 dB.
                        single = locate(signal[1]+noise, method, 1, cfg, variant)
                        metric = evaluate(positions[1:], single['positions'])
                        row = dict(case=key, gap_db=0, method=method, variant=variant,
                                   mode='actual', kind='weak', both20=all(metric['success']['20']),
                                   strong20=None, weak20=metric['success']['20'][0],
                                   strong_error_m=None, weak_error_m=metric['errors_m'][0],
                                   gospa_m=metric['gospa_m'], **single)
                        rows.append(row)
                        write(out/f'{key}_{method}_{variant}_weak.json', row)
            print(f'完成 {key}，累计 {len(rows)} 次诊断求解，{time.perf_counter()-start:.1f} 秒', flush=True)
        for path, identity in identities.items():
            checked_bytes(Path(path), identity)
        write(out/'report.json', dict(status='COMPLETED', observations=len(cache), solves=len(rows),
              elapsed_seconds=time.perf_counter()-start, identities_unchanged=True,
              rows=rows, diagnostics=diagnostic_rows, configurations_frozen=True,
              scope='Diagnostic only; oracle rows excluded from actual-data comparisons'))
        fields = [k for k in rows[0] if k not in ('positions', 'counts')]
        with (out/'per_case.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        print(str(out), flush=True)
    except Exception as exc:
        write(out/'failure.json', dict(status='FAILED', error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
