"""Finite-window model audit at known simulation positions; no localization.

Read frozen clean source components, noise and IQ. Compare N/1, N/4 and N/8
DFTs with their own frequency axes and normalization. Truth is diagnostic only.
"""
from pathlib import Path
from datetime import datetime
import argparse
import csv
import hashlib
import io
import json
import sys
import time
import numpy as np
from ....g8.physics import C, RECEIVERS

ROOT = Path(__file__).resolve().parents[5]
BASE = ROOT/'outputs_e2e/unified/e2e_g10/r4'
STATIONS = RECEIVERS.copy()
HALF_BAND = 7e6
TASKS = [(g, 200, gap) for g in range(4) for gap in (0, 15)]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def array_sha(data):
    return digest(np.ascontiguousarray(data).tobytes())


def checked_bytes(path, expected=None):
    data = path.resolve(strict=True).read_bytes()
    if expected is not None and digest(data) != expected:
        raise RuntimeError('Identity mismatch: '+str(path))
    return data


def read(path, expected=None):
    return json.loads(checked_bytes(path, expected).decode('utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def resolve(path):
    p = Path(path)
    return (p if p.is_absolute() else ROOT/p).resolve(strict=True)


def atom(positions, freq):
    distance = np.linalg.norm(positions[None]-STATIONS[:, None], axis=-1)
    return np.exp(-2j*np.pi*freq[:, None, None]*distance[None]/C)/distance[None]


def project(a, data):
    return data-a@(np.linalg.pinv(a, rcond=1e-12)@data)


def check_numerics():
    rng = np.random.default_rng(20261018)
    p = np.array([[-200., 100.], [700., -650.]])
    f = np.linspace(-HALF_BAND, HALF_BAND, 17)
    a = atom(p, f)
    coef = rng.normal(size=(17, 2, 8))+1j*rng.normal(size=(17, 2, 8))
    y = a@coef
    r = project(a, y)
    assert np.linalg.norm(r)/np.linalg.norm(y) < 1e-12
    loop = np.stack([y[i]-a[i]@np.linalg.lstsq(a[i], y[i], rcond=1e-12)[0]
                     for i in range(len(f))])
    np.testing.assert_allclose(r, loop, atol=1e-14)
    noise = rng.normal(size=(4, 8192))+1j*rng.normal(size=(4, 8192))
    for segments in (1, 4, 8):
        length = 8192//segments
        z = noise.reshape(4, segments, length)
        spec = np.fft.fft(z, axis=-1)/length
        np.testing.assert_allclose(np.sum(abs(spec)**2), np.sum(abs(z)**2)/length, rtol=1e-12)
    return dict(passed=True, exact_frequency_model_relative_error=float(np.linalg.norm(r)/np.linalg.norm(y)),
                checks=['exact model recovery', 'batched vs direct LS', 'segment Parseval scaling'])


def measure(signal, noise, positions, fs, segments, sigma2):
    k, l, n = signal.shape
    length = n//segments
    frequency = np.fft.fftfreq(length, 1/fs)
    keep = abs(frequency) <= HALF_BAND
    f = frequency[keep]
    z = (np.fft.fft(signal.reshape(k, l, segments, length), axis=-1)/length)[..., keep]
    z = z.transpose(0, 3, 1, 2)  # source, frequency, receiver, snapshot
    w = (np.fft.fft(noise.reshape(l, segments, length), axis=-1)/length)[..., keep].transpose(2, 0, 1)
    a = atom(positions, f)
    total = z.sum(axis=0)
    clean_res = project(a, total)
    noise_res = project(a, w)
    # Relative rank test must confirm K directions at every retained frequency.
    singular = np.linalg.svd(a, compute_uv=False)
    ranks = (singular > singular[:, :1]*1e-12).sum(axis=1)
    assert np.all(ranks == k)
    residual_power = float(np.sum(abs(clean_res)**2))
    source_power = np.sum(abs(z)**2, axis=(1, 2, 3))
    individual = [float(np.sum(abs(project(a[:, :, j:j+1], z[j]))**2)) for j in range(k)]
    expected_noise = float((l-k)*len(f)*segments*sigma2/length)
    covariance = total@total.swapaxes(-1, -2).conj()/segments
    eigenvalues = np.linalg.eigvalsh(covariance).clip(0)
    rank_tail = float(eigenvalues[:, :l-k].sum()/eigenvalues.sum())
    return dict(segments=segments, segment_samples=length, segment_us=length/fs*1e6,
                retained_bins=len(f), expected_bin_noise=sigma2/length,
                clean_model_residual_fraction=residual_power/float(np.sum(abs(total)**2)),
                clean_model_residual_to_expected_projected_noise=residual_power/expected_noise,
                projected_noise_to_expectation=float(np.sum(abs(noise_res)**2))/expected_noise,
                source_a_model_residual_fraction=individual[0]/float(source_power[0]),
                source_b_model_residual_fraction=individual[1]/float(source_power[1]),
                source_a_model_residual_to_source_b_energy=individual[0]/float(source_power[1]),
                clean_covariance_beyond_k_fraction=rank_tail,
                max_relative_delay_us=float(np.ptp(np.linalg.norm(positions[:, None]-STATIONS, axis=-1), axis=1).max()/C*1e6))


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.limit <= 8:
        raise ValueError('This bounded audit has exactly eight possible observations')
    start = time.perf_counter()
    checks = check_numerics()
    pointer_path = BASE/'preparation.json'
    pointer = read(pointer_path)
    prep = resolve(pointer['out'])
    contract = read(prep/'contract.json', pointer['contract_sha'])
    physics_path = ROOT/'统一模型代码/gates/g8/physics.py'
    checked_bytes(physics_path, contract['sources'][str(physics_path.relative_to(ROOT))])
    ref = contract['reference']
    source = resolve(ref['out'])
    read(source/'contract.json', ref['contract_sha'])
    manifest = read(source/'input_manifest.json', ref['input_manifest_sha'])
    assert source.is_relative_to((ROOT/'outputs_e2e').resolve())
    identities = {str(prep/'contract.json'): pointer['contract_sha'],
                  str(source/'contract.json'): ref['contract_sha'],
                  str(source/'input_manifest.json'): ref['input_manifest_sha'],
                  str(physics_path): digest(physics_path.read_bytes()),
                  str(Path(__file__).resolve()): digest(Path(__file__).read_bytes())}
    cache = []
    for group, repeat, gap in TASKS[:args.limit]:
        key = f'{group}_{repeat}_{gap}'
        registered = manifest[key]
        meta_path = source/'inputs'/f'{key}.json'
        meta = read(meta_path, registered['metadata_sha'])
        payload_path = (source/meta['path']).resolve(strict=True)
        assert payload_path.is_relative_to(source)
        assert registered['payload_sha'] == meta['sha256']
        payload = checked_bytes(payload_path, registered['payload_sha'])
        with np.load(io.BytesIO(payload), allow_pickle=False) as z:
            signal, noise, positions = z['signal'], z['noise'], z['positions']
        assert signal.shape == (2, 4, 8192) and noise.shape == (4, 8192)
        assert array_sha(signal) == meta['components_sha'] and array_sha(noise) == meta['noise_sha']
        iq = signal.sum(axis=0)+noise
        assert array_sha(iq) == contract['inputs'][key]['input_sha']
        np.testing.assert_array_equal(positions, contract['inputs'][key]['truth'])
        assert meta['sample_rate_hz'] == 50e6 and meta['center_hz'] == 0.
        identities[str(meta_path)] = registered['metadata_sha']
        identities[str(payload_path)] = registered['payload_sha']
        cache.append((key, signal, noise, positions, meta))
    out = BASE/('segment_check_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    assert out.resolve().is_relative_to(BASE.resolve()) and not out.exists()
    out.mkdir()
    write(out/'contract.json', dict(inputs=identities, tasks=TASKS[:args.limit], segments=[1, 4, 8],
          propagation_speed_m_s=C, receivers=STATIONS.tolist(),
          half_band_hz=HALF_BAND, source_positions_used_for_model_diagnostic_only=True,
          localization_solves=0, scope='Finite-window audit; not a performance ranking or parameter selection.'))
    rows = []
    for key, signal, noise, positions, meta in cache:
        for segments in (1, 4, 8):
            rows.append(dict(case=key, group=meta['group'], gap_db=meta['gap_db'],
                             **measure(signal, noise, positions, meta['sample_rate_hz'], segments, meta['noise_variance'])))
    assert all(digest(checked_bytes(Path(p))) == h for p, h in identities.items())
    elapsed = time.perf_counter()-start
    with (out/'per_case.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write(out/'report.json', dict(status='COMPLETED_MODEL_DIAGNOSTIC', observations=len(cache),
          elapsed_seconds=elapsed, checks=checks, rows=rows, identities_unchanged=True,
          note='Nonzero clean residual measures model approximation, not localization error; no solver ranking recomputed.'))
    summaries = []
    for segments in (1, 4, 8):
        part = [x for x in rows if x['segments'] == segments]
        summaries.append(dict(segments=segments,
            clean_residual_fraction_range=[min(x['clean_model_residual_fraction'] for x in part), max(x['clean_model_residual_fraction'] for x in part)],
            model_to_noise_range=[min(x['clean_model_residual_to_expected_projected_noise'] for x in part), max(x['clean_model_residual_to_expected_projected_noise'] for x in part)]))
    lines = ['# 分段模型一致性核验', '', f'完成{len(cache)}条既有观测，耗时{elapsed:.3f}秒；未运行定位优化。', '',
             '已知真实位置仅用于核验频域模型。采用原始干净源分量和噪声，分别按实际段长进行DFT及带限；保留频点数不替代段长。', '',
             '|分段数|干净混合观测残差占比范围|模型残差/预计投影后噪声范围|', '|---|---|---|']
    for s in summaries:
        lo, hi = s['clean_residual_fraction_range']; nl, nh = s['model_to_noise_range']
        lines.append(f"|{s['segments']}|{lo:.6%}—{hi:.6%}|{nl:.4f}—{nh:.4f}|")
    lines += ['', '该检查量化有限窗近似的实际偏差，不把偏差直接归因为某个定位方法的失败，也不据此调整正式参数。',
              '身份、投影代数与归一化检查通过；详细逐例记录见per_case.csv和report.json。']
    (out/'运行摘要.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write(BASE/'segment_check_latest.json', dict(out=str(out), complete=args.limit == 8,
                                              report_sha=digest((out/'report.json').read_bytes())))
    print(json.dumps(dict(out=str(out), elapsed_seconds=elapsed, summaries=summaries), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
