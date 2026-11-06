"""Finite source-level audit and paired-input checks; no method ranking."""
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import fitz
import numpy as np

from xia_reference import (C, score, scalar_score, literal_bandwidth,
                           decode_literal, relative_delays, cross_spectrum)

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0, str(PROJECT))
from 统一模型代码.gates.g8.scenes import rrc  # noqa: E402

STATIONS = np.array([[-400., -300.], [450., -250.], [380., 400.],
                     [-350., 350.], [20., -460.]])
POSITIONS = np.array([[-260., -180.], [230., 120.]])
FS, RATE, BAND, N = 50e6, 10e6, 12.5e6, 8192


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              allow_nan=False, default=convert), encoding='utf-8')


def paired_components(seed=2026102701):
    rng = np.random.default_rng(seed)
    distance = np.linalg.norm(POSITIONS[:, None] - STATIONS[None], axis=-1)
    delays = distance / C
    extra = rng.uniform(.2e-6, 2e-6, (2, 5, 2))
    gain = 500 / distance * np.exp(1j * rng.uniform(-np.pi, np.pi, (2, 5)))
    reflected_gain = gain[:, :, None] * 10**(np.array([-7., -15.])/20)
    reflected_gain *= np.exp(1j * rng.uniform(-np.pi, np.pi, (2, 5, 2)))
    # Long independent symbols before/after the observation; never periodic shifts.
    origin = -256
    bits = rng.choice([-1., 1.], (2, int(N * RATE / FS) + 512))
    timing = rng.random(2)
    carrier_phase = rng.uniform(-np.pi, np.pi, 2)
    pulse_energy = np.sum(rrc(np.arange(-10, 10, .001))**2) * .001

    def waveform(q, tau):
        u = (np.arange(N) / FS - tau) * RATE - timing[q]
        base = np.floor(u).astype(int)
        value = np.zeros(N)
        for offset in range(-10, 11):
            symbols = base + offset
            if (symbols - origin).min() < 0 or (symbols-origin).max() >= bits.shape[1]:
                raise RuntimeError('Missing waveform guard symbols')
            value += bits[q, symbols-origin] * rrc(u-symbols)
        return value * np.exp(1j * carrier_phase[q]) / np.sqrt(pulse_energy)

    los = np.empty((2, 5, N), complex)
    mp = np.zeros_like(los)
    for q in range(2):
        for m in range(5):
            los[q, m] = gain[q, m] * waveform(q, delays[q, m])
            for j in range(2):
                mp[q, m] += reflected_gain[q, m, j] * waveform(q, delays[q, m]+extra[q, m, j])
    # Ensemble power, not a per-realization observation renormalization.
    power = np.mean(abs(gain)**2, axis=1)
    scale = np.sqrt(power[1] / power[0])
    los[0] *= scale
    mp[0] *= scale
    gain[0] *= scale
    reflected_gain[0] *= scale
    sigma2 = power[1] / (10**(.6) * BAND / FS)
    noise = np.sqrt(sigma2/2) * (rng.normal(size=(5, N)) + 1j*rng.normal(size=(5, N)))
    truth = dict(positions=POSITIONS, direct_delays=delays, extra_delays=extra,
                 gain=gain, reflected_gain=reflected_gain, sigma2=sigma2)
    return los, mp, noise, truth


def preprocess(iq):
    frequency = np.fft.fftfreq(N, 1/FS)
    keep = abs(frequency) <= 7e6
    return (np.fft.fft(iq*np.hanning(N), axis=-1)/N)[:, keep], frequency[keep]


def run():
    if PROJECT != Path('F:/DPD_Strong_Weak').resolve():
        raise RuntimeError('Wrong project root')
    pdf = next(Path('F:/Zotero/storage/USPFK577').glob('*.pdf')).resolve(strict=True)
    source = [HERE/'preflight.py', HERE/'xia_reference.py', HERE/'执行合同.md',
              PROJECT/'统一模型代码/gates/g8/scenes.py', PROJECT/'统一模型代码/gates/g8/physics.py']
    identities = {str(p): sha(p) for p in source}
    pdf_hash = sha(pdf)
    out = HERE / ('audit_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    for p in source[:3]:
        (out/p.name).write_bytes(p.read_bytes())
    write(out/'preflight.json', {'sources': identities, 'pdf': str(pdf), 'pdf_sha256': pdf_hash,
                                 'python': platform.python_version(), 'numpy': np.__version__,
                                 'mode': 'SOURCE_AUDIT_AND_INPUT_TESTS_NOT_PERFORMANCE'})
    doc = fitz.open(pdf)
    doc[2].get_pixmap(matrix=fitz.Matrix(1.6, 1.6)).save(str(out/'source_page_2857.png'))
    rng = np.random.default_rng(1701)
    # Exact frequency-domain LOS algebra, not the final nonperiodic measurement.
    f = np.fft.fftfreq(64, 1/1e6)
    tau = relative_delays(POSITIONS[:1], STATIONS)[0]
    s = 1 + .2*rng.normal(size=64) + .2j*rng.normal(size=64)
    y = np.exp(-2j*np.pi*tau[:, None]*f[None]) * s[None]
    trial = np.vstack([POSITIONS, rng.uniform(-900, 900, (4, 2))])
    vector, _ = score(trial, STATIONS, y, f, 64)
    scalar = np.array([scalar_score(p, STATIONS, y, f, 64) for p in trial])
    err = float(np.max(abs(vector-scalar)))
    scaled, _ = score(trial, STATIONS, y*(2+3j), f, 64)
    zeros_guard = False
    try:
        cross_spectrum(np.zeros((5, 64), complex))
    except ValueError:
        zeros_guard = True
    particles = np.array([[0., 0.], [10., 0.], [20., 0.], [1200., 0.], [1210., 0.]])
    literal = literal_bandwidth(particles, particles[0])
    # A positive epsilon is ONLY a countercheck, not adopted in the algorithm.
    positive = literal_bandwidth(particles, particles[0], epsilon=.01)
    decoded = decode_literal(trial, vector)
    checks = {'eq9_vector_scalar': err < 1e-12,
              'exact_single_los_score': abs(vector[0]-1) < 1e-12,
              'phase_sign_prefers_true_over_registered_offsets': bool(vector[0] > max(vector[1:])),
              'global_complex_scale_invariant': bool(np.allclose(vector, scaled, rtol=1e-12)),
              'undefined_reference_rejected': zeros_guard,
              'cumulative_counts_monotone': all(t['increment'] >= 0 for t in literal['trace']),
              'epsilon_zero_cannot_stop_confirmed': literal['status'] == 'LITERAL_RULE_CANNOT_STOP',
              'positive_epsilon_countercheck': positive['status'] == 'STOP_RULE_MET',
              'no_fabricated_two_source_output': decoded['positions'] is None}
    los, mp, noise, truth = paired_components()
    np.savez_compressed(out/'paired_components.npz', los=los, reflection=mp, noise=noise,
                        **truth)
    input_hash = sha(out/'paired_components.npz')
    cases = []
    inputs = {}
    public = {'stations_m': STATIONS, 'fs_hz': FS, 'samples': N, 'symbol_rate_hz': RATE,
              'bandwidth_hz': BAND, 'processing_band_hz': [-7e6, 7e6],
              'window': 'hann', 'snr_definition': 'ensemble LOS station-average in nominal signal band',
              'B_inband_snr_db': 6., 'power_gap_db': 0., 'bounds_m': [-1000, 1000]}
    for name, ids in [('A', [0]), ('B', [1]), ('dual', [0, 1])]:
        for channel in ['los', 'mp']:
            iq = los[ids].sum(axis=0) + noise
            if channel == 'mp':
                iq += mp[ids].sum(axis=0)
            processed, frequency = preprocess(iq)
            key = name+'_'+channel
            inputs[key] = iq
            np.savez_compressed(out/(key+'_observation.npz'), iq=iq, y=processed, frequency=frequency)
            cases.append({'case': key, 'k': len(ids), 'selected_bins': len(frequency),
                          'finite': bool(np.isfinite(processed).all()),
                          'observation_sha256': sha(out/(key+'_observation.npz'))})
    checks.update({
        'six_prepared_inputs_finite': all(v['finite'] for v in cases),
        'LOS_source_removal_keeps_noise': bool(np.allclose(inputs['dual_los']-inputs['A_los'], los[1])),
        'MP_source_removal_keeps_noise': bool(np.allclose(inputs['dual_mp']-inputs['A_mp'], los[1]+mp[1])),
        'adding_MP_changes_only_reflection': bool(np.allclose(inputs['dual_mp']-inputs['dual_los'], mp.sum(0))),
        'positive_excess_delay': bool(np.all(truth['extra_delays'] > 0)),
        'same_public_band_all_cases': len(set(v['selected_bins'] for v in cases)) == 1,
        'input_identity_unchanged': sha(out/'paired_components.npz') == input_hash,
        'sources_unchanged': all(sha(Path(p)) == h for p, h in identities.items()),
        'paper_identity_unchanged': sha(pdf) == pdf_hash})
    result = {'status': 'XIA_LITERAL_DECODER_BLOCKED_PERFORMANCE_NOT_STARTED',
              'checks': checks, 'all_diagnostic_checks_pass': all(checks.values()),
              'score_max_abs_difference': err, 'literal_epsilon_zero': literal,
              'positive_epsilon_not_adopted': positive, 'decode_example': decoded,
              'cases': cases, 'public': public, 'input_sha256': input_hash,
              'localization_solves': 0, 'method_ranking_available': False}
    write(out/'result.json', result)
    if not all(checks.values()):
        raise RuntimeError('Diagnostic test failure; preserve this output')
    print('中文日志核验：公式与配对输入检查完成；原文双源解码未放行。', flush=True)
    print(json.dumps({'out': str(out), 'checks': len(checks), 'status': result['status']}, ensure_ascii=False))


if __name__ == '__main__':
    run()
