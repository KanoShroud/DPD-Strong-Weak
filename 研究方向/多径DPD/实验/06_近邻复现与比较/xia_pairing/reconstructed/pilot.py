"""Frozen, source-disclosed Xia reconstruction. No outcome-driven tuning."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'recheck'))
import xia_checked as xia  # noqa: E402

FS, N, SPS, C = 1e6, 2000, 5, 299792458.
SEED = 2026102707
DT = np.array([0., 3., 8., 11., 13., 21.])*1e-6
DB = np.array([0., -7., -15., -22., -24., -19.])


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(path, value):
    def cast(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    default=cast, allow_nan=False), encoding='utf-8')


def generate(index):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, index]))
    stations = np.vstack([np.zeros(2), rng.uniform(-3000, 3000, (4, 2))])
    truth = rng.uniform(-3000, 3000, (2, 2))
    d = np.linalg.norm(truth[:, None]-stations[None], axis=-1)
    if np.any(d == 0): raise ValueError('Zero distance; do not silently redraw')
    loss = 42.6+20*np.log10(900.)+26*np.log10(d/1000)+rng.normal(0, 5, (2, 5))
    raw_gain = 10**(-loss/20)*np.exp(1j*rng.uniform(-np.pi, np.pi, (2, 5)))
    scale = np.sqrt(np.mean(abs(raw_gain[0])**2))
    gain = raw_gain/scale
    phase = rng.uniform(-np.pi, np.pi, (2, 5, 6))
    phase[:, :, 0] = 0
    taps = gain[:, :, None]*10**(DB/20)*np.exp(1j*phase)
    bits = rng.choice([-1., 1.], (2, N//SPS))
    offsets = rng.integers(0, SPS, 2)
    waves = np.array([np.roll(np.repeat(bits[q], SPS), offsets[q]) for q in range(2)])
    spectrum = np.fft.fft(waves, axis=-1)/N
    f = np.fft.fftfreq(N, 1/FS)
    tau = (d-d[:, :1])/C
    component = taps[:, :, :, None]*spectrum[:, None, None]*np.exp(
        -2j*np.pi*(tau[:, :, None]+DT)[..., None]*f)
    noise_iq = (rng.normal(size=(5, N))+1j*rng.normal(size=(5, N)))/np.sqrt(2)
    noise_f = np.fft.fft(noise_iq, axis=-1)/N
    return dict(stations=stations, truth=truth, loss_db=loss, gain=gain, raw_gain=raw_gain,
                common_scale=scale, bits=bits, offsets=offsets, waves=waves, spectrum=spectrum,
                taps=taps, component=component, noise_iq=noise_iq, noise_f=noise_f, frequency=f)


def observe(z, snr, sir, paths):
    amplitude = np.array([1., 10**(-sir/20)])
    per_source = z['component'][:, :, :paths].sum(axis=2)*amplitude[:, None, None]
    sigma2 = 10**(-snr/10)
    y = per_source.sum(0)+np.sqrt(sigma2)*z['noise_f']
    return y, per_source, sigma2


def evaluate(pred, truth):
    if pred is None:
        return {'complete': False, 'error_m': None, 'rmse_m': None, 'both20': False,
                'first_peak_target_error_m': None}
    p = np.asarray(pred)
    assert p.shape == truth.shape == (2, 2)
    cost = np.sum((p[:, None]-truth[None])**2, axis=-1)
    i, j = linear_sum_assignment(cost)
    error = np.empty(2)
    error[j] = np.sqrt(cost[i, j])
    return {'complete': True, 'error_m': error.tolist(), 'rmse_m': float(np.sqrt(np.mean(error**2))),
            'both20': bool(np.all(error <= 20)),
            'first_peak_target_error_m': float(np.linalg.norm(p[0]-truth[0]))}


def select_output(result):
    complete = [s for s in result['steps'] if s['output'].get('positions') is not None]
    if not complete:
        return None, None, False
    step = complete[-1]
    return step['output']['positions'], step['round'], result['positions'] is None


def tests():
    z = generate(9000)
    y, source, sigma = observe(z, 5., 3., 3)
    y_sir, source_sir, _ = observe(z, 5., -3., 3)
    y_los, source_los, _ = observe(z, 5., 3., 1)
    checks = {'rectpulse_unit_power': np.all(np.mean(abs(z['waves'])**2, axis=1) == 1),
              'target_LOS_average_power_one': np.isclose(np.mean(abs(z['gain'][0])**2), 1),
              'relative_gain_common_scaling': np.allclose(z['gain']*z['common_scale'], z['raw_gain']),
              'sir_only_changes_source2': np.allclose(source_sir[0], source[0]),
              'sir_noise_unchanged': np.allclose(y-source.sum(0), y_sir-source_sir.sum(0)),
              'paths_noise_unchanged': np.allclose(y-source.sum(0), y_los-source_los.sum(0)),
              'exact_fft_roundtrip': np.allclose(np.fft.fft(np.fft.ifft(y*N, axis=-1), axis=-1)/N, y),
              'noise_parseval': np.isclose(np.mean(abs(z['noise_iq'])**2), N*np.mean(abs(z['noise_f'])**2)),
              'noise_variance_unit_control': abs(np.mean(abs(z['noise_iq'])**2)-1) < .05,
              'snr_definition': np.isclose(10*np.log10(1/sigma), 5),
              'nonzero_noisy_reference': bool(np.all(abs(y[0]) > 0)),
              'one_to_one_swap': evaluate(z['truth'][::-1], z['truth'])['rmse_m'] == 0}
    # Translation and station relabeling preserve physical differences; reference
    # is kept at index 0. This checks the scoring axis, not performance.
    p = z['truth']
    v = xia.scores(p, z['stations'], y, z['frequency'], N)
    v2 = xia.scores(p+100., z['stations']+100., y, z['frequency'], N)
    permutation = [0, 3, 1, 4, 2]
    vp = xia.scores(p, z['stations'][permutation], y[permutation], z['frequency'], N)
    checks['score_translation'] = np.allclose(v, v2, rtol=1e-10, atol=0)
    checks['score_station_permutation'] = np.allclose(v, vp, rtol=1e-10, atol=0)
    fake = {'positions': None, 'steps': [{'round': 1, 'output': {'positions': [[1,2],[3,4]]}},
                                         {'round': 2, 'output': {'positions': None}}]}
    selected, round_id, kept = select_output(fake)
    checks['retain_last_complete_without_truth'] = selected == [[1,2],[3,4]] and round_id == 1 and kept
    return {k: bool(v) for k,v in checks.items()}


def main(limit=None):
    out = ROOT/('pilot_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    source_files = [Path(__file__), ROOT/'实现约定.md', ROOT.parent/'recheck/xia_checked.py',
                    ROOT.parent/'xia_reference.py']
    identities = {str(p.resolve(strict=True)): sha(p) for p in source_files}
    for p in source_files: (out/p.name).write_bytes(p.read_bytes())
    check = tests()
    write(out/'checks.json', check)
    assert all(check.values()), check
    cases = [(snr, sir, 3) for snr in [-10., 5., 20.] for sir in [-3., 3.]]+[(5., 3., 1)]
    jobs = [(i, snr, sir, paths) for i in range(12) for snr, sir, paths in cases]
    if limit is not None: jobs = jobs[:limit]
    pdf = next(Path('F:/Zotero/storage/USPFK577').glob('*.pdf')).resolve(strict=True)
    write(out/'preflight.json', {'sources': identities, 'paper_sha': sha(pdf),
                                'seed': SEED, 'jobs': jobs, 'limit': limit,
                                'scope': 'DISCLOSED_RECONSTRUCTION_PILOT_NOT_AUTHOR_CODE',
                                'snr': 'five-station target LOS average received SNR',
                                'sir': 'transmit power ratio', 'threads': 2,
                                'paths_G': 'total including LOS', 'source_labels_only_for_evaluation': True})
    rows = []
    cached = {}
    started = time.perf_counter()
    for count, (i, snr, sir, paths) in enumerate(jobs, 1):
        if i not in cached:
            z = generate(i)
            cache = out/f'input_{i:02d}.npz'
            np.savez_compressed(cache, **z)
            cached = {i: (z, sha(cache))}
        z, data_sha = cached[i]
        y, components, sigma2 = observe(z, snr, sir, paths)
        obs_sha = hashlib.sha256(y.tobytes()).hexdigest()
        seed = int(np.random.SeedSequence([SEED, 700, i]).generate_state(1)[0])
        result = xia.solve(y, z['frequency'], z['stations'], 2, N, 3000, seed=seed)
        pred, round_id, retained = select_output(result)
        loss = evaluate(pred, z['truth'])
        target_los = np.mean(abs(z['gain'][0])**2)
        other_los = np.mean(abs(z['gain'][1])**2)*10**(-sir/10)
        row = {'realization': i, 'snr_db': snr, 'sir_db': sir, 'paths': paths,
               'input_sha': data_sha, 'observation_sha': obs_sha, 'particle_seed': seed,
               'evaluation': loss, 'seconds': result['seconds'], 'stop_reason': result['status'],
               'selected_round': round_id, 'retained_previous': retained,
               'terminal_complete': result['positions'] is not None,
               'actual_los_received_sir_db': float(10*np.log10(target_los/other_los)),
               'target_los_snr_db': float(10*np.log10(target_los/sigma2)),
               'actual_total_target_snr_db': float(10*np.log10(N*np.mean(abs(components[0])**2)/sigma2)),
               'realized_noise_power': float(sigma2*np.mean(abs(z['noise_iq'])**2))}
        name = f'r{i:02d}_snr{snr:g}_sir{sir:g}_G{paths}'
        write(out/(name+'.json'), {**row, 'output': result, 'selected_positions': pred})
        rows.append(row)
        print(f'原文重建短测 {count}/{len(jobs)} | 已输出{sum(r["evaluation"]["complete"] for r in rows)} | '
              f'累计{time.perf_counter()-started:.1f}s | {name} {result["status"]}', flush=True)
    write(out/'result.json', {'rows': rows, 'seconds': time.perf_counter()-started,
                              'checks': check, 'sources_unchanged': all(sha(p)==h for p,h in identities.items()),
                              'paper_unchanged': sha(pdf)==read_paper_sha(out),
                              'new_solves': len(rows), 'scope': '12_REALIZATIONS_PILOT_NOT_1000_MC'})
    print(str(out), flush=True)


def read_paper_sha(out):
    return json.loads((out/'preflight.json').read_text(encoding='utf-8'))['paper_sha']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    with threadpool_limits(limits=2): main(args.limit)
