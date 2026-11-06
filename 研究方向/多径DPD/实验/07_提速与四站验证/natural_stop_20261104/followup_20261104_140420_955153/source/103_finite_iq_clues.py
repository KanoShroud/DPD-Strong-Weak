"""Finite-IQ lag extraction controls. No source labels enter extract()."""
from pathlib import Path
from datetime import datetime
import argparse
import json
import time

import numpy as np
from scipy.fft import fft, ifft, fftfreq, fftshift, ifftshift
from scipy.signal import find_peaks, correlate
from threadpoolctl import threadpool_limits

from geometry_association_gate import ROOT, SOURCE, STATIONS, PAIRS, C, B, sha, write, signature

FS = 50e6
N = 8192
RS = 10e6
BETA = .25
SPAN = 32
UPSAMPLE = 16
SEED = 20261029219
PROTOCOL = ROOT/'有限IQ时差提取协议.md'
TOLS = [.1, .25, .5]
COUNTS = [2, 6, 18]


def rrc(x):
    xx = np.asarray(x, dtype=float)
    zero = abs(xx) < 1e-10
    singular = abs(abs(xx)-1/(4*BETA)) < 1e-10
    safe = ~(zero | singular)
    y = np.empty_like(xx)
    z = xx[safe]
    y[safe] = (np.sin(np.pi*(1-BETA)*z)+4*BETA*z*np.cos(np.pi*(1+BETA)*z))/(np.pi*z*(1-(4*BETA*z)**2))
    y[zero] = 1+BETA*(4/np.pi-1)
    y[singular] = BETA/np.sqrt(2)*((1+2/np.pi)*np.sin(np.pi/(4*BETA))+(1-2/np.pi)*np.cos(np.pi/(4*BETA)))
    return y


def synthesize(bits, first, offset, times, span=SPAN):
    sample = (times-offset)*RS
    k = np.floor(sample).astype(int)[:, None]+np.arange(-span+1, span+1)[None]
    assert k.min() >= first and k.max() < first+len(bits)
    x = sample[:, None]-k
    return np.sum(bits[k-first]*rrc(x), axis=1)


def generate(case, layout, rep):
    rng = np.random.default_rng(SEED+10000*layout+rep)
    # Guards depend on public domain and allowed delay bounds, not source truth.
    maxdelay = np.sqrt(2)*6000/C + 40/B
    first = -int(np.ceil(maxdelay*RS))-2*SPAN-4
    last = int(np.ceil(N/FS*RS))+2*SPAN+4
    bits = rng.choice([-1., 1.], (2, last-first+1))
    offsets = rng.uniform(0., 1/RS, 2)
    t = np.arange(N)/FS
    los = np.zeros((2, 5, N), complex)
    reflections = np.zeros_like(los)
    distances = np.linalg.norm(STATIONS[:, None]-case['p'][None], axis=2)
    for q in range(2):
        for m in range(5):
            for path in range(3):
                delay = distances[m, q]/C+case['d'][m, q, path]/B
                value = case['g'][m, q, path]*synthesize(bits[q], first, offsets[q], t-delay)
                if path == 0:
                    los[q, m] = value
                else:
                    reflections[q, m] += value
    power = np.mean(abs(los)**2, axis=2)
    variance = np.mean(power[1])/(10**.6*(B/FS))
    noise = np.sqrt(variance/2)*(rng.normal(size=(5, N))+1j*rng.normal(size=(5, N)))
    los_snr_db = 10*np.log10(power/(variance*(B/FS)))
    return dict(los=los, reflected=reflections, noise=noise, bits=bits, first=first,
                offsets=offsets, variance=variance, los_snr_db=los_snr_db,
                nominal_average_snr_db=10*np.log10(np.mean(power, axis=1)/(variance*(B/FS))))


def peak_candidates(lag, curve):
    indices = list(find_peaks(curve)[0])
    if curve[0] >= curve[1]:
        indices.append(0)
    if curve[-1] >= curve[-2]:
        indices.append(len(curve)-1)
    indices = sorted(indices, key=lambda i: (-curve[i], lag[i]))[:max(COUNTS)]
    candidates, heights = [], []
    step = lag[1]-lag[0]
    for i in indices:
        displacement = 0.
        if 0 < i < len(curve)-1:
            a, b, c = curve[i-1:i+2]
            den = a-2*b+c
            if den < 0:
                displacement = float(np.clip(.5*(a-c)/den, -.5, .5))
        candidates.append(float(np.clip(lag[i]+displacement*step, lag[0], lag[-1])))
        heights.append(float(curve[i]))
    return dict(lag_s=candidates, heights=heights)


def extract(y, stations=STATIONS, save_curves=False):
    """Only IQ plus public sampling/geometry, no gains or source count/labels."""
    begin = time.perf_counter()
    n = y.shape[1]
    length = 2**int(np.ceil(np.log2(2*n-1)))
    freq = fftfreq(length, 1/FS)
    mask = abs(freq) <= B/2
    spectrum = fft(y, n=length, axis=1)
    long_length = length*UPSAMPLE
    all_lags = np.arange(-long_length//2, long_length//2)/(FS*UPSAMPLE)
    results = {'CC_ABS': {}, 'PHAT_ABS': {}}
    saved = {}
    for m, station_n in PAIRS:
        csd = spectrum[station_n]*spectrum[m].conj()
        denom = np.maximum(abs(csd), max(float(abs(csd).max()), 1e-30)*1e-12)
        bound = np.linalg.norm(stations[station_n]-stations[m])/C
        keep = abs(all_lags) <= bound
        lags = all_lags[keep]
        for method, weights in [('CC_ABS', csd*mask), ('PHAT_ABS', (csd/denom)*mask)]:
            padded = np.pad(fftshift(weights), ((long_length-length)//2,)*2)
            curve = abs(fftshift(ifft(ifftshift(padded))))[keep]*UPSAMPLE
            if method == 'CC_ABS':
                curve /= n
            results[method][f'{m}_{station_n}'] = peak_candidates(lags, curve)
            if save_curves:
                saved[f'{method}_{m}_{station_n}_lag_s'] = lags
                saved[f'{method}_{m}_{station_n}_curve'] = curve
    return results, saved, time.perf_counter()-begin


def metrics(extracted, truth, active):
    ranges = np.linalg.norm(STATIONS[:, None]-truth[None], axis=2)
    answer = {}
    for method, pairs in extracted.items():
        answer[method] = {}
        for count in COUNTS:
            errors = []
            for m, n in PAIRS:
                candidates = np.array(pairs[f'{m}_{n}']['lag_s'][:count])
                assert len(candidates) > 0
                target = (ranges[n, active]-ranges[m, active])/C
                errors.append(np.min(abs(candidates[:, None]-target[None]), axis=0)*C)
            errors = np.asarray(errors)
            tolerances = {}
            for tol in TOLS:
                hits = errors <= tol*C/B
                tolerances[str(tol)] = dict(pair_source_coverage=float(hits.mean()),
                                            hits_by_source=hits.sum(axis=0),
                                            reference_hits_by_source=hits[:4].sum(axis=0),
                                            all_reference_sources=bool(hits[:4].all()),
                                            all_pairs_sources=bool(hits.all()))
            answer[method][str(count)] = dict(nearest_lag_error_m=errors,
                                              nearest_lag_rmse_m=float(np.sqrt(np.mean(errors**2))),
                                              tolerances=tolerances)
    return answer


def selftest():
    checks = {}
    checks['rrc_even_and_singular'] = bool(np.allclose(rrc(np.arange(-5, 6)/4), rrc(np.arange(5, -6, -1)/4)) and np.isfinite(rrc([0, 1, -1])).all())
    rng = np.random.default_rng(901)
    bits = rng.choice([-1.,1.], 1000)
    t = np.arange(2048)/FS
    s32 = synthesize(bits, -200, .31/RS, t, 32)
    s64 = synthesize(bits, -200, .31/RS, t, 64)
    trunc = float(np.linalg.norm(s32-s64)/np.linalg.norm(s64))
    checks['rrc_truncation_below_0_002'] = trunc < .002
    x = rng.normal(size=128)+1j*rng.normal(size=128)
    y = rng.normal(size=128)+1j*rng.normal(size=128)
    padded = fftshift(ifft(fft(y,256)*fft(x,256).conj()))
    checks['linear_correlation_conjugation'] = bool(np.max(abs(padded[1:]-correlate(y,x,mode='full',method='direct'))) < 1e-11)
    impulse = np.zeros((5, 512), complex)
    for m in range(5):
        impulse[m, 100+11*m] = np.exp(.7j*m)
    extracted, _, _ = extract(impulse)
    checks['positive_lag_and_interp'] = all(abs(extracted[method]['0_1']['lag_s'][0]-11/FS) < 1e-12 for method in extracted)
    rotated, _, _ = extract(impulse*np.exp(1j*np.arange(5)*1.37)[:, None])
    checks['phase_invariant_primary_peak'] = all(abs(extracted[method]['0_1']['lag_s'][0]-rotated[method]['0_1']['lag_s'][0]) < 1e-12 for method in extracted)
    noise = (rng.normal(size=(5,N))+1j*rng.normal(size=(5,N)))/np.sqrt(2)
    nf = fft(noise, axis=1)/N
    empirical = np.sum(abs(nf[:, abs(fftfreq(N,1/FS)) <= B/2])**2,axis=1).mean()
    checks['band_noise_fft_scale'] = abs(empirical/(B/FS)-1) < .06
    assert all(checks.values()), checks
    return dict(checks=checks, truncation_relative_l2=trunc, empirical_unit_noise_band_power=empirical)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bases', type=int, default=20)
    args = parser.parse_args()
    test = selftest()
    old = json.loads((SOURCE/'result.json').read_text(encoding='utf-8'))
    cases = [r for r in old['rows'] if r['paths'] == 3][:args.bases]
    files = [Path(__file__), PROTOCOL, ROOT/'geometry_association_gate.py', SOURCE/'result.json']
    files += [SOURCE/(r['tag']+'.npz') for r in cases]
    hashes = {str(p.resolve(strict=True)): sha(p) for p in files}
    assert all(Path(p).is_relative_to(ROOT.resolve()) for p in hashes)
    out = ROOT/('finite_iq_clues_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    (out/'source').mkdir()
    for p in files[:3]:
        (out/'source'/p.name).write_bytes(p.read_bytes())
    write(out/'protocol.json', dict(kind='FINITE_IQ_LAG_EXTRACTION_NOT_LOCALIZATION',
                                   hashes=hashes, selftest=test, fs=FS, n=N, symbol_rate=RS,
                                   bandwidth=B, rolloff=BETA, span=SPAN, seed=SEED,
                                   no_hann=True, band_hz=[-B/2,B/2], counts=COUNTS, tolerances=TOLS,
                                   stations=STATIONS, planned_bases=args.bases))
    rows = []
    checks = dict(test['checks'])
    begin = time.perf_counter()
    print('OUTPUT', out, flush=True)
    for row in cases:
        base_tag = f'L{row["layout"]}_R{row["realization"]}'
        case = dict(np.load(SOURCE/(row['tag']+'.npz')))
        assert signature(case['y'], case['h'], case['s'], case['g'], case['d']) == row['identity']
        data = generate(case, row['layout'], row['realization'])
        assert abs(data['nominal_average_snr_db'][1]-6) < 1e-10
        np.savez_compressed(out/(base_tag+'_components.npz'), **data, truth=case['p'])
        los = data['los']
        multi = los+data['reflected']
        for paths, source in [('LOS', los), ('MP', multi)]:
            for label, active in [('A',[0]), ('B',[1]), ('AB',[0,1])]:
                clean = np.sum(source[active], axis=0)
                for noise_label, noise in [('clean', np.zeros_like(clean)), ('snr6', data['noise'])]:
                    iq = clean+noise
                    tag = f'{base_tag}_{paths}_{label}_{noise_label}'
                    save = row['realization'] == 0 and label == 'AB'
                    extracted, curves, seconds = extract(iq, save_curves=save)
                    metric = metrics(extracted, case['p'], active)
                    rec = dict(tag=tag, layout=row['layout'], realization=row['realization'],
                               paths=paths, signal=label, noise=noise_label, active_sources=active,
                               extraction=extracted, metrics=metric, seconds=seconds,
                               iq_sha=signature(iq), los_snr_db=data['los_snr_db'])
                    write(out/(tag+'.json'), rec)
                    if save:
                        np.savez_compressed(out/(tag+'_curves.npz'), **curves)
                    rows.append(rec)
        # Pairing and arbitrary fixed receiver phase checked on a mixed noisy input.
        mixed = multi.sum(axis=0)+data['noise']
        original, _, _ = extract(mixed)
        rotated, _, _ = extract(mixed*np.exp(1j*np.arange(5)*.73)[:, None])
        checks[base_tag+'/phase'] = all(np.allclose(original[method][key]['lag_s'], rotated[method][key]['lag_s'],atol=1e-12,rtol=0)
                                      for method in original for key in original[method])
        checks[base_tag+'/finite'] = bool(np.isfinite(mixed).all())
        checks[base_tag+'/paired_sum'] = bool(np.max(abs((multi[0]+data['noise'])+(multi[1]+data['noise'])-data['noise']-mixed)) < 1e-12)
        assert all(checks.values()), checks
        print(f'{base_tag}: {len(rows)} observations; phase/pairing PASS', flush=True)
    checks.update({p: sha(p) == h for p,h in hashes.items()})
    assert all(checks.values())
    write(out/'result.json', dict(rows=rows, checks=checks, completed=len(rows),
                                 seconds=time.perf_counter()-begin,
                                 kind='FINITE_IQ_LAG_EXTRACTION_NOT_LOCALIZATION'))
    print('COMPLETE', out, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
