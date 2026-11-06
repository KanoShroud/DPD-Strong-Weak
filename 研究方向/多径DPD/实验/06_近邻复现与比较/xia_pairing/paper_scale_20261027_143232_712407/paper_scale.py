"""Original numerical scale positive controls, not a Figure-2 reproduction."""
from datetime import datetime
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
import inclusive_solver as xia
from los_2d import evaluate
from preflight import rrc, sha, write
from xia_reference import C

ROOT = Path(__file__).resolve().parent


def main():
    out = ROOT/('paper_scale_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    rng = np.random.default_rng(270201)
    stations = np.array([[0., 0.], [2500., 2400.], [-2200., 2300.],
                         [-2400., -2500.], [2300., -2200.]])
    truth = np.array([[2000., 1000.], [-1000., -2000.]])
    fs, rate, n = 1e6, 200e3, 2000
    distance = np.linalg.norm(truth[:, None]-stations[None], axis=-1)
    # Paper Eq17; no noise here, so an unreported transmit/noise calibration
    # cannot masquerade as a performance loss. Unknown phase convention disclosed.
    shadow = rng.normal(0, 5, (2, 5))
    path_loss_db = 42.6+20*np.log10(900.)+26*np.log10(distance/1000)+shadow
    gain = 10**(-path_loss_db/20)*np.exp(1j*rng.uniform(-np.pi, np.pi, (2, 5)))
    signal = np.zeros((2, 5, n), complex)
    for q in range(2):
        bits = rng.choice([-1., 1.], int(n*rate/fs)+512)
        timing = rng.random()
        phase = np.exp(1j*rng.uniform(-np.pi, np.pi))
        for m in range(5):
            u = (np.arange(n)/fs-distance[q, m]/C)*rate-timing
            base = np.floor(u).astype(int)
            wave = np.zeros(n)
            for offset in range(-10, 11):
                symbols = base+offset
                wave += bits[symbols+256]*rrc(u-symbols)
            signal[q, m] = gain[q, m]*phase*wave
    cfg = {'scope': 'NOISELESS_LOS_ORIGINAL_NUMERICAL_SCALE_NOT_FULL_REPRODUCTION',
           'fs_hz': fs, 'symbol_rate_hz': rate, 'n': n, 'particles': 1000,
           'bound_m': 3000, 'delta_h_m': 400, 'resampling_covariance_m2': 100,
           'rounds': 3, 'filter': 'all FFT bins, rectangular record as Eq9',
           'unspecified_choices': 'fixed public geometry, RRC beta=.25 span20, independent constant phases',
           'gain_model': 'Eq17 COST231 with independent 5dB shadow, equal transmit power',
           'algorithm_source': {str(ROOT/name): sha(ROOT/name) for name in
                                ['paper_scale.py', 'inclusive_solver.py', 'xia_reference.py']}}
    write(out/'preflight.json', cfg)
    for name in ['paper_scale.py', 'inclusive_solver.py', 'xia_reference.py']:
        (out/name).write_bytes((ROOT/name).read_bytes())
    np.savez_compressed(out/'input.npz', signal=signal, stations=stations, truth=truth,
                        path_loss_db=path_loss_db, gain=gain)
    f = np.fft.fftfreq(n, 1/fs)
    rows = []
    start = time.perf_counter()
    for name, ids in [('A', [0]), ('B', [1]), ('dual', [0, 1])]:
        y = np.fft.fft(signal[ids].sum(axis=0), axis=-1)/n
        result = xia.solve(y, f, stations, len(ids), n, bound=3000, seed=2701)
        row = {'case': name, 'output': result, 'evaluation': evaluate(result['positions'], truth[ids])}
        write(out/(name+'.json'), row)
        rows.append({'case': name, 'status': result['status'], 'seconds': result['seconds'],
                     **row['evaluation']})
    write(out/'result.json', {'rows': rows, 'seconds': time.perf_counter()-start,
                              'input_sha256': sha(out/'input.npz'),
                              'source_unchanged': all(sha(Path(p)) == h for p,h in cfg['algorithm_source'].items())})
    print('原文尺度控制完成：'+str(out), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
