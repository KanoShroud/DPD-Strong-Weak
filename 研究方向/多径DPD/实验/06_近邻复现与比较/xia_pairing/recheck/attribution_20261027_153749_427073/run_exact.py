"""Small exact-Eq4 controls. No original-paper performance claim."""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import xia_checked as xia

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from xia_reference import C, scalar_score  # noqa: E402
from los_2d import evaluate  # noqa: E402
from preflight import write, sha  # noqa: E402


def tests():
    p = ROOT.parent/'pair_20261027_143405_945211/A_los_XIA.json'
    st = json.loads(p.read_text(encoding='utf-8'))['output']['steps'][-1]
    v = np.array(st['scores'])
    t = xia.threshold(v)
    # Artificial histogram dominated by the sixth bin: true excess is retained.
    high = np.r_[0., np.full(20, .55), 1.]
    ph = np.array([[0., 0.], [10., 0.], [1200., 0.], [1210., 0.]])
    ph = np.vstack([ph, np.array([[3000.+i, 0.] for i in range(100)])])
    vv = np.r_[[10., 9., 9.5, 9.], np.zeros(100)]
    checks = {'endpoint_exact_max_retained': t['modal_bin'] == 5 and t['keep'][v.argmax()],
              'true_excess_not_capped': not xia.threshold(high)['keep'].any(),
              'approved_cluster_two_groups': xia.decode(ph, vv, 2, 400)['status'] == 'OK',
              'scale_invariance': np.array_equal(xia.threshold(v*1e5)['keep'], t['keep'])}
    return {k: bool(val) for k, val in checks.items()}


def main():
    source_input = ROOT.parent/'paper_scale_20261027_143232_712407/input.npz'
    old = json.loads((source_input.parent/'result.json').read_text(encoding='utf-8'))
    assert sha(source_input) == old['input_sha256']
    out = ROOT/('exact_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    files = [ROOT/'xia_checked.py', Path(__file__), ROOT/'执行协议.md',
             ROOT.parent/'xia_reference.py', ROOT.parent/'los_2d.py', ROOT.parent/'preflight.py']
    identity = {str(p): sha(p) for p in files}
    for p in files:
        (out/p.name).write_bytes(p.read_bytes())
    checks = tests()
    assert all(checks.values()), checks
    with np.load(source_input) as z:
        signal, gain, truth, stations = [z[k] for k in ['signal', 'gain', 'truth', 'stations']]
    fs, n = 1e6, 2000
    f = np.fft.fftfreq(n, 1/fs)
    s = np.fft.fft(signal[:, 0], axis=-1)/n/gain[:, 0, None]
    dist = np.linalg.norm(truth[:, None]-stations[None], axis=-1)
    tau = (dist-dist[:, :1])/C
    direct = gain[:, :, None]*s[:, None]*np.exp(-2j*np.pi*tau[:, :, None]*f)
    rng = np.random.default_rng(270271)
    delays = np.array([3, 8, 11, 13, 21])*1e-6
    a = 10**(np.array([-7., -15., -22., -24., -19.])/20)
    extra_gain = gain[:, :, None]*a*np.exp(1j*rng.uniform(-np.pi, np.pi, (2, 5, 5)))
    reflection = np.sum(extra_gain[:, :, :, None]*s[:, None, None]
                        *np.exp(-2j*np.pi*(tau[:, :, None]+delays)[:, :, :, None]*f), axis=2)
    y = direct[0]
    val = xia.scores(truth[:1], stations, y, f, n)[0]
    expected = np.prod(abs(gain[0]/gain[0, 0]))
    checks['exact_single_score'] = bool(np.isclose(val, expected, rtol=1e-12))
    checks['scalar_score_agrees'] = bool(np.isclose(val, scalar_score(truth[0], stations, y, f, n), rtol=1e-12))
    checks['roundtrip_fft'] = bool(np.allclose(np.fft.fft(np.fft.ifft(direct*n, axis=-1), axis=-1)/n, direct))
    checks['LOS_remove_source'] = bool(np.allclose(direct.sum(0)-direct[0], direct[1]))
    checks['no_zero_reference_bins'] = bool(all(np.all(abs(z[0])>0) for z in [direct[0], direct[1], direct.sum(0)]))
    assert all(checks.values()), checks
    write(out/'preflight.json', {'checks': checks, 'sources': identity, 'input': str(source_input),
                                'input_sha256': sha(source_input), 'public': {'fs': fs, 'n': n,
                                'bound': 3000, 'particles': 1000, 'h': 400, 'std': 10,
                                'window': 'none', 'frequency': 'all signed FFT bins',
                                'signal_model': 'exact Eq4 using saved BPSK reference spectrum',
                                'noise': 'zero', 'reflection_delays_s': delays.tolist(),
                                'reflection_db': [-7, -15, -22, -24, -19]}})
    np.savez_compressed(out/'components.npz', direct=direct, reflection=reflection, frequency=f,
                        stations=stations, truth=truth, source=s, gain=gain, extra_gain=extra_gain)
    rows = []
    for label, ids in [('A', [0]), ('B', [1]), ('dual', [0, 1])]:
        for channel in ['los', 'mp']:
            obs = direct[ids].sum(0)+(reflection[ids].sum(0) if channel == 'mp' else 0)
            result = xia.solve(obs, f, stations, len(ids), n, 3000)
            record = {'case': label+'_'+channel, 'output': result,
                      'evaluation': evaluate(result['positions'], truth[ids]),
                      'observation_sha': hashlib.sha256(obs.tobytes()).hexdigest()}
            write(out/(record['case']+'.json'), record)
            compact = {k: v for k, v in record.items() if k != 'output'}
            compact.update(status=result['status'], seconds=result['seconds'])
            rows.append(compact)
            print(json.dumps(compact, ensure_ascii=False), flush=True)
    write(out/'result.json', {'rows': rows, 'checks': checks,
                              'sources_unchanged': all(sha(Path(p)) == h for p,h in identity.items()),
                              'components_sha': sha(out/'components.npz'),
                              'input_unchanged': sha(source_input) == old['input_sha256']})
    print(str(out), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
