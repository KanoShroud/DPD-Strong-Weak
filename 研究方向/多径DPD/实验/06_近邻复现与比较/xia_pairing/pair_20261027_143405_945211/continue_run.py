"""Authorized small control and paired comparison, no truth inside solvers."""
import argparse
import hashlib
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize._numdiff import approx_derivative
from threadpoolctl import threadpool_limits

import inclusive_solver as xia
import los_2d as los
from preflight import STATIONS, N, preprocess, write, sha, PROJECT
from xia_reference import relative_delays

ROOT = Path(__file__).resolve().parent
INPUT = ROOT/'audit_20261027_135521_131734'


def tests():
    points = np.array([[0., 0.], [10., 0.], [20., 0.], [1200., 0.], [1210., 0.]])
    # Add low-valued points to exercise threshold and cluster on high values.
    p = np.vstack([points, np.array([[4000.+i, 0.] for i in range(100)])])
    v = np.r_[[10., 9., 8., 9.5, 9.], np.zeros(100)]
    decoded = xia.decode(p, v, 2, 400)
    empty = xia.decode(p[:3], np.array([10., 9., 8.]), 2, 400)
    rng = np.random.default_rng(27019)
    truth = np.array([[-260., -180.], [230., 120.]])
    f = np.fft.fftfreq(64, 1/50e6)
    tau = relative_delays(truth, STATIONS)
    gain = rng.normal(size=(2, 5))+1j*rng.normal(size=(2, 5))
    gain /= gain[:, :1]
    source = rng.normal(size=(2, 64))+1j*rng.normal(size=(2, 64))
    response = gain[:, :, None]*np.exp(-2j*np.pi*tau[:, :, None]*f[None, None])
    y = np.sum(response*source[:, None], axis=0)
    model = los.Model2D(y, f, STATIONS)
    idx = los.columns(2, 5)
    vector = los.base.pack(los.xyz(truth), gain)[idx]
    exact = model.original.value_grad(los.expand(vector, 2, 5), 2)[0]
    vector[:4] += [.003, -.002, -.004, .005]
    full = los.expand(vector, 2, 5)
    r, jac = model.original.residual_jac(full, 2)
    jac = jac[:, idx]
    numeric = approx_derivative(lambda x: model.original.residual_jac(los.expand(x, 2, 5), 2)[0], vector)
    error = np.linalg.norm(jac-numeric)/np.linalg.norm(numeric)
    value, grad = model.original.value_grad(full, 2)
    checks = {'inclusive_finite_two_clusters': decoded['status'] == 'OK',
              'inclusive_empty_cluster_no_fallback': empty['positions'] is None,
              'noiseless_LS_exact_algebra': exact < 1e-20,
              '2d_residual_jacobian': error < 1e-5,
              '2d_gradient_consistency': np.max(abs(2*jac.T@r-grad[idx])) < 1e-10,
              'coordinate_source_swap_evaluation': los.evaluate(truth[::-1], truth)['rmse_m'] == 0,
              'value_residual_agreement': abs(value-r@r) < 1e-12}
    return {'checks': {k: bool(v) for k, v in checks.items()},
            'all_pass': bool(all(checks.values())), 'jac_relative_error': float(error)}


def run(stage):
    out = ROOT/(stage+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    source = [ROOT/name for name in ['inclusive_solver.py', 'los_2d.py', 'continue_run.py',
                                      'preflight.py', 'xia_reference.py', '继续执行协议.md']]
    source += [los.BASE_PATH]
    identity = {str(p): sha(p) for p in source}
    for p in source:
        (out/p.name).write_bytes(p.read_bytes())
    original = json.loads((INPUT/'result.json').read_text(encoding='utf-8'))
    path = INPUT/'paired_components.npz'
    assert sha(path) == original['input_sha256']
    pdf = next(Path('F:/Zotero/storage/USPFK577').glob('*.pdf'))
    write(out/'preflight.json', {'source': identity, 'input': str(path), 'input_sha256': sha(path),
                                'pdf': str(pdf), 'pdf_sha256': sha(pdf),
                                'python': platform.python_version(), 'numpy': np.__version__,
                                'scipy': scipy.__version__, 'project': str(PROJECT), 'threads': 2})
    test = tests()
    write(out/'tests.json', test)
    if not test['all_pass']:
        raise RuntimeError('Preserved failed numerical tests')
    if stage == 'tests':
        print(json.dumps({'out': str(out), **test}, ensure_ascii=False))
        return
    with np.load(path) as data:
        direct, reflection, noise = data['los'], data['reflection'], data['noise']
        truth = data['positions']
    jobs = [(name+'_'+ch, ids, ch) for name, ids in [('A', [0]), ('B', [1]), ('dual', [0, 1])]
            for ch in (['los'] if stage == 'control' else ['los', 'mp'])]
    rows = []
    started = time.perf_counter()
    for j, (name, ids, channel) in enumerate(jobs):
        iq = direct[ids].sum(axis=0)
        if stage != 'control':
            iq += noise
        if channel == 'mp':
            iq += reflection[ids].sum(axis=0)
        y, frequency = preprocess(iq)
        observation_sha = hashlib.sha256(iq.tobytes()).hexdigest()
        np.savez_compressed(out/(name+'_input.npz'), iq=iq, y=y, frequency=frequency)
        methods = ['XIA', 'LOS_LS'] if j % 2 == 0 else ['LOS_LS', 'XIA']
        for method in methods:
            if method == 'XIA':
                output = xia.solve(y, frequency, STATIONS, len(ids), N)
            else:
                output = los.solve(y, frequency, STATIONS, len(ids))
            row = {'stage': stage, 'case': name, 'method': method, 'k': len(ids),
                   'observation_sha256': observation_sha, 'output': output,
                   'evaluation': los.evaluate(output['positions'], truth[ids])}
            write(out/(name+'_'+method+'.json'), row)
            rows.append(row)
            print(json.dumps({'case': name, 'method': method, 'seconds': output['seconds'],
                              'status': output['status'], **row['evaluation']}, ensure_ascii=False), flush=True)
    write(out/'result.json', {'stage': stage, 'count': len(rows), 'seconds': time.perf_counter()-started,
                              'source_unchanged': all(sha(Path(p)) == h for p, h in identity.items()),
                              'input_unchanged': sha(path) == original['input_sha256'],
                              'rows': [{key: value for key, value in r.items() if key != 'output'} |
                                       {'seconds': r['output']['seconds'], 'status': r['output']['status']}
                                       for r in rows]})
    print('完成：'+str(out), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['tests', 'control', 'pair'])
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.stage)
