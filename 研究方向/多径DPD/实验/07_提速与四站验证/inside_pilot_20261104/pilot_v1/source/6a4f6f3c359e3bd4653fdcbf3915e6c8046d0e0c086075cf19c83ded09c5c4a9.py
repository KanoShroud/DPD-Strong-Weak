"""Pre-run numerical, physical-unit, and four-station adaptation gates."""
import json
import time
import numpy as np
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits
import scene as cfg
import adapters as ad


def main():
    start = time.perf_counter()
    contract = cfg.contract()
    expected = json.loads(json.dumps(contract, default=cfg.clean))
    if (cfg.HERE/'contract.json').exists():
        assert cfg.read(cfg.HERE/'contract.json') == expected
    else:
        cfg.write(cfg.HERE/'contract.json', contract)
    identity = ad.source_hashes()
    engine = ad.Engine()
    bank = cfg.make_bank(contract['pilot_layouts'][0], 'pilot', 0, ad.wave.synthesize, ad.wave.rrc)
    iq, powers = cfg.observation(bank, cfg.CONDITIONS[-1])
    checks = {}
    checks['geometry_all_registered'] = all(cfg.geometry_ok(v['p']) for phase in ('pilot', 'formal') for v in contract[phase+'_layouts'])
    checks['independent_layouts'] = len({cfg.digest(np.asarray(v['p'])) for phase in ('pilot', 'formal') for v in contract[phase+'_layouts']}) == 16
    checks['formal_count'] = len(contract['formal_layouts'])*20*len(cfg.CONDITIONS) == 3120
    checks['pilot_count'] = len(contract['pilot_layouts'])*len(cfg.CONDITIONS) == 52
    checks['2p4GHz'] = cfg.FC == 2.4e9
    checks['iq_shape_finite'] = iq.shape == (4, 8192) and np.isfinite(iq).all()
    checks['rho_and_individual_power'] = np.allclose(powers.sum(axis=2), 1.5) and bool(np.all(powers < 1))
    checks['nominal_SNR'] = bool(np.allclose(10*np.log10(np.mean(abs(bank['gains'])**2, axis=0)/(cfg.NOISE_PSD*cfg.B)), 6., atol=1e-12))
    checks['ensemble_pulse_unit'] = abs(bank['pulse_norm2']-1) < .005
    freq = np.fft.fftfreq(cfg.N, 1/cfg.FS); mask = abs(freq) <= cfg.B/2
    noise_fft = np.fft.fft(bank['noise'], axis=-1, norm='ortho')
    checks['physical_noise_total'] = abs(np.mean(abs(bank['noise'])**2)/cfg.NOISE_VARIANCE-1) < .03
    checks['physical_noise_band'] = abs(np.mean(abs(noise_fft[:, mask])**2)/cfg.NOISE_VARIANCE-1) < .05
    checks['physical_noise_outband'] = abs(np.mean(abs(noise_fft[:, ~mask])**2)/cfg.NOISE_VARIANCE-1) < .03
    f, y, n, sigma = ad.b.fc.cal.spectral(iq)
    checks['frequencies'] = len(f) == 2049 and y.shape == (2049, 4)
    x = np.r_[np.array([3600., 1100., -700., -1300.])/ad.b.ref.UNIT,
              np.ones(6), np.full(6, .1), .1, .04, 3.3, -.15, .1, 8.1]
    paths = [(1, 0), (3, 1)]
    r, j = engine.evaluate(x, f[::59], y[::59], paths)
    eps = 1e-5
    fd = np.column_stack([(engine.evaluate(x+eps*d, f[::59], y[::59], paths)[0]-
                           engine.evaluate(x-eps*d, f[::59], y[::59], paths)[0])/(2*eps) for d in np.eye(len(x))])
    fd_error = float(np.linalg.norm(fd-j)/np.linalg.norm(j))
    checks['four_station_Jacobian'] = fd_error < 3e-6
    value, gradient = engine.objective(x, f[::59], y[::59], paths)
    checks['objective_gradient'] = np.isclose(value, .5*r@r) and np.allclose(gradient, j.T@r, atol=1e-10)
    scaled = engine.objective(x, f[::59], y[::59]*1e6, paths)
    checks['physical_scale_objective'] = np.isclose(value, scaled[0], rtol=1e-12) and np.allclose(gradient, scaled[1], rtol=1e-10, atol=1e-12)
    checks['position_domain'] = np.array_equal(engine.ref.project(x, n)[:4], x[:4])
    zz = x.copy(); zz[0] = 6000/ad.b.ref.UNIT
    checks['projection_4500'] = engine.ref.project(zz, n)[0] == 4500/ad.b.ref.UNIT
    rng = np.random.default_rng(2026110427)
    checks['convex_projection'] = True
    for _ in range(30):
        z = rng.normal(size=len(x))*100
        p = engine.ref.project(z, n); q = engine.ref.project(rng.normal(size=len(x))*100, n)
        checks['convex_projection'] &= bool((z-p)@(q-p) < 1e-7)
    cost, candidates, meta = engine.search(x, paths, f, y, n, mask, sigma, 1)
    _, scaled_candidates, scaled_meta = engine.search(x, paths, f, y*1e6, n, mask, sigma*1e12, 1)
    checks['FFT_direct_search'] = all(meta['checks'].values()) and all(scaled_meta['checks'].values())
    checks['four_station_search_count'] = len(candidates) == 8*(8192-1)
    checks['search_scale_invariance'] = np.allclose(candidates[:, 4], scaled_candidates[:, 4], atol=1e-7, rtol=1e-7)
    p = np.array([[3600., 1100.], [-700., -1300.]])
    lags = engine.predicted(p)[0].T
    roots = engine.triples(lags)
    checks['outside_noiseless_geometric_root'] = bool(np.all(np.min(np.linalg.norm(roots[:, None]-p[None], axis=2), axis=0) < 1e-4))
    # Exact LOS model; known start is only a formula/optimizer unit control.
    xt = x[:16].copy()
    hh = engine.core(xt, f, y, [])[0]
    symbols = rng.normal(size=(len(f), 2))+1j*rng.normal(size=(len(f), 2))
    exact_y = (hh@symbols[:, :, None])[:, :, 0]
    initial = xt.copy(); initial[:4] += np.array([.02, -.01, -.02, .01])
    xx, state = engine.los_fit(initial, f, exact_y, n)
    checks['exact_noiseless_LOS_local_control'] = state['stationary'] and state['cost'] < 1e-10 and np.max(abs(xx[:4]-xt[:4]))*ad.b.ref.UNIT < .01
    peaks = engine.extract(iq)[0]['CC_ABS']
    bigger = engine.extract(iq*1e6)[0]['CC_ABS']
    checks['CC_scale_invariance'] = all(np.allclose(peaks[k]['lag_s'], bigger[k]['lag_s'], rtol=0, atol=1e-13) for k in peaks)
    kernel = ad.kernel_module.CrossSpectrumKernel(y.T, f, n, cfg.FS)
    pp = rng.uniform(-cfg.BOUND, cfg.BOUND, (32, 2))
    direct = ad.xia.scores(pp, cfg.STATIONS, y.T, f, n)
    fast = kernel.scores(pp, cfg.STATIONS)
    checks['Xia_four_station_direct'] = np.allclose(fast, direct, rtol=1e-9, atol=1e-13*max(direct))
    bigkernel = ad.kernel_module.CrossSpectrumKernel(y.T*1e6, f, n, cfg.FS)
    checks['Xia_scale_invariance'] = np.allclose(fast, bigkernel.scores(pp, cfg.STATIONS), rtol=1e-9, atol=1e-13*max(direct))
    seed = 2026110428
    fastflow = ad.xia_solve(iq, seed, particles=1000)
    rawflow = ad.xia.solve(y.T, f, cfg.STATIONS, 2, n, cfg.BOUND, seed=seed, particles=1000, delta_h=400., std=10.)
    checks['Xia_flow_exact'] = fastflow['p'] == rawflow['positions'] and len(fastflow['steps']) == len(rawflow['steps'])
    for a, z in zip(fastflow['steps'], rawflow['steps']):
        checks['Xia_flow_exact'] &= a['points_sha'] == cfg.digest(np.asarray(z['points']))
        for key in ('status', 'positions', 'retained', 'modal_bin', 'histogram', 'trace'):
            checks['Xia_flow_exact'] &= a['output'].get(key) == z['output'].get(key)
        for key in ('threshold', 'minimum', 'maximum'):
            if key in a['output']:
                checks['Xia_flow_exact'] &= bool(np.isclose(a['output'][key], z['output'][key], rtol=1e-10, atol=1e-13*max(direct)))
    # Independent full LP versus sparse edge-pruned implementation at capacity6.
    co = rng.uniform(0, 3, (8, 6)); nn, jj = co.shape; width = nn*jj+nn+jj
    ae = np.zeros((nn+1, width)); au = np.zeros((jj+nn*jj, width))
    for i in range(nn):
        ae[i, i*jj:(i+1)*jj] = 1; ae[i, nn*jj+i] = 1
    ae[-1, nn*jj+nn:] = 1
    for k in range(jj):
        au[k, k:nn*jj:jj] = 1
    for i in range(nn):
        for k in range(jj):
            au[jj+i*jj+k, i*jj+k] = 1; au[jj+i*jj+k, nn*jj+nn+k] = -1
    full = linprog(np.r_[co.ravel(), np.ones(nn+jj)], A_eq=ae, b_eq=np.r_[np.ones(nn), 2],
                   A_ub=au, b_ub=np.r_[np.full(jj, 6), np.zeros(nn*jj)], bounds=(0, 1), method='highs')
    _, _, _, small = ad.flood.pruned_lp(co, capacity=6, known_k=2)
    checks['Flood_capacity6_dense'] = full.success and abs(full.fun-small['objective']) < 1e-8
    checks['source_identity'] = all(cfg.sha(p) == h for p, h in identity.items())
    result = dict(passed=all(checks.values()), checks=checks, FD_relative=fd_error,
                  hashes=identity, seconds=time.perf_counter()-start, xia_design=ad.xia_design(),
                  physical_noise_band_dbm=10*np.log10(cfg.NOISE_PSD*cfg.B*1000),
                  actual_snr_db=bank['realized_los_snr_db'], scope='numeric and generator gates, not performance evidence')
    cfg.write(cfg.HERE/'correctness.json', result)
    cfg.write(cfg.HERE/f'correctness_attempt_{time.time_ns()}.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'hashes'}, default=cfg.clean, ensure_ascii=False, indent=2))
    assert result['passed'], {k: v for k, v in checks.items() if not v}


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
