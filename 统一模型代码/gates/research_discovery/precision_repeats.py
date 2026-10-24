"""G11-P4: fixed-geometry, paired random repeats; RMSE before threshold success.

Frozen numerical solvers are reused. Truth is passed only to offline evaluation
and a separately timed local diagnostic, never to the common blind frontend.
"""
import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.research_discovery.band_validation import (
    BASE, INPUT, FS, BAND, RIDGE, Family, Problem, digest, write,
)
from 统一模型代码.gates.research_discovery.slepian_reference import reconstruct, optimize
from 统一模型代码.gates.g10.r1.scenes import make
from 统一模型代码.gates.g10.r3.solvers import Model
from 统一模型代码.gates.g10.r2.solvers import local
from 统一模型代码.gates.g10.physics import grid
from 统一模型代码.gates.g10.methods import peaks
from 统一模型代码.gates.g10.r4.paper_checks.assignment_metrics import (
    evaluate, self_test, REVISION,
)

N = 1024
GROUPS = (0, 2, 3)
GAPS = (0, 15)
SEED = 20261023040
METHODS = ('periodic', 'hann_after_band', 'finite_fourier')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def matched_vectors(truth, predicted):
    metrics = evaluate(truth, predicted)
    if metrics['matched'] != 2 or not metrics['valid_output']:
        raise ArithmeticError('Missing/nonfinite output: preserve failure, do not exclude trial')
    vectors = np.empty((2, 2))
    for i, j in metrics['assignment']:
        vectors[i] = np.asarray(predicted)[j]-truth[i]
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), metrics['errors_m'])
    return metrics, vectors.tolist()


def decompose(vectors):
    e = np.asarray(vectors)
    mean = e.mean(axis=0)
    mse = np.mean(np.sum(e**2, axis=-1), axis=0)
    dispersion = np.mean(np.sum((e-mean)**2, axis=-1), axis=0)
    mean_square = np.sum(mean**2, axis=-1)
    np.testing.assert_allclose(mse, mean_square+dispersion, rtol=1e-12, atol=1e-9)
    return dict(rmse_m=np.sqrt(mse).tolist(), mean_vector_m=mean.tolist(),
                mean_displacement_m=np.sqrt(mean_square).tolist(),
                centered_rms_m=np.sqrt(dispersion).tolist(),
                # Signed unbiased estimate of population bias squared; not clipped.
                bias_square_unbiased_m2=(mean_square-dispersion/max(len(e)-1, 1)).tolist())


def common_frontend(iq, config):
    """No truth argument. All methods receive these same five initial pairs."""
    torch.cuda.synchronize()
    tick = time.perf_counter()
    fam = Family(N)
    band_iq = reconstruct(fam.analysis(iq), N, fam.keep)
    m = Model(band_iq, sample_rate_hz=FS, half_band_hz=BAND)
    points = grid(20)
    first = points[peaks(points, m.conditional(points), 1)[0]]
    first, _, _ = local(m, first[None], config)
    fam.setup('direct')
    pb = Problem(fam, band_iq)
    coeff, inner = pb.solve(first, 'direct')
    if not inner['sufficient']:
        raise ArithmeticError('Common finite-fit coefficient certificate failed')
    _, rows, _ = pb.parts(first)
    residual = pb.z-pb.forward(rows, coeff)
    residual_model = Model(fam.synthesis(residual), sample_rate_hz=FS, half_band_hz=BAND)
    ids = peaks(points, residual_model.conditional(points), 5)
    starts = [np.vstack([first, points[j]]) for j in ids]
    assert len(starts) == 5
    torch.cuda.synchronize()
    return band_iq, pb, starts, dict(seconds=time.perf_counter()-tick,
        first=first.tolist(), inner=inner, rank=fam.q, band_points=len(fam.keep))


def compact(result):
    infos = result.pop('inner')
    certificates = [v['coefficient_certificate'] for v in infos if 'coefficient_certificate' in v]
    if any(v.get('sufficient') is False for v in infos):
        raise ArithmeticError('Finite-model coefficient certificate failed')
    result['inner_certificate_max'] = max(certificates) if certificates else None
    result['gradient_pass_1e8'] = result['gradient_max'] <= 1e-8
    return result


def solve_all(iq, config, truth, order):
    band_iq, pb, starts, common = common_frontend(iq, config)
    results = {}
    for method in order:
        torch.cuda.synchronize()
        tick = time.perf_counter()
        if method == 'finite_fourier':
            callback = lambda p: pb.evaluate(p, 'direct')
        else:
            window = np.hanning(N) if method == 'hann_after_band' else np.ones(N)
            model = Model(band_iq*window, sample_rate_hz=FS, half_band_hz=BAND)

            def callback(p):
                value, gradient = model.joint(p, True)
                return value, gradient, None, {}
        prepared = time.perf_counter()-tick
        branches = [compact(optimize(callback, [start])) for start in starts]
        selected = int(np.argmin([v['objective'] for v in branches]))
        best = branches[selected]
        torch.cuda.synchronize()
        method_seconds = time.perf_counter()-tick
        metrics, vectors = matched_vectors(truth, best['positions'])
        # Diagnostic timing begins only AFTER blind result is fixed.
        diagnostic_tick = time.perf_counter()
        diagnostic = compact(optimize(callback, [truth]))
        torch.cuda.synchronize()
        diagnostic['seconds'] = time.perf_counter()-diagnostic_tick
        diagnostic['metrics'], diagnostic['error_vectors_m'] = matched_vectors(truth, diagnostic['positions'])
        results[method] = dict(branches=branches, selected=selected,
            positions=best['positions'], objective=best['objective'],
            gradient_max=best['gradient_max'], optimizer_success=best['optimizer_success'],
            message=best['message'], metrics=metrics, error_vectors_m=vectors,
            seconds=common['seconds']+method_seconds, own_seconds=method_seconds,
            preparation_seconds=prepared, local_diagnostic=diagnostic)
    return dict(starts=[p.tolist() for p in starts], common=common, order=list(order), results=results)


def checks(out, layouts):
    checks_done = self_test()
    rng = np.random.default_rng(SEED)
    e = rng.normal(size=(20, 2, 2))*5
    decompose(e)
    truth = np.array([[20., 3.], [-200., -350.]])
    _, a = matched_vectors(truth, (truth+e[0])[::-1])
    np.testing.assert_allclose(a, e[0])
    x = make(0, 26998, gap=0., n=8192, snr=6., fs=FS, positions=layouts[0], seed=SEED)
    y = make(0, 26998, gap=15., n=8192, snr=6., fs=FS, positions=layouts[0], seed=SEED)
    np.testing.assert_array_equal(x['noise'], y['noise'])
    np.testing.assert_array_equal(x['signal'][1], y['signal'][1])
    np.testing.assert_allclose(x['signal'][0]*10**(15/20), y['signal'][0], rtol=1e-12)
    z = np.fft.fft(x['signal'].sum(axis=0)[:, :N]+x['noise'][:, :N], axis=-1)/np.sqrt(N)
    keep = np.flatnonzero(abs(np.fft.fftfreq(N, 1/FS)) <= BAND)
    b = reconstruct(z[:, keep], N, keep)
    np.testing.assert_allclose(np.fft.fft(b, axis=-1)[:, keep]/np.sqrt(N), z[:, keep], atol=1e-12)
    payload = dict(passed=True, evaluation=checks_done, decomposition=True,
        identical_weak_and_noise=True, source_A_scaling=True, unitary_band_roundtrip=True,
        unicode='中文路径与日志检查通过')
    write(out/'checks.json', payload)
    assert load(out/'checks.json') == payload
    print('接口与配对检查通过', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('tests', 'pilot', 'run'))
    args = parser.parse_args()
    previous = load(INPUT/'protocol.json')
    identities = load(INPUT/'input_identities.json')
    files = [INPUT/'inputs'/f'{g}_25000.npz' for g in GROUPS]
    for path in files:
        assert path.resolve(strict=True).is_relative_to(ROOT/'outputs_e2e')
        assert digest(path) == identities[str(path)]
    layouts = previous['layouts']
    for g, path in zip(GROUPS, files):
        with np.load(path) as f:
            np.testing.assert_array_equal(f['positions'], layouts[g])
    out = BASE/'precision_repeats_20261023'/(args.mode+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    assert not out.exists() and out.resolve().is_relative_to(ROOT/'outputs_e2e')
    out.mkdir(parents=True)
    sources = {str(p): digest(p) for p in (ROOT/'统一模型代码/gates').rglob('*.py')}
    for name in sources:
        p = Path(name)
        dst = out/'source_snapshot'/p.relative_to(ROOT/'统一模型代码/gates')
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(p.read_bytes())
    repeats = [26999] if args.mode != 'run' else list(range(27000, 27020))
    protocol = dict(mode=args.mode, seed=SEED, repeats=repeats, groups=list(GROUPS),
        layouts=layouts, gaps=list(GAPS), n=N, generator_n=8192, fs=FS, band_hz=BAND,
        nominal_generator_weak_inband_snr_db=6, renormalize_crop=False, ridge=RIDGE,
        methods=list(METHODS), evaluation=REVISION, prior_protocol_sha=digest(INPUT/'protocol.json'),
        sources=sources, reference_inputs={str(p): digest(p) for p in files},
        primary='untruncated A/B/pooled RMSE; all trials included',
        frontend='periodic first estimate; finite fit residual; five common starts; refine every branch',
        local_diagnostic='truth start for EVERY trial; excluded from blind metrics and deployment timing',
        inference='exploratory independent waveform/noise repeats on three selected fixed development geometries',
        bootstrap=dict(count=2000, seed=20261023041, unit='realization within geometry, shared across methods and gaps'),
        optimization=dict(maxiter=100, ftol=1e-12, gtol=1e-8, maxls=25),
        timing='common preparation charged to each; own five branches; diagnostic excluded; GPU sync; CPU threads=2')
    write(out/'protocol.json', protocol)
    print('结果目录：', out, flush=True)
    tick = time.perf_counter()
    raw_ids = {}
    records = []
    try:
        with threadpool_limits(limits=2):
            checks(out, layouts)
            if args.mode != 'tests':
                warm = time.perf_counter()
                torch.zeros(1, device='cuda')
                torch.cuda.synchronize()
                write(out/'warmup.json', dict(seconds=time.perf_counter()-warm))
                (out/'inputs').mkdir()
                (out/'cases').mkdir()
                for g in GROUPS:
                    for rep in repeats:
                        scene = make(g, rep, gap=0., n=8192, snr=6., fs=FS, positions=layouts[g], seed=SEED)
                        raw = out/'inputs'/f'{g}_{rep}.npz'
                        np.savez_compressed(raw, signal=scene['signal'], noise=scene['noise'], positions=scene['positions'])
                        raw_ids[str(raw)] = digest(raw)
                        write(out/'input_identities.json', raw_ids)
                        for gap in GAPS:
                            scale = 10**(gap/20)
                            iq = (scale*scene['signal'][0]+scene['signal'][1]+scene['noise'])[:, :N]
                            shift = (g+rep+int(gap == 15)) % 3
                            order = METHODS[shift:]+METHODS[:shift]
                            row = solve_all(iq, previous['config'], scene['positions'], order)
                            row.update(group=g, realization=rep, gap=gap, truth=scene['positions'].tolist(),
                                       input_sha=raw_ids[str(raw)], iq_sha=__import__('hashlib').sha256(iq.tobytes()).hexdigest())
                            name = f'{g}_{rep}_{gap}'
                            write(out/'cases'/f'{name}.json', row)
                            records.append(row)
                            write(out/'results.json', records)
                            elapsed = time.perf_counter()-tick
                            total = len(GROUPS)*len(repeats)*2
                            write(out/'progress.json', dict(completed=len(records), total=total, seconds=elapsed))
                            print(f'精度复核 {len(records)}/{total} | {elapsed/60:.1f} min | {name}', flush=True)
        assert all(digest(p) == v for p, v in sources.items())
        assert all(digest(p) == v for p, v in raw_ids.items())
        assert all(digest(p) == identities[str(p)] for p in files)
        assert digest(INPUT/'protocol.json') == protocol['prior_protocol_sha']
        write(out/'audit.json', dict(completed=True, inputs_and_sources_unchanged=True,
            seconds=time.perf_counter()-tick, observations=len(records), blind_outputs=len(records)*3,
            diagnostic_outputs=len(records)*3, formal=False))
    except Exception:
        write(out/'failure.json', dict(traceback=traceback.format_exc(), seconds=time.perf_counter()-tick))
        raise
    print('完成：', out, flush=True)


if __name__ == '__main__':
    main()
