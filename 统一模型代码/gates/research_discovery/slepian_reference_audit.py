"""Bounded post-hoc diagnostic, never substituted for blind results."""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.special import roots_legendre
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.research_discovery.slepian_reference import (
    SlepianFamily, SlepianProblem, optimize, reconstruct, FS, BAND, INPUT, BASE,
    Family, Problem, input_data, digest, write,
)


def actual_size_audit(out):
    records = []
    for n in (1024, 4096):
        # Analytic covariance identity verifies quadrature across the entire interval.
        # Includes every extreme lag and random interior pairs, not only a small N proxy.
        from 统一模型代码.gates.g8.physics import RECEIVERS, C
        sites = np.asarray(RECEIVERS)
        delay = np.linalg.norm(sites[:, None]-sites[None], axis=-1).max()*FS/C
        duration = n-1+2*delay
        rng = np.random.default_rng(20261023031)
        times = np.r_[-duration/2, duration/2, rng.uniform(-duration/2, duration/2, 62)]
        order = int(np.ceil(4*BAND/FS*duration))+32
        errors = []
        for factor in (1, 2):
            nodes, weight = roots_legendre(order*factor)
            frequencies, weight = nodes*BAND/FS, weight*BAND/FS
            e = np.exp(2j*np.pi*times[:, None]*frequencies[None])
            approximate = (e*weight)@e.conj().T
            exact = 2*BAND/FS*np.sinc(2*BAND/FS*(times[:, None]-times[None]))
            errors.append(float(np.linalg.norm(approximate-exact)/np.linalg.norm(exact)))
        assert max(errors) < 1e-9
        records.append(dict(n=n, quadrature=order, covariance_errors=errors))
    fam = SlepianFamily(1024, 1e-6)
    fine = SlepianFamily(1024, 1e-6, 2)
    p = np.array([[173., 117.], [-137., -213.]])
    a, b = fam.matrix(p)[::37], fine.matrix(p)[::37]
    frame_error = float(np.linalg.norm(a@a.conj().T-b@b.conj().T)/np.linalg.norm(b@b.conj().T))
    assert fam.rank == fine.rank and frame_error < 1e-7
    write(out/'actual_size_numerics.json', dict(passed=True, rows=records,
                                               n1024_truncated_frame_error=frame_error))


def closure(out, pilots):
    from 统一模型代码.gates.g10.r3.solvers import Model
    from 统一模型代码.gates.g10.physics import grid
    from 统一模型代码.gates.g10.methods import peaks
    rows = []
    for pilot in pilots:
        for row in json.loads((pilot/'results.json').read_text(encoding='utf-8')):
            failed = [m for m, r in row['results'].items() if not r['metrics']['both20']]
            # Avoid tripling equivalent Slepian controls: representative fixed 1e-6 only.
            methods = [m for m in failed if not m.startswith('slepian_') or m == 'slepian_1e-06']
            if not methods:
                continue
            n, gap = row['n'], row['gap']
            iq, truth = input_data(INPUT/'inputs'/(row['case']+'.npz'), n, gap)
            keep = np.flatnonzero(abs(np.fft.fftfreq(n, 1/FS)) <= BAND)
            band_iq = reconstruct(np.fft.fft(iq, axis=-1)[:, keep]/np.sqrt(n), n, keep)
            starts = [np.asarray(x) for x in row['starts']]
            # Conventional fitted-waveform subtraction (R025 relation), no true position.
            # Same supplemental starts offered to every diagnostic method.
            prefix_tick = time.perf_counter()
            finite_family = Family(n); finite_family.setup('direct')
            finite_pb = Problem(finite_family, band_iq)
            first = starts[0][:1]
            coefficient, _ = finite_pb.solve(first, 'direct')
            _, response, _ = finite_pb.parts(first)
            residual = finite_pb.z-finite_pb.forward(response, coefficient)
            residual_model = Model(finite_family.synthesis(residual), sample_rate_hz=FS, half_band_hz=BAND)
            points = grid(20)
            indices = peaks(points, residual_model.conditional(points), 5)
            residual_starts = [np.vstack([first, points[j]]) for j in indices]
            prefix_seconds = time.perf_counter()-prefix_tick
            for method in methods:
                tick = time.perf_counter()
                if method == 'slepian_1e-06':
                    family = SlepianFamily(n, 1e-6)
                    problem = SlepianProblem(family, band_iq)
                    callback = problem.evaluate
                elif method == 'finite_fourier':
                    family = Family(n); family.setup('direct')
                    problem = Problem(family, band_iq)
                    callback = lambda p: problem.evaluate(p, 'direct')
                else:
                    window = np.hanning(n) if method == 'hann_after_band' else np.ones(n)
                    model = Model(band_iq*window, sample_rate_hz=FS, half_band_hz=BAND)

                    def callback(p):
                        value, grad = model.joint(p, True)
                        return value, grad, None, {}
                truth_start = optimize(callback, [truth.copy()])
                branches = [optimize(callback, [p]) for p in starts]
                selected = int(np.argmin([r['objective'] for r in branches]))
                residual_result = optimize(callback, residual_starts)
                from 统一模型代码.gates.g10.r4.paper_checks.assignment_metrics import evaluate
                truth_start['metrics'] = evaluate(truth, truth_start['positions'])
                residual_result['metrics'] = evaluate(truth, residual_result['positions'])
                for branch in branches:
                    branch['metrics'] = evaluate(truth, branch['positions'])
                record = dict(pilot=str(pilot), n=n, case=row['case'], gap=gap, method=method,
                    truth_initialized=truth_start, all_five_refined=branches,
                    conventional_residual= residual_result,
                    residual_starts=[p.tolist() for p in residual_starts],
                    residual_prefix_seconds=prefix_seconds,
                    selected_by_objective=selected, seconds=time.perf_counter()-tick,
                    nearest_initial_weak_m=float(min(np.linalg.norm(p-truth[1], axis=1).min() for p in starts)),
                    use='post-hoc diagnostic; not independent, not replacing blind result')
                rows.append(record)
                write(out/'closure.json', rows)
                print('失败归因', n, row['case'], gap, method,
                      '真值初始', truth_start['metrics']['errors_m'],
                      '五支精修', branches[selected]['metrics']['errors_m'],
                      '常规有限残余', residual_result['metrics']['errors_m'], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pilots', nargs='+', type=Path)
    args = parser.parse_args()
    for p in args.pilots:
        assert p.resolve().is_relative_to(ROOT/'outputs_e2e')
        assert json.loads((p/'audit.json').read_text(encoding='utf-8'))['completed']
    out = BASE/'slepian_reference_20261023'/('closure_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir()
    inputs = {str(p/'results.json'): digest(p/'results.json') for p in args.pilots}
    declared = json.loads((INPUT/'input_identities.json').read_text(encoding='utf-8'))
    for p in (INPUT/'inputs').glob('*_25000.npz'):
        assert digest(p) == declared[str(p)]
        inputs[str(p)] = digest(p)
    sources = {str(p): digest(p) for p in (ROOT/'统一模型代码/gates').rglob('*.py')}
    for name in sources:
        p = Path(name); target = out/'source_snapshot'/p.relative_to(ROOT/'统一模型代码/gates')
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(p.read_bytes())
    write(out/'protocol.json', dict(input_identities=inputs, source_identities=sources,
        post_hoc=True, rule='all blind failures; one fixed Slepian epsilon, all five branches, truth start, conventional finite-fit residual candidates',
        no_new_algorithm=True, truth_only_diagnostic=True))
    print('结果目录', out, flush=True)
    tick = time.perf_counter()
    with threadpool_limits(limits=2):
        actual_size_audit(out)
        closure(out, args.pilots)
    assert all(digest(p) == identity for p, identity in {**inputs, **sources}.items())
    write(out/'audit.json', dict(completed=True, identities_unchanged=True, seconds=time.perf_counter()-tick))
    print('完成', out, flush=True)


if __name__ == '__main__':
    main()
