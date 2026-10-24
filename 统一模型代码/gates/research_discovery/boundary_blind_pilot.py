"""Complete paired blind finite-record solvers with shared candidate rules."""
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.research_discovery.finite_fft_operator import FiniteOperator
from 统一模型代码.gates.research_discovery.finite_fft_blind import derivative_action
from 统一模型代码.gates.research_discovery.boundary_schur_audit import prepare, schur
from 统一模型代码.gates.research_discovery.residual_competition import HOLDOUT
from 统一模型代码.gates.research_discovery.window_baselines import weight
from 统一模型代码.gates.g10.r3.solvers import Model, Trace
from 统一模型代码.gates.g10.r4.solvers import ap
from 统一模型代码.gates.g10.r2.solvers import local
from 统一模型代码.gates.g10.physics import grid
from 统一模型代码.gates.g10.methods import peaks
from 统一模型代码.gates.g11.precision_diagnostic import sha, write
from 统一模型代码.gates.g10.r4.paper_checks.assignment_metrics import evaluate


class Direct:
    def __init__(self, op):
        delta = (op.f[None]-op.f[:, None])/50e6
        self.kernel = op.n*np.sinc(op.n*delta)/np.sinc(delta)*np.exp(1j*np.pi*(op.n-1)*delta)

    def solve(self, op, y):
        rows = (op.phase*op.gains[:, :, None]).transpose(1, 0, 2).reshape(4, -1)
        matrix = np.tile(self.kernel, (op.k, op.k))*(rows.conj().T@rows)
        matrix.flat[::matrix.shape[0]+1] += 1e-6
        return cho_solve(cho_factor(matrix, lower=True), op.H@y)


def conventional(iq, cfg, window):
    torch.cuda.synchronize()
    start = time.perf_counter()
    trace = Trace(lambda: None, history=False)
    model = Model(iq*weight(window, iq.shape[-1], 50e6), trace.check,
                  sample_rate_hz=50e6, half_band_hz=7e6)
    trace.model = model
    detail = ap(model, trace, 'AP_5_JOINT', copy.deepcopy(cfg))
    torch.cuda.synchronize()
    return dict(positions=detail['output_positions'], seconds=time.perf_counter()-start)


def solve(iq, cfg, backend):
    torch.cuda.synchronize()
    tick = time.perf_counter()
    trace = Trace(lambda: None, history=False)
    model = Model(iq, trace.check, sample_rate_hz=50e6, half_band_hz=7e6)
    trace.model = model
    points = grid(20)
    first = points[peaks(points, model.conditional(points), 1)[0]]
    first, _, _ = local(model, first[None], cfg)
    n = iq.shape[-1]
    y = iq.ravel()
    energy = np.vdot(y, y).real
    single = FiniteOperator(first, n=n)
    direct = Direct(single)
    c = direct.solve(single, y)
    residual = (y-single@c).reshape(iq.shape)
    residual_model = Model(residual, trace.check, sample_rate_hz=50e6, half_band_hz=7e6)
    indices = peaks(points, residual_model.conditional(points), 5)
    starts = [np.vstack([first, p]) for p in points[indices]]
    prefix_seconds = time.perf_counter()-tick
    preparation = None
    h = None
    if backend == 'boundary':
        h, preparation = prepare(FiniteOperator(starts[0], n=n))
    cache, normals = {}, []

    def objective(x):
        key = np.asarray(x).tobytes()
        if key not in cache:
            p = np.asarray(x).reshape(2, 2)
            op = FiniteOperator(p, n=n)
            coeff = direct.solve(op, y) if backend == 'direct' else schur(op, y, h)
            r = y-op@coeff
            value = float((np.vdot(r, r).real+1e-6*np.vdot(coeff, coeff).real)/energy)
            derivative = derivative_action(op, p, coeff)
            gradient = -2*(r.conj()@derivative).real/energy
            normal = -op.H@r+1e-6*coeff
            normals.append(float(np.linalg.norm(normal)/np.linalg.norm(op.H@y)))
            cache[key] = value, gradient
        return cache[key]

    scores = [objective(p.ravel())[0] for p in starts]
    best = int(np.argmin(scores))
    opt = minimize(objective, starts[best].ravel(), jac=True, method='L-BFGS-B',
        bounds=[(-1000., 1000.)]*4, options=dict(maxiter=100, ftol=1e-12, gtol=1e-8, maxls=25))
    final = opt.x.reshape(2, 2) if np.isfinite(opt.fun) and opt.fun <= scores[best] else starts[best]
    value, gradient = objective(final.ravel())
    torch.cuda.synchronize()
    return dict(positions=final.tolist(), seconds=time.perf_counter()-tick,
        prefix_seconds=prefix_seconds, preparation=preparation, first=first.tolist(),
        starts=[p.tolist() for p in starts], selected=best, scores=scores,
        objective=value, gradient_max=float(abs(gradient).max()),
        max_relative_normal_residual=max(normals), objective_calls=len(cache),
        iterations=int(opt.nit), optimizer_success=bool(opt.success), message=str(opt.message))


def main():
    out = HOLDOUT.parent/('round21_blind_boundary_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    assert not out.exists() and out.resolve().is_relative_to(ROOT/'outputs_e2e')
    cfg = json.loads((HOLDOUT/'protocol.json').read_text(encoding='utf-8'))['config']
    files = sorted((HOLDOUT/'inputs').glob('*_24000.npz'))
    inputs = {str(p.resolve(strict=True)): sha(p) for p in files}
    sources = {str(p): sha(p) for p in (ROOT/'统一模型代码/gates/research_discovery').glob('*.py')}
    out.mkdir()
    write(out/'protocol.json', dict(inputs=inputs, sources=sources, n=4096, gaps=[15, 30],
        complete_cost=True, reused_development_data=True, config=cfg,
        joint=dict(maxiter=100, ftol=1e-12, gtol=1e-8, maxls=25), ridge=1e-6,
        relative_compression_bound=1e-6, candidate_rule='common finite ridge SIC plus five residual peaks',
        targets='identical full-time finite objective for direct and boundary; AP uses its usual windowed frequency criterion',
        methods=['direct', 'boundary', 'rect', 'hann', 'tukey_geometry']))
    torch.cuda.synchronize()
    rows = []
    with threadpool_limits(limits=2):
        for index, path in enumerate(files):
            with np.load(path, allow_pickle=False) as z:
                signal, noise, truth = z['signal'][..., :4096], z['noise'][:, :4096], z['positions']
            for gap in (15, 30):
                iq = signal[0]*10**(gap/20)+signal[1]+noise
                results = {}
                for backend in (['direct', 'boundary'] if (index+(gap == 30)) % 2 == 0 else ['boundary', 'direct']):
                    result = solve(iq, cfg, backend)
                    result['metrics'] = evaluate(truth, result['positions'])
                    results[backend] = result
                np.testing.assert_allclose(results['direct']['starts'], results['boundary']['starts'], atol=1e-9, rtol=0)
                for window in ('rect', 'hann', 'tukey_geometry'):
                    result = conventional(iq, cfg, window)
                    result['metrics'] = evaluate(truth, result['positions'])
                    results[window] = result
                rows.append(dict(case=path.stem, gap=gap, results=results))
                write(out/'results.json', rows)
                print(path.stem, gap, {k: (v['metrics']['both20'], round(v['seconds'], 2)) for k, v in results.items()}, flush=True)
    for path, identity in {**inputs, **sources}.items():
        assert sha(path) == identity
    write(out/'audit.json', dict(completed=True, cases=len(rows), inputs_unchanged=True,
        sources_unchanged=True, shared_candidates_passed=True, independent_data=False))
    print(out, flush=True)


if __name__ == '__main__':
    main()
