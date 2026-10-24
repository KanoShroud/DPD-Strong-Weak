"""Conventional matrix-free finite-record baseline with position derivatives."""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator, lsmr
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.research_discovery.finite_fft_operator import FiniteOperator, SOURCE, ATTR
from 统一模型代码.gates.research_discovery.finite_residual_candidates import candidates
from 统一模型代码.gates.g8.physics import C, RECEIVERS
from 统一模型代码.gates.g11.precision_diagnostic import sha, write
from 统一模型代码.gates.g10.r4.paper_checks.assignment_metrics import evaluate


def derivative_action(op, p, coeff):
    p = np.asarray(p).reshape(2, 2)
    dvec = p[:, None]-np.asarray(RECEIVERS)[None]
    d = np.linalg.norm(dvec, axis=-1)
    unit = dvec/d[:, :, None]
    weights = d**-2/np.sum(d**-2, axis=1)[:, None]
    log_gain = -unit/d[:, :, None]+np.sum(weights[:, :, None]*unit/d[:, :, None], axis=1)[:, None]
    nearest = np.argmin(d, axis=1)
    dtau = (unit-unit[np.arange(2), nearest][:, None])/C
    coeff = np.asarray(coeff).reshape(2, op.width)
    base = op.phase*coeff[:, None]
    result = []
    for k in range(2):
        for axis in range(2):
            spec = np.zeros((4, op.length), dtype=complex)
            multiplier = log_gain[k, :, axis][:, None]-2j*np.pi*dtau[k, :, axis][:, None]*op.f[None]
            spec[:, op.indices] = base[k]*multiplier
            wave = np.fft.ifft(spec, axis=-1)*op.length
            result.append((wave[:, :op.n]*op.gains[k, :, None]).ravel())
    return np.column_stack(result)


def solve(iq, first, damp=0., bandlimited=False):
    tick = time.perf_counter()
    screen = candidates(iq, first, bandlimited=bandlimited)
    n = iq.shape[-1]
    keep = np.flatnonzero(abs(np.fft.fftfreq(n, 1/50e6)) <= 7e6)

    def analysis(v):
        return (np.fft.fft(np.asarray(v).reshape(4, n), axis=-1)[:, keep]/np.sqrt(n)).ravel()

    def synthesis(v):
        spectrum = np.zeros((4, n), dtype=complex)
        spectrum[:, keep] = np.asarray(v).reshape(4, len(keep))
        return (np.fft.ifft(spectrum, axis=-1)*np.sqrt(n)).ravel()

    y = analysis(iq.ravel()) if bandlimited else iq.ravel()
    energy = np.vdot(y, y).real
    cache, inner = {}, []

    def objective(x):
        key = np.asarray(x).tobytes()
        if key not in cache:
            p = np.asarray(x).reshape(2, 2)
            full_op = FiniteOperator(p)
            op = LinearOperator((len(y), full_op.shape[1]),
                matvec=lambda v: analysis(full_op@v),
                rmatvec=lambda v: full_op.H@synthesis(v), dtype=np.complex128) if bandlimited else full_op
            tol = 1e-9 if damp else 1e-6
            result = lsmr(op, y, damp=damp, atol=tol, btol=tol, maxiter=1000 if damp else 500, conlim=1e8)
            residual = y-op@result[0]
            value = float((np.vdot(residual, residual).real+damp**2*np.vdot(result[0], result[0]).real)/energy)
            deriv = derivative_action(full_op, p, result[0])
            if bandlimited:
                deriv = np.column_stack([analysis(deriv[:, j]) for j in range(4)])
            g = -2*(residual.conj()@deriv).real/energy
            cache[key] = (value, g)
            inner.append(dict(stop=int(result[1]), iterations=int(result[2])))
        return cache[key]

    starts = [np.vstack([np.asarray(first).reshape(1, 2), p]) for p in screen['positions']]
    scores = [objective(p.ravel())[0] for p in starts]
    index = int(np.argmin(scores))
    opt = minimize(objective, starts[index].ravel(), jac=True, method='L-BFGS-B',
        bounds=[(-1000, 1000)]*4, options=dict(maxiter=40, ftol=1e-10, gtol=1e-7, maxls=15))
    final = opt.x.reshape(2, 2) if np.isfinite(opt.fun) and opt.fun <= scores[index] else starts[index]
    return dict(positions=final.tolist(), initial=starts[index].tolist(), selected_index=index,
        scores=scores, objective=objective(final.ravel())[0], seconds=time.perf_counter()-tick,
        optimizer_success=bool(opt.success), message=str(opt.message), iterations=int(opt.nit),
        gradient_max=float(abs(objective(final.ravel())[1]).max()), inner=inner, bandlimited=bandlimited,
        note='gradient is an envelope approximation at iterative LS coefficients')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--damped', action='store_true')
    args = parser.parse_args()
    damp = 1e-3 if args.damped else 0.
    prefix = 'round13_ridge_' if args.damped else 'round12_fft_blind_'
    out = SOURCE.parent/(prefix+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    assert not out.exists() and out.resolve().is_relative_to(ROOT/'outputs_e2e')
    protected = {str(p): sha(p) for p in (ROOT/'统一模型代码/gates').rglob('*.py')}
    inputs = {str(p): sha(p) for folder in (SOURCE, ATTR) for p in folder.rglob('*') if p.is_file()}
    out.mkdir()
    write(out/'protocol.json', dict(cases='four layouts realization22000 gap15', maxiter=40,
        inner_lsmr=dict(atol=1e-9 if damp else 1e-6, btol=1e-9 if damp else 1e-6,
            maxiter=1000 if damp else 500, conlim=1e8, damp=damp),
        criterion='residual squared plus fixed damp squared coefficient norm; normalized by input energy',
        no_truth_in_solver=True, conventional_baseline=True, prefix_cost_excluded=True,
        protected=protected, inputs=inputs))
    p = np.array([[173., 117.], [-137., -213.]])
    op = FiniteOperator(p)
    rng = np.random.default_rng(20261022012)
    coeff = rng.normal(size=op.shape[1])+1j*rng.normal(size=op.shape[1])
    analytic = derivative_action(op, p, coeff)
    errors = []
    for j in range(4):
        delta = np.zeros((2, 2))
        delta.flat[j] = .001
        numerical = (FiniteOperator(p+delta)@coeff-FiniteOperator(p-delta)@coeff)/.002
        errors.append(float(np.linalg.norm(numerical-analytic[:, j])/np.linalg.norm(numerical)))
    assert max(errors) < 1e-7
    write(out/'derivative_check.json', dict(relative_errors=errors, tolerance=1e-7))
    records = []
    with threadpool_limits(limits=2):
        for path in sorted((ATTR/'pairs').glob('*.json')):
            row = json.loads(path.read_text(encoding='utf-8'))
            if row['gap'] != 15 or not row['case'].endswith('_22000'):
                continue
            _, group, rep = row['case'].split('_')
            with np.load(SOURCE/'inputs'/f'{group}_{rep}.npz', allow_pickle=False) as z:
                iq = z['signal'][0, :, :1024]*10**(15/20)+z['signal'][1, :, :1024]+z['noise'][:, :1024]
            detail = solve(iq, row['results']['rect']['ap1']['detail']['first_position'], damp=damp)
            metric = evaluate(row['truth'], detail['positions'])
            records.append(dict(case=row['case'], detail=detail, metrics=metric))
            write(out/'results.json', records)
            print(row['case'], metric['errors_m'], detail['message'], flush=True)
    for path, identity in {**protected, **inputs}.items():
        assert sha(path) == identity, path
    write(out/'audit.json', dict(completed=True, n=len(records), both20=sum(r['metrics']['both20'] for r in records),
        inputs_unchanged=True, sources_unchanged=True, development_only=True))
    print(out, flush=True)


if __name__ == '__main__':
    main()
