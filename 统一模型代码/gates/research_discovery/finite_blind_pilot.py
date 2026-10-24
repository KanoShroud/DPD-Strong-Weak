"""Small conventional finite-record SIC+joint-LS pipeline, no novelty claim."""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.linalg import svd
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.research_discovery.finite_extension_probe import SOURCE, ATTR, dictionary
from 统一模型代码.gates.research_discovery.finite_residual_candidates import candidates
from 统一模型代码.gates.g11.precision_diagnostic import sha, write
from 统一模型代码.gates.g10.r4.paper_checks.assignment_metrics import evaluate


class Criterion:
    def __init__(self, iq):
        self.y = iq.ravel()
        self.energy = np.vdot(self.y, self.y).real
        self.calls = 0
        self.fallbacks = 0

    def evaluate(self, positions, gradient=False):
        p = np.asarray(positions).reshape(2, 2)
        a = dictionary(p)
        self.calls += 1
        try:
            u, s, vh = svd(a, full_matrices=False, lapack_driver='gesdd')
        except np.linalg.LinAlgError:
            self.fallbacks += 1
            u, s, vh = svd(a, full_matrices=False, lapack_driver='gesvd')
        rank = int((s > 1e-6*s[0]).sum())
        rhs = u[:, :rank].conj().T@self.y
        residual = self.y-u[:, :rank]@rhs
        val = float(np.vdot(residual, residual).real/self.energy)
        if not gradient:
            return val
        coef = vh[:rank].conj().T@(rhs/s[:rank])
        g = []
        for j in range(4):
            delta = np.zeros((2, 2))
            delta.flat[j] = .001
            da = (dictionary(p+delta)-dictionary(p-delta))/.002
            g.append(float(-2*np.vdot(residual, da@coef).real/self.energy))
        return val, np.asarray(g)


def solve(iq, first):
    tick = time.perf_counter()
    screen = candidates(iq, first)
    criterion = Criterion(iq)
    starts = [np.vstack([np.asarray(first).reshape(1, 2), p]) for p in screen['positions']]
    values = [criterion.evaluate(p) for p in starts]
    index = int(np.argmin(values))
    initial = starts[index]
    opt = minimize(lambda x: criterion.evaluate(x, True), initial.ravel(), jac=True,
        method='L-BFGS-B', bounds=[(-1000, 1000)]*4,
        options=dict(maxiter=15, ftol=1e-10, gtol=1e-7, maxls=10))
    final = opt.x.reshape(2, 2) if np.isfinite(opt.fun) and opt.fun <= values[index] else initial
    return dict(positions=final.tolist(), initial=initial.tolist(), candidates=screen,
        candidate_values=values, selected_index=index, before=values[index],
        after=criterion.evaluate(final), optimizer_success=bool(opt.success),
        message=str(opt.message), iterations=int(opt.nit), factorizations=criterion.calls,
        svd_fallbacks=criterion.fallbacks, seconds=time.perf_counter()-tick,
        gradient='approximate envelope with matrix central differences; actual objective line search')


def main():
    out = SOURCE.parent/('round10_blind_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    assert not out.exists() and out.resolve().is_relative_to(ROOT/'outputs_e2e')
    protected = {str(p): sha(p) for p in (ROOT/'统一模型代码/gates').rglob('*.py')}
    inputs = {str(p): sha(p) for folder in (SOURCE, ATTR) for p in folder.rglob('*') if p.is_file()}
    out.mkdir()
    write(out/'protocol.json', dict(cases='four layouts realization22000 gap15',
        first='reuse saved blind first estimate, prefix cost excluded and not speed benchmark',
        candidate_count=5, grid=20, maxiter=15, rcond=1e-6,
        no_truth_in_solver=True, baseline_only=True, protected=protected, inputs=inputs))
    records = []
    with threadpool_limits(limits=2):
        for path in sorted((ATTR/'pairs').glob('*.json')):
            row = json.loads(path.read_text(encoding='utf-8'))
            if row['gap'] != 15 or not row['case'].endswith('_22000'):
                continue
            _, group, rep = row['case'].split('_')
            with np.load(SOURCE/'inputs'/f'{group}_{rep}.npz', allow_pickle=False) as z:
                iq = z['signal'][0, :, :1024]*10**(15/20)+z['signal'][1, :, :1024]+z['noise'][:, :1024]
            detail = solve(iq, row['results']['rect']['ap1']['detail']['first_position'])
            metrics = evaluate(row['truth'], detail['positions'])
            records.append(dict(case=row['case'], detail=detail, metrics=metrics))
            write(out/'results.json', records)
            print(row['case'], metrics['errors_m'], detail['message'], flush=True)
    for path, identity in {**protected, **inputs}.items():
        assert sha(path) == identity, path
    write(out/'audit.json', dict(completed=True, n=len(records),
        both20=sum(r['metrics']['both20'] for r in records), sources_unchanged=True,
        inputs_unchanged=True, development_only=True))
    print(out, flush=True)


if __name__ == '__main__':
    main()
