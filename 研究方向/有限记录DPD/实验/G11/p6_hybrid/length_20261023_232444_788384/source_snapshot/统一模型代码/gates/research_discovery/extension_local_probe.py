"""Local attainability diagnostic with central-difference truncated LS objective."""
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
from 统一模型代码.gates.research_discovery.extension_stability import extended, SOURCE, ATTR
from 统一模型代码.gates.g11.precision_diagnostic import sha, write
from 统一模型代码.gates.g10.r4.paper_checks.assignment_metrics import evaluate


def main():
    out = SOURCE.parent/('round8_local_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    assert not out.exists() and out.resolve().is_relative_to(ROOT/'outputs_e2e')
    protected = {str(p): sha(p) for p in (ROOT/'统一模型代码/gates').rglob('*.py')}
    inputs = {str(p): sha(p) for folder in (SOURCE, ATTR) for p in folder.rglob('*') if p.is_file()}
    out.mkdir()
    write(out/'protocol.json', dict(cases=['new_1_22000', 'new_3_22000'], gap=15,
        initial='truth plus [[5,-3],[-4,6]], diagnostic only', guard=2, rcond=1e-6,
        gradient='central difference 0.01m, no naive envelope gradient', maxiter=15,
        ftol=1e-11, gtol=1e-7, bounded_positions=[-1000, 1000],
        protected=protected, inputs=inputs))
    records = []
    with threadpool_limits(limits=2):
        for path in sorted((ATTR/'pairs').glob('*.json')):
            row = json.loads(path.read_text(encoding='utf-8'))
            if row['gap'] != 15 or row['case'] not in ('new_1_22000', 'new_3_22000'):
                continue
            _, group, rep = row['case'].split('_')
            with np.load(SOURCE/'inputs'/f'{group}_{rep}.npz', allow_pickle=False) as z:
                y = (z['signal'][0, :, :1024]*10**(15/20)+z['signal'][1, :, :1024]+z['noise'][:, :1024]).ravel()
            cache, ranks = {}, {}
            energy = np.vdot(y, y).real

            def value(x):
                key = np.asarray(x).tobytes()
                if key not in cache:
                    a = extended(np.asarray(x).reshape(2, 2), 2)
                    u, s, _ = svd(a, full_matrices=False, lapack_driver='gesdd')
                    rank = int((s > s[0]*1e-6).sum())
                    q = u[:, :rank]
                    residual = y-q@(q.conj().T@y)
                    cache[key] = float(np.vdot(residual, residual).real/energy)
                    ranks[key] = rank
                return cache[key]

            def objective(x):
                center = value(x)
                grad = np.zeros(4)
                for j in range(4):
                    delta = np.eye(4)[j]*.01
                    grad[j] = (value(x+delta)-value(x-delta))/.02
                return center, grad

            truth = np.asarray(row['truth'])
            start = (truth+np.array([[5., -3.], [-4., 6.]])).ravel()
            tick = time.perf_counter()
            before = value(start)
            opt = minimize(objective, start, jac=True, method='L-BFGS-B',
                bounds=[(-1000, 1000)]*4,
                options=dict(maxiter=15, ftol=1e-11, gtol=1e-7, maxls=10))
            assert np.isfinite(opt.fun) and opt.fun <= before+1e-10
            item = dict(case=row['case'], initial=start.reshape(2, 2).tolist(),
                positions=opt.x.reshape(2, 2).tolist(), before=before, after=float(opt.fun),
                metrics=evaluate(truth, opt.x.reshape(2, 2)), iterations=int(opt.nit),
                success=bool(opt.success), message=str(opt.message), gradient_max=float(abs(opt.jac).max()),
                factorizations=len(cache), visited_ranks=sorted(set(ranks.values())),
                seconds=time.perf_counter()-tick, diagnostic_only=True)
            records.append(item)
            write(out/'results.json', records)
            print(row['case'], item['metrics']['errors_m'], item['message'], flush=True)
    for path, identity in {**protected, **inputs}.items():
        assert sha(path) == identity, path
    write(out/'audit.json', dict(completed=True, cases=len(records), sources_unchanged=True,
        inputs_unchanged=True, diagnostic_only=True))
    print(out, flush=True)


if __name__ == '__main__':
    main()
