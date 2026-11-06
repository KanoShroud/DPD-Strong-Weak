"""Check naive full-LS envelope gradient against truncated-SVD objective.

Development diagnostics only; no blind localization or new gradient claim.
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.linalg import svd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.research_discovery.extension_stability import extended, SOURCE, ATTR
from 统一模型代码.gates.g11.precision_diagnostic import sha, write

CONDS = (1e-6, 1e-8)


def fit(p, y, derivative=False):
    a = extended(p, 2)
    u, s, vh = svd(a, full_matrices=False, lapack_driver='gesdd')
    energy = np.vdot(y, y).real
    results = []
    for cond in CONDS:
        rank = int((s > cond*s[0]).sum())
        q = u[:, :rank]
        rhs = q.conj().T@y
        coef = vh[:rank].conj().T@(rhs/s[:rank])
        residual = y-q@rhs
        item = dict(value=float(np.vdot(residual, residual).real/energy), rank=rank)
        if derivative:
            gradient = []
            for j in range(4):
                delta = np.zeros((2, 2))
                delta.flat[j] = .001
                da = (extended(p+delta, 2)-extended(p-delta, 2))/.002
                gradient.append(float(-2*np.vdot(residual, da@coef).real/energy))
            item['naive_envelope_gradient'] = gradient
        results.append(item)
    return results


def main():
    out = SOURCE.parent/('round7_gradient_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    assert not out.exists() and out.resolve().is_relative_to(ROOT/'outputs_e2e')
    protected = {str(p): sha(p) for p in (ROOT/'统一模型代码/gates').rglob('*.py')}
    inputs = {str(p): sha(p) for folder in (SOURCE, ATTR) for p in folder.rglob('*') if p.is_file()}
    out.mkdir()
    offset = np.array([[5., -3.], [-4., 6.]])
    write(out/'protocol.json', dict(cases=['new_1_22000', 'new_3_22000'], gap=15,
        position='truth plus fixed offset, diagnostic only', offset=offset.tolist(),
        rconds=CONDS, matrix_difference_step_m=.001, scalar_difference_steps_m=[.1, .01],
        purpose='Do not assume envelope derivative survives numerical rank truncation',
        protected=protected, inputs=inputs))
    records = []
    tick = time.perf_counter()
    with threadpool_limits(limits=2):
        for path in sorted((ATTR/'pairs').glob('*.json')):
            row = json.loads(path.read_text(encoding='utf-8'))
            if row['gap'] != 15 or row['case'] not in ('new_1_22000', 'new_3_22000'):
                continue
            _, group, rep = row['case'].split('_')
            with np.load(SOURCE/'inputs'/f'{group}_{rep}.npz', allow_pickle=False) as z:
                y = (z['signal'][0, :, :1024]*10**(15/20)+z['signal'][1, :, :1024]+z['noise'][:, :1024]).ravel()
            p = np.asarray(row['truth'])+offset
            center = fit(p, y, True)
            differences = {}
            for step in (.1, .01):
                gradients = [[], []]
                ranks = [[], []]
                for j in range(4):
                    delta = np.zeros((2, 2))
                    delta.flat[j] = step
                    plus, minus = fit(p+delta, y), fit(p-delta, y)
                    for i in range(2):
                        gradients[i].append((plus[i]['value']-minus[i]['value'])/(2*step))
                        ranks[i].append([plus[i]['rank'], minus[i]['rank']])
                differences[str(step)] = dict(gradients=gradients, ranks=ranks)
            record = dict(case=row['case'], positions=p.tolist(), center=center, differences=differences)
            records.append(record)
            write(out/'results.json', records)
            print(row['case'], '梯度核验完成', flush=True)
    for path, identity in {**protected, **inputs}.items():
        assert sha(path) == identity, path
    write(out/'audit.json', dict(completed=True, cases=2, seconds=time.perf_counter()-tick,
        sources_unchanged=True, inputs_unchanged=True, diagnostic_only=True))
    print(out, flush=True)


if __name__ == '__main__':
    main()
