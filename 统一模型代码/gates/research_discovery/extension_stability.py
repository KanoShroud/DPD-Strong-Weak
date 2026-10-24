"""Predeclared Fourier-extension length/rank sensitivity, no parameter selection."""
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
from 统一模型代码.gates.g11.precision_diagnostic import sha, write
from 统一模型代码.gates.g8.physics import C, RECEIVERS
from 统一模型代码.gates.research_discovery.finite_extension_probe import SOURCE, ATTR, dictionary


def extended(positions, guard):
    n, fs = 1024, 50e6
    length = n+guard*int(np.ceil(fs*1000/C))
    k = int(np.floor(7e6*length/fs))
    f = np.arange(-k, k+1)*fs/length
    blocks = []
    for pos in positions:
        d = np.linalg.norm(np.asarray(RECEIVERS)-pos, axis=1)
        t = np.arange(n)[None, :]/fs-(d-d.min())[:, None]/C
        a = np.exp(2j*np.pi*t.reshape(-1, 1)*f[None])
        a *= np.repeat(1/d, n)[:, None]
        a /= np.linalg.norm(a, axis=0)[None]
        blocks.append(a)
    return np.concatenate(blocks, axis=1)


def main():
    out = SOURCE.parent/('round6_stability_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    assert not out.exists() and out.resolve().is_relative_to(ROOT/'outputs_e2e')
    protected = {str(p): sha(p) for p in (ROOT/'统一模型代码/gates').rglob('*.py')}
    inputs = {str(p): sha(p) for folder in (SOURCE, ATTR) for p in folder.rglob('*') if p.is_file()}
    out.mkdir()
    write(out/'protocol.json', dict(guards=[1, 2, 3], rconds=[1e-6, 1e-8, 1e-10],
        input_roots=[str(SOURCE), str(ATTR)], cases='four layouts, realization22000, gap15',
        diagnostic_only=True, selection_by_error_forbidden=True, protected=protected, inputs=inputs))
    records = []
    tick = time.perf_counter()
    with threadpool_limits(limits=2):
        for path in sorted((ATTR/'pairs').glob('*.json')):
            row = json.loads(path.read_text(encoding='utf-8'))
            if row['gap'] != 15 or not row['case'].endswith('_22000'):
                continue
            _, group, rep = row['case'].split('_')
            with np.load(SOURCE/'inputs'/f'{group}_{rep}.npz', allow_pickle=False) as z:
                clean = z['signal'][0, :, :1024]*10**(15/20)+z['signal'][1, :, :1024]
                y = np.column_stack([clean.ravel(), (clean+z['noise'][:, :1024]).ravel()])
            for label, p in [('truth', row['truth']), ('blind_ap5', row['results']['rect']['ap5']['positions'])]:
                for guard in (1, 2, 3):
                    start = time.perf_counter()
                    a = extended(p, guard)
                    if guard == 2:
                        np.testing.assert_allclose(a, dictionary(np.asarray(p)), atol=1e-14)
                    u, s, _ = svd(a, full_matrices=False, lapack_driver='gesdd')
                    for cond in (1e-6, 1e-8, 1e-10):
                        rank = int((s > s[0]*cond).sum())
                        q = u[:, :rank]
                        residual = y-q@(q.conj().T@y)
                        fractions = np.sum(abs(residual)**2, axis=0)/np.sum(abs(y)**2, axis=0)
                        assert np.isfinite(fractions).all()
                        records.append(dict(case=row['case'], label=label, guard=guard, rcond=cond,
                            rank=rank, residual_energy_fraction=fractions.tolist(),
                            shape=list(a.shape), elapsed_including_reused_svd=time.perf_counter()-start))
            write(out/'results.json', records)
            print(row['case'], '稳定性点对完成', flush=True)
    contrasts = []
    for case in sorted({r['case'] for r in records}):
        for guard in (1, 2, 3):
            for cond in (1e-6, 1e-8, 1e-10):
                rows = {r['label']: r for r in records if r['case'] == case and r['guard'] == guard and r['rcond'] == cond}
                delta = np.asarray(rows['blind_ap5']['residual_energy_fraction'])-rows['truth']['residual_energy_fraction']
                contrasts.append(dict(case=case, guard=guard, rcond=cond, wrong_minus_true=delta.tolist()))
    write(out/'contrasts.json', contrasts)
    for path, identity in {**protected, **inputs}.items():
        assert sha(path) == identity, path
    write(out/'audit.json', dict(completed=True, factorizations=24, contrasts=len(contrasts),
        true_better_clean=sum(r['wrong_minus_true'][0] > 0 for r in contrasts),
        true_better_noisy=sum(r['wrong_minus_true'][1] > 0 for r in contrasts),
        seconds=time.perf_counter()-tick, sources_unchanged=True, inputs_unchanged=True))
    print(out, flush=True)


if __name__ == '__main__':
    main()
