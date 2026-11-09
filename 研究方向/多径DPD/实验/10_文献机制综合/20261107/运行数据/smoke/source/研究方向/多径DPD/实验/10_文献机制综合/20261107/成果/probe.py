"""Source-blind two-branch feasibility probe and algebraic unit checks."""
import argparse
import hashlib
import importlib.util
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'AGENTS.md').exists())
BASE = ROOT/'研究方向/多径DPD'
OLD = BASE/'实验/09_文献后机制核验/20261107'
spec = importlib.util.spec_from_file_location('previous_gate', OLD/'成果/run_gate.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def algebra():
    from itertools import combinations
    rng = np.random.default_rng(2026110711)
    discrepancies, invariance = [], []
    for _ in range(100):
        h = rng.normal(size=(4, 2))+1j*rng.normal(size=(4, 2))
        y = rng.normal(size=4)+1j*rng.normal(size=4)
        z = np.column_stack((h, y))
        den = np.linalg.det(h.conj().T@h).real
        residual = y-h@np.linalg.lstsq(h, y, rcond=None)[0]
        lhs = np.vdot(residual, residual).real
        rhs = sum(abs(np.linalg.det(z[list(ids)]))**2 for ids in combinations(range(4), 3))/den
        discrepancies.append(abs(lhs-rhs)/max(1, lhs))
        t = np.array([[1., .3j], [.2, 1.]])
        rr = y-h@t@np.linalg.lstsq(h@t, y, rcond=None)[0]
        invariance.append(abs(np.vdot(rr, rr).real-lhs)/max(1, lhs))
    h = np.array([[1, 2], [3, 5]], dtype=complex)
    s = np.array([2, 4j])
    y = h@s
    cross = h[1, 0]*y[0]-h[0, 0]*y[1]
    predicted = (h[1, 0]*h[0, 1]-h[0, 0]*h[1, 1])*s[1]
    result = dict(seed=2026110711, trials=100,
                  minors_vp_max_relative_error=max(discrepancies),
                  mixing_invariance_max_relative_error=max(invariance),
                  double_source_cross_term_abs=float(abs(cross)),
                  cross_identity_error=float(abs(cross-predicted)),
                  scope='algebra_only_not_localization_performance')
    assert max(discrepancies+invariance) < 1e-11
    assert abs(cross-predicted) < 1e-12 and abs(cross) > 1
    # A free per-frequency column h=y fits y exactly; no geometric information.
    v = np.array([1+1j, 2, 3j, 4])
    free_h = np.column_stack((v, np.ones(4)))
    result['free_channel_residual'] = float(np.linalg.norm(v-free_h@np.linalg.lstsq(free_h, v, rcond=None)[0]))
    # Nested subspaces: LOS ordering can reverse after nuisance expansion.
    e = np.eye(4)
    yy = e[:, 0]
    a = np.column_stack((e[:, 1], e[:, 2]))
    b = np.column_stack(((e[:, 0]+e[:, 1])/np.sqrt(2), e[:, 2]))
    loss = lambda hh: float(np.linalg.norm(yy-hh@np.linalg.lstsq(hh, yy, rcond=None)[0])**2)
    result['nested_subspace_counterexample'] = dict(los_a=loss(a), los_b=loss(b),
        expanded_a=loss(np.column_stack((a, yy))), expanded_b=loss(b))
    return result


def run(out, limit):
    out.mkdir(parents=True, exist_ok=False)
    files = list((BASE/'代码/mp_dpd').rglob('*.py'))+[Path(__file__), HERE/'执行合同.md', OLD/'成果/run_gate.py']
    hashes = {str(p): sha(p) for p in files}
    for p in files:
        gate.copy_checked(p, out/'source'/p.relative_to(ROOT))
    gate.cfg.write(out/'preflight.json', dict(source_hashes=hashes, threads=2, limit=limit))
    gate.cfg.write(out/'algebra.json', algebra())
    rows, start = [], time.perf_counter()
    previous = gate.read(OLD/'运行数据/common18/rows.json')[:limit]
    for i, old in enumerate(previous):
        if time.perf_counter()-start > 1800:
            raise TimeoutError('Operational stage limit; no algorithm convergence claim')
        tag = old['tag']
        src = OLD/'运行数据/common18/inputs'/tag/'iq.npz'
        assert sha(src) == old['input_sha']
        local = out/'inputs'/tag/'iq.npz'
        gate.copy_checked(src, local)
        with np.load(local, allow_pickle=False) as z:
            iq = z['iq']
        t0 = time.perf_counter()
        engine = gate.api.create_engine()
        front = engine.frontend(iq)
        reg = engine.estimate_function.__globals__['bc'].regions
        selection = reg.alternate_pair(front['refined'], front['reward'])
        front_seconds = time.perf_counter()-t0
        solutions = {}
        for branch in (['first', 'second'] if i % 2 == 0 else ['second', 'first']):
            fresh = gate.api.create_engine()
            original = "pairs = [selection['first'], selection['second']] if two_regions else [fc.engine.joint_select(geom['reward'])[0]]"
            fresh.estimate_function = gate.api.runtime().ad.clone(fresh.estimate_function,
                [(original, 'pairs = [blind_pair]')], {'blind_pair': selection[branch]})
            print('START', tag, branch, flush=True)
            solved = fresh.mp(iq)
            gate.cfg.write(out/f'{tag}_{branch}_decision.json', solved)
            f, y, n, _ = fresh.fc.cal.spectral(iq)
            x = np.asarray(solved['final']['x'])
            value, gradient = fresh.objective(x, f, y, solved['paths'])
            pg = float(np.max(abs(x-fresh.ref.project(x-gradient, n))))
            assert np.isclose(value, solved['final']['cost'], rtol=1e-9, atol=1e-12)
            solutions[branch] = dict(p=solved['final']['p'], cost=value, pg=pg,
                K=len(solved['paths']), normal=solved['normal'], stop=solved['stop'],
                seconds=solved['seconds'], los_cost=solved['los']['cost'])
            print('END', tag, branch, solved['stop'], flush=True)
        # Freeze no-truth decisions before reading evaluation truth.
        a, b = solutions['first'], solutions['second']
        dominates = lambda u, v: u['cost'] <= v['cost']+1e-12 and u['K'] <= v['K'] and (u['cost'] < v['cost']-1e-12 or u['K'] < v['K'])
        choice = 'unresolved'
        if a['normal'] and b['normal']:
            if dominates(a, b):
                choice = 'first'
            elif dominates(b, a):
                choice = 'second'
        decision = dict(tag=tag, condition=old['condition'], branches=solutions,
            pareto_choice=choice, selection=selection, input_sha=sha(local),
            seconds_including_repeated_front=front_seconds+a['seconds']+b['seconds'])
        gate.cfg.write(out/f'{tag}_blind_decision.json', decision)
        truth_path = Path(old['truth_evaluation_file'])
        assert sha(truth_path) == old['truth_file_sha']
        with np.load(truth_path, allow_pickle=False) as z:
            truth = z['p']
        for item in solutions.values():
            item['errors_m'] = gate.errors(item['p'], truth)
        decision['original'] = old['methods']['MP']
        rows.append(decision)
        gate.cfg.write(out/'rows.json', rows)
        assert sha(local) == old['input_sha']
    assert all(sha(p) == h for p, h in hashes.items())
    gate.cfg.write(out/'completion.json', dict(count=len(rows), source_unchanged=True,
        seconds=time.perf_counter()-start, scope='development_feasibility_not_new_algorithm_validation'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert 1 <= args.limit <= 18
    assert args.out.resolve().is_relative_to(HERE.parent/'运行数据')
    with threadpool_limits(limits=2):
        run(args.out.resolve(), args.limit)
