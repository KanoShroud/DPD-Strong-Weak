#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exploratory branch identifiability probe for DPD-Strong-Weak.

No source truth is used by the blind fitting code. The original estimator and
physical generator are not changed. Results are DEVELOPMENT diagnostics only.

For a PyCharm one-click run, execute 一键运行_第20轮.py directly (no arguments).
This module remains importable for unit tests and optional command-line audit.

The frozen common18 raw IQ data must exist in the original project workspace.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

SCHEMA = "BRANCH_DISCRIMINABILITY_EXPLORATORY_V1"
# These are diagnostic choices, not physical priors or tuned estimator settings.
MAX_DELAY_SECONDS = 20e-6
FREQ_BLOCK_BINS = 8
SEED = 2026110901


def digest(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def filehash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value))


def save(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".pending")
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, default=canonical, indent=2), encoding="utf-8")
    tmp.replace(path)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fold_mask(f: np.ndarray, fold: int) -> np.ndarray:
    """Two pre-specified interleaved spectral-block folds; no outcome tuning."""
    assert fold in (0, 1)
    return (np.arange(len(f)) // FREQ_BLOCK_BINS) % 2 == fold


def train_only_iq(iq: np.ndarray, f: np.ndarray, train: np.ndarray, fs: float) -> np.ndarray:
    """For candidate generation, mask held-out bins *before* GCC frontend.

    This is a diagnostic data split, not the unchanged production frontend.
    Finite-record leakage/correlation still prevents i.i.d. CV claims.
    """
    n = iq.shape[1]
    allbins = np.fft.fftfreq(n, 1 / fs)
    idx = np.rint(f * n / fs).astype(int) % n
    assert np.max(abs(allbins[idx] - f)) < 1e-5
    mask = np.zeros(n, bool)
    mask[idx[train]] = True
    z = np.fft.fft(iq, axis=1, norm="ortho")
    masked = np.where(mask[None, :], z, 0)
    iq_train = np.fft.ifft(masked, axis=1, norm="ortho")
    check = np.fft.fft(iq_train, axis=1, norm="ortho")
    assert np.allclose(check[:, mask], z[:, mask], rtol=1e-10, atol=1e-11 * max(1., abs(z).max()))
    assert np.max(abs(check[:, ~mask])) < 1e-10 * max(1., abs(z).max())
    return iq_train


def branch_approx_score(engine, p: np.ndarray, f, y, n) -> tuple[np.ndarray, float]:
    """Blind LOS nuisance initialization at fixed ordered source positions."""
    gains = np.column_stack([engine.rank_one(y, f, pp)[0] for pp in p])
    # Source permutation is intentionally not identified in this representation.
    ref = engine.reference
    free = [m for m in range(engine.M) if m != ref]
    g = gains[free]
    x = np.r_[p.ravel() / engine.ref.UNIT, g.real.ravel(), g.imag.ravel()]
    assert len(x) == engine.D and np.all(np.isfinite(x))
    x = engine.ref.project(x, n)
    score = engine.objective(x, f, y, [])[0]
    return x, float(score)


def fixed_nuisance_fit(engine, x0, paths, f, y, n, maxiter=70):
    """Optimize ALL allowed nuisance variables, with 2 positions kept EXACTLY fixed.

    Same physical per-path unit-disk constraints and 20-us delay range for every
    branch. This is a conditional numerical fit, NOT global variable projection.
    """
    p0 = np.array(x0[:4], copy=True)
    x0 = engine.ref.project(np.asarray(x0, float), n)
    assert np.array_equal(x0[:4], p0)
    D = engine.D
    assert len(x0) == D + 3 * len(paths)
    cache = [None, None]
    evaluations = 0

    def evaluate(z):
        nonlocal evaluations
        if cache[0] is None or not np.array_equal(z, cache[0]):
            x = np.r_[p0, z]
            val, grad = engine.objective(x, f, y, paths)
            if not np.isfinite(val) or not np.all(np.isfinite(grad)):
                raise FloatingPointError("nonfinite conditional loss/gradient")
            cache[:] = [z.copy(), (float(val), np.asarray(grad[4:]))]
            evaluations += 1
        return cache[1]

    bounds = [(None, None)] * (D - 4)
    for _ in paths:
        bounds.extend([(None, None), (None, None), (0., MAX_DELAY_SECONDS * engine.ref.B)])
    constraints = []
    if paths:
        def cons(z):
            return 1. - np.sum(z[D-4:].reshape(-1, 3)[:, :2]**2, axis=1)
        def jac(z):
            a = np.zeros((len(paths), len(z)))
            for i in range(len(paths)):
                k = D-4+3*i
                a[i, k:k+2] = -2*z[k:k+2]
            return a
        constraints = [dict(type="ineq", fun=cons, jac=jac)]
    start = time.perf_counter()
    try:
        opt = minimize(evaluate, x0[4:], jac=True, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options=dict(maxiter=maxiter, ftol=1e-11, disp=False))
        candidate = np.r_[p0, opt.x]
        projected = engine.ref.project(candidate, n)
        # Reject nonphysical output; never silently replace or relax constraints.
        feasible = float(np.linalg.norm(candidate - projected)) < 1e-7
        cnew, gnew = engine.objective(candidate, f, y, paths)
        cold = engine.objective(x0, f, y, paths)[0]
        accepted = bool(feasible and np.isfinite(cnew) and cnew <= cold + 1e-10)
        x = candidate if accepted else x0.copy()
        cost, grad = engine.objective(x, f, y, paths)
        direction = np.r_[np.zeros(4), -np.asarray(grad)[4:]]
        pg = np.max(abs(x[4:] - engine.ref.project(x + direction, n)[4:]))
        return x, dict(cost=float(cost), accepted=accepted, reason=str(opt.message),
                       iterations=int(opt.nit), calls=evaluations,
                       conditional_projected_gradient=float(pg),
                       nuisance_stationary=bool(pg <= 1e-7),
                       closure=float(np.linalg.norm(x-engine.ref.project(x, n))),
                       elapsed_seconds=time.perf_counter()-start)
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
        cost = engine.objective(x0, f, y, paths)[0]
        return x0.copy(), dict(cost=float(cost), accepted=False, reason=repr(exc),
                               iterations=0, calls=evaluations,
                               conditional_projected_gradient=None,
                               nuisance_stationary=False, closure=0.,
                               elapsed_seconds=time.perf_counter()-start)


def propose_path(engine, x, paths, f, y, n):
    """A blind fixed-position channel-path proposal from *joint* IQ residual.

    Condition on BOTH current unknown waveforms. No source is subtracted
    permanently; physical per-link path ratio is projected to the unit disk.
    No individual cross-spectrum components or path truth are supplied.
    """
    h, inv, s, residual, base, *_ = engine.core(x, f, y, paths)
    del h, inv
    fs = engine.ref.FS
    max_idx = min(int(round(MAX_DELAY_SECONDS * fs)), n - 1)
    indices = np.rint(f * n / fs).astype(int) % n
    assert len(np.unique(indices)) == len(indices)
    best = None
    for m in range(engine.M):
        for q in range(2):
            signal = base[:, m, q] * s[:, q]
            den = float(np.vdot(signal, signal).real)
            if den < 1e-24: continue
            padded = np.zeros(n, complex)
            padded[indices] = signal.conj() * residual[:, m]
            correlations = np.fft.ifft(padded) * n
            c = correlations[1:max_idx+1]
            alpha = c / den
            alpha /= np.maximum(1., abs(alpha))
            improvement = 2*np.real(alpha.conj()*c) - den*abs(alpha)**2
            j = int(np.argmax(improvement))
            if best is None or improvement[j] > best[0]:
                best = (float(improvement[j]), int(m), int(q), int(j+1), complex(alpha[j]))
    if best is None:
        return (0, 0), np.array([0., 0., .25]), dict(predicted_gain=0., status="zero_atom_fallback")
    val, m, q, delay, amp = best
    return (m, q), np.array([amp.real, amp.imag, delay/fs*engine.ref.B]), dict(
        predicted_gain=val, status="matched_joint_residual", delay_samples=delay)


def fit_branch(engine, position_pair, f, y, n, k_budget, maxiter):
    x, init_score = branch_approx_score(engine, position_pair, f, y, n)
    paths = []
    steps = []
    x, state = fixed_nuisance_fit(engine, x, paths, f, y, n, maxiter=maxiter)
    steps.append(dict(K=0, state=state))
    for _ in range(k_budget):
        key, coeff, proposal = propose_path(engine, x, paths, f, y, n)
        oldcost = engine.objective(x, f, y, paths)[0]
        expanded = engine.ref.project(np.r_[x, coeff], n)
        pp = paths + [key]
        # Zero-amplitude new path keeps current model feasible and nested.
        if engine.objective(expanded, f, y, pp)[0] > oldcost + 1e-10:
            expanded[-3:-1] = 0.
        updated, state = fixed_nuisance_fit(engine, expanded, pp, f, y, n, maxiter=maxiter)
        assert np.array_equal(updated[:4], x[:4])
        x, paths = updated, pp
        steps.append(dict(K=len(paths), path=key, proposal=proposal, state=state))
    return x, paths, dict(initial_LOS_cost=init_score, steps=steps,
                          training_cost=float(engine.objective(x, f, y, paths)[0]))


def select_blind_pairs(engine, pool, f, y, n, max_points, max_pairs):
    # Down-select points by a deterministic rank spectrum (not by true position).
    # Pair selection is stratified by its *training-only* LOS objective to avoid
    # selecting exclusively the best LOS hypothesis.
    original = np.asarray(pool, float)
    count = len(original)
    assert original.shape == (count, 2) and count >= 2
    scores = np.array([engine.rank_one(y, f, p)[2] for p in original])  # explained below
    # rank_one returns (gain, waveform, fit); fit is not a scalar.
    energy = np.sum(abs(y)**2)
    singles = np.array([float(np.sum(abs(y-fit)**2)/energy) for fit in scores])
    order = np.argsort(singles, kind="stable")
    positions = np.unique(np.rint(np.linspace(0, count-1, min(count, max_points))).astype(int))
    ids = [int(order[j]) for j in positions]
    points = original[ids]
    proposed = []
    for i in range(len(points)):
        for j in range(len(points)):
            x, approx = branch_approx_score(engine, points[[i,j]], f, y, n)
            proposed.append(dict(i=i,j=j,los=float(approx),x=x))
    proposed.sort(key=lambda t:(t['los'], t['i'], t['j']))
    if len(proposed) <= max_pairs: return points, proposed
    # Stratify over the full training-only LOS loss ordering. Quartile coverage
    # is a computational design, NOT a physical rank or a correctness claim.
    chosen = []
    rank_positions = np.unique(np.rint(np.linspace(0, len(proposed)-1, max_pairs)).astype(int))
    for k in rank_positions:
        chosen.append(proposed[int(k)])
    return points, chosen


def project_root_from_file(here: Path) -> Path:
    return next(p for p in here.resolve().parents if (p/"AGENTS.md").exists())


def get_r14(base):
    path = base/"实验/14_公开信道记忆核验/20261107/成果/run.py"
    sp = importlib.util.spec_from_file_location("branch_probe_r14", path)
    mod = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = mod
    sp.loader.exec_module(mod)
    return mod


def test_instance(base, case_index, out, max_points, max_pairs, k_budget, maxiter):
    r14 = get_r14(base)
    old = r14.OLD/"运行数据/common18"
    records = load(old/"rows.json")
    if case_index >= len(records): raise ValueError("case index outside frozen common18")
    record = records[case_index]
    tag = record["tag"]
    iq_file = old/"inputs"/tag/"iq.npz"
    if not iq_file.exists(): raise FileNotFoundError("Frozen IQ not accessible: "+str(iq_file))
    if filehash(iq_file) != record["input_sha"]:
        raise RuntimeError("Frozen IQ hash differs from registered record")
    with np.load(iq_file, allow_pickle=False) as data:
        assert list(data.files) == ["iq"], "IQ input must not contain truth"
        iq = data["iq"]
    identity = dict(iq_file=str(iq_file), iq_sha=filehash(iq_file), iq_array_sha=digest(iq),
                    source_script_sha=filehash(Path(__file__)),
                    r14_sha=filehash(Path(r14.__file__)), condition=record["condition"], tag=tag)
    e = r14.engine_for("CAP_COUNTED")
    f,y,n,_ = e.fc.cal.spectral(iq)
    assert iq.shape == (4,n)
    fold_results=[]
    for fold in range(2):
        train=fold_mask(f,fold)
        training_iq=train_only_iq(iq,f,train,e.ref.FS)
        front=e.frontend(training_iq)  # Blind selection uses training IQ only.
        points, pairs=select_blind_pairs(e,front["refined"],f[train],y[train],n,max_points,max_pairs)
        records2=[]
        for num,trial in enumerate(pairs):
            p=points[[trial['i'],trial['j']]]
            x,paths,state=fit_branch(e,p,f[train],y[train],n,k_budget,maxiter)
            validation=float(e.objective(x,f[~train],y[~train],paths)[0])
            records2.append(dict(candidate_index=num, pair=[trial['i'],trial['j']],
                  positions=p, paths=paths, x=x, training=state,
                  validation_cost=validation,
                  conditional_pg=state['steps'][-1]['state']['conditional_projected_gradient'],
                  numerical_stationary=state['steps'][-1]['state']['nuisance_stationary']))
        fold_results.append(dict(fold=fold, train_bins=int(train.sum()), val_bins=int((~train).sum()),
              frontend_candidate_count=len(front['refined']), full_observed_candidate_pool=front['refined'],
              tested_points=points,
              tested_pairs=records2, training_spectral_sha=digest(y[train]),
              validation_spectral_sha=digest(y[~train])))
        print(tag,"fold",fold,"candidates",len(records2),"done",flush=True)
    dst=out/"blind"/(tag+".json")
    save(dst,dict(schema=SCHEMA,scope="DEVELOPMENT_NOT_VALIDATION",identity=identity,folds=fold_results,
         settings=dict(max_points=max_points,max_pairs=max_pairs,K=k_budget,conditional_maxiter=maxiter,
                       frequency_block_bins=FREQ_BLOCK_BINS,max_delay_seconds=MAX_DELAY_SECONDS)))
    # The blind decision file exists on disk BEFORE evaluation truth is loaded.
    return dst, record


def evaluation_only(decision_path, record, out, oracle_control=True):
    raw=load(decision_path)
    with np.load(record["truth_evaluation_file"],allow_pickle=False) as data:
        truth=np.asarray(data["p"])
    assert filehash(Path(record["truth_evaluation_file"])) == record["truth_file_sha"]
    from scipy.optimize import linear_sum_assignment
    def err(est):
        p=np.asarray(est)
        d2=np.sum((p[:,None]-truth[None])**2,axis=2)
        row,col=linear_sum_assignment(d2)
        return float(np.sqrt(np.mean(d2[row,col])))
    evaluated=[]
    # Oracle-control is strictly post-decision and NEVER changes blind branch
    # retention, ranking, or parameters. It tests the unchosen near-truth pool.
    engine=None
    if oracle_control:
        root=project_root_from_file(Path(__file__))
        base=root/"研究方向/多径DPD"
        r14=get_r14(base)
        engine=r14.engine_for("CAP_COUNTED")
        iqpath=Path(raw["identity"]["iq_file"])
        assert filehash(iqpath)==raw["identity"]["iq_sha"]
        with np.load(iqpath,allow_pickle=False) as data: iq=data["iq"]
        f,y,n,_=engine.fc.cal.spectral(iq)
    for fold in raw["folds"]:
        entries=[]
        for b in fold["tested_pairs"]:
            entries.append(dict(candidate_index=b["candidate_index"],rmse_m=err(b["positions"]),
                           validation_cost=b["validation_cost"],
                           conditional_pg=b["conditional_pg"],
                           numerical_stationary=b["numerical_stationary"]))
        best=min(entries,key=lambda b:b["validation_cost"])
        comparable=[q for q in entries if q["numerical_stationary"]]
        best_comparable=(min(comparable,key=lambda b:b["validation_cost"])
                         if comparable else None)
        nearby=[p for p in entries if p["rmse_m"]<=20.]
        oracle=None
        if oracle_control:
            candidate_pool=np.asarray(fold["full_observed_candidate_pool"],float)
            if len(candidate_pool):
                errgrid=np.sum((candidate_pool[:,None,:]-truth[None,:,:])**2,axis=2)
                d01=errgrid[:,0,None]+errgrid[None,:,1]
                d10=errgrid[:,1,None]+errgrid[None,:,0]
                dist2=np.minimum(d01,d10)
                ix=np.unravel_index(np.argmin(dist2),dist2.shape)
                pp=candidate_pool[list(ix)]
                train=fold_mask(f,fold["fold"])
                x,paths,state=fit_branch(engine,pp,f[train],y[train],n,
                              raw["settings"]["K"],raw["settings"]["conditional_maxiter"])
                oracle=dict(oracle_selection="NEAREST_POOL_PAIR_USING_GROUND_TRUTH_ONLY_AFTER_BLIND_SAVING",
                            pool_indices=list(ix),positions=pp,rmse_m=err(pp),
                            training_cost=state["training_cost"],
                            validation_cost=float(engine.objective(x,f[~train],y[~train],paths)[0]),
                            conditional_pg=state["steps"][-1]["state"]["conditional_projected_gradient"],
                            numerical_stationary=state["steps"][-1]["state"]["nuisance_stationary"],
                            paths=paths)
        evaluated.append(dict(fold=fold["fold"],candidates=entries,oracle_pool_control=oracle,
                   near_true_20m_count=len(nearby),chosen_by_validation=best,
                   chosen_among_stationary_only=best_comparable,
                   conditional_numerically_stationary_count=len(comparable),
                   near_true_validation_cost_range=[min((p["validation_cost"] for p in nearby),default=None),
                                                    max((p["validation_cost"] for p in nearby),default=None)]))
    dst=out/"evaluation"/(raw["identity"]["tag"]+".json")
    save(dst,dict(scope="OFFLINE_TRUTH_ATTRIBUTION_NOT_BLIND_RESULT",decision_file=str(decision_path),
                  decision_sha=filehash(decision_path),true_positions=truth,folds=evaluated))
    return dst


def selftest():
    rng=np.random.default_rng(2026110902)
    m,n=4,101
    f=np.fft.fftfreq(n,1/50e6)
    c=np.exp(1j*rng.normal(size=(n,m,2)))
    x=rng.normal(size=(n,2))+1j*rng.normal(size=(n,2))
    y=np.einsum("fmq,fq->fm",c,x)
    s=np.linalg.pinv(c)@y[:,:,None]
    s=s[:,:,0]
    assert np.linalg.norm(y-np.einsum("fmq,fq->fm",c,s))<1e-11
    z=np.zeros((4,n),complex)
    z[:,:]=rng.normal(size=(4,n))+1j*rng.normal(size=(4,n))
    train=fold_mask(f,0)
    w=train_only_iq(z,f,train,50e6)
    wf=np.fft.fft(w,axis=1,norm="ortho")
    orig=np.fft.fft(z,axis=1,norm="ortho")
    indices=np.arange(n)
    assert np.max(abs(wf[:,indices[train]]-orig[:,indices[train]]))<1e-12
    assert np.max(abs(wf[:,indices[~train]]))<1e-12
    # The exact projection is invariant under a nonsingular source-basis change.
    T=np.array([[1.+.2j,.3-.1j],[.6+.1j,1.-.3j]])
    cp=c@T
    norm=lambda v: float(sum(np.linalg.norm(y[i]-v[i]@np.linalg.pinv(v[i])@y[i])**2 for i in range(n)))
    assert abs(norm(c)-norm(cp))<1e-20
    result={"selftest":"PASS","frequency_bins":n,"train_bins":int(train.sum()),
            "projection_basis_invariance_error":abs(norm(c)-norm(cp)),
            "no_truth_needed":True}
    print(json.dumps(result,ensure_ascii=False,indent=2));return result


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--selftest",action="store_true")
    a.add_argument("--pilot",action="store_true")
    a.add_argument("--case-indices",default="0,8,16")
    a.add_argument("--max-points",type=int,default=6)
    a.add_argument("--max-pairs",type=int,default=6)
    a.add_argument("--path-budget",type=int,default=4)
    a.add_argument("--maxiter",type=int,default=70)
    a.add_argument("--skip-oracle-control",action="store_true")
    a.add_argument("--output",default=None)
    args=a.parse_args()
    if args.selftest: selftest();return
    if not args.pilot: a.error("Use 一键运行_第20轮.py in PyCharm or explicit --pilot; formal batch disabled")
    if not (2<=args.max_points<=16 and 1<=args.max_pairs<=32 and 0<=args.path_budget<=16 and 1<=args.maxiter<=300):
        a.error("pilot limits exceeded")
    repo=project_root_from_file(Path(__file__))
    if str(repo) not in sys.path:
        sys.path.insert(0,str(repo))
    base=repo/"研究方向/多径DPD"
    indices=[int(s.strip()) for s in args.case_indices.split(",")]
    if len(indices)!=len(set(indices)) or not indices or any(i<0 or i>=18 for i in indices):
        a.error("specify distinct frozen common18 case indices in [0,17]")
    out=Path(args.output) if args.output else base/"实验/20_分支可区分性核验/20261109/运行数据"
    out.mkdir(parents=True,exist_ok=False)
    source_files=list((base/"代码/mp_dpd").rglob("*.py")) + [Path(__file__),
       base/"实验/14_公开信道记忆核验/20261107/成果/run.py",
       base/"实验/09_文献后机制核验/20261107/成果/run_gate.py"]
    source_hashes={str(p.resolve()):filehash(p) for p in source_files}
    save(out/"protocol.json",dict(schema=SCHEMA,case_indices=indices,script_sha=filehash(Path(__file__)),
           source_hashes=source_hashes, settings=vars(args),
           no_truth_for_estimator=True,formal_batch=False))
    with threadpool_limits(limits=2):
        finished=[]
        for idx in indices:
            file,record=test_instance(base,idx,out,args.max_points,args.max_pairs,args.path_budget,args.maxiter)
            finished.append((file,record))
        # Only after ALL blind results are safely written do we open truth banks.
        evaluation=[evaluation_only(file,record,out,oracle_control=not args.skip_oracle_control) for file,record in finished]
    changed=[path for path,h in source_hashes.items() if filehash(Path(path))!=h]
    if changed: raise RuntimeError("Source identity drift during pilot: "+str(changed))
    save(out/"completion.json",dict(schema=SCHEMA,blind_files=[str(t[0]) for t in finished],
               evaluation_files=evaluation,scope="DEVELOPMENT_ONLY",
               source_hashes_verified=True,source_sha=filehash(Path(__file__))))
    print("Exploratory pilot completed:",out)


if __name__ == "__main__": main()
