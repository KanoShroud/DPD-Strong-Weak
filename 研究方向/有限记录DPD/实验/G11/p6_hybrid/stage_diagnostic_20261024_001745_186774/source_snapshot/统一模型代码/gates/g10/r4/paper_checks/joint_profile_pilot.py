"""Bounded common-stop and same-candidate joint-profile pilot; no frozen edits."""
import argparse
import copy
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from . import joint_profile_math as joint
from .spectral_window_check import load_inputs
from .segment_model_check import BASE, read, write, checked_bytes
from ...physics import grid
from ...methods import peaks
from ...metrics import evaluate
from ...r1.physics import small_fit
from ...r2.solvers import local
from ...r3.solvers import Model, Trace, projected_gradient
from ...r3.common_stop import refine_pair
from ..adaptation_review import run as native_vp, refine as vp_step
from ..solvers import run as frozen_ap

METHODS = ('AP_LBFGS', 'AP_VP_BLOCK', 'AP_VP_JOINT')


def synced_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def ap_candidates(model, cfg):
    points = grid(cfg['grid'])
    first = points[peaks(points, model.conditional(points), cfg['top'])[0]]
    first, _, _ = local(model, first[None], cfg)
    ids = peaks(points, model.conditional(points, first), cfg['top'])
    candidates = np.asarray([np.vstack([first, points[i]]) for i in ids])
    if candidates.shape != (5, 2, 2):
        raise RuntimeError('Exactly five candidates required')
    return candidates


def run_ap(iq, cfg, method, expected=None):
    started = synced_time()
    trace = Trace(lambda: None)
    model = Model(iq, trace.check, sample_rate_hz=cfg['sample_rate_hz'], half_band_hz=cfg['half_band_hz'])
    trace.model = model
    candidates = ap_candidates(model, cfg)
    if expected is not None:
        np.testing.assert_array_equal(candidates, expected)
    after_candidates = synced_time()
    branches = []
    for rank, start in enumerate(candidates, 1):
        if method == 'AP_LBFGS':
            p, value, detail = refine_pair(model, trace, start, 'AP_1_JOINT', cfg)
        else:
            p, value, detail = joint.refine(model, start, cfg, block=method == 'AP_VP_BLOCK')
        branches.append(dict(rank=rank, positions=p.tolist(), objective=value, **detail))
    best = min(branches, key=lambda b: (b['objective'], b['rank']))
    # All variants pay for an identical independent final certificate.
    value, gradient = model.joint(best['positions'], True)
    done = synced_time()
    trace.model = None
    return dict(method=method, positions=best['positions'], objective=value,
                projected_gradient=projected_gradient(best['positions'], gradient),
                stop_reason=best['stop_reason'], seconds=done-started,
                preparation_seconds=after_candidates-started, refinement_seconds=done-after_candidates,
                counts=copy.deepcopy(model.counts), candidates=candidates.tolist(), branches=branches)


def common_vp(iq, cfg, guard=lambda: None):
    """Preserve native updates; only replace second-source cycle stop.

    Full LS/gradient certificates are read-only: they never feed coefficients
    into the cyclic trajectory. Final joint LS still occurs after cyclic stop.
    First-source phase retains its original energy stopping condition.
    """
    start = synced_time()
    trace = Trace(guard)
    model = Model(iq, trace.check, sample_rate_hz=cfg['sample_rate_hz'], half_band_hz=cfg['half_band_hz'])
    trace.model = model
    residual, positions, atoms, coeffs = model.x, [], [], []
    cycles, diagnostics, native_checkpoint = [], [], None
    certificate_seconds = 0.
    for source in range(2):
        points = grid(cfg['grid'])
        initial = points[int(np.argmax(model.conditional(points, residual=residual)))]
        p, a, q, _ = vp_step(model, initial, residual, cfg, True)
        positions.append(p)
        atoms.append(a)
        coeffs.append(q)
        residual = residual-a*q[:, None]
        reason = 'ITERATION_LIMIT'
        no_motion = 0
        for cycle in range(cfg['bnomp_cycles']):
            guard()
            before = float(residual.abs().square().sum().cpu()/model.energy.cpu())
            old_positions = np.array(positions)
            for slot in range(len(positions)):
                data = residual+atoms[slot]*coeffs[slot][:, None]
                p, a, q, _ = vp_step(model, positions[slot], data, cfg, True)
                positions[slot], atoms[slot], coeffs[slot] = p, a, q
                residual = data-a*q[:, None]
            after = float(residual.abs().square().sum().cpu()/model.energy.cpu())
            model.counts['iterations'] += 1
            energy_stable = abs(before-after) <= cfg['bnomp_energy_tolerance']
            if source == 0:
                if energy_stable:
                    reason = 'NATIVE_ENERGY_STABLE'
                    break
                continue
            certificate_start = time.perf_counter()
            value, gradient = model.joint(positions, True)
            pg = projected_gradient(positions, gradient)
            certificate_seconds += time.perf_counter()-certificate_start
            record = dict(cycle=cycle+1, objective=value, projected_gradient=pg,
                          energy_change=abs(before-after), seconds=synced_time()-start)
            if energy_stable and native_checkpoint is None:
                native_checkpoint = dict(**record, positions=np.array(positions).tolist())
            if cycle < 5 or (cycle+1) % 25 == 0 or pg <= 1e-8 or energy_stable and native_checkpoint['cycle'] == cycle+1:
                diagnostics.append(record)
            if pg <= 1e-8:
                reason = 'COMMON_GRADIENT_REACHED'
                break
            no_motion = no_motion+1 if np.array_equal(old_positions, positions) else 0
            if no_motion >= 2:
                reason = 'NUMERICAL_STAGNATION'
                break
        matrix = torch.stack(atoms, -1)
        scale = torch.linalg.vector_norm(matrix, dim=1)
        fitted, residual = small_fit(matrix/scale[:, None, :], model.x)
        fitted = fitted/scale
        coeffs = [fitted[:, i] for i in range(len(positions))]
        model.counts['coefficient_joint_refits'] = model.counts.get('coefficient_joint_refits', 0)+1
        cycles.append(dict(source=source+1, cycles=cycle+1, stop_reason=reason))
    value, gradient = model.joint(positions, True)
    done = synced_time()
    trace.model = None
    return dict(method='VP_COMMON', positions=np.array(positions).tolist(), objective=value,
                projected_gradient=projected_gradient(positions, gradient), stop_reason=reason,
                seconds=done-start, common_check_seconds=certificate_seconds,
                counts=model.counts, native_checkpoint=native_checkpoint,
                cycles=cycles, diagnostics=diagnostics)


def prepare(limit):
    cache, _, identities = load_inputs(limit)
    pointer_path = BASE/'formal_run.json'
    pointer = read(pointer_path)
    formal = Path(pointer['out']).resolve(strict=True)
    contract = read(formal/'contract.json', pointer['contract_sha'])
    identities[str(formal/'contract.json')] = pointer['contract_sha']
    ap_cfg = copy.deepcopy(contract['configurations']['AP_5_JOINT'])
    vp_cfg = copy.deepcopy(contract['configurations']['PROFILE_DIAG'])
    assert ap_cfg['gtol'] == 1e-8 and ap_cfg['common_ftol'] == 0.
    assert vp_cfg['bnomp_cycles'] == 10000
    return cache, ap_cfg, vp_cfg, identities


def report(out, rows, elapsed, status):
    write(out/'report.json', dict(status=status, rows=rows, elapsed_seconds=elapsed,
                                  scope='Finite development pilot; not formal independent evidence'))
    summaries = []
    for name in ('VP_NATIVE', 'VP_COMMON')+METHODS:
        subset = [r for r in rows if r['method'] == name]
        if not subset:
            continue
        summaries.append(dict(method=name, n=len(subset),
            success=sum(all(r['metrics']['success']['20']) for r in subset),
            common_precision=sum(r['projected_gradient'] <= 1e-8 for r in subset),
            mean_seconds=float(np.mean([r['seconds'] for r in subset])),
            weak_rmse_m=float(np.sqrt(np.mean([r['metrics']['errors_m'][1]**2 for r in subset]))),
            mean_gradient=float(np.mean([r['projected_gradient'] for r in subset]))))
    with (out/'summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
        if summaries:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    return summaries


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=8)
    parser.add_argument('--checks-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.limit <= 8:
        raise ValueError('The approved bounded pilot has eight development observations')
    cache, ap_cfg, vp_cfg, identities = prepare(args.limit)
    out = BASE/('joint_profile_pilot_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    assert not out.exists() and out.resolve().is_relative_to(BASE.resolve())
    out.mkdir()
    write(out/'contract.json', dict(inputs=identities, ap_config=ap_cfg, vp_config=vp_cfg,
          tasks=[c[0] for c in cache], methods=['VP_NATIVE', 'VP_COMMON', *METHODS],
          newton=dict(maxiter=ap_cfg['local_maxiter'], hessian_floor=1e-9,
                      step_cap_m=50., armijo=1e-4, maxls=ap_cfg['maxls']),
          common_gradient=1e-8, coordinates='metres', true_positions='evaluation only',
          common_vp='native coefficient trajectory retained; joint certificates do not feed back',
          budget_seconds=10800, checks_only=args.checks_only,
          revision='EQUAL_MEMO_CHECK_COST_COUNTERBALANCED_V2'))
    started = time.perf_counter()
    rows = []
    print(f'结果目录：{out}', flush=True)
    try:
        key, signal, noise, _, _ = cache[0]
        iq = signal.sum(0)+noise
        m = Model(iq)
        # Reduce only the synthetic derivative-check frequency set, never solver inputs.
        m.f, m.x = m.f[::67], m.x[::67]
        m.n, m.energy = len(m.f), m.x.abs().square().sum()
        math_checks = joint.check(m)
        baseline = frozen_ap(iq, 'AP_5_JOINT', ap_cfg)
        replay = run_ap(iq, ap_cfg, 'AP_LBFGS')
        np.testing.assert_array_equal(replay['candidates'], baseline['mechanism']['candidate_list'])
        np.testing.assert_allclose(replay['positions'], baseline['positions'], atol=1e-6)
        # Independently check full-band derivatives at a nonstationary initial branch.
        m = Model(iq)
        p = np.asarray(replay['candidates'][0])
        v, g, _ = joint.derivatives(m, p)
        # Do not let the new derivative cache turn this into a self-comparison.
        m.memo.clear()
        v0, g0 = m.joint(p, True)
        np.testing.assert_allclose(v, v0, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(g, g0, rtol=1e-6, atol=1e-10)
        _, _, full_h = joint.derivatives(m, p)
        xx = torch.tensor(p.ravel(), device=m.device, dtype=torch.float64, requires_grad=True)
        auto_h = torch.autograd.functional.hessian(m.joint_tensor, xx).cpu().numpy()
        np.testing.assert_allclose(full_h, auto_h, rtol=1e-6, atol=1e-10)
        write(out/'checks.json', dict(passed=True, analytic_autograd_finite_difference=math_checks,
                                      original_ap_reproduced=True, full_band_gradient=True,
                                      full_band_hessian_max_error=float(np.max(abs(full_h-auto_h))),
                                      independent_cache_bypass=True))
        print('推导、有限差分、全频梯度与原AP输出核验通过。', flush=True)
        if args.checks_only:
            report(out, rows, time.perf_counter()-started, 'CHECKS_ONLY')
            return
        for case_index, (key, signal, noise, truth, _) in enumerate(cache):
            if time.perf_counter()-started > 10800:
                report(out, rows, time.perf_counter()-started, 'PAUSED_BUDGET')
                print('已到准备运行预算，未启动下一例。', flush=True)
                return
            iq = signal.sum(0)+noise
            def run_native():
                start = synced_time()
                result = native_vp(iq, vp_cfg, True)
                result['seconds'] = synced_time()-start
                result.update(method='VP_NATIVE', objective=result['native_objective'])
                return result

            # Balance which VP stopping variant runs first, without changing its updates.
            if case_index % 2:
                common = common_vp(iq, vp_cfg)
                native = run_native()
            else:
                native = run_native()
                common = common_vp(iq, vp_cfg)
            # Drop verbose local per-step logs; final certificates and cycle summaries retained.
            native.pop('updates', None)
            native.pop('history', None)
            native['metrics'] = evaluate(truth, native['positions'])
            native['case'] = key
            rows.append(native)
            write(out/f'{key}_VP_NATIVE.json', native)
            if common['native_checkpoint'] is not None:
                np.testing.assert_allclose(common['native_checkpoint']['positions'], native['positions'], atol=1e-5)
            common.update(case=key, metrics=evaluate(truth, common['positions']))
            rows.append(common)
            write(out/f'{key}_VP_COMMON.json', common)
            # Rotate order; each method independently pays for identical candidate generation.
            ordered = METHODS[case_index % 3:]+METHODS[:case_index % 3]
            expected = None
            for method in ordered:
                result = run_ap(iq, ap_cfg, method, expected)
                expected = np.asarray(result['candidates'])
                result.update(case=key, metrics=evaluate(truth, result['positions']))
                rows.append(result)
                write(out/f'{key}_{method}.json', result)
            report(out, rows, time.perf_counter()-started, 'RUNNING')
            print(f'完成 {key}：{len(rows)} 次求解，累计 {time.perf_counter()-started:.1f} 秒', flush=True)
        for path, identity in identities.items():
            checked_bytes(Path(path), identity)
        summaries = report(out, rows, time.perf_counter()-started, 'COMPLETED')
        write(out/'final_audit.json', dict(passed=True, identities_unchanged=True,
              same_candidates=True, final_outputs=len(rows), no_formal_results_replaced=True))
        for summary in summaries:
            print(summary, flush=True)
    except Exception as exc:
        write(out/'failure.json', dict(status='FAILED', error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
