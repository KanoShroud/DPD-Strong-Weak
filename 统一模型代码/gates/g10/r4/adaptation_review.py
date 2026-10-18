"""Bounded BNOMP mechanism review, never replacing the frozen reference.

Five predeclared variants on eight already-used observations. Profile Newton
is a separate diagnostic; discovery and post-cycle joint-LS schedule stay fixed.
"""
from pathlib import Path
from datetime import datetime
import copy
import csv
import io
import sys
import time
import traceback
import numpy as np
import torch
from ...g8.physics import C
from ..physics import grid, atom
from ..r1.physics import small_fit
from ..r1.runtime import Ledger
from ..r3.solvers import Model, Trace, projected_gradient
from ..metrics import evaluate
from .bnomp import coefficient, raw_response, fixed_derivatives
from .config import ROOT, BASE
from .runtime import read, write, sha, seal, check_contract, load, lock, require_exclusive, resource_check

TASKS = tuple((g, 200, gap, 'mixed') for g in range(4) for gap in (0, 15))
VARIANTS = ('FIXED_STRICT', 'FIXED_E7', 'FIXED_E6', 'FIXED_NO_INNER_EXIT', 'PROFILE_STRICT')
REVISION = 'G10_R4_BNOMP_MECHANISM_V1'


def profile_derivatives(m, p, data, q):
    """Exact Schur complement over per-frequency complex coefficient q.

    Requires q=argmin ||data-a(p)q||². Its real/imag Hessian block is
    2||a_f||² I. Eliminate those blocks without assembling an F-sized matrix.
    """
    value, gradient, fixed = fixed_derivatives(m, p, data, q)
    z = torch.as_tensor(p, dtype=torch.float64, device=m.device)
    delta = z[None]-m.stations
    distance = torch.linalg.vector_norm(delta, dim=1)
    d = distance.clamp_min(1.)
    u = torch.where((distance > 1.)[:, None], delta/d[:, None], torch.zeros_like(delta))
    k = 2*torch.pi*m.f[:, None]/C
    a = torch.exp(-1j*k*d[None])/d[None]
    da = a[:, :, None]*(-1/d[None]-1j*k)[:, :, None]*u[None]
    r = data-a*q[:, None]
    cross = (a.conj()[:, :, None]*da).sum(1)*q[:, None]
    res = (r.conj()[:, :, None]*da).sum(1)
    real = 2*(cross-res).real
    imag = 2*(cross+res).imag
    denom = 2*a.abs().square().sum(1)*m.energy
    correction = (torch.einsum('fi,fj,f->ij', real, real, 1/denom)
                  +torch.einsum('fi,fj,f->ij', imag, imag, 1/denom))
    m.counts['frequency_response_points'] += m.n
    m.counts['profile_schur_calls'] = m.counts.get('profile_schur_calls', 0)+1
    return value, gradient, fixed-correction.cpu().numpy()


def refine(m, initial, data, cfg, profile=False):
    """Instrumented frozen update, or explicitly labeled profiled alternative."""
    p = np.asarray(initial, float).copy()
    a = raw_response(m, p).squeeze(-1)
    q = coefficient(a, data)
    details = []
    for _ in range(cfg['bnomp_newton_steps']):
        m.check()
        fn = profile_derivatives if profile else fixed_derivatives
        value, g, h = fn(m, p, data, q)
        if not np.isfinite(value) or not np.isfinite(g).all() or not np.isfinite(h).all():
            raise ArithmeticError('Nonfinite Newton state')
        h = (h+h.T)/2
        eig = float(np.linalg.eigvalsh(h).min())
        h += np.eye(2)*max(0., 1e-9-eig)
        direction = -np.linalg.solve(h, g)
        norm = float(np.linalg.norm(direction))
        direction *= min(1., cfg['bnomp_step_cap_m']/max(norm, 1e-30))
        accepted, shift, clipped = False, 0., False
        for scale in (1., .5, .25, .125, .0625):
            trial = p+scale*direction
            proposal = np.clip(trial, -1000., 1000.)
            clipped = bool(np.any(trial != proposal))
            candidate = raw_response(m, proposal).squeeze(-1)
            trial_q = coefficient(candidate, data) if profile else q
            after = float((data-candidate*trial_q[:, None]).abs().square().sum().cpu()/m.energy.cpu())
            if after <= value:
                shift = float(np.linalg.norm(proposal-p))
                p, a, accepted = proposal, candidate, True
                q = coefficient(a, data)
                break
        details.append(dict(before=value, after=after, accepted=accepted, step_m=shift,
            min_hessian_eigenvalue=eig, floor_active=eig < 1e-9,
            cap_active=norm > cfg['bnomp_step_cap_m'], backtracked=scale < 1.,
            scale=scale, boundary_clipped=clipped, raw_direction_m=norm,
            inner_exit=bool(accepted and shift < cfg['bnomp_min_step_m'])))
        if not accepted or shift < cfg['bnomp_min_step_m']:
            break
    return p, a, q, details


def run(iq, cfg, profile=False, guard=lambda: None):
    t = Trace(guard)
    with t.stage('input_fft_transfer_band_selection'):
        m = Model(iq, t.check, sample_rate_hz=cfg['sample_rate_hz'], half_band_hz=cfg['half_band_hz'])
        t.model = m
    residual, positions, atoms, coeffs = m.x, [], [], []
    updates, cycles, discovery = [], [], []
    with t.stage('discovery_newton_cyclic_then_joint_ls'):
        for source in range(2):
            points = grid(cfg['grid'])
            initial = points[int(np.argmax(m.conditional(points, residual=residual)))]
            p, a, q, detail = refine(m, initial, residual, cfg, profile)
            positions.append(p); atoms.append(a); coeffs.append(q)
            residual = residual-a*q[:, None]
            updates.extend(detail)
            discovery.append(dict(source=source, grid_position=initial.tolist(), position=p.tolist()))
            reason = 'ITERATION_LIMIT'
            for cycle in range(cfg['bnomp_cycles']):
                before = float(residual.abs().square().sum().cpu()/m.energy.cpu())
                for slot in range(len(positions)):
                    data = residual+atoms[slot]*coeffs[slot][:, None]
                    p, a, q, detail = refine(m, positions[slot], data, cfg, profile)
                    positions[slot], atoms[slot], coeffs[slot] = p, a, q
                    residual = data-a*q[:, None]
                    updates.extend(detail)
                after = float(residual.abs().square().sum().cpu()/m.energy.cpu())
                t.iteration(positions, after, phase='cyclic', discovered=source+1,
                            cycle=cycle+1, energy_before=before)
                m.counts['iterations'] += 1
                if abs(before-after) <= cfg['bnomp_energy_tolerance']:
                    reason = 'NATIVE_ENERGY_STABLE'
                    break
            matrix = torch.stack(atoms, -1)
            scale = torch.linalg.vector_norm(matrix, dim=1)
            fitted, residual = small_fit(matrix/scale[:, None, :], m.x)
            fitted = fitted/scale
            coeffs = [fitted[:, i] for i in range(len(positions))]
            m.counts['coefficient_joint_refits'] = m.counts.get('coefficient_joint_refits', 0)+1
            cycles.append(dict(discovered=source+1, cycles=cycle+1, stop_reason=reason))
            if source == 1:
                t.save(positions, m.joint(positions))
    torch.cuda.synchronize()
    core, counts = t.elapsed(), copy.deepcopy(m.counts)
    p = np.asarray(positions)
    start = time.perf_counter()
    val, grad = m.joint(p, True)
    torch.cuda.synchronize()
    cert = time.perf_counter()-start
    return dict(positions=p.tolist(), native_objective=val, projected_gradient=projected_gradient(p, grad),
        config=cfg, profile=profile, seconds=core, wall_solver_seconds=core+t.recording_seconds,
        certificate_seconds=cert, recording_seconds=t.recording_seconds, counts=counts,
        native_cycles=cycles, discovery=discovery, updates=updates, history=t.history,
        stop_reason='ITERATION_LIMIT' if any(c['stop_reason']=='ITERATION_LIMIT' for c in cycles) else 'NATIVE_ENERGY_STABLE')


def derivative_checks(cache, originals):
    records = []
    for task in TASKS[::2]:
        iq = cache[task][0]
        old = originals[task]['result']
        p = np.asarray(old['history'][0]['positions'][0])
        m = Model(iq, device='cpu')
        a = raw_response(m, p).squeeze(-1)
        q = coefficient(a, m.x)
        v, g, hf = fixed_derivatives(m, p, m.x, q)
        vp, gp, hp = profile_derivatives(m, p, m.x, q)
        z = torch.tensor(p, dtype=torch.float64, requires_grad=True)
        def objective(x):
            aa = atom(x.reshape(1, 2), m.f, m.stations, normalized=False).squeeze(-1)
            qq = coefficient(aa, m.x)
            return (m.x-aa*qq[:, None]).abs().square().sum()/m.energy
        vg = objective(z)
        ag = torch.autograd.grad(vg, z)[0].detach().numpy()
        ah = torch.autograd.functional.hessian(objective, z).detach().numpy()
        fd = np.array([(float(objective(z.detach()+torch.eye(2)[i]*.001))
                        -float(objective(z.detach()-torch.eye(2)[i]*.001)))/.002 for i in range(2)])
        np.testing.assert_allclose(vp, v, rtol=0, atol=1e-14)
        np.testing.assert_allclose(gp, g, rtol=0, atol=1e-14)
        np.testing.assert_allclose(gp, ag, rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(hp, ah, rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(gp, fd, rtol=1e-3, atol=1e-9)
        correction_eigs = np.linalg.eigvalsh(hf-hp)
        assert correction_eigs.min() >= -1e-12
        records.append(dict(task=list(task), position=p.tolist(), gradient=gp.tolist(),
            fixed_hessian=hf.tolist(), profile_hessian=hp.tolist(),
            fixed_eigenvalues=np.linalg.eigvalsh(hf).tolist(), profile_eigenvalues=np.linalg.eigvalsh(hp).tolist(),
            removed_curvature_eigenvalues=correction_eigs.tolist(),
            fixed_trial_m=float(np.linalg.norm(np.linalg.solve(hf, g))),
            profile_trial_m=float(np.linalg.norm(np.linalg.solve(hp, gp))),
            autograd_hessian_max_error=float(abs(hp-ah).max()), finite_difference_max_error=float(abs(gp-fd).max())))
    return dict(passed=True, records=records, meaning='same positions/current LS coefficients; trial steps, not accepted localization results')


def equivalence(new, old):
    np.testing.assert_array_equal(new['positions'], old['positions'])
    assert new['native_objective'] == old['native_objective']
    assert new['counts'] == old['counts']
    assert len(new['history']) == len(old['history'])
    for a, b in zip(new['history'], old['history']):
        np.testing.assert_array_equal(a['positions'], b['positions'])
        assert a['native_objective'] == b['native_objective']
    assert len(new['updates']) == len(old['mechanism']['fixed_q_updates'])
    for a, b in zip(new['updates'], old['mechanism']['fixed_q_updates']):
        assert all(a[x] == b[y] for x,y in [('before','before'),('after','fixed_q_after'),('accepted','accepted'),('step_m','step_m')])


def summarize(out, rows):
    summary = {}
    ref = {tuple(r['task']):r for r in rows if r['variant']=='FIXED_STRICT'}
    for variant in VARIANTS:
        rr = [r for r in rows if r['variant']==variant]
        updates = [u for r in rr for u in r['result']['updates']]
        summary[variant] = dict(cases=len(rr), both20=sum(all(r['metrics']['success']['20']) for r in rr),
            weak_rmse=float(np.sqrt(np.mean([r['metrics']['errors_m'][1]**2 for r in rr]))),
            mean_seconds=float(np.mean([r['result']['wall_solver_seconds'] for r in rr])),
            mean_derivative_calls=float(np.mean([r['result']['counts']['fixed_q_derivative_calls'] for r in rr])),
            cycles=[{str(c['discovered']):c['cycles'] for c in r['result']['native_cycles']} for r in rr],
            stopped=sum(r['result']['stop_reason']=='NATIVE_ENERGY_STABLE' for r in rr),
            steps=len(updates), small_accepted_steps=sum(u['accepted'] and u['step_m']<.01 for u in updates),
            safeguards={k:sum(u[k] for u in updates) for k in ('floor_active','cap_active','backtracked','boundary_clipped','inner_exit')},
            rejected=sum(not u['accepted'] for u in updates),
            max_movement_from_strict_m=float(max(np.linalg.norm(np.asarray(r['result']['positions'])-
                np.asarray(ref[tuple(r['task'])]['result']['positions']),axis=1).max() for r in rr)),
            max_objective_above_strict=float(max(r['result']['native_objective']-
                ref[tuple(r['task'])]['result']['native_objective'] for r in rr)))
    report = dict(status='COMPLETED_BNOMP_MECHANISM_REVIEW', summaries=summary,
        scope='eight already-used observations, not independent validation or formal method ranking',
        formal_released=False, instrumented_timing=True, baseline_replaced=False)
    write(out/'comparison_report.json', report)
    buff=io.StringIO(newline=''); writer=csv.writer(buff)
    writer.writerow(['layout','repeat','gap','variant','both20','strong_error_m','weak_error_m','objective','gradient','stage1','stage2','derivative_calls','seconds'])
    for r in rows:
        v=r['result']
        writer.writerow(r['task'][:3]+[r['variant'],all(r['metrics']['success']['20']),*r['metrics']['errors_m'],
            v['native_objective'],v['projected_gradient'],*[c['cycles'] for c in v['native_cycles']],
            v['counts']['fixed_q_derivative_calls'],v['wall_solver_seconds']])
    (out/'per_case.csv').write_text(buff.getvalue(),encoding='utf-8-sig')
    return report


def main():
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):
            stream.reconfigure(encoding='utf-8',errors='replace')
    require_exclusive(); resource_check()
    pointer=read(BASE/'preparation.json'); prep=Path(pointer['out']).resolve(strict=True)
    if sha(prep/'contract.json')!=pointer['contract_sha']:
        raise RuntimeError('Preparation identity changed')
    original=check_contract(prep)
    stopped=Path(read(BASE/'stop_review.json')['out']).resolve(strict=True)
    audit=read(stopped/'final_audit_report.json')
    cache, originals, hashes = {}, {}, {}
    for task in TASKS:
        _, iq, truth, identity=load(original, task)
        path=stopped/'rows'/f'{task[0]}_{task[1]}_{task[2]}_BNOMP_REF.json'
        hashes[str(path)]=sha(path)
        if hashes[str(path)]!=audit['rows'][path.name]:
            raise RuntimeError('Frozen reference row changed')
        originals[task]=read(path); cache[task]=(iq,truth,identity)
    out=(BASE/('adaptation_review_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    if out.exists() or not out.is_relative_to(BASE.resolve()):
        raise RuntimeError('Fresh output required')
    out.mkdir()
    contract=dict(revision=REVISION, tasks=TASKS, variants=VARIANTS, source_sha=sha(__file__),
        reference=str(prep), stopped_reference=str(stopped), frozen_rows=hashes,
        source=str(Path(__file__).relative_to(ROOT)), original_contract_sha=sha(prep/'contract.json'),
        base_cfg=originals[TASKS[0]]['result']['config'], schedule_seconds=1800,
        changes={'FIXED_E7':{'bnomp_energy_tolerance':1e-7},'FIXED_E6':{'bnomp_energy_tolerance':1e-6},
                 'FIXED_NO_INNER_EXIT':{'bnomp_min_step_m':0.},'PROFILE_STRICT':{'local_update':'profiled_coefficient_schur_newton'}},
        formal_started=False, truth_used_only_for_offline_metrics=True)
    write(out/'contract.json',contract)
    (out/'adaptation_review.py').write_bytes(Path(__file__).read_bytes())
    print('BNOMP机制核验目录：'+str(out),flush=True)
    with lock(out):
        clock=Ledger(out)
        try:
            write(out/'derivative_checks.json',derivative_checks(cache, originals))
            # The common warmup is not included in per-variant timing.
            warm=copy.deepcopy(contract['base_cfg']); warm['bnomp_cycles']=1
            run(cache[TASKS[0]][0],warm,guard=clock.check)
            rows=[]
            for index,task in enumerate(TASKS):
                iq,truth,identity=cache[task]
                order=VARIANTS[index%5:]+VARIANTS[:index%5]
                for variant in order:
                    if clock.total()>contract['schedule_seconds']:
                        raise RuntimeError('Finite diagnostic schedule reached; next solve not started')
                    require_exclusive()
                    cfg=copy.deepcopy(contract['base_cfg'])
                    cfg.update({k:v for k,v in contract['changes'].get(variant,{}).items() if k!='local_update'})
                    clock.flush(dict(task=list(task),variant=variant))
                    result=run(iq,cfg,variant=='PROFILE_STRICT',clock.check)
                    if variant=='FIXED_STRICT':
                        equivalence(result,originals[task]['result'])
                    for u in result['updates']:
                        if u['accepted'] and u['after']>u['before']:
                            raise ArithmeticError('Accepted objective increase')
                    row=dict(task=list(task),variant=variant,input_sha=identity,result=result,truth=truth.tolist(),
                             metrics=evaluate(truth,result['positions']))
                    row['record_sha']=seal(row)
                    write(out/'rows'/f'{task[0]}_{task[1]}_{task[2]}_{variant}.json',row)
                    rows.append(row); clock.flush()
                print(f'完成布局{task[0]} / {task[2]}dB：{len(rows)}/40项，累计{clock.total():.1f}s',flush=True)
            report=summarize(out,rows)
            check_contract(prep)
            if sha(__file__)!=contract['source_sha'] or sha(prep/'contract.json')!=contract['original_contract_sha']:
                raise RuntimeError('Source identity changed')
            for path,digest in hashes.items():
                if sha(path)!=digest:
                    raise RuntimeError('Frozen result changed')
            for task in TASKS:
                load(original,task)
            saved={}
            for path in sorted((out/'rows').glob('*.json')):
                row=read(path)
                assert row['record_sha']==seal(row)
                assert row['metrics']==evaluate(row['truth'],row['result']['positions'])
                saved[path.name]=sha(path)
            assert len(saved)==40
            write(out/'final_audit_report.json',dict(passed=True,solves=40,rows=saved,
                frozen_sources_inputs_and_results_unchanged=True, instrumented_reference_exact=True,
                derivatives_passed=True, accepted_objectives_nonincreasing=True, metrics_replayed=True))
            clock.flush()
            write(out/'status.json',dict(status=report['status'],seconds=clock.total(),formal_released=False))
            write(BASE/'adaptation_review.json',dict(out=str(out),status=report['status'],contract_sha=sha(out/'contract.json')))
            print('机制核验完成；原文参照和正式配置均未替换。',flush=True)
        except BaseException:
            write(out/'status.json',dict(status='FAILED_DIAGNOSTIC_RETAINED',error=traceback.format_exc(),formal_released=False))
            clock.flush()
            raise


if __name__=='__main__':
    main()
