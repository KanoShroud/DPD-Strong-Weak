"""Bounded equivalent-curvature audit, not another localization algorithm.

Both implementations use the frozen conditional solver and its safeguards.
Only the algebraic evaluation of the exact derivatives changes. The temporary
in-process class override is restored after each call; frozen files are untouched.
"""
import argparse
import copy
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from ...runtime import read, write, sha, resource_check
from ..runtime import require_exclusive, lock
from ..config import BASE
from . import vp_conditional as candidate
from .vp_route_diagnostic import TASKS, prepare_inputs, sources, now
from .assignment_metrics import evaluate


FROZEN = {
    'vp_conditional.py': '63d7ad1740e0efc2c51f2ab703cbaa4e0f04010ac4535a4fe8ae06057332778c',
    'vp_route_diagnostic.py': 'a389edd1705696d11ae980e41d329467729d47f7068f1a8116a8b7b3c6df2ab0',
}
SCHUR = candidate.ConditionalBlock


class QuotientBlock(SCHUR):
    """Analytic derivatives of sum_f |b_f^H y_f|^2 / ||b_f||^2.

Shares the physical response and rank-one projection, not the Schur derivative
formula. Residual-form objective evaluation is shared for stable acceptance.
No dense FL-by-FL matrix, autodiff, finite differences, or inverse is timed.
"""
    def evaluate(self, position, derivatives=False):
        if not derivatives:
            return super().evaluate(position, False)
        m = self.m
        b, db, ddb = self.response_derivatives(position)
        d = b.abs().square().sum(1)
        if bool((d < 1e-24).any()):
            raise ArithmeticError('Rank-deficient conditional response')
        z = (b.conj()*self.data).sum(1)
        dz = (db.conj()*self.data[:, :, None]).sum(1)
        ddz = (ddb.conj()*self.data[:, :, None, None]).sum(1)
        n = z.abs().square()
        dn = 2*(z.conj()[:, None]*dz).real
        ddn = 2*(dz.conj()[:, :, None]*dz[:, None, :]
                 +z.conj()[:, None, None]*ddz).real
        dd = 2*(b.conj()[:, :, None]*db).sum(1).real
        ddd = 2*(torch.einsum('fli,flj->fij', db.conj(), db)
                 +(b.conj()[:, :, None, None]*ddb).sum(1)).real
        ds = dn/d[:, None]-n[:, None]*dd/d[:, None]**2
        dds = (ddn/d[:, None, None]
               -(dn[:, :, None]*dd[:, None, :]+dd[:, :, None]*dn[:, None, :]
                 +n[:, None, None]*ddd)/d[:, None, None]**2
               +2*n[:, None, None]*dd[:, :, None]*dd[:, None, :]/d[:, None, None]**3)
        residual = self.data-b*(z/d)[:, None]
        value = residual.abs().square().sum()/m.energy
        m.counts['conditional_hessians'] = m.counts.get('conditional_hessians', 0)+1
        return (float(value.cpu()), (-ds.sum(0)/m.energy).cpu().numpy(),
                (-dds.sum(0)/m.energy).cpu().numpy())


def execute(iq, cls, cfgs):
    with patch.object(candidate, 'ConditionalBlock', cls):
        return candidate.solve(iq, 'FULL_CONDITIONAL', copy.deepcopy(cfgs['PROFILE_DIAG']),
                               copy.deepcopy(cfgs['AP_1_JOINT']))


def math_checks(iq, cfg):
    model = candidate.Model(iq, sample_rate_hz=cfg['sample_rate_hz'],
                            half_band_hz=cfg['half_band_hz'])
    positions = [np.array([[-210., 131.], [670., -460.]]),
                 np.array([[130., -250.], [-310., 220.]])]
    rows = []
    for p in positions:
        x = torch.tensor(p.ravel(), dtype=torch.float64, device=model.device, requires_grad=True)
        value = model.joint_tensor(x)
        gradient = torch.autograd.grad(value, x)[0].detach().cpu().numpy()
        hessian = torch.autograd.functional.hessian(model.joint_tensor, x).cpu().numpy()
        for slot in (0, 1):
            blocks = [cls(model, p[1-slot][None]) for cls in (SCHUR, QuotientBlock)]
            triples = [b.evaluate(p[slot], True) for b in blocks]
            ss = slice(slot*2, slot*2+2)
            for v, g, h in triples:
                np.testing.assert_allclose(v, float(value.detach().cpu()), rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(g, gradient[ss], rtol=1e-6, atol=1e-10)
                np.testing.assert_allclose(h, hessian[ss, ss], rtol=1e-5, atol=1e-10)
            v0, g0, h0 = triples[0]
            v1, g1, h1 = triples[1]
            rows.append(dict(positions=p.tolist(), slot=slot, objective_difference=abs(v0-v1),
                             gradient_difference=float(np.max(abs(g0-g1))),
                             hessian_difference=float(np.max(abs(h0-h1)))))
    # Equal warmup and interleaved microbenchmarks; no full-solver speed claim.
    blocks = [cls(model, positions[0][1][None]) for cls in (SCHUR, QuotientBlock)]
    for b in blocks:
        for _ in range(5):
            b.evaluate(positions[0][0], True)
    timing = []
    for repeat in range(12):
        for index in (repeat % 2, 1-repeat % 2):
            started = now()
            for _ in range(25):
                blocks[index].evaluate(positions[0][0], True)
            timing.append(dict(repeat=repeat, variant=('SCHUR', 'QUOTIENT')[index],
                               per_call_seconds=(now()-started)/25))
    return dict(passed=True, records=rows, kernel_timing=timing)


def summarize(out):
    pairs = [read(p) for p in sorted((out/'pairs').glob('*.json'))]
    summary = []
    for name in ('SCHUR', 'QUOTIENT'):
        rows = [r for p in pairs for r in p['rows'] if r['variant'] == name]
        if rows:
            summary.append(dict(variant=name, solves=len(rows),
                success=sum(r['metrics']['both20'] for r in rows),
                mean_seconds=float(np.mean([r['result']['complete_seconds'] for r in rows])),
                median_seconds=float(np.median([r['result']['complete_seconds'] for r in rows])),
                mean_refinement_seconds=float(np.mean([r['result']['refinement_seconds'] for r in rows])),
                common_gradient_count=sum(r['result']['projected_gradient'] <= 1e-8 for r in rows)))
    comparisons = [p['comparison'] for p in pairs]
    return dict(summary=summary, pairs=len(pairs), unique_observations=len({p['case'] for p in pairs}),
                comparisons=comparisons, scope='Equivalent implementation audit; repetitions are timing repeats, not independent localization trials')


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    for name, identity in FROZEN.items():
        assert sha(Path(__file__).with_name(name)) == identity, name
    torch.set_num_threads(20)
    require_exclusive(); resource_check()
    tasks, repeats = TASKS, 3
    cache, cfgs, identities = prepare_inputs(tasks)
    source_ids = sources()
    if args.resume:
        out = args.resume.resolve(strict=True)
        assert out.is_relative_to(BASE.resolve()) and out != BASE.resolve()
        contract = read(out/'contract.json')
        assert contract['sources'] == source_ids and contract['inputs'] == identities
        assert contract['tasks'] == [list(t) for t in tasks] and contract['repeats'] == repeats
    else:
        out = BASE/('vp_conditional_audit_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        assert not out.exists() and out.resolve().is_relative_to(BASE.resolve())
        out.mkdir(); (out/'pairs').mkdir()
        write(out/'contract.json', dict(tasks=tasks, repeats=repeats, inputs=identities,
              sources=source_ids, configs=cfgs, frozen_candidate=FROZEN,
              scope='16 existing development observations; equivalent-curvature computational audit, not formal performance comparison',
              budget_seconds=600, expected_pairs=48,
              common='Same initialization, physical derivatives, residual-form loss, safeguards, 1e-8 projected-gradient stop',
              difference='Analytic Schur curvature versus analytic conditional-quotient differentiation'))
        shutil.copyfile(__file__, out/'audit_source.py')
    print('等价计算核验目录：'+str(out), flush=True)
    started = time.perf_counter()
    previous = read(out/'status.json').get('cumulative_seconds', 0.) if (out/'status.json').exists() else 0.
    with lock(out):
        try:
            iq = cache[tasks[-1]][0]
            if not (out/'math_checks.json').exists():
                write(out/'math_checks.json', math_checks(iq, cfgs['PROFILE_DIAG']))
            for cls in (SCHUR, QuotientBlock):
                execute(iq, cls, cfgs)
            warmup = time.perf_counter()-started
            done = 0
            for repeat in range(repeats):
                for index, task in enumerate(tasks):
                    key = '_'.join(map(str, task))
                    path = out/'pairs'/f'{key}_repeat{repeat}.json'
                    if path.exists():
                        continue
                    if previous+time.perf_counter()-started > 540:
                        break
                    require_exclusive(); resource_check()
                    iq, truth, _ = cache[task]
                    order = [('SCHUR', SCHUR), ('QUOTIENT', QuotientBlock)]
                    if (repeat+index) % 2:
                        order.reverse()
                    rows = []
                    for name, cls in order:
                        result = execute(iq, cls, cfgs)
                        rows.append(dict(variant=name, result=result, metrics=evaluate(truth, result['positions'])))
                    a, b = (next(r for r in rows if r['variant'] == name) for name in ('SCHUR', 'QUOTIENT'))
                    np.testing.assert_array_equal(a['result']['initial_positions'], b['result']['initial_positions'])
                    comparison = dict(case=key, repeat=repeat,
                        max_position_difference_m=float(np.max(abs(np.array(a['result']['positions'])-np.array(b['result']['positions'])))),
                        objective_difference=abs(a['result']['objective']-b['result']['objective']),
                        same_success=a['metrics']['both20'] == b['metrics']['both20'],
                        same_cycles=a['result']['cycles'] == b['result']['cycles'],
                        same_attempts=a['result']['attempts'] == b['result']['attempts'])
                    write(path, dict(case=key, repeat=repeat, task=task, rows=rows, comparison=comparison))
                    done += 1
                    if args.limit and done >= args.limit:
                        break
                if args.limit and done >= args.limit:
                    break
            assert candidate.ConditionalBlock is SCHUR
            assert sources() == source_ids
            assert all(sha(p) == h for p, h in identities.items())
            report = summarize(out)
            write(out/'report.json', report)
            write(out/'status.json', dict(status='COMPLETED' if report['pairs'] == 48 else 'PARTIAL',
                  cumulative_seconds=previous+time.perf_counter()-started,
                  warmup_this_invocation_seconds=warmup, sources_inputs_unchanged=True))
            print(json.dumps(report['summary'], ensure_ascii=False), flush=True)
        except BaseException as exc:
            write(out/'status.json', dict(status='FAILED_RETAINED', error=repr(exc),
                  cumulative_seconds=previous+time.perf_counter()-started))
            raise


if __name__ == '__main__':
    main()
