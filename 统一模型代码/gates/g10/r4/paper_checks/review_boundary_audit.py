"""Read-only solver audit: synthetic boundary tests and frozen branch replay."""
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import torch

from ...physics import Model, atom, fit
from ..bnomp import coefficient, fixed_derivatives
from ..adaptation_review import profile_derivatives
from ..config import BASE
from ...runtime import read, sha, write
from .assignment_metrics import evaluate, REVISION
from migration_paths import input_path


def boundary_tests():
    rng = np.random.default_rng(20261020)
    m = Model(rng.normal(size=(4, 64)) + 1j*rng.normal(size=(4, 64)), device='cpu')
    m.f = torch.linspace(-7e6, 7e6, 64, dtype=torch.float64)
    station = m.stations[0].numpy()
    rows = []
    for offset in ([0., 0.], [.3, .4], [1.001, 0.], [2., 1.], [200., 123.]):
        pos = station + offset
        a = atom(torch.tensor(pos[None]), m.f, m.stations, normalized=False).squeeze(-1)
        q = coefficient(a, m.x).detach()
        for profile in (False, True):
            fn = profile_derivatives if profile else fixed_derivatives
            v, g, h = fn(m, pos, m.x, q)

            def objective(p):
                response = atom(p[None], m.f, m.stations, normalized=False).squeeze(-1)
                qq = coefficient(response, m.x) if profile else q
                return abs(m.x-response*qq[:, None]).square().sum()/m.energy

            z = torch.tensor(pos, requires_grad=True)
            auto_v = objective(z)
            auto_g = torch.autograd.grad(auto_v, z)[0].detach().numpy()
            # norm has an undefined second derivative exactly at the station;
            # finite differences remain inside the flat clipped response there.
            step = 1e-5
            fd_h = np.column_stack([
                (fn(m, pos+np.eye(2)[i]*step, m.x,
                    coefficient(atom(torch.tensor((pos+np.eye(2)[i]*step)[None]),
                                     m.f, m.stations, normalized=False).squeeze(-1), m.x)
                    if profile else q)[1] -
                 fn(m, pos-np.eye(2)[i]*step, m.x,
                    coefficient(atom(torch.tensor((pos-np.eye(2)[i]*step)[None]),
                                     m.f, m.stations, normalized=False).squeeze(-1), m.x)
                    if profile else q)[1])/(2*step) for i in range(2)])
            np.testing.assert_allclose(v, float(auto_v.detach()), atol=1e-12)
            np.testing.assert_allclose(g, auto_g, atol=1e-10, rtol=1e-6)
            np.testing.assert_allclose(h, fd_h, atol=1e-8, rtol=2e-4)
            rows.append(dict(offset=offset, profile=profile,
                             gradient_max_error=float(abs(g-auto_g).max()),
                             hessian_max_error=float(abs(h-fd_h).max())))
    # Compare actual conditional score with normalized-column SVD improvement.
    first = np.array([[120., 170.]])
    a = m.response(first)
    base_energy = abs(fit(a, m.x)[1]).square().sum()/m.energy
    conditional = []
    for displacement in (0., 1e-9, 1e-7, 1e-5, .01, 1., 30.):
        p = first[0]+[displacement, 0.]
        score = float(m.conditional(p[None], first)[0])
        joint = m.joint(np.vstack([first, p]))
        gain = float(base_energy)-joint
        conditional.append(dict(displacement_m=displacement, conditional_score=score,
                                svd_improvement=gain, absolute_difference=abs(score-gain)))
        assert np.isfinite(score) and score >= 0
        if displacement == 0. or displacement >= .01:
            np.testing.assert_allclose(score, gain, atol=1e-9, rtol=1e-6)
    return dict(derivatives=rows, conditional=conditional,
                caveat='Exactly d=1 is nondifferentiable; no unique derivative is claimed. '
                       'Near duplicate columns may use different numerical rank thresholds; '
                       'these diagnostics are not a claim about historical optimizer paths.')


def branch_replay(out):
    pointer_path = BASE/'formal_run.json'
    pointer = read(pointer_path)
    source = input_path(pointer['out']).resolve(strict=True)
    assert source.is_relative_to(BASE.resolve()) and source != out
    assert sha(source/'contract.json') == pointer['contract_sha']
    audit_path = source/'final_audit_report.json'
    audit = read(audit_path)
    assert audit['passed']
    identities = {str(pointer_path): sha(pointer_path), str(audit_path): sha(audit_path),
                  str(source/'contract.json'): sha(source/'contract.json')}
    rows = []
    for path in sorted((source/'rows').glob('*_AP_5_JOINT.json')):
        digest = sha(path)
        assert digest == audit['rows'][path.name]
        identities[str(path)] = digest
        row = read(path)
        mechanism = row['result']['mechanism']
        branches = mechanism['branches']
        assert len(branches) == 5
        chosen = min(branches, key=lambda b: (b['objective'], b['rank']))
        assert chosen['rank'] == mechanism['selected_rank']
        np.testing.assert_allclose(chosen['positions'], row['result']['positions'], atol=1e-8, rtol=0)
        ranks = [b['rank'] for b in branches if evaluate(row['truth'], b['positions'])['both20']]
        selected = evaluate(row['truth'], chosen['positions'])['both20']
        rows.append(dict(case=path.stem.removesuffix('_AP_5_JOINT'),
                         successful_ranks=ranks, selected_rank=chosen['rank'],
                         oracle_success=bool(ranks), selected_success=selected,
                         first_success=1 in ranks,
                         selection_loss=bool(ranks) and not selected,
                         all_branches_failed=not ranks))
    assert len(rows) == 2400
    assert all(sha(Path(p)) == digest for p, digest in identities.items())
    counts = {k: sum(r[k] for r in rows) for k in
              ('oracle_success', 'selected_success', 'first_success',
               'selection_loss', 'all_branches_failed')}
    write(out/'branches.json', dict(protocol=REVISION, n=len(rows), counts=counts,
                                   rows=rows, source_sha256=identities))
    return counts


def main():
    out = BASE/('review_boundary_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    result = dict(source_sha256={str(Path(__file__)): sha(Path(__file__))})
    try:
        result['boundary'] = boundary_tests()
        write(out/'boundary.json', result)
        result['branch_counts'] = branch_replay(out)
        result['status'] = 'COMPLETED'
    except Exception as error:
        result.update(status='FAILED', error=repr(error))
        raise
    finally:
        write(out/'report.json', result)
        print(json.dumps(dict(out=str(out), status=result.get('status'),
                              counts=result.get('branch_counts')), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
