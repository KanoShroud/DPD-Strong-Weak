"""One bounded, offline SBL output audit; no optimizer or truth-driven grouping."""
import time
import numpy as np
import torch
from scipy.spatial import cKDTree
from ..physics import sbl_posterior, shortlist
from ..metrics import evaluate
from .physics import Model
from .config import HISTORY_BASE
from .runtime import fresh, preflight, read, write, sha, load_scene, finish_audit
from .scenes import observation, digest


def near_duplicate_groups(points, tolerance=.05):
    """Complete-link bounded diameter; tolerance is existing solver stability, not 20m."""
    points = np.asarray(points)
    labels = np.arange(len(points))
    groups = {i: [i] for i in range(len(points))}
    for a, b in sorted(cKDTree(points).query_pairs(tolerance)):
        ia, ib = int(labels[a]), int(labels[b])
        if ia == ib:
            continue
        members = groups[ia]+groups[ib]
        pp = points[members]
        if np.linalg.norm(pp[:, None]-pp[None], axis=-1).max() <= tolerance:
            groups[ia] = members
            labels[members] = ia
            del groups[ib]
    return list(groups.values())


def duplicate_identity_test():
    """Identical columns: covariance and summed posterior mean are invariant."""
    rng = np.random.default_rng(1201)
    a = torch.tensor(rng.normal(size=(16, 4, 2))+1j*rng.normal(size=(16, 4, 2)))
    x = torch.tensor(rng.normal(size=(16, 4))+1j*rng.normal(size=(16, 4)))
    gamma = torch.tensor([.7, .3], dtype=torch.float64)
    mu, _, cov, value = sbl_posterior(a, x, gamma, .2)
    split = torch.tensor([.1, .2, .4, .3], dtype=torch.float64)
    other, _, cov2, value2 = sbl_posterior(a[:, :, [0, 0, 0, 1]], x, split, .2)
    combined = other[:, :3].sum(1)
    np.testing.assert_allclose(cov, cov2, atol=1e-12)
    np.testing.assert_allclose(value, value2, atol=1e-12)
    np.testing.assert_allclose(mu[:, 0], combined, atol=1e-12)
    # This is why summing independent point powers omits cross terms.
    naive = abs(other[:, :3]).square().sum(1).mean()
    coherent = abs(combined).square().mean()
    np.testing.assert_allclose(float(naive/coherent), (.1**2+.2**2+.4**2)/.7**2, atol=1e-12)
    return dict(passed=True, covariance_max_error=float(abs(cov-cov2).max()),
                evidence_max_error=float(abs(value-value2).max()),
                naive_power_over_coherent=float(naive/coherent))


def main():
    source = (HISTORY_BASE/'native_probe_20261014_153914').resolve(strict=True)
    row_path, map_path = source/'rows/2_SBL.json', source/'maps/2_SBL.json'
    previous_audit = read(source/'final_audit_report.json')
    if sha(row_path) != previous_audit['rows'][row_path.name]:
        raise RuntimeError('Historical row identity changed')
    scene = load_scene(source, '2_0_0')
    row, native = read(row_path), read(map_path)
    iq, truth, k = observation(scene, 'weak')
    if digest(iq) != row['input_sha']:
        raise RuntimeError('Historical consumed IQ mismatch')
    evidence_inputs = {str(p): sha(p) for p in (row_path, map_path, source/'inputs/2_0_0.npz')}
    out = fresh('sbl_output_check')
    preflight(out, dict(stage='OFFLINE_OUTPUT_CHECK', input_identity=evidence_inputs,
                        grouping_tolerance_m=.05, no_optimizer=True,
                        map_note='Historical map lacked registered hash; hash now + posterior replay verifies consistency'))
    print('OUT='+str(out), flush=True)
    check = duplicate_identity_test()
    pp, gamma = np.asarray(native['points']), torch.tensor(native['gamma'], dtype=torch.float64, device='cuda')
    m = Model(iq, sample_rate_hz=100e6)
    weights = torch.tensor(native['spectral_weight'], device=m.device, dtype=torch.float64)
    groups = near_duplicate_groups(pp)
    ids = [g for g in groups if len(g) > 1]
    merged_pp = np.asarray([np.average(pp[g], axis=0, weights=gamma[g].cpu().numpy()) for g in groups])
    gp = np.asarray(native['power']).copy()
    coherent_group = {}
    power = torch.zeros_like(gamma)
    original_nll, merged_nll, covariance_change = 0., 0., 0.
    start = time.perf_counter()
    for lo in range(0, m.n, 128):
        f, w = m.f[lo:lo+128], weights[lo:lo+128]
        a = m.response(pp, f)*w.sqrt()[:, None, None]
        mu, _, cov, nll = sbl_posterior(a, m.x[lo:lo+128], gamma, 1/m.n)
        power += (w[:, None]*abs(mu)**2).sum(0)/m.n
        cov_new = cov.clone()
        for g in ids:
            # Reconstruct station-space posterior mean, including cross terms.
            reconstruction = (a[:, :, g]*mu[:, None, g]).sum(-1)
            coherent_group[g[0]] = coherent_group.get(g[0], 0.)+float(abs(reconstruction).square().sum().cpu())/m.n
            center = np.average(pp[g], axis=0, weights=gamma[g].cpu().numpy())
            merged_a = m.response(center[None], f)*w.sqrt()[:, None, None]
            cov_new -= (a[:, :, g]*gamma[g][None, None])@a[:, :, g].mH
            cov_new += merged_a@merged_a.mH*gamma[g].sum()
        delta = torch.linalg.matrix_norm(cov_new-cov)/torch.linalg.matrix_norm(cov)
        covariance_change = max(covariance_change, float(delta.max().cpu()))
        chol = torch.linalg.cholesky(cov_new)
        x = m.x[lo:lo+128]
        invx = torch.cholesky_solve(x[..., None], chol).squeeze(-1)
        nll2 = 2*torch.log(chol.diagonal(dim1=1, dim2=2).real).sum(1)+(x.conj()*invx).sum(1).real
        original_nll += float(nll.sum().cpu())/m.n
        merged_nll += float(nll2.sum().cpu())/m.n
    np.testing.assert_allclose(power.cpu(), gp, rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(original_nll, row['result']['history'][-1]['native_objective'], atol=1e-8)
    gc = np.asarray([coherent_group[g[0]] if len(g) > 1 else gp[g[0]] for g in groups])
    gg = np.asarray([float(gamma[g].sum().cpu()) for g in groups])
    variants = [('legacy_posterior_point', pp, gp), ('literature_gamma_point', pp, gamma.cpu().numpy()),
                ('candidate_group_coherent', merged_pp, gc), ('candidate_group_gamma', merged_pp, gg)]
    results = []
    for name, points, score in variants:
        take = shortlist(points, score, k)
        estimated = points[take]
        results.append(dict(name=name, positions=estimated.tolist(), metrics=evaluate(truth, estimated)))
    report = dict(duplicate_identity=check, outputs=results, tolerance_m=.05,
                  near_duplicate_groups=[dict(ids=g, points=pp[g].tolist()) for g in ids],
                  original_nll=original_nll, merged_nll=merged_nll,
                  max_relative_covariance_change=covariance_change,
                  posterior_replay_passed=True, seconds=time.perf_counter()-start,
                  candidate_not_default=True, no_population_performance_claim=True)
    write(out/'report.json', report)
    for path, expected in evidence_inputs.items():
        if sha(path) != expected:
            raise RuntimeError('Evidence input changed during audit')
    finish_audit(out)
    print('RESULT '+str([(r['name'], r['metrics']['errors_m']) for r in results]), flush=True)
    print('DONE '+str(out), flush=True)


if __name__ == '__main__':
    main()
