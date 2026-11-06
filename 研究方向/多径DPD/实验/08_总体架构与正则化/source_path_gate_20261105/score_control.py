"""All observation-generated pairs, fixed blind gains, joint waveform LS control."""
import time
import numpy as np
from threadpoolctl import threadpool_limits
from diagnose import HERE, INPUT, cfg, ad, stable, data


def scores(e, iq, points):
    f, y, _, _ = ad.b.fc.cal.spectral(iq)
    channels = []
    for p in points:
        gain = e.engine.rank_one(y, f, p)[0]
        distances = np.linalg.norm(p-e.stations, axis=1)
        channels.append(np.exp(-2j*np.pi*f[:, None]*distances[None]/cfg.C)*gain[None])
    h = np.asarray(channels)
    norm = np.linalg.norm(h, axis=2)
    u = h/norm[..., None]
    dot = np.einsum('cfm,fm->cf', u.conj(), y)
    total = float(np.vdot(y, y).real)
    ids, costs = [], []
    fallback_bins = 0
    for i in range(len(points)-1):
        cross = np.einsum('fm,jfm->jf', u[i].conj(), u[i+1:])
        denom = 1-abs(cross)**2
        scale_ratio = np.maximum(norm[i+1:]/norm[i], norm[i]/norm[i+1:])
        bad = (denom <= 1e-5) | (scale_ratio >= 1e8)
        reduction = abs(dot[i])[None]**2+np.divide(abs(dot[i+1:]-cross.conj()*dot[i])**2,
            denom, out=np.zeros_like(denom), where=~bad)
        jj, ff = np.nonzero(bad)
        if len(jj):
            matrix = np.stack([h[i, ff], h[i+1+jj, ff]], axis=2)
            inverse = np.linalg.pinv(matrix, rcond=1e-12)
            residual = y[ff]-(matrix@(inverse@y[ff, :, None]))[:, :, 0]
            reduction[jj, ff] = np.sum(abs(y[ff])**2-abs(residual)**2, axis=1)
        fallback_bins += len(jj)
        cost = 1-reduction.sum(axis=1)/total
        ids.extend((i, j) for j in range(i+1, len(points)))
        costs.extend(cost.tolist())
    ids, costs = np.asarray(ids), np.asarray(costs)
    assert np.isfinite(costs).all() and costs.min() > -1e-10
    check_ids = np.unique(np.r_[np.linspace(0, len(ids)-1, 31, dtype=int), np.argmin(costs)])
    error = 0.
    for k in check_ids:
        i, j = ids[k]
        matrix = np.stack([h[i], h[j]], axis=2)
        residual = y-(matrix@(np.linalg.pinv(matrix, rcond=1e-12)@y[:, :, None]))[:, :, 0]
        direct = float(np.vdot(residual, residual).real/total)
        error = max(error, abs(direct-costs[k]))
    assert error < 1e-9, error
    return ids, costs, dict(direct_error=error, fallback_bins=fallback_bins, checks=len(check_ids))


def main():
    src = HERE/'run_v1'
    assert cfg.read(src/'summary.json')['completed'] == 24
    stable.inside.rr.check_hashes(cfg.read(src/'identity.json'))
    out = HERE/'score_control_v1'
    out.mkdir(exist_ok=False)
    hashes = ad.source_hashes()
    hashes.update({str(p.resolve()): cfg.sha(p) for p in
        [HERE/'diagnose.py', HERE/'score_control.py', HERE/'score_control_protocol.md']})
    cfg.write(out/'identity.json', hashes)
    rows = []
    for path in sorted(src.glob('*_blind_pool.json')):
        tag = path.name.removesuffix('_blind_pool.json')
        pool = cfg.read(path)
        iq = data(tag)
        assert cfg.digest(iq) == pool['iq_sha']
        points = np.asarray(pool['geometry']['refined'])
        start = time.perf_counter()
        ids, cost, checks = scores(stable.Engine(), iq, points)
        elapsed = time.perf_counter()-start
        best = int(np.argmin(cost))
        # Persist observation-only score/decision before truth-based assessment.
        cfg.write(out/(tag+'_scores.json'), dict(ids=ids, cost=cost, winner=ids[best],
            checks=checks, seconds=elapsed, iq_sha=cfg.digest(iq), pool_sha=cfg.sha(path)))
        diagnostic = cfg.read(src/(tag+'_diagnostic_selection.json'))
        nearest = sorted(diagnostic['nearest_ids'])
        idx = np.flatnonzero(np.all(ids == nearest, axis=1)).item()
        with np.load(INPUT/'banks'/f'{tag.split("_")[1]}_00.npz', allow_pickle=False) as z:
            truth = z['p']
        err = stable.inside.rr.evaluation(points[ids[best]], truth)
        row = dict(tag=tag, pairs=len(ids), best_ids=ids[best], selected_errors=err,
            selected_rmse=float(np.sqrt(np.mean(np.square(err)))), best_cost=cost[best],
            nearest_cost=cost[idx], nearest_strict_rank=int(np.sum(cost < cost[idx]-1e-10))+1,
            near_ties=int(np.sum(abs(cost-cost[idx]) <= 1e-10)), seconds=elapsed, **checks)
        rows.append(row)
        cfg.write(out/'rows.json', rows)
        print(row, flush=True)
    stable.inside.rr.check_hashes(hashes)
    cfg.write(out/'summary.json', dict(rows=rows, identity_unchanged=True,
        not_full_ML=True, not_multipath_score=True, diagnostic_only=True))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
