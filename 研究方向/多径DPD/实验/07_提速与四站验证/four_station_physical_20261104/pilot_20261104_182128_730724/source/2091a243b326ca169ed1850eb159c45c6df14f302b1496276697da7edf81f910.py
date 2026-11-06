"""Audit completed pilot; issue explicit ready or held formal manifest."""
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import scene as cfg
import adapters as ad


def main():
    out = Path(cfg.read(cfg.HERE/'pilot_latest.json')['path']).resolve(strict=True)
    summary = cfg.read(out/'summary.json')
    identity = cfg.read(out/'identity.json')['hashes']
    checks = {'pilot_complete': summary['completed'] == 52 and summary['finished'],
              'source_identity': all(cfg.sha(p) == h for p, h in identity.items()),
              'correctness': cfg.read(cfg.HERE/'correctness.json')['passed']}
    engine = ad.Engine()
    for folder in sorted((out/'cases').iterdir()):
        row = cfg.read(folder/'evaluation.json')
        with np.load(folder/'iq.npz', allow_pickle=False) as z:
            iq = z['iq']
        checks[folder.name+'/iq'] = cfg.digest(iq) == row['iq_sha']
        f, y, n, _ = ad.b.fc.cal.spectral(iq)
        result = cfg.read(folder/'MP_decision.json')
        if result['status'] != 'returned':
            continue
        solved = result['result']
        for model, paths in [('los', []), ('final', solved['paths'])]:
            state = solved[model]; x = np.asarray(state['x'])
            value, g = engine.objective(x, f, y, paths)
            pg = float(np.max(abs(x-engine.ref.project(x-g, n))))
            checks[folder.name+'/'+model+'/objective'] = abs(value-state['cost']) < 1e-9
            checks[folder.name+'/'+model+'/PG'] = abs(pg-state['projected_gradient']) < 1e-8
            checks[folder.name+'/'+model+'/honest'] = bool(state['stationary']) == bool(pg <= 1e-7)
            checks[folder.name+'/'+model+'/feasible'] = np.linalg.norm(x-engine.ref.project(x, n)) < 1e-7
    reasons = list(summary['review_flags'])
    if not all(checks.values()):
        reasons.append('Correctness or identity audit failed')
    result = dict(formal_allowed=not reasons, reasons=reasons, pilot=str(out), checks=checks,
                  hashes=ad.source_hashes(), summary=summary, formal_cases=3120,
                  no_formal_data_generated=True)
    cfg.write(cfg.HERE/'release.json', result)
    print('FORMAL_ALLOWED', result['formal_allowed'], 'REASONS', reasons)
    print('AUDIT', len(checks), all(checks.values()), 'FORECAST_HOURS', summary['forecast_formal_seconds']/3600)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
