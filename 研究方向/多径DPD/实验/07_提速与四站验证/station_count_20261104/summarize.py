"""Offline station-count evidence replay; no new localization or retuning."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import backend as b

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    paired = HERE / 'paired_20261104_164010_435626'
    runs = [HERE / 'kernel_20261104_163907_617079',
            HERE / 'pilot_20261104_163918_865443', paired]
    checks = {}
    for folder in runs:
        identity = read(folder / 'identity.json')['hashes']
        checks[folder.name + '/identity'] = all(b.ref.sha(p) == h for p, h in identity.items())
        result = read(folder / ('kernel.json' if folder.name.startswith('kernel') else 'results.json'))
        checks[folder.name + '/checks'] = all(result['checks'].values())
        checks[folder.name + '/postflight'] = read(folder / 'postflight.json')['unchanged']
    rows = read(paired / 'results.json')['rows']
    groups = {}
    for group in ('L2', 'L3', 'ALL'):
        groups[group] = {}
        for method in ('FIVE_COMMON', 'FOUR_COMMON'):
            selected = [r for r in rows if r['method'] == method and (group == 'ALL' or r['case'].startswith(group))]
            errors = np.array([r['errors'] for r in selected])
            times = np.array([r['seconds'] for r in selected])
            groups[group][method] = dict(n=len(selected), RMSE=float(np.sqrt(np.mean(errors**2))),
                mean_seconds=float(times.mean()), max_seconds=float(times.max()),
                normal=sum(r['normal'] for r in selected), below3=int(sum(times <= 3)),
                below5=int(sum(times <= 5)), LOS_RMSE=float(np.sqrt(np.mean([r['LOS_RMSE']**2 for r in selected]))))
    # Inspect the retained nonstationary output, without resuming its optimizer.
    engine = b.Backend([1, 2, 3, 4], 0)
    result = read(paired / 'L2_R01_FOUR_COMMON.json')
    f, y, n, _ = b.fc.cal.spectral(b.rg.inputs('L2_R01')[engine.ids])
    x = np.array(result['final']['x']); paths = result['paths']
    ff, yy = f[::59], y[::59]
    residual, jac = engine.evaluate(x, ff, yy, paths)
    eps = 1e-5
    fd = np.column_stack([(engine.evaluate(x + eps*d, ff, yy, paths)[0] -
                           engine.evaluate(x - eps*d, ff, yy, paths)[0])/(2*eps) for d in np.eye(len(x))])
    relative = float(np.linalg.norm(fd-jac)/np.linalg.norm(jac))
    checks['nonstationary_final_Jacobian_FD'] = relative < 3e-6
    residual, jac = engine.evaluate(x, f, y, paths)
    pg = float(np.max(abs(x-engine.ref.project(x-jac.T@residual, n))))
    checks['nonstationary_honest'] = pg > 1e-7 and not result['normal']
    diagnosis = dict(case='L2_R01', method='FOUR_COMMON', FD_relative=relative, PG=pg,
        reason=result['final']['reason'], iterations=result['final']['iterations'],
        optimizer_resumed=False, scope='Derivative and stop consistency; not a complete causal diagnosis')
    index = {(r['case'], r['method']): r for r in rows}
    worse = sum(index[c, 'FOUR_COMMON']['RMSE'] > index[c, 'FIVE_COMMON']['RMSE'] for c in {r['case'] for r in rows})
    assert all(checks.values()), checks
    b.fc.write(HERE / '配对汇总与审计.json', dict(groups=groups, checks=checks,
        diagnosis=diagnosis, four_worse_cases=worse, rows=rows, all_pass=True))
    print(json.dumps(dict(groups=groups, diagnosis=diagnosis, four_worse_cases=worse,
                          checks=checks), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
