"""Read saved main-batch evidence only; never invoke a localization solver."""
from pathlib import Path
from collections import Counter
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.g10.r2.band.runtime import read, write, sha, verify, load_input, validate, tag
from 统一模型代码.gates.g10.metrics import evaluate
from 统一模型代码.gates.g10.r1.methods import at_time

OUT = Path(__file__).resolve().parent
RUN = OUT.parent/'independent_20261016_151236'
SNAP = RUN/'main_batch_report'
c = read(RUN/'contract.json')
audit = read(SNAP/'final_audit_report.json')
marker = read(RUN/'main_batch_ready.json')
verify(c['sources']); verify(c['band_sources'])
for rel, expected in (c['sources'] | c['band_sources']).items():
    assert sha(RUN/'source_snapshot'/rel) == expected
assert sha(RUN/'input_manifest.json') == c['input_manifest_sha']
jobs = [j for j in c['jobs'] if j['method'] != 'SBL']
assert len(jobs) == 2540
expected = {tag(j['task'], j['method'], j['variant'])+'.json' for j in jobs}
assert set(audit['rows']) == expected == set(marker['main_rows'])
rows, identities = [], {}
for j in jobs:
    name = tag(j['task'], j['method'], j['variant'])+'.json'
    assert sha(RUN/'rows'/name) == audit['rows'][name] == marker['main_rows'][name]
    key = tuple(j['task'])
    if key not in identities:
        identities[key] = load_input(RUN, j['task'])[-1]
    row = read(RUN/'rows'/name)
    validate(row, j, identities[key])
    assert np.isfinite(row['result']['positions']).all()
    assert row['metrics']['predictions'] == row['metrics']['targets']
    assert row['scene']['role'] == 'independent'
    rows.append(row)


def stats(items):
    err = np.array([r['metrics']['errors_m'] for r in items])
    flags = np.array([r['metrics']['success']['20'] for r in items])
    seconds = np.array([r['result']['seconds'] for r in items])
    return dict(n=len(items), both=int(flags.all(1).sum()),
        strong=int(flags[:, 0].sum()), weak=int(flags[:, -1].sum()),
        rmse=float(np.sqrt(np.mean(err**2))), strong_rmse=float(np.sqrt(np.mean(err[:, 0]**2))),
        weak_rmse=float(np.sqrt(np.mean(err[:, -1]**2))),
        weak_p95=float(np.quantile(err[:, -1], .95)), weak_max=float(err[:, -1].max()),
        strong_max=float(err[:, 0].max()), gospa=float(np.mean([r['metrics']['gospa_m'] for r in items])),
        seconds=float(seconds.mean()), time_p95=float(np.quantile(seconds, .95)), time_max=float(seconds.max()),
        stops=dict(Counter(r['result']['stop_reason'] for r in items)))


mixed = [r for r in rows if r['kind'] == 'mixed']
methods = list(dict.fromkeys(r['method'] for r in mixed))
summary = {m:stats([r for r in mixed if r['method'] == m]) for m in methods}
report = read(SNAP/'comparison_report.json')
for s in report['summaries']:
    if s['kind'] == 'mixed':
        computed = summary[s['method']]
        assert computed['both'] == s['success_count']
        for a, b in [('weak_rmse', 'weak_rmse_m'), ('strong_rmse', 'strong_rmse_m'), ('gospa', 'gospa_m'), ('seconds', 'seconds')]:
            assert abs(computed[a]-s[b]) < 1e-10
lookup = {(tuple(r['task']), r['method']):r for r in mixed}
for task in {tuple(r['task']) for r in mixed}:
    for m in ['NOMP_EXTRA12', 'NOMP_JOINT']:
        np.testing.assert_allclose(lookup[task, m]['result']['mechanism']['prefix_positions'],
            lookup[task, 'NOMP']['result']['mechanism']['prefix_positions'], rtol=0, atol=1e-8)
    for suffix in ['ALT', 'JOINT']:
        np.testing.assert_allclose(lookup[task, 'AP_1_'+suffix]['result']['mechanism']['branches'][0]['final_positions'],
            lookup[task, 'AP_5_'+suffix]['result']['mechanism']['branches'][0]['final_positions'], rtol=0, atol=1e-8)
gap = {m:{str(g):stats([r for r in mixed if r['method'] == m and r['task'][2] == g]) for g in [0,3,6,9,12,15]} for m in methods}
groups = {m:{str(g):stats([r for r in mixed if r['method'] == m and r['task'][0] == g]) for g in range(4)} for m in methods}
anytime = {}
for m in methods:
    items = [r for r in mixed if r['method'] == m]
    anytime[m] = {str(t):sum(all(evaluate(r['truth'], at_time(r['result'], t))['success']['20']) for r in items)
                  for t in [.25,.5,1,2,3,5,15,30,60]}
pairs = []
for candidate, baseline in [('AP_1_JOINT','AP_1_ALT'),('AP_5_JOINT','AP_5_ALT'),('AP_5_JOINT','AP_1_JOINT'),
                              ('NOMP_JOINT','NOMP'),('NOMP_EXTRA12','NOMP'),('AP_FULL','AP_1_JOINT')]:
    rescued=harmed=0; differences=[]; improvements=[]
    for task in {tuple(r['task']) for r in mixed}:
        a,b=lookup[task,candidate],lookup[task,baseline]
        sa,sb=all(a['metrics']['success']['20']),all(b['metrics']['success']['20'])
        rescued+=int(sa and not sb);harmed+=int(sb and not sa)
        differences.append(a['result']['seconds']-b['result']['seconds'])
        improvements.append(b['result']['events'][-1]['native_objective']-a['result']['events'][-1]['native_objective'])
    pairs.append(dict(candidate=candidate,baseline=baseline,rescued=rescued,harmed=harmed,
        mean_seconds_difference=float(np.mean(differences)),objective_improvement_max=float(max(improvements))))
branch = {}
for m in ['AP_5_ALT','AP_5_JOINT']:
    items=[r for r in mixed if r['method']==m]
    improves=[r['result']['mechanism']['branches'][0]['final_objective']-
              min(b['final_objective'] for b in r['result']['mechanism']['branches']) for r in items]
    branch[m]=dict(selected_rank=dict(Counter(r['result']['mechanism']['selected_rank'] for r in items)),
        objective_improvement_max=max(improves), improvements_over_1e_6=sum(x>1e-6 for x in improves))
operations = {}
for m in methods:
    items=[r for r in mixed if r['method']==m]
    operations[m]={k:float(np.mean([r['result']['counts'].get(k,0) for r in items]))
        for k in ['candidate_points','frequency_response_points','joint_evaluations','gradient_calls','iterations']}
isolated={m:{k:stats(z) for k in ['weak','strong'] if (z:=[r for r in rows if r['method']==m and r['kind']==k])} for m in methods}
facts=dict(proof=dict(passed=True,rows=len(rows),observations=len(identities),metric_recalculation=True,
    source_and_input_identity=True,paired_prefix=True,no_solver_executed=True),
    source=str(RUN),snapshot=str(SNAP),snapshot_state=read(SNAP/'status.json'),
    source_summary_sha=sha(SNAP/'comparison_report.json'),environment=c['environment'],layouts=c['layouts'],
    summary=summary,isolated=isolated,power_gap=gap,layout_results=groups,anytime=anytime,pairs=pairs,
    branch_diagnostics=branch,operations=operations,
    algorithm_seconds=sum(r['result']['seconds'] for r in rows),
    input_energy_min=min(v['band_energy']['minimum_fraction'] for v in read(RUN/'input_manifest.json').values()))
write(OUT/'analysis.json',facts)
print('VERIFIED',facts['proof'])
for m,v in summary.items():print(m,v)
print('PAIRS',pairs)
print('BRANCH',branch)
print('ANYTIME',anytime)
print('TOTAL',facts['algorithm_seconds'],facts['snapshot_state']['seconds'])
