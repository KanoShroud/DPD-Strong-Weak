"""Saved-position reevaluation only; frozen v1 files and Ho report remain untouched."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..metrics import evaluate as evaluate_v1, partial, aggregate
from ...g8.physics import RECEIVERS

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'outputs_e2e/unified'
OUT = BASE / 'e2e_g9/eval_2pctR_20261011_01'
R = float(np.linalg.norm(RECEIVERS[:, None]-RECEIVERS[None], axis=-1).max())
RADIUS = .02*R
PROTOCOL = 'EVAL-20261011-v2'
IDENTITIES = {}


from migration_paths import input_path  # noqa: E402

def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    path = input_path(path)
    data = path.read_bytes()
    IDENTITIES[str(path.relative_to(ROOT))] = sha(data)
    result = json.loads(data)
    if isinstance(result, dict) and 'row_sha256' in result:
        payload = {k: v for k, v in result.items() if k != 'row_sha256'}
        # Match the frozen runner's JSON serialization exactly.
        expected = sha(json.dumps(payload, sort_keys=True).encode('utf-8'))
        assert expected == result['row_sha256'], path
    return result


def evaluate(truth, predicted):
    result = evaluate_v1(truth, predicted)
    a = np.asarray(truth).reshape(-1, 2)
    b = np.asarray(predicted).reshape(-1, 2)
    d = np.linalg.norm(a[:, None]-b[None], axis=-1)
    valid = [x for x in partial(len(a), len(b)) if all(d[i, j] <= RADIUS for i, j in x)]
    pairs = min(valid, key=lambda x: (-len(x), sum(d[i, j]**2 for i, j in x), x))
    flags = [False]*len(a)
    for i, _ in pairs:
        flags[i] = True
    result['success']['20'] = flags
    result['protocol'] = PROTOCOL
    assert sum(flags) <= sum(result['success']['100'])
    return result


def summary(rows):
    result = aggregate(rows)
    for radius in ('20', '100'):
        flags = [x['metrics']['success'][radius] for x in rows]
        for name, values in [('strong', [f[0] for f in flags if len(f) == 2]),
                             ('weak', [f[1] for f in flags if len(f) == 2]),
                             ('both', [all(f) for f in flags if len(f) == 2]),
                             ('single', [f[0] for f in flags if len(f) == 1])]:
            if values:
                result[name+'_'+radius] = {'successes': sum(values), 'total': len(values),
                                          'rate': float(np.mean(values))}
    return result


def geometry(group):
    rng = np.random.default_rng(np.random.SeedSequence([2026101009, 1, group, 0]))
    while True:
        pos = rng.uniform(-1000, 1000, (2, 2))
        if np.linalg.norm(pos[:, None]-RECEIVERS, axis=-1).min() >= 150 and np.linalg.norm(pos[0]-pos[1]) >= 300:
            return pos


def test():
    assert R == 1000 and RADIUS == 20
    assert evaluate([[0, 0]], [[20, 0]])['success']['20'] == [True]
    assert evaluate([[0, 0]], [[20.001, 0]])['success']['20'] == [False]
    assert evaluate([[0, 0], [100, 0]], [[100, 0], [0, 0]])['success']['20'] == [True, True]
    assert sum(evaluate([[0, 0], [10, 0]], [[5, 0]])['success']['20']) == 1
    assert evaluate([[0, 0]], [])['success']['20'] == [False]


def main():
    test()
    assert not OUT.exists(), 'Use a new output directory; never overwrite evidence.'
    protected = ROOT/'outputs_e2e/reports/kc_ho_20261012'
    ho = {str(p): sha(p.read_bytes()) for p in protected.rglob('*') if p.is_file()}
    report = dict(protocol=PROTOCOL, baseline_definition='maximum pairwise receiver distance',
                  receivers_m=RECEIVERS.tolist(), R_m=R, radius_m=RADIUS,
                  gospa=dict(p=2, alpha=2, c_m=100), no_inference_or_training=True)
    detailed = []
    g9 = BASE/'e2e_g9'
    scores = [read(p) for p in sorted((g9/'r1_failure_20261011_01/score_rows').glob('*.json'))]
    truth = {(x['group'], x['realization']): x['truth'] for x in scores}
    formal = defaultdict(list)
    for p in sorted((g9/'formal_20261011_repaired/rows').glob('*.json')):
        row = read(p)
        t = truth[row['group'], row['realization']]
        if row['kind'] != 'mixed':
            t = [t[0 if row['kind'] == 'strong' else 1]]
        for factor in (.5, 1., 2.):
            budget = row['budget_seconds']*factor/2
            events = [e for e in row['events'] if e['seconds'] <= budget]
            pred = events[-1]['positions'] if events else []
            m = evaluate(t, pred)
            if factor == 2:
                assert np.allclose(m['errors_m'], row['metrics']['errors_m']), p
            item = dict(group=row['group'], realization=row['realization'], N=row['N'],
                        method=row['method'], kind=row['kind'], factor=factor, metrics=m)
            detailed.append(item)
            formal[f"{row['kind']}/{row['method']}/{factor}"].append(item)
            formal[f"{row['kind']}/{row['method']}/{factor}/N{row['N']}"].append(item)
    report['formal'] = {k: summary(v) for k, v in formal.items()}
    ci = {}
    for factor in (.5, 1., 2.):
        for control in ('A', 'B'):
            cs = formal[f'mixed/C/{factor}']; bs = formal[f'mixed/{control}/{factor}']
            means = []
            for group in sorted({x['group'] for x in cs}):
                means.append(np.mean([x['metrics']['success']['20'][1] for x in cs if x['group']==group])
                             -np.mean([x['metrics']['success']['20'][1] for x in bs if x['group']==group]))
            rng = np.random.default_rng(2026101009)
            boot = np.mean(np.asarray(means)[rng.integers(len(means), size=(2000,len(means)))], axis=1)
            ci[f'C-{control}/{factor}'] = dict(delta=float(np.mean(means)), ci95=np.quantile(boot,[.025,.975]).tolist(), unit='layout', replicates=2000)
    report['formal_weak20_paired_bootstrap'] = ci
    qual = defaultdict(list)
    for p in sorted((g9/'repair_prepare_20261011_01/rows').glob('qual*.json')):
        row=read(p); err=row['metrics']['errors_m'][0]
        qual[f"snr{row['snr']}/source{row['source']}/N{row['N']}"].append(err)
    report['qualification'] = {k:dict(total=len(v), success20=sum(e is not None and e<=20 for e in v), success100=sum(e is not None and e<=100 for e in v)) for k,v in qual.items()}
    failures = [x for x in scores if not evaluate(x['truth'],x['predicted'])['success']['20'][1]]
    report['r1_all'] = dict(total=len(scores), failures20=len(failures),
        estimated_truth_higher=sum(x['estimated']['truth_higher'] for x in failures),
        true_strong_truth_higher=sum(x['true_strong']['truth_higher'] for x in failures),
        estimated_oracle20=sum(x['estimated']['oracle_error_m']<=20 for x in failures))
    dev = defaultdict(list)
    for dirname in ('r1_failure_20261011_01','r2_coverage_dev_20261011_01','r2_coverage_shortlist_dev_20261011_01'):
        sub = 'dense_rows' if dirname.startswith('r1') else 'rows'
        for p in sorted((g9/dirname/sub).glob('*.json')):
            row=read(p)
            if dirname.startswith('r1'):
                variants={k:row[k] for k in ('estimated','true_strong')}
            elif 'shortlist' in dirname:
                variants={'top32':row['result']}
            else:
                variants=row['results']
            for name,v in variants.items():
                dev[f"{dirname}/{row['selection']}/{name}"].append(v['error_m'])
    report['fixed_historical_selection']={k:dict(total=len(v), weak20=sum(e<=20 for e in v),weak100=sum(e<=100 for e in v)) for k,v in dev.items()}
    fresh=defaultdict(list)
    for p in sorted((g9/'r2_coverage_fresh_20261011_01/rows').glob('*.json')):
        row=read(p); t=geometry(row['group'])
        for method in ('B','D'):
            v=row[method]; m=evaluate(t,v['positions'])
            assert np.allclose(m['errors_m'],v['metrics']['errors_m'],rtol=0,atol=1e-9),p
            item=dict(group=row['group'],N=row['N'],method=method,metrics=m,seconds=v['seconds'])
            fresh[method].append(item);fresh[f"{method}/N{row['N']}"].append(item)
    report['r2_fresh']={k:summary(v) for k,v in fresh.items()}
    g8=defaultdict(list)
    for row in read(BASE/'e2e_g8/r5_projection_20261006_01/rows.json'):
        for name,v in row['methods'].items():
            if 'positions' in v:
                g8[name].append(dict(metrics=evaluate(row['positions'],v['positions'])))
    report['g8_r5_success_only']={k:{n:v for n,v in summary(rows).items() if n.endswith(('_20','_100')) and isinstance(v,dict)} for k,rows in g8.items()}
    g8r6=defaultdict(list)
    for row in read(BASE/'e2e_g8/r6_failure_20261008_01/rows.json'):
        g8r6['old'].append(row['r5_second_error'])
        for k in ('A','B','C','D'):
            if k in row and 'error_m' in row[k]:g8r6[k].append(row[k]['error_m'])
    report['g8_r6_selected_weak_only']={k:dict(total=len(v),weak20=sum(e<=20 for e in v),weak100=sum(e<=100 for e in v)) for k,v in g8r6.items()}
    assert all(sha((ROOT/p).read_bytes())==h for p,h in IDENTITIES.items())
    assert ho=={str(p):sha(p.read_bytes()) for p in protected.rglob('*') if p.is_file()}
    OUT.mkdir(parents=True)
    for name,data in [('report.json',report),('formal_rows.json',detailed),('input_manifest.json',IDENTITIES),
                      ('final_audit_report.json',dict(status='PASS', input_files=len(IDENTITIES),protected_ho_files=len(ho),protected_ho_unchanged=True, tests='PASS', geometry_matches_saved_errors=True))]:
        (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(OUT),'r2':report['r2_fresh'],'qualification':report['qualification']},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
