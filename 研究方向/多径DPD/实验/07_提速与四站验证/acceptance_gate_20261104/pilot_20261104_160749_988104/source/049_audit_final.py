"""Read-only evidence replay with an isolated audit output; no new localization."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def main():
    final = HERE/'followup_20261104_140420_955153'
    first = HERE/'followup_20261104_140108_370562'
    checks = {}; count = 0; histories = []
    for p in sorted(HERE.glob('*/results.json')):
        d = read(p); count += len(d['rows'])
        histories.append(dict(folder=p.parent.name, completed_calls=len(d['rows']),
            complete=d['complete'], postflight=read(p.parent/'postflight.json')))
    rows = read(final/'results.json')['rows']
    for r in rows:
        name = f"{r['case']}_{r['method']}"
        a, b = read(final/(name+'.json')), read(first/(name+'.json'))
        checks[name+'/positions_repeat'] = bool(np.array_equal(a['final']['p'], b['final']['p']))
        checks[name+'/support_repeat'] = a['paths'] == b['paths']
        checks[name+'/objective_repeat'] = a['final']['cost'] == b['final']['cost']
        checks[name+'/stop_repeat'] = a['stop'] == b['stop']
        for key, ok in read(final/(name+'_audit.json')).items():
            checks[name+'/'+key] = ok
        if a['normal']:
            checks[name+'/strict_normal'] = a['final']['stationary'] and a['final']['projected_gradient'] <= 1e-7 and a['stop'] == 'no_detected_increment'
    for p, sha in read(final/'identity.json')['hashes'].items():
        checks['sha/'+p] = hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha
    checks['clean_postflight'] = read(final/'postflight.json')['unchanged']
    checks['all_28_accounted'] = len(rows) == 28
    kernel = read(HERE/'kernel_20261104_135314_964025/kernel.json')
    checks['all_kernel_checks'] = kernel['all_pass']
    output = dict(checks=checks, check_count=len(checks), all_pass=all(checks.values()),
        completed_calls_across_attempts=count, history=histories,
        final_natural=sum(r['normal'] for r in rows), final_total=len(rows),
        negative_results_preserved=True)
    (HERE/'最终审计.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print({k: v for k, v in output.items() if k not in ('checks', 'history')})
    print('Failed:', [k for k, v in checks.items() if not v])
    assert output['all_pass']


if __name__ == '__main__':
    main()
