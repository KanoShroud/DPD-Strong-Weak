"""Rebuild short-paper figures and audit saved results; never run a locator."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('FC_VP', 'VP_VP')
STYLE = {'FC_VP': ('#3C5488', '--', 'FC\u2013VP'),
         'VP_VP': ('#E64B35', '-', 'VP\u2013VP')}
AUTHORS = (
    ('Benjian Hao', '0000-0002-6676-7306', 'bjhao@xidian.edu.cn'),
    ('Xinlei Wang', '0000-0002-6855-3686', 'wangxinlei@stu.xidian.edu.cn'),
    ('Zan Li', '0000-0002-5207-6504', 'zanli@xidian.edu.cn'))
JOINT_TOL = 1e-10  # Descriptive numerical equality, not a significance test.
SMALL_STEP_M = .01  # Frozen local-exit threshold; not a new stopping rule.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_trajectories(project):
    """Replay saved certificates and steps, with one summary per observation."""
    project = project.resolve(strict=True)
    manifest = json.loads((ROOT / 'data/source_manifest.json').read_text(encoding='utf-8'))
    audit_path = project / 'paper/data/review_supplement_20261020/audit.json'
    assert sha(audit_path) == manifest['source_files'][str(audit_path)]
    source_audit = json.loads(audit_path.read_text(encoding='utf-8'))
    local = ROOT / 'data/per_case.csv'
    original = project / 'paper/data/review_supplement_20261020/per_case.csv'
    assert sha(local) == manifest['source_files'][str(original)]
    saved = {(r['case'], r['method']): r for r in read_csv(local) if r['stage'] == 'mechanism'}
    sources, rows = {}, []
    for name, expected in source_audit['source_files'].items():
        path = Path(name)
        if path.parent.name != 'pairs' or not path.name.startswith('mechanism_'):
            continue
        path = path.resolve(strict=True)
        assert path.is_relative_to(project / 'outputs_e2e') and sha(path) == expected
        sources[str(path)] = expected
        pair = json.loads(path.read_text(encoding='utf-8'))
        case = '_'.join(map(str, pair['task']))
        assert pair['stage'] == 'mechanism' and pair['iq_sha']
        assert {r['method'] for r in pair['rows']} == set(METHODS)
        assert pair['rows'][0]['result']['config'] == pair['rows'][1]['result']['config']
        for record in pair['rows']:
            method, result = record['method'], record['result']
            ref = saved[(case, method)]
            assert result['stop_reason'] == ref['stop_reason']
            for source, target in (('native_objective', 'objective'), ('projected_gradient', 'gradient')):
                np.testing.assert_allclose(result[source], float(ref[target]), rtol=1e-14, atol=1e-16)
            updates = result['updates']
            assert len(updates) == int(ref['updates']) and updates
            accepted = [u for u in updates if u['accepted']]
            small = [u for u in accepted if u['step_m'] < SMALL_STEP_M]
            guards = ('floor_active', 'cap_active', 'backtracked', 'boundary_clipped')
            unguarded_small = [u for u in small if not any(u[g] for g in guards)]
            ratios = [u['step_m']/u['raw_direction_m'] for u in accepted if u['raw_direction_m'] > 0]
            assert accepted and ratios
            assert all(u['step_m'] == 0 for u in updates if not u['accepted'])
            unmodified = [u for u in accepted if not any(u[g] for g in guards)]
            for u in unmodified:
                np.testing.assert_allclose(u['step_m'], u['raw_direction_m'], rtol=1e-6, atol=1e-10)
            cycles = sum(c['cycles'] for c in result['native_cycles'])
            assert cycles == int(ref['cycles'])
            row = dict(case=case, method=method, layout=ref['layout'], repeat=ref['repeat'],
                       gap=ref['gap'], objective=float(ref['objective']), gradient=float(ref['gradient']),
                       strong_error_m=float(ref['strong_error_m']), weak_error_m=float(ref['weak_error_m']),
                       both20=ref['both20'] == 'True', seconds=float(ref['seconds']),
                       attempts=len(updates), accepted=len(accepted), rejected=len(updates)-len(accepted),
                       small_accepted=len(small), small_fraction=len(small)/len(accepted),
                       unguarded_small=len(unguarded_small),
                       unguarded_fraction_of_small=len(unguarded_small)/len(small) if small else None,
                       accepted_to_direction_median=float(np.median(ratios)),
                       accepted_zero_direction=sum(u['raw_direction_m'] == 0 for u in accepted),
                       cycles=cycles, second_source_cycles=result['native_cycles'][-1]['cycles'])
            for guard in guards:
                row[guard+'_count'] = sum(u[guard] for u in updates)
                row[guard+'_fraction'] = row[guard+'_count']/len(updates)
            rows.append(row)
    assert len(sources) == 240 and len(rows) == len(saved) == 480
    rows.sort(key=lambda r: (int(r['layout']), int(r['repeat']), int(r['gap']), r['method']))
    paired = {m: {r['case']: r for r in rows if r['method'] == m} for m in METHODS}
    assert paired['FC_VP'].keys() == paired['VP_VP'].keys()
    cases = list(paired['FC_VP'])
    delta = np.array([paired['FC_VP'][k]['objective']-paired['VP_VP'][k]['objective'] for k in cases])
    quantiles = lambda values: dict(zip(('min', 'q25', 'median', 'q75', 'p95', 'max'),
                                      map(float, np.quantile(values, [0, .25, .5, .75, .95, 1]))))
    largest = cases[int(np.argmax(delta))]
    terminal = dict(delta_definition='J_FC_minus_J_VP', equality_tolerance=JOINT_TOL,
                    improved=int((delta > JOINT_TOL).sum()),
                    numerically_equal=int((abs(delta) <= JOINT_TOL).sum()),
                    worsened=int((delta < -JOINT_TOL).sum()), delta=quantiles(delta),
                    gradient_lower=sum(paired['VP_VP'][k]['gradient'] < paired['FC_VP'][k]['gradient'] for k in cases),
                    same_success_identity=all(paired['VP_VP'][k]['both20'] == paired['FC_VP'][k]['both20'] for k in cases),
                    largest_difference={m: paired[m][largest] for m in METHODS})
    summaries = {}
    for method in METHODS:
        rr = list(paired[method].values())
        fields = ('gradient', 'small_fraction', 'small_accepted', 'accepted_to_direction_median', 'second_source_cycles',
                  'attempts', 'accepted', 'rejected') + tuple(g+'_fraction' for g in guards)
        summaries[method] = {field: quantiles([r[field] for r in rr]) for field in fields}
        summaries[method]['observations_with_small_steps'] = sum(r['small_accepted'] > 0 for r in rr)
        summaries[method]['total_small_accepted'] = sum(r['small_accepted'] for r in rr)
        summaries[method]['total_unguarded_small'] = sum(r['unguarded_small'] for r in rr)
        summaries[method]['unguarded_fraction_of_small'] = quantiles(
            [r['unguarded_fraction_of_small'] for r in rr if r['small_accepted']])
        summaries[method]['total_accepted'] = sum(r['accepted'] for r in rr)
        summaries[method]['total_rejected'] = sum(r['rejected'] for r in rr)
        summaries[method]['observations_without_any_guard'] = sum(
            all(r[g+'_count'] == 0 for g in guards) for r in rr)
    report = dict(passed=True, pairs=240, no_locator_run=True,
                  protocol=manifest['protocol'], terminal=terminal, per_observation=summaries,
                  small_step_m=SMALL_STEP_M,
                  definitions=dict(objective='Full joint residual after fitting all source coefficients, divided by retained input energy.',
                      gradient='Infinity norm of box-KKT projected gradient of the same joint objective; metre coordinates.',
                      small_fraction='Accepted updates below 0.01 m divided by accepted updates, separately per observation.',
                      direction='After Hessian eigenvalue repair, before step cap/backtracking/boundary clipping.',
                      guard_fraction='Triggered attempts divided by all attempts, separately per observation.',
                      aggregation='Descriptive quantiles across observations, not an independence claim or an update-level significance test.'),
                  source_audit=dict(path=str(audit_path), sha256=sha(audit_path)),
                  per_case_sha256=sha(local), source_files=sources, script_sha256=sha(__file__))
    target = ROOT / 'data/terminal_step_per_case.csv'
    write_csv(target, rows)
    report['derived_csv_sha256'] = sha(target)
    (ROOT / 'data/terminal_step_audit.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('pairs', 'terminal', 'per_observation')}, ensure_ascii=False))


def import_evidence(project):
    src = project / 'paper/data/review_supplement_20261020'
    audit = json.loads((src / 'audit.json').read_text(encoding='utf-8'))
    assert audit['status'] == 'COMPLETED_OFFLINE_AUDIT'
    manifest = dict(protocol=audit['protocol'], source_files={},
                    literature_sources=[], paired=audit['paired'],
                    configurations=audit['frozen_configs'],
                    geometry_seed=audit['geometry_seed'],
                    waveform_seed=audit['waveform_seed'])
    for name in ('per_case.csv', 'summary.csv'):
        path = src / name
        assert sha(path) == sha(ROOT / 'data' / name)
        manifest['source_files'][str(path)] = sha(path)
    for path in (src / 'audit.json',
                 project / '统一模型代码/gates/g10/r4/adaptation_review.py',
                 project / '统一模型代码/gates/g10/r4/bnomp.py',
                 project / '统一模型代码/gates/g10/r4/paper_checks/review_supplement.py'):
        manifest['source_files'][str(path)] = sha(path)
    # The long manuscript is maintained concurrently and may absorb its supplement.
    # It is a writing reference, not a numerical dependency of this short paper.
    manifest['writing_references'] = {
        str(path): sha(path) for path in (
            project / 'paper/paper_submission.tex', project / 'paper/paper_supplement.tex')
        if path.exists()}
    # Verify the source audit's individual identities without executing solvers.
    checked = 0
    for field, content in audit.items():
        if isinstance(content, dict):
            for path, expected in content.items():
                if isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected):
                    p = Path(path)
                    if p.is_absolute() and p.exists():
                        assert sha(p) == expected, str(p)
                        checked += 1
    manifest['source_identity_checks'] = checked
    layouts = []
    raw = project / 'outputs_e2e/unified/e2e_g10/r4/formal_recovered_20261017_093236_136920'
    for i in range(20):
        metadata_path = raw / 'inputs' / f'{i}_400_0.json'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        manifest['source_files'][str(metadata_path)] = sha(metadata_path)
        for k, position in enumerate(metadata['positions']):
            layouts.append(dict(stage='mechanism', layout=i, source=k,
                                x_m=position[0], y_m=position[1]))
    for i, entry in enumerate(audit['layouts']):
        for k, position in enumerate(entry['positions']):
            layouts.append(dict(stage='geometry', layout=i, source=k,
                                x_m=position[0], y_m=position[1]))
    write_csv(ROOT / 'data/source_positions.csv', layouts)
    manifest['realizations'] = dict(mechanism=[400, 410], geometry=list(range(1000, 1005)))
    manifest['mechanism_seeds'] = dict(geometry=20261017041, waveform=20261017042)
    manifest['template_hashes'] = {}
    for name in ('cas-sc.cls', 'cas-common.sty', 'cas-model2-names.bst'):
        p = project / 'paper/template' / name
        assert sha(p) == sha(ROOT / 'template' / name)
        manifest['template_hashes'][name] = sha(p)
    # Subset the already verified BibTeX without copying unrelated references.
    selected = ('ChenHudsonYao2002', 'WeissAmar2005', 'Mamandipoor2016',
                'Yang2022', 'Zeng2025', 'GolubPereyra1973', 'ITURP5252024')
    bib = (project / 'paper/references/references.bib').read_text(encoding='utf-8')
    entries = re.split(r'(?m)(?=^@)', bib)
    chosen = [entry.strip() for entry in entries
              if any(re.match(r'@\w+\{' + key + ',', entry) for key in selected)]
    assert len(chosen) == len(selected)
    (ROOT / 'references/references.bib').write_text('\n\n'.join(chosen)+'\n', encoding='utf-8')
    manifest['literature_sources'] = [
        {'key': key, 'source': 'paper/references/references.bib'} for key in selected]
    manifest['public_checks'] = [
        'https://arxiv.org/html/1509.01942v2#S2.SS1',
        'https://doi.org/10.1016/j.jsv.2021.116549',
        'https://doi.org/10.1109/ACCESS.2025.3645128',
        'https://doi.org/10.1137/0710036']
    diagnostic = ROOT / 'data/development_diagnostic.json'
    if diagnostic.exists():
        source = project / ('outputs_e2e/unified/e2e_g10/r4/'
                            'adaptation_review_20261017_011824_276518/comparison_report.json')
        assert sha(diagnostic) == sha(source)
        manifest['development_diagnostic'] = dict(
            local_file='data/development_diagnostic.json', source=str(source),
            sha256=sha(diagnostic),
            scope='Eight development observations, FIXED_STRICT / FC-FC; not the 240 formal FC-VP pairs.')
    (ROOT / 'data/source_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def check_documents():
    import fitz
    en = (ROOT / 'main.tex').read_text(encoding='utf-8')
    zh = (ROOT / 'main_zh.tex').read_text(encoding='utf-8')
    blocks = r'\\begin\{(equation|align)\}(.*?)\\end\{\1\}'
    normalize = lambda text: re.sub(r'\s+', '', text)
    equations = [[normalize(body) for _, body in re.findall(blocks, text, re.S)]
                 for text in (en, zh)]
    assert equations[0] == equations[1] and len(equations[0]) == 8
    get_cites = lambda text: re.findall(r'\\cite\{([^}]+)\}', text)
    assert get_cites(en) == get_cites(zh)
    citations = set(','.join(get_cites(en)).split(','))
    bib = (ROOT / 'references/references.bib').read_text(encoding='utf-8')
    assert citations <= set(re.findall(r'@\w+\{([^,]+),', bib))
    result_lines = [[line.strip() for line in text.splitlines()
                     if re.match(r'^(FC--VP|VP--VP|AP1|AP5|VP-BNOMP) &', line)]
                    for text in (en, zh)]
    assert result_lines[0] == result_lines[1] and len(result_lines[0]) == 5
    abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', en, re.S).group(1)
    abstract_words = len(abstract.split())
    assert abstract_words <= 250
    manifest = json.loads((ROOT / 'data/source_manifest.json').read_text(encoding='utf-8'))
    diagnostic_ref = manifest['development_diagnostic']
    diagnostic_file = ROOT / diagnostic_ref['local_file']
    assert sha(diagnostic_file) == diagnostic_ref['sha256']
    diagnostic = json.loads(diagnostic_file.read_text(encoding='utf-8'))
    fixed = diagnostic['summaries']['FIXED_STRICT']
    assert diagnostic['status'] == 'COMPLETED_BNOMP_MECHANISM_REVIEW'
    assert fixed['cases'] == 8 and fixed['steps'] == 21436 and fixed['rejected'] == 0
    assert fixed['small_accepted_steps'] == 17935
    assert all(fixed['safeguards'][key] == 0 for key in (
        'floor_active', 'cap_active', 'backtracked', 'boundary_clipped'))
    small_percent = 100 * fixed['small_accepted_steps'] / fixed['steps']
    terminal_file = ROOT / 'data/terminal_step_audit.json'
    terminal = json.loads(terminal_file.read_text(encoding='utf-8'))
    terminal_csv = ROOT / 'data/terminal_step_per_case.csv'
    assert sha(terminal_csv) == terminal['derived_csv_sha256']
    assert sha(ROOT / 'data/per_case.csv') == terminal['per_case_sha256']
    assert sha(__file__) == terminal['script_sha256']
    terminal_rows = read_csv(terminal_csv)
    assert len(terminal_rows) == 480 and terminal['pairs'] == 240
    pairs = {m: {r['case']: r for r in terminal_rows if r['method'] == m} for m in METHODS}
    keys = list(pairs['FC_VP'])
    assert len(keys) == 240 and pairs['FC_VP'].keys() == pairs['VP_VP'].keys()
    delta = np.array([float(pairs['FC_VP'][k]['objective'])-float(pairs['VP_VP'][k]['objective']) for k in keys])
    assert int((delta > JOINT_TOL).sum()) == terminal['terminal']['improved'] == 199
    assert int((abs(delta) <= JOINT_TOL).sum()) == terminal['terminal']['numerically_equal'] == 41
    assert int((delta < -JOINT_TOL).sum()) == terminal['terminal']['worsened'] == 0
    assert all(float(pairs['VP_VP'][k]['gradient']) < float(pairs['FC_VP'][k]['gradient']) for k in keys)
    scientific = lambda value: (lambda mantissa, exponent:
                               mantissa+r'\times10^{'+str(int(exponent))+'}')(*f'{value:.2e}'.split('e'))
    for method in METHODS:
        rr = list(pairs[method].values())
        summary = terminal['per_observation'][method]
        for field in ('gradient', 'small_fraction', 'second_source_cycles'):
            values = np.array([float(r[field]) for r in rr])
            for key, q in (('median', .5), ('p95', .95)):
                np.testing.assert_allclose(np.quantile(values, q), summary[field][key], rtol=1e-14)
        for text in (en, zh):
            assert f"{100*summary['small_fraction']['median']:.2f}" in text
            assert scientific(summary['gradient']['median']) in text
            assert scientific(summary['gradient']['p95']) in text
    for text in (en, zh):
        assert all(token in text for token in ('21,436', '17,935', f'{small_percent:.2f}'))
        assert all(token in text for token in ('359,717', '89.5', '197.85', '1647', '199', '41'))
        assert re.findall(r'\\author\[inst1\]\{([^}]+)\}', text) == [a[0] for a in AUTHORS]
        for name, orcid, email in AUTHORS:
            assert rf'\author[inst1]{{{name}}}[orcid={orcid}]' in text
            assert rf'\ead{{{email}}}' in text
        assert r'\author[inst1]{Zan Li}[orcid=0000-0002-5207-6504]'+'\n'+r'\cormark[1]' in text
        assert r'\cortext[cor1]{Corresponding author.}' in text
        assert 'State Key Laboratory of Integrated Service Networks, School of Telecommunications Engineering, Xidian University' in text
        assert r'\shortauthors{B. Hao et al.}' in text
        assert text.count('% TODO(CRediT):') == 3
        assert 'Declaration of generative AI use' not in text and '生成式AI使用声明' not in text
        assert 'ChatGPT' not in text and 'Codex' not in text
    for name, value in manifest['template_hashes'].items():
        assert sha(ROOT / 'template' / name) == value
    report = dict(bilingual_equation_blocks=8, bilingual_result_rows=5,
                  same_citations=True, abstract_words=abstract_words,
                  template_files_unchanged=True, documents={}, figures={},
                  authors=AUTHORS, corresponding_author='Zan Li', affiliation='Xidian University',
                  ai_declaration_removed=True,
                  terminal_step_audit_sha256=sha(terminal_file),
                  development_diagnostic=dict(cases=fixed['cases'], steps=fixed['steps'],
                      small_steps=fixed['small_accepted_steps'], small_percent=small_percent,
                      matched_source_sha256=diagnostic_ref['sha256']))
    for name in ('main', 'main_zh'):
        tex = (ROOT / (name+'.tex')).read_text(encoding='utf-8')
        labels = re.findall(r'\\label\{([^}]+)\}', tex)
        refs = re.findall(r'\\(?:eqref|ref)\{([^}]+)\}', tex)
        assert len(labels) == len(set(labels)) and set(refs) <= set(labels)
        assert '\\documentclass[a4paper,12pt,review]{cas-sc}' in tex
        assert not any(s in tex for s in ('margin=', '@fleqnfalse', '\\cs_set', '\\resizebox'))
        log = (ROOT / 'build' / (name+'.log')).read_text(encoding='utf-8', errors='replace')
        assert 'Missing character' not in log
        assert not re.search(r'(Reference|Citation).*undefined|There were undefined', log)
        warnings = re.findall(r'Overfull.*', log)
        # CAS 2.4 reports an internal front-matter box wider than its vbox.
        # It occurs before body text; page-coordinate checks and rendering
        # independently test whether any printed content is actually clipped.
        assert all('129.88684pt' in warning for warning in warnings), warnings
        document = fitz.open(ROOT / (name+'.pdf'))
        for author, orcid, email in AUTHORS:
            assert author in document[0].get_text()
            assert email in document[0].get_text()
            assert orcid in document[0].get_text()
        assert 'Corresponding author.' in document[0].get_text()
        assert 'Xidian University' in re.sub(r'\s+', ' ', document[0].get_text())
        if name == 'main':
            assert len(document) <= 10
        outside = []
        for i, page in enumerate(document):
            assert abs(page.rect.width-595.28) < .1 and abs(page.rect.height-841.89) < .1
            text = page.get_text()
            assert f'Page {i+1} of {len(document)}' in text
            for block in page.get_text('dict')['blocks']:
                for line in block.get('lines', []):
                    for span in line['spans']:
                        box = fitz.Rect(span['bbox'])
                        if box.x0 < -1 or box.y0 < -1 or box.x1 > page.rect.width+1 or box.y1 > page.rect.height+1:
                            outside.append((i+1, span['text']))
        assert not outside, outside
        report['documents'][name] = dict(pages=len(document), a4=True, labels=len(labels),
            sha256=sha(ROOT / (name+'.pdf')), source_sha256=sha(ROOT / (name+'.tex')),
            page_bounds_ok=True, template_frontmatter_box_warnings=warnings)
    for name in ('error_cdf', 'cost_cdf'):
        doc = fitz.open(ROOT / 'figures' / (name+'.pdf'))
        assert sum(len(p.get_images()) for p in doc) == 0
        fonts = doc[0].get_fonts()
        assert fonts and all(len(doc.extract_font(f[0])[3]) > 0 for f in fonts)
        report['figures'][name] = dict(vector=True, embedded_fonts=True)
    report['passed'] = True
    (ROOT / 'build/document_check.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--import-from', type=Path)
    parser.add_argument('--check-documents', action='store_true')
    parser.add_argument('--audit-trajectories', type=Path, metavar='PROJECT',
                        help='Audit saved formal pairs only; never execute a solver.')
    args = parser.parse_args()
    if args.audit_trajectories:
        audit_trajectories(args.audit_trajectories)
        return
    if args.check_documents:
        check_documents()
        return
    if args.import_from:
        import_evidence(args.import_from)
    rows = read_csv(ROOT / 'data/per_case.csv')
    saved = read_csv(ROOT / 'data/summary.csv')
    assert len(rows) == 1920
    calculated = []
    for s in saved:
        rr = [r for r in rows if (r['stage'], r['method']) == (s['stage'], s['method'])]
        assert len(rr) == int(s['n'])
        strong = np.array([float(r['strong_error_m']) for r in rr])
        weak = np.array([float(r['weak_error_m']) for r in rr])
        times = np.array([float(r['seconds']) for r in rr])
        assert np.isfinite(np.concatenate((strong, weak, times))).all()
        both = (strong <= 20.) & (weak <= 20.)
        assert int(both.sum()) == int(s['both_count'])
        for key, value in (
            ('strong_rmse_m', np.sqrt(np.mean(strong**2))),
            ('weak_rmse_m', np.sqrt(np.mean(weak**2))),
            ('mean_seconds', times.mean()),
            ('median_seconds', np.median(times)),
            ('p95_seconds', np.quantile(times, .95))):
            np.testing.assert_allclose(value, float(s[key]), rtol=1e-12, atol=1e-12)
        calculated.append(s)
    paired = {m: {r['case']: r for r in rows if r['stage'] == 'mechanism' and r['method'] == m}
              for m in METHODS}
    assert len(paired[METHODS[0]]) == len(paired[METHODS[1]]) == 240
    assert paired[METHODS[0]].keys() == paired[METHODS[1]].keys()
    assert all(paired[METHODS[0]][k]['both20'] == paired[METHODS[1]][k]['both20']
               for k in paired[METHODS[0]])
    fc = np.array([float(r['seconds']) for r in paired['FC_VP'].values()])
    vp = np.array([float(paired['VP_VP'][k]['seconds']) for k in paired['FC_VP']])
    saving = 100 * (1-vp.mean()/fc.mean())
    assert round(saving, 2) == 83.77
    assert int((vp < fc).sum()) == 170
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.labelsize': 9, 'legend.fontsize': 8.5,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    exported = []
    for name, fields, labels in (
        ('error_cdf', ('strong_error_m', 'weak_error_m'),
         ('Strong-source error (m)', 'Weak-source error (m)')),
        ('cost_cdf', ('seconds', 'cycles'), ('Complete runtime (s)', 'Cyclic sweeps'))):
        fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.35), layout='constrained')
        plotrows = []
        for j, (ax, field, label) in enumerate(zip(axes, fields, labels)):
            for method in METHODS:
                values = np.sort([float(r[field]) for r in paired[method].values()])
                prob = np.arange(1, len(values)+1)/len(values)
                color, line, title = STYLE[method]
                ax.step(values, prob, where='post', color=color, ls=line, lw=1.7,
                        label=title)
                plotrows.extend(dict(panel=j+1, method=method, value=float(x), cdf=float(y))
                                for x, y in zip(values, prob))
            ax.set_xscale('log')
            ax.set_ylim(0, 1.025)
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.set_xlabel(label)
            ax.set_ylabel('Empirical CDF')
            ax.grid(alpha=.16, which='major')
            ax.text(.03, .95, f'({chr(97+j)})', transform=ax.transAxes,
                    va='top', fontweight='bold')
            ax.legend(loc='lower right', frameon=False)
            if name == 'error_cdf':
                ax.axvline(20, color='#555555', lw=.8, ls=':')
                ax.set_xlim(.003 if j == 0 else .01, 30 if j == 0 else 1000)
        fig.savefig(ROOT / 'figures' / (name+'.pdf'), bbox_inches='tight')
        fig.savefig(ROOT / 'figures' / (name+'.png'), dpi=600, bbox_inches='tight')
        plt.close(fig)
        write_csv(ROOT / 'figures' / (name+'.csv'), plotrows)
        exported.append(name)
    report = dict(passed=True, rows=len(rows), mechanism_pairs=240,
                  new_geometry_observations=480, same_success_identity=True,
                  mean_saving_percent=float(saving), vp_faster=170,
                  figures=exported, evaluated_from='saved positions and complete runtime',
                  no_locator_run=True)
    (ROOT / 'build/data_check.json').write_text(
        json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
