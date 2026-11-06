"""Version-aware checks for the approved evaluation/cost manuscript revision."""
import hashlib
import argparse
import csv
import json
import re
from pathlib import Path
from relocated_evidence import local_path

import fitz
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(local_path(p).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='qa/assignment_cost_20261019')
    parser.add_argument('--submission', action='store_true', help='Also audit the English review manuscript and supplement')
    parser.add_argument('--integrated', action='store_true',
                        help='Audit the complete bilingual manuscripts without supplements')
    args = parser.parse_args()
    if args.integrated or not (HERE/'paper_supplement_zh.tex').exists():
        validate_integrated(args.out)
        return
    text=(HERE/'paper_draft.tex').read_text(encoding='utf-8')
    extra = HERE/'paper_supplement_zh.tex'
    combined = text + ('\n'+extra.read_text(encoding='utf-8') if extra.exists() else '')
    old=(HERE/'archive/assignment_cost_20261019/paper_draft.tex').read_text(encoding='utf-8')
    labels=re.findall(r'\\label\{([^}]+)\}',combined)
    refs=re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',combined)
    assert len(labels)==len(set(labels)) and not set(refs)-set(labels)
    bib=(HERE/'references/references.bib').read_text(encoding='utf-8')
    cites=set(','.join(re.findall(r'\\cite\{([^}]+)\}',combined)).split(','))-{''}
    assert not cites-set(re.findall(r'@\w+\{([^,]+),',bib))
    assert not re.search(r'GOSPA|eq:gospa|Rahmathullah2017|\\(?:input|include)\{',combined)
    eq=r'\\begin\{(equation|align)\}.*?\\end\{\1\}'
    protected=[m.group() for m in re.finditer(eq,old,re.S) if 'eq:gospa' not in m.group()]
    assert all(b in combined for b in protected), 'Theory equation missing from main manuscript and supplement'
    report=json.loads((HERE/'data/assignment_cost_20261019/report.json').read_text(encoding='utf-8'))
    for path,h in report['source_files'].items():assert sha(path)==h,path
    for name in ('common','core'):
        rows=(HERE/f'data/assignment_cost_20261019/{name}_table_rows.tex').read_text(encoding='utf-8')
        assert rows.strip() in text, 'Table differs from audited export: '+name
    figures=re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',combined)
    for rel in figures:
        p=(HERE/rel).resolve(strict=True)
        assert p.is_relative_to(HERE.resolve()), 'Nonstandalone figure reference'
    vector_assets = validate_vector_assets()
    log=(HERE/'build/paper_draft.log').read_text(encoding='utf-8',errors='replace')
    issues=re.findall(r'^!.*|^.*(?:undefined references|Citation .* undefined|Reference .* undefined|Missing character:|Overfull \\hbox).*',log,re.M)
    assert not issues,issues
    doc=fitz.open(HERE/'paper_draft.pdf')
    assert all('GOSPA' not in p.get_text() for p in doc)
    assert all('??' not in p.get_text() for p in doc)
    if extra.exists():
        extra_log=(HERE/'build/paper_supplement_zh.log').read_text(encoding='utf-8',errors='replace')
        extra_issues=re.findall(r'^!.*|^.*(?:undefined references|Citation .* undefined|Reference .* undefined|Missing character:|Overfull \\hbox).*',extra_log,re.M)
        assert not extra_issues,extra_issues
        with fitz.open(HERE/'paper_supplement_zh.pdf') as extra_pdf:
            assert all('??' not in page.get_text() for page in extra_pdf)
    supplement_checked = False
    if r'\label{tab:curvaturepair}' in text:
        supplement = json.loads((HERE/'data/review_supplement_20261020/audit.json').read_text(encoding='utf-8'))
        assert supplement['solves'] == 1920 and supplement['historical_path_equivalent_checks'] == 480
        assert supplement['exporter_sha'] == sha(HERE/'scripts/prepare_assignment_cost.py')
        for path, h in supplement['source_files'].items():
            assert sha(path) == h, path
        named = {(r['stage'], r['method']): r for r in supplement['summary']}
        for m, label in [('AP_1_JOINT','单候选AP'), ('AP_5_JOINT','五候选MCJ-AP'),
                         ('PROFILE_DIAG','VP-BNOMP')]:
            s = named['geometry', m]
            expected = (f"{label} & {s['both_count']}/480 & {s['strong_rmse_m']:.3f} & "
                        f"{s['weak_rmse_m']:.3f} & {s['mean_seconds']:.3f} & "
                        f"{s['median_seconds']:.3f} & {s['p95_seconds']:.3f}")
            assert expected in text, expected
        for m, label in [('FC_VP','FC--VP'), ('VP_VP','VP--VP')]:
            s = named['mechanism', m]
            expected = (f"{label} & {s['both_count']}/240 & {s['strong_rmse_m']:.3f} & "
                        f"{s['weak_rmse_m']:.3f} & {s['mean_seconds']:.3f} & "
                        f"{s['mean_updates']:.2f} & {s['mean_cycles']:.2f}")
            assert expected in text, expected
        saving = supplement['paired']['mechanism']
        assert f"{saving['mean_time_saving_percent']:.2f}\\%" in text
        assert f"{saving['saving_ci95'][0]:.2f}\\%--{saving['saving_ci95'][1]:.2f}\\%" in text
        supplement_checked = True
    english_result = validate_submission(bib) if args.submission else None
    result=dict(passed=True, manuscript_sha=sha(HERE/'paper_draft.tex'),
                pdf_sha=sha(HERE/'paper_draft.pdf'), protocol=report['protocol'],
                theory_equations_preserved=len(protected), labels=len(labels), citations=len(cites),
                figures=len(figures), pages=len(doc), source_identity=True,
                tables_match_saved_evidence=True, gospa_removed_from_published_pdf=True,
                old_evidence_preserved=True, supplement_checked=supplement_checked,
                english_submission=english_result,
                vector_assets=vector_assets,
                visual_review='See separate per-page render review')
    out=HERE/args.out;out.mkdir(parents=True,exist_ok=True)
    (out/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


def validate_vector_assets(names=None):
    """Reject raster PDFs and unembedded fonts in all active figure assets."""
    assets = set()
    for name in names or ('paper_draft', 'paper_submission', 'paper_supplement', 'paper_supplement_zh'):
        path = HERE/(name+'.tex')
        if path.exists():
            assets.update(re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',
                                     path.read_text(encoding='utf-8')))
    checked = []
    for relative in sorted(assets):
        path = (HERE/relative).resolve(strict=True)
        assert path.is_relative_to(HERE.resolve()) and path.suffix == '.pdf', relative
        with fitz.open(path) as pdf:
            assert len(pdf) == 1, relative
            page = pdf[0]
            assert page.get_drawings() and not page.get_images(full=True), relative
            fonts = page.get_fonts(full=True)
            assert fonts and all(pdf.extract_font(f[0])[3] for f in fonts), relative
            assert all(f[2] != 'Type3' for f in fonts), relative
            checked.append(dict(file=relative, sha256=sha(path),
                                raster_images=0, embedded_fonts=len(fonts)))
    return checked


def validate_submission(bib, integrated=False):
    names = ('paper_submission',) if integrated else ('paper_submission', 'paper_supplement')
    texts = {n: (HERE/(n+'.tex')).read_text(encoding='utf-8') for n in names}
    results = {}
    for name, src in texts.items():
        labs = re.findall(r'\\label\{([^}]+)\}', src)
        refs = re.findall(r'\\(?:eqref|ref)\{([^}]+)\}', src)
        assert len(labs) == len(set(labs)) and not set(refs)-set(labs), name
        cites = set(','.join(re.findall(r'\\cite\{([^}]+)\}', src)).split(','))-{''}
        assert not cites-set(re.findall(r'@\w+\{([^,]+),', bib))
        assert not re.search(r'GOSPA|Rahmathullah2017|\\(?:input|include)\{', src)
        for rel in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', src):
            assert (HERE/rel).resolve(strict=True).is_relative_to(HERE.resolve())
        log = (HERE/'build'/(name+'.log')).read_text(encoding='utf-8', errors='replace')
        issues = manuscript_log_issues(log, integrated)
        assert not issues, (name, issues)
        pdf = fitz.open(HERE/(name+'.pdf'))
        assert all('??' not in p.get_text() and 'GOSPA' not in p.get_text() for p in pdf)
        width, height = (192*72/25.4, 262*72/25.4) if integrated else (595.276, 841.89)
        assert all(abs(p.rect.width-width) < 1 and abs(p.rect.height-height) < 1 for p in pdf)
        expected_class = (r'\documentclass[a4paper,fleqn,review]{cas-sc}' if integrated
                          else r'\documentclass[a4paper,12pt,review]{cas-sc}')
        assert expected_class in src
        results[name] = dict(pages=len(pdf), template_native=integrated,
                             page_mm=[192,262] if integrated else [210,297],
                             review_double_spacing=True,
                             manuscript_sha=sha(HERE/(name+'.tex')), pdf_sha=sha(HERE/(name+'.pdf')))
    assert results['paper_submission']['pages'] <= 30, 'English review manuscript exceeds 30 pages'
    en = texts['paper_submission']
    def rows(filename):
        with (HERE/'data/assignment_cost_20261019'/filename).open(encoding='utf-8-sig', newline='') as f:
            return {r['method']:r for r in csv.DictReader(f) if r['group']=='all'}
    maps = {
        'common_summary.csv': [('LS','LS peaks'), ('MUSIC','WB-MUSIC'), ('MVDR','MVDR-known'),
          ('AP_FULL','Full AP'), ('BNOMP_REF','BNOMP (FC--FC)' if integrated else 'Fixed BNOMP'), ('PROFILE_DIAG','VP-BNOMP'),
          ('MULTISTART_FIXED','Multistart ML'), ('GA','GA-ML'), ('SBL','WB-SBL'), ('AP_5_JOINT','MCJ-AP ($M=5$)')],
        'core_summary.csv': [('AP_1_JOINT','AP ($M=1$)'),('AP_2_JOINT','AP ($M=2$)'),
          ('AP_5_JOINT','MCJ-AP ($M=5$)'),('PROFILE_DIAG','VP-BNOMP')]}
    for filename, pairs in maps.items():
        rr = rows(filename)
        for method, label in pairs:
            r = rr[method]
            places = 4
            expected = (f"{label} & {r['both_count']} & {float(r['strong_rmse_m']):.3f} & "
                        f"{float(r['weak_rmse_m']):.3f} & {float(r['mean_seconds']):.{places}f} & "
                        f"{float(r['median_seconds']):.{places}f} & {float(r['p95_seconds']):.{places}f}")
            assert expected in en, expected
    audited = json.loads((HERE/'data/review_supplement_20261020/audit.json').read_text(encoding='utf-8'))
    for s in audited['summary']:
        if s['stage']=='geometry':
            label={'AP_1_JOINT':'AP ($M=1$)','AP_5_JOINT':'MCJ-AP ($M=5$)','PROFILE_DIAG':'VP-BNOMP'}[s['method']]
            expected=(f"{label} & {s['both_count']} & {s['strong_rmse_m']:.3f} & {s['weak_rmse_m']:.3f} & "
                      f"{s['mean_seconds']:.3f} & {s['median_seconds']:.3f} & {s['p95_seconds']:.3f}")
        else:
            label={'FC_VP':'FC--VP','VP_VP':'VP--VP'}[s['method']]
            expected=(f"{label} & {s['both_count']}/240 & {s['strong_rmse_m']:.3f} & {s['weak_rmse_m']:.3f} & "
                      f"{s['mean_seconds']:.3f} & {s['mean_updates']:.2f} & {s['mean_cycles']:.2f}")
        assert expected in en, expected
    with (HERE/'data/submission/new_layouts.csv').open(encoding='utf-8-sig', newline='') as f:
        coords=list(csv.DictReader(f))
    assert len(coords)==16
    for i,(saved,row) in enumerate(zip(audited['layouts'],coords),1):
        assert int(row['layout_one_based'])==i and int(row['layout_zero_based'])==i-1
        actual=[float(row[k]) for k in ('source_a_x_m','source_a_y_m','source_b_x_m','source_b_y_m')]
        expected=saved['positions'][0]+saved['positions'][1]
        assert actual==expected
        texrow=f"{i} & {saved['category']+1} & " + ' & '.join(f'{v:.6f}' for v in expected)
        assert texrow in texts['paper_submission' if integrated else 'paper_supplement']
    for claim in ('31.36','83.77','79.92','170','390','2376'):
        assert claim in en, claim
    results['metrics_and_coordinates_match_saved_results'] = True
    return results


def manuscript_log_issues(log, native_template=False):
    """Native CAS 2.4 overlays keywords in a zero-width box; audit it separately."""
    issues = re.findall(
        r'^!.*|^.*(?:undefined references|Citation .* undefined|Reference .* undefined|'
        r'Missing character:|Overfull \\[hv]box).*', log, re.M)
    if native_template:
        # Exact, known template-only diagnostic, not a generic overfull waiver.
        issues = [s for s in issues if not re.fullmatch(
            r'Overfull \\hbox \(117\.0831pt too wide\) detected at line \d+', s)]
    return issues


def validate_candidate_ranks(texts):
    """Replay saved branch positions and verify the pre-run AP5 configuration chain."""
    base = HERE.parent/'outputs_e2e/unified/e2e_g10/r4'
    evidence = base/'review_boundary_20261020_122047_490569/branches.json'
    saved = json.loads(evidence.read_text(encoding='utf-8'))
    assert saved['protocol'] == 'EVAL-20261019-v3-SINGLE-SQUARED-ASSIGNMENT'
    assert saved['n'] == len(saved['rows']) == 2400
    for path, expected in saved['source_sha256'].items():
        assert sha(path) == expected, path
    run = base/'formal_recovered_20261017_093236_136920'
    audited = {r['case']: r for r in saved['rows']}
    counts = {str(i): 0 for i in range(1, 6)}
    counts['none'] = 0
    recovered, selected_success, selection_loss = [], 0, 0

    def success(truth, positions):
        truth, positions = np.asarray(truth, float), np.asarray(positions, float)
        assert truth.shape == positions.shape == (2, 2)
        assert np.isfinite(positions).all()
        cost = np.sum((truth[:, None, :]-positions[None, :, :])**2, axis=2)
        a, b = linear_sum_assignment(cost)
        return bool(np.all(np.sqrt(cost[a, b]) <= 20.0))

    for case, expected in audited.items():
        path = run/'rows'/(case+'_AP_5_JOINT.json')
        row = json.loads(path.read_text(encoding='utf-8'))
        mechanism = row['result']['mechanism']
        branches = mechanism['branches']
        assert [b['rank'] for b in branches] == [1, 2, 3, 4, 5]
        ranks = [b['rank'] for b in branches if success(row['truth'], b['positions'])]
        assert ranks == expected['successful_ranks'], case
        chosen = success(row['truth'], row['result']['positions'])
        assert chosen == expected['selected_success'], case
        assert mechanism['selected_rank'] == expected['selected_rank'], case
        counts[str(min(ranks)) if ranks else 'none'] += 1
        selected_success += int(chosen)
        selection_loss += int(bool(ranks) and not chosen)
        if ranks and 1 not in ranks:
            recovered.append(tuple(map(int, case.split('_'))))
    assert counts == dict(zip(('1','2','3','4','5','none'), (2280,0,1,8,87,24)))
    assert selected_success == 2376 and selection_loss == 0 and len(recovered) == 96
    assert {r[0] for r in recovered} == {3}
    assert len({r[1] for r in recovered}) == 18
    for src in texts.values():
        assert '& 2280 & 0 & 1 & 8 & 87 & 24' in src

    names = ('formal_ready_20261017_023133_631280',
             'formal_20261017_023343_260288',
             'formal_recovered_20261017_093236_136920',
             'review_supplement_formal_20261020_122624_540128')
    paths = {n: base/n/'contract.json' for n in names}
    contracts = {n: json.loads(p.read_text(encoding='utf-8')) for n,p in paths.items()}
    ready, original, resumed, independent = [contracts[n] for n in names]
    assert ready['stage'] == 'formal_release' and not ready['formal_observations_generated']
    assert original['preparation_sha'] == sha(paths[names[0]])
    assert resumed['io_recovery']['previous_contract_sha'] == sha(paths[names[1]])
    configs = [c['configurations']['AP_5_JOINT'] for c in (ready, original, resumed)]
    assert configs[0] == configs[1] == configs[2] and configs[0]['top'] == 5
    assert independent['configs']['AP_5_JOINT'] == configs[0]
    return dict(observations=2400, first_success_rank=counts,
                selected_success=selected_success, selection_loss=selection_loss,
                recovered=96, recovered_configuration_one_based=4, recovered_realizations=18,
                evidence_sha256=sha(evidence), evidence_sources_checked=len(saved['source_sha256']),
                formal_ap5_configuration_unchanged=True,
                new_source_configuration_ap5_unchanged=True,
                contract_sha256={n:sha(p) for n,p in paths.items()})


def terminal_quality(texts, audit):
    """Compare saved joint terminal certificates; never rerun a solver."""
    path = HERE/'data/review_supplement_20261020/per_case.csv'
    with path.open(encoding='utf-8-sig', newline='') as stream:
        saved = {(r['case'], r['method']): r for r in csv.DictReader(stream)
                 if r['stage'] == 'mechanism'}
    rows = []
    for raw, expected_sha in sorted(audit['source_files'].items()):
        p = local_path(raw)
        if p.parent.name != 'pairs' or not p.name.startswith('mechanism_'):
            continue
        assert sha(p) == expected_sha
        pair = json.loads(p.read_text(encoding='utf-8'))
        assert pair['stage'] == 'mechanism' and pair['iq_sha']
        case = '_'.join(map(str, pair['task']))
        methods = {r['method']: r for r in pair['rows']}
        assert set(methods) == {'FC_VP', 'VP_VP'}
        values = {}
        for method, row in methods.items():
            result, csvrow = row['result'], saved[case, method]
            assert result['stop_reason'] == 'NATIVE_ENERGY_STABLE'
            assert result['native_objective'] == float(csvrow['objective'])
            assert result['projected_gradient'] == float(csvrow['gradient'])
            assert row['metrics']['both20'] == (csvrow['both20'] == 'True')
            assert result['config'] == methods['FC_VP']['result']['config']
            values[method] = result
        fc, vp = values['FC_VP'], values['VP_VP']
        rows.append(dict(case=case, configuration_one_based=pair['task'][0]+1,
            realization=pair['task'][1], power_gap_db=pair['task'][2], iq_sha=pair['iq_sha'],
            fc_joint_objective=fc['native_objective'], vp_joint_objective=vp['native_objective'],
            fc_minus_vp_objective=fc['native_objective']-vp['native_objective'],
            fc_projected_gradient=fc['projected_gradient'], vp_projected_gradient=vp['projected_gradient'],
            fc_success=methods['FC_VP']['metrics']['both20'],
            vp_success=methods['VP_VP']['metrics']['both20'], source=str(p), source_sha256=expected_sha))
    assert len(rows) == 240 and len(saved) == 480 and len({r['case'] for r in rows}) == 240
    delta = np.asarray([r['fc_minus_vp_objective'] for r in rows])
    gradients = np.asarray([[r['fc_projected_gradient'], r['vp_projected_gradient']] for r in rows])
    assert np.isfinite(delta).all() and np.isfinite(gradients).all()
    comparisons = {}
    for tol in (1e-10, 1e-8):
        comparisons[str(tol)] = dict(vp_lower=int(np.sum(delta > tol)),
            approximately_equal=int(np.sum(np.abs(delta) <= tol)), vp_higher=int(np.sum(delta < -tol)))
    biggest = rows[int(np.argmax(np.abs(delta)))]
    result = dict(n=240, objective='Full joint normalized projection residual J',
        gradient='Infinity norm of box-projected joint gradient, original meter coordinates',
        tolerance_role='Absolute descriptive comparison tolerance, not a solver stop or significance test',
        objective_comparisons=comparisons,
        absolute_objective_difference_median_p95_max=np.quantile(np.abs(delta), [.5,.95,1]).tolist(),
        gradient_medians=np.median(gradients, axis=0).tolist(),
        vp_gradient_smaller=int(np.sum(gradients[:,1] < gradients[:,0])),
        largest_difference=biggest, original_csv_sha256=sha(path),
        original_audit_sha256=sha(HERE/'data/review_supplement_20261020/audit.json'))
    assert comparisons['1e-10'] == dict(vp_lower=199, approximately_equal=41, vp_higher=0)
    assert comparisons['1e-08'] == dict(vp_lower=167, approximately_equal=73, vp_higher=0)
    assert result['vp_gradient_smaller'] == 240
    assert biggest['case'] == '3_410_0' and not biggest['fc_success'] and not biggest['vp_success']
    def sci(value):
        mantissa, exponent = f'{value:.2e}'.split('e')
        return mantissa+r'\times10^{'+str(int(exponent))+'}'
    numbers = result['absolute_objective_difference_median_p95_max']+result['gradient_medians']
    for text in texts.values():
        assert all(sci(v) in text for v in numbers)
        assert '199' in text and '41' in text and '167' in text and '73' in text
    return result, rows


def validate_integrated(out_relative):
    """Validate content parity and saved evidence, without running localization."""
    for name in ('paper_supplement', 'paper_supplement_zh'):
        for suffix in ('.tex', '.pdf'):
            assert not (HERE/(name+suffix)).exists(), 'Archive obsolete supplement: '+name+suffix
    texts = {name: (HERE/(name+'.tex')).read_text(encoding='utf-8')
             for name in ('paper_submission', 'paper_draft')}
    bib = (HERE/'references/references.bib').read_text(encoding='utf-8')
    bib_keys = set(re.findall(r'@\w+\{([^,]+),', bib))
    en, zh = texts['paper_submission'], texts['paper_draft']
    for name, src in texts.items():
        labels = re.findall(r'\\label\{([^}]+)\}', src)
        refs = re.findall(r'\\(?:eqref|ref)\{([^}]+)\}', src)
        assert len(labels) == len(set(labels)) and not set(refs)-set(labels), name
        cites = set(','.join(re.findall(r'\\cite\{([^}]+)\}', src)).split(','))-{''}
        assert not cites-bib_keys, name
        assert not re.search(
            r'Supplement|补充材料|paper_supplement|GOSPA|\\externaldocument|'
            r'\\(?:input|include)\{|\\geometry|@fleqnfalse|__make_.*caption|'
            r'__first_footerline|\\(?:setstretch|linespread)', src), name
        log = (HERE/'build'/(name+'.log')).read_text(encoding='utf-8', errors='replace')
        assert not manuscript_log_issues(log, True), (name, manuscript_log_issues(log, True))
        with fitz.open(HERE/(name+'.pdf')) as pdf:
            assert all('??' not in p.get_text() and 'GOSPA' not in p.get_text() for p in pdf)
    assert r'\documentclass[a4paper,fleqn]{cas-sc}' in zh
    assert r'\documentclass[a4paper,fleqn,review]{cas-sc}' in en
    for src in texts.values():
        authors = [('Benjian Hao', '0000-0002-6676-7306', 'bjhao@xidian.edu.cn'),
                   ('Xinlei Wang', '0000-0002-6855-3686', 'wangxinlei@stu.xidian.edu.cn'),
                   ('Zan Li', '0000-0002-5207-6504', 'zanli@xidian.edu.cn')]
        assert re.findall(r'\\author\[inst1\]\{([^}]+)\}', src) == [a[0] for a in authors]
        for author, orcid, email in authors:
            assert rf'\author[inst1]{{{author}}}[orcid={orcid}]' in src
            assert rf'\ead{{{email}}}' in src
        assert r'\cormark[1]' in src and r'\cortext[cor1]{Corresponding author.}' in src
        assert 'State Key Laboratory of Integrated Service Networks, School of Telecommunications Engineering, Xidian University' in src
        assert "city={Xi'an}, postcode={710071}, country={China}" in src
        assert not re.search(r'\\ead\{\[|\\credit\{', src)
        assert 'same current conditional coefficient estimate, gradient, discovery sequence' not in src
    patterns = {
        'display_math': r'\\begin\{(?:equation|align)\}.*?\\end\{(?:equation|align)\}',
        'labels': r'\\label\{[^}]+\}',
        'citations': r'\\cite\{[^}]+\}',
        'figures': r'\\includegraphics(?:\[[^\]]*\])?\{[^}]+\}',
    }
    parity = {}
    for key, pattern in patterns.items():
        left, right = re.findall(pattern, en, re.S), re.findall(pattern, zh, re.S)
        assert left == right, 'Bilingual mismatch: '+key
        parity[key] = len(left)
    def numeric_rows(src):
        found = []
        for line in src.splitlines():
            if '&' in line and re.search(r' & \d', line):
                cells = line.split('&')[1:]
                if len(cells) > 1 and all(re.match(r'\s*[-\d]', c) for c in cells):
                    found.append(cells)
        return found
    assert numeric_rows(en) == numeric_rows(zh), 'Bilingual numeric table mismatch'
    report = json.loads((HERE/'data/assignment_cost_20261019/report.json').read_text(encoding='utf-8'))
    audit = json.loads((HERE/'data/review_supplement_20261020/audit.json').read_text(encoding='utf-8'))
    assert audit['solves'] == 1920 and audit['historical_path_equivalent_checks'] == 480
    for record in (report, audit):
        for path, expected in record['source_files'].items():
            assert sha(path) == expected, path
    assert audit['exporter_sha'] == sha(HERE/'scripts/prepare_assignment_cost.py')
    candidate_ranks = validate_candidate_ranks(texts)
    terminal, terminal_rows = terminal_quality(texts, audit)
    english = validate_submission(bib, integrated=True)
    vectors = validate_vector_assets(tuple(texts))
    template_files = {n: sha(HERE/'template'/n) for n in ('cas-sc.cls','cas-common.sty')}
    with fitz.open(HERE/'paper_draft.pdf') as pdf:
        zh_pages = len(pdf)
    result = dict(
        passed=True, protocol=report['protocol'], english=english, chinese_pages=zh_pages,
        bilingual_parity=parity, numeric_table_rows=len(numeric_rows(en)),
        candidate_rank_evidence=candidate_ranks,
        terminal_quality_evidence=terminal,
        source_identity=True, no_external_supplement=True, template_sha256=template_files,
        vector_assets=vectors, manuscript_sha256={n:sha(HERE/(n+'.tex')) for n in texts},
        pdf_sha256={n:sha(HERE/(n+'.pdf')) for n in texts},
        native_template_notes=[
            'CAS 2.4 native keyword-overlay overfull diagnostic is checked by exact signature and visual review.',
            'Three authors, corresponding author, affiliation, ORCIDs and emails supplied by user; funding and CRediT remain unfilled.'],
        visual_review='Separate final per-page review required.')
    out = HERE/out_relative
    out.mkdir(parents=True, exist_ok=True)
    with (out/'terminal_quality.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(terminal_rows[0]))
        writer.writeheader()
        writer.writerows(terminal_rows)
    (out/'validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__=='__main__':main()
