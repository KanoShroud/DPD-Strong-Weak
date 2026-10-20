"""Version-aware checks for the approved evaluation/cost manuscript revision."""
import hashlib
import argparse
import csv
import json
import re
from pathlib import Path
from relocated_evidence import local_path

import fitz

HERE = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(local_path(p).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='qa/assignment_cost_20261019')
    parser.add_argument('--submission', action='store_true', help='Also audit the English review manuscript and supplement')
    args = parser.parse_args()
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


def validate_vector_assets():
    """Reject raster PDFs and unembedded fonts in all active figure assets."""
    assets = set()
    for name in ('paper_draft', 'paper_submission', 'paper_supplement', 'paper_supplement_zh'):
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


def validate_submission(bib):
    names = ('paper_submission', 'paper_supplement')
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
        issues = re.findall(r'^!.*|^.*(?:undefined references|Citation .* undefined|Reference .* undefined|Missing character:|Overfull \\hbox).*', log, re.M)
        assert not issues, (name, issues)
        pdf = fitz.open(HERE/(name+'.pdf'))
        assert all('??' not in p.get_text() and 'GOSPA' not in p.get_text() for p in pdf)
        assert all(abs(p.rect.width-595.276) < 1 and abs(p.rect.height-841.89) < 1 for p in pdf)
        assert r'\documentclass[a4paper,12pt,review]{cas-sc}' in src
        results[name] = dict(pages=len(pdf), a4=True, review_double_spacing=True,
                             manuscript_sha=sha(HERE/(name+'.tex')), pdf_sha=sha(HERE/(name+'.pdf')))
    assert results['paper_submission']['pages'] <= 30, 'English review manuscript exceeds 30 pages'
    en = texts['paper_submission']
    def rows(filename):
        with (HERE/'data/assignment_cost_20261019'/filename).open(encoding='utf-8-sig', newline='') as f:
            return {r['method']:r for r in csv.DictReader(f) if r['group']=='all'}
    maps = {
        'common_summary.csv': [('LS','LS peaks'), ('MUSIC','WB-MUSIC'), ('MVDR','MVDR-known'),
          ('AP_FULL','Full AP'), ('BNOMP_REF','Fixed BNOMP'), ('PROFILE_DIAG','VP-BNOMP'),
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
        assert texrow in texts['paper_supplement']
    for claim in ('31.36','83.77','79.92','170','390','2376'):
        assert claim in en, claim
    results['metrics_and_coordinates_match_saved_results'] = True
    return results


if __name__=='__main__':main()
