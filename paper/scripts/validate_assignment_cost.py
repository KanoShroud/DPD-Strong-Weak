"""Version-aware checks for the approved evaluation/cost manuscript revision."""
import hashlib
import argparse
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
    args = parser.parse_args()
    text=(HERE/'paper_draft.tex').read_text(encoding='utf-8')
    old=(HERE/'archive/assignment_cost_20261019/paper_draft.tex').read_text(encoding='utf-8')
    labels=re.findall(r'\\label\{([^}]+)\}',text)
    refs=re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',text)
    assert len(labels)==len(set(labels)) and not set(refs)-set(labels)
    bib=(HERE/'references/references.bib').read_text(encoding='utf-8')
    cites=set(','.join(re.findall(r'\\cite\{([^}]+)\}',text)).split(','))-{''}
    assert not cites-set(re.findall(r'@\w+\{([^,]+),',bib))
    assert not re.search(r'GOSPA|eq:gospa|Rahmathullah2017|\\(?:input|include)\{',text)
    eq=r'\\begin\{(equation|align)\}.*?\\end\{\1\}'
    protected=[m.group() for m in re.finditer(eq,old,re.S) if 'eq:gospa' not in m.group()]
    assert all(b in text for b in protected), 'Unapproved theory equation change'
    report=json.loads((HERE/'data/assignment_cost_20261019/report.json').read_text(encoding='utf-8'))
    for path,h in report['source_files'].items():assert sha(path)==h,path
    for name in ('common','core'):
        rows=(HERE/f'data/assignment_cost_20261019/{name}_table_rows.tex').read_text(encoding='utf-8')
        assert rows.strip() in text, 'Table differs from audited export: '+name
    figures=re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',text)
    for rel in figures:
        p=(HERE/rel).resolve(strict=True)
        assert p.is_relative_to(HERE.resolve()), 'Nonstandalone figure reference'
    log=(HERE/'build/paper_draft.log').read_text(encoding='utf-8',errors='replace')
    issues=re.findall(r'^!.*|^.*(?:undefined references|Citation .* undefined|Reference .* undefined|Missing character:|Overfull \\hbox).*',log,re.M)
    assert not issues,issues
    doc=fitz.open(HERE/'paper_draft.pdf')
    assert all('GOSPA' not in p.get_text() for p in doc)
    assert all('??' not in p.get_text() for p in doc)
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
    result=dict(passed=True, manuscript_sha=sha(HERE/'paper_draft.tex'),
                pdf_sha=sha(HERE/'paper_draft.pdf'), protocol=report['protocol'],
                theory_equations_preserved=len(protected), labels=len(labels), citations=len(cites),
                figures=len(figures), pages=len(doc), source_identity=True,
                tables_match_saved_evidence=True, gospa_removed_from_published_pdf=True,
                old_evidence_preserved=True, supplement_checked=supplement_checked,
                visual_review='See separate per-page render review')
    out=HERE/args.out;out.mkdir(parents=True,exist_ok=True)
    (out/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
