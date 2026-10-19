"""Check the current inline manuscript and approved evidence additions."""
from pathlib import Path
import hashlib
import json
import re
import csv
import numpy as np
from validate_draft import math_checks

PAPER = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    path = PAPER/'paper_draft.tex'
    text = path.read_text(encoding='utf-8')
    old = (PAPER/'archive/evidence_revision_20261018/paper_draft.tex').read_text(encoding='utf-8')
    labels = re.findall(r'\\label\{([^}]+)\}', text)
    references = re.findall(r'\\(?:eqref|ref)\{([^}]+)\}', text)
    cites = set(','.join(re.findall(r'\\cite\{([^}]+)\}', text)).split(','))-{''}
    bib = (PAPER/'references/references.bib').read_text(encoding='utf-8')
    keys = set(re.findall(r'@\w+\{([^,]+),', bib))
    assert len(labels) == len(set(labels)) and not set(references)-set(labels)
    assert not cites-keys
    assert not re.search(r'\\(?:input|include)\{', text)
    equation = r'\\begin\{(equation|align)\}.*?\\end\{\1\}'
    # Existing scientific equations must be byte-identical after text-only revision.
    blocks = [m.group() for m in re.finditer(equation, old, re.S)]
    assert all(block in text for block in blocks)
    tables = re.findall(r'% BEGIN (tables/[^\r\n]+)\s*(.*?)% END \1', old, re.S)
    assert all(body in text for _, body in tables)
    assert 'GA与SBL分别有208条和14条' in text
    evidence = PAPER/'data/mechanism_evidence'
    report = json.loads((evidence/'comparison_report.json').read_text(encoding='utf-8'))
    assert report['summaries']['FIXED_STRICT']['steps'] == 21436
    assert not any(report['summaries']['FIXED_STRICT']['safeguards'][key]
                   for key in ('floor_active', 'cap_active', 'backtracked', 'boundary_clipped'))
    with (evidence/'mechanism_summary.csv').open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            assert f"{float(row['seconds']):.3f}" in text
            assert f"{float(row['derivative_calls']):.1f}" in text
    segment = json.loads((PAPER/'data/segment_model_report.json').read_text(encoding='utf-8'))
    assert segment['observations'] == 8 and segment['checks']['passed']
    sources = json.loads((PAPER/'data/segment_model_sources.json').read_text(encoding='utf-8'))
    assert sources['propagation_speed_m_s'] == 299792458.
    assert np.asarray(sources['receivers']).shape == (4, 2)
    for count in (1, 4, 8):
        rows = [r for r in segment['rows'] if r['segments'] == count]
        fractions = [r['clean_model_residual_fraction']*100 for r in rows]
        noise = [r['clean_model_residual_to_expected_projected_noise'] for r in rows]
        assert f'{min(fractions):.3f}--{max(fractions):.3f}' in text
        assert f'{min(noise):.3f}--{max(noise):.3f}' in text
    audit = json.loads((evidence/'export_audit.json').read_text(encoding='utf-8'))
    for p, identity in (audit['identities'] | sources['inputs']).items():
        assert sha(Path(p)) == identity, p
    log = (PAPER/'build/paper_draft.log').read_text(encoding='utf-8', errors='replace')
    errors = re.findall(r'^!.*|^.*(?:undefined references|Citation .* undefined|Reference .* undefined|Missing character:|Overfull \\hbox).*', log, re.M)
    assert not errors, errors
    math = math_checks()
    checks = dict(passed=True, manuscript_sha=sha(path), pdf_sha=sha(PAPER/'paper_draft.pdf'),
                  original_equations_preserved=len(blocks), original_result_tables_preserved=len(tables),
                  labels=len(labels), citations=len(cites), math=math, latex_errors=errors,
                  source_identities_unchanged=True, new_localization_solves=0,
                  diagnostic_status='Finite-window mismatch measured; no causal solver failure attribution.')
    (PAPER/'qa/evidence_revision/validation.json').write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in checks.items() if k != 'math'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
