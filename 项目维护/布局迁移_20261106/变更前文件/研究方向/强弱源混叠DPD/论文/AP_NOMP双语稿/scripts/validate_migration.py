"""Check that CAS migration preserves the complete paper and local evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
OLD = PAPER / 'archive/pre_cas_20261018'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content(text):
    # Page breaks are presentation-only; CAS can flow appendices continuously.
    text = text.replace(r'\clearpage', '')
    text = text.replace(r'\input{tables/generated_', r'\input{sections/generated_')
    text = re.sub(r'(\\begin\{(?:table|figure)\}\[)pos=', r'\1', text)
    text = text.replace(r'\begin{abstract}[摘要]', r'\begin{abstract}')
    if r'\begin{keywords}' in text:
        text = re.sub(r'\\begin\{keywords\}\[关键词\]\s*(.*?)\s*\\end\{keywords\}',
                      lambda m: r'\noindent\textbf{关键词：}'+m[1].replace(r' \sep ', '；'), text, flags=re.S)
    return re.sub(r'\s+', '', text)


def main():
    manifest = json.loads((PAPER / 'qa/migration_manifest.json').read_text(encoding='utf-8'))
    for rel, expected in manifest['archive_files'].items():
        if sha(OLD / rel) != expected:
            raise AssertionError('Archived manuscript changed: '+rel)
    for src in (OLD / 'sections').glob('*.tex'):
        dst = PAPER / ('tables' if src.name.startswith('generated_') else 'sections') / src.name
        if content(src.read_text(encoding='utf-8')) != content(dst.read_text(encoding='utf-8')):
            raise AssertionError('Paper content changed beyond format adaptation: '+src.name)
    for src in (OLD / 'figures').glob('formal_*.csv'):
        if sha(src) != sha(PAPER / 'figures' / src.name):
            raise AssertionError('Figure data changed: '+src.name)
    for src in (OLD / 'sources').glob('*.csv'):
        if sha(src) != sha(PAPER / 'data' / src.name):
            raise AssertionError('Statistical CSV changed: '+src.name)
    if sha(OLD / 'references.bib') != sha(PAPER / 'references/references.bib'):
        raise AssertionError('Bibliography changed')
    for name in ('cas-sc.cls', 'cas-common.sty', 'cas-model2-names.bst'):
        if sha(PAPER / 'template' / name) != sha(ROOT / 'els-cas-templates' / name):
            raise AssertionError('Publisher template modified: '+name)
    main_text = (PAPER / 'cas-sc-sample.tex').read_text(encoding='utf-8')
    for name in ('abstract','introduction','model','methods','experiments','conclusion','appendix'):
        if r'\input{\PaperRoot sections/'+name+'}' not in main_text:
            raise AssertionError('Missing section: '+name)
    if r'\documentclass[a4paper,fleqn]{cas-sc}' not in main_text:
        raise AssertionError('CAS class not active')
    forbidden = ('J.K. Krishnan', 'This is a specimen', 'Author biography', 'quadrupole exciton')
    if any(s in main_text for s in forbidden):
        raise AssertionError('Template example content remains')
    entry = (ROOT / 'els-cas-templates/cas-sc-sample.tex').read_text(encoding='utf-8')
    if r'\input{../paper/cas-sc-sample.tex}' not in entry:
        raise AssertionError('Original requested entry does not load maintained paper')
    result = dict(passed=True, all_sections_preserved=True, all_table_values_preserved=True,
                  figure_csv_preserved=True, bibliography_preserved=True, publisher_files_unchanged=True,
                  original_entry_redirects_to_active_paper=True, archived_files=len(manifest['archive_files']))
    (PAPER / 'qa/migration_checks.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
