"""Compare revision assets against the preserved draft; no solver or IQ access."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
BEFORE = HERE / 'qa/revision1_before'
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    unchanged = []
    targets = [*BEFORE.glob('figures/*.csv'), *BEFORE.glob('sections/generated_*.tex'),
               BEFORE / 'sources/method_parameters.csv', BEFORE / 'sources/paired_summary.csv',
               BEFORE / 'sources/citation_metadata.json']
    for old in targets:
        rel = old.relative_to(BEFORE)
        if sha(old) != sha(HERE / rel):
            raise AssertionError(f'Numerical table, plot data or existing metadata changed: {rel}')
        unchanged.append(str(rel))
    old = json.loads((BEFORE / 'sources/source_manifest.json').read_text(encoding='utf-8'))
    new = json.loads((HERE / 'sources/source_manifest.json').read_text(encoding='utf-8'))
    if old['files'] != new['files']:
        raise AssertionError('Evidence file set/identity changed')
    for rel, expected in new['files'].items():
        path = (ROOT / rel).resolve(strict=True)
        if not path.is_relative_to(ROOT / 'outputs_e2e/unified/e2e_g10') or sha(path) != expected:
            raise AssertionError(f'Changed or unregistered input: {rel}')
    key_pattern = r'@\w+\{([^,]+),'
    oldkeys = set(re.findall(key_pattern, (BEFORE / 'references.bib').read_text(encoding='utf-8')))
    newkeys = set(re.findall(key_pattern, (HERE / 'references.bib').read_text(encoding='utf-8')))
    if newkeys - oldkeys != {'ITURP5252024'} or oldkeys - newkeys:
        raise AssertionError('Unexpected bibliography change')
    sections = '\n'.join(p.read_text(encoding='utf-8') for p in (HERE / 'sections').glob('*.tex'))
    if any(token in sections for token in (r'\begin{proposition}', r'\begin{proof}', r'\ref{prop:')):
        raise AssertionError('Obsolete theorem environment/reference remains')
    report = dict(passed=True, preserved_draft_pdf_sha256=sha(BEFORE / 'main.pdf'),
                  identical_numeric_assets=unchanged, unchanged_evidence_files=len(new['files']),
                  added_citation='ITURP5252024', removed_citations=[],
                  scope='Presentation/derivation revision; no new localization or performance data.')
    (HERE / 'qa/revision1_checks.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
