"""Version-aware formal manuscript checks; no solver/IQ access."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
BEFORE = HERE / 'qa/formal_revision_before'
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old = json.loads((BEFORE / 'sources/source_manifest.json').read_text(encoding='utf-8'))
    prior = json.loads((HERE / 'qa/paired_visual_revision_before/sources/source_manifest.json').read_text(encoding='utf-8'))
    new = json.loads((HERE / 'sources/source_manifest.json').read_text(encoding='utf-8'))
    # Authorized formal data replace old paper tables; historical input hashes
    # stay protected, rather than falsely requiring the new statistics to agree.
    for rel, expected in old['files'].items():
        if sha(ROOT / rel) != expected:
            raise AssertionError('Historical evidence changed: '+rel)
    for rel, expected in prior['files'].items():
        if new['files'].get(rel) != expected or sha(ROOT / rel) != expected:
            raise AssertionError('Prior formal paper evidence changed: '+rel)
    for rel, expected in new['files'].items():
        path = (ROOT / rel).resolve(strict=True)
        if not path.is_relative_to(ROOT / 'outputs_e2e') or sha(path) != expected:
            raise AssertionError(f'Changed or unregistered input: {rel}')
        if '/inputs/' in rel or path.suffix == '.npz' or 'profile_full_' in rel:
            raise AssertionError('Reserved PROFILE experiment or raw IQ was consumed')
    if sha(HERE / 'references.bib') != sha(BEFORE / 'references.bib'):
        raise AssertionError('Unapproved bibliography change')
    for name in ('model.tex', 'methods.tex', 'appendix.tex'):
        pattern = r'\\begin\{(?:equation|align)\}.*?\\end\{(?:equation|align)\}'
        previous = re.findall(pattern, (BEFORE / 'sections' / name).read_text(encoding='utf-8'), re.S)
        current = re.findall(pattern, (HERE / 'sections' / name).read_text(encoding='utf-8'), re.S)
        if current != previous:
            raise AssertionError('Unapproved mathematical change: '+name)
    assets = json.loads((HERE / 'qa/formal_asset_checks.json').read_text(encoding='utf-8'))
    if (not assets['passed'] or assets['row_metrics_replayed'] != 16320
            or assets['paired_rows_replayed'] != 4800 or not assets['paired_common_candidates']):
        raise AssertionError('Incomplete formal result replay')
    sections = '\n'.join(p.read_text(encoding='utf-8') for p in (HERE / 'sections').glob('*.tex'))
    if any(token in sections for token in (r'\begin{proposition}', r'\begin{proof}', r'\ref{prop:')):
        raise AssertionError('Obsolete theorem environment/reference remains')
    if any(x in sections for x in ('扩大比较正在运行', '未截断RMSE', '预留布局分组')):
        raise AssertionError('Obsolete/disallowed prose remains')
    exp = (HERE / 'sections/experiments.tex').read_text(encoding='utf-8')
    if len(re.findall(r'\\pending\{', exp)) != 1:
        raise AssertionError('Only final PROFILE content remains reserved')
    report = dict(passed=True, preserved_draft_pdf_sha256=sha(BEFORE / 'main.pdf'),
                  old_evidence_files=len(old['files']), formal_evidence_files=len(new['files']),
                  prior_formal_evidence_files=len(prior['files']),
                  formal_metrics=assets, bibliography_unchanged=True, mathematical_derivations_unchanged=True,
                  scope='Approved paired-result and prose/figure update; no new localization or parameter tuning.')
    (HERE / 'qa/formal_revision_checks.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
