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
    new = json.loads((HERE / 'sources/source_manifest.json').read_text(encoding='utf-8'))
    # Authorized formal data replace old paper tables; historical input hashes
    # stay protected, rather than falsely requiring the new statistics to agree.
    for rel, expected in old['files'].items():
        if sha(ROOT / rel) != expected:
            raise AssertionError('Historical evidence changed: '+rel)
    for rel, expected in new['files'].items():
        path = (ROOT / rel).resolve(strict=True)
        if not path.is_relative_to(ROOT / 'outputs_e2e') or sha(path) != expected:
            raise AssertionError(f'Changed or unregistered input: {rel}')
        if 'paper_pair_formal' in rel or '/inputs/' in rel or path.suffix == '.npz':
            raise AssertionError('Reserved experiment or raw IQ was consumed')
    if sha(HERE / 'references.bib') != sha(BEFORE / 'references.bib'):
        raise AssertionError('Unapproved bibliography change')
    if sha(HERE / 'sections/model.tex') != sha(BEFORE / 'sections/model.tex'):
        raise AssertionError('Observation model changed')
    assets = json.loads((HERE / 'qa/formal_asset_checks.json').read_text(encoding='utf-8'))
    if not assets['passed'] or assets['row_metrics_replayed'] != 16320:
        raise AssertionError('Incomplete formal result replay')
    sections = '\n'.join(p.read_text(encoding='utf-8') for p in (HERE / 'sections').glob('*.tex'))
    if any(token in sections for token in (r'\begin{proposition}', r'\begin{proof}', r'\ref{prop:')):
        raise AssertionError('Obsolete theorem environment/reference remains')
    if any(x in sections for x in ('扩大比较正在运行', '未截断RMSE', '预留布局分组')):
        raise AssertionError('Obsolete/disallowed prose remains')
    exp = (HERE / 'sections/experiments.tex').read_text(encoding='utf-8')
    if len(re.findall(r'\\pending\{', exp)) != 2:
        raise AssertionError('Expected two reserved sections')
    report = dict(passed=True, preserved_draft_pdf_sha256=sha(BEFORE / 'main.pdf'),
                  old_evidence_files=len(old['files']), formal_evidence_files=len(new['files']),
                  formal_metrics=assets, bibliography_unchanged=True, observation_model_unchanged=True,
                  scope='Approved formal-result update; no new localization or parameter tuning.')
    (HERE / 'qa/formal_revision_checks.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
