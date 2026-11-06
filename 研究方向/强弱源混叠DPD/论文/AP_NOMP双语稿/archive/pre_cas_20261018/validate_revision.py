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
    predecessor = json.loads((HERE / 'qa/profile_stratified_revision_before/sources/source_manifest.json').read_text(encoding='utf-8'))
    new = json.loads((HERE / 'sources/source_manifest.json').read_text(encoding='utf-8'))
    # Authorized formal data replace old paper tables; historical input hashes
    # stay protected, rather than falsely requiring the new statistics to agree.
    for rel, expected in old['files'].items():
        if sha(ROOT / rel) != expected:
            raise AssertionError('Historical evidence changed: '+rel)
    for rel, expected in prior['files'].items():
        if new['files'].get(rel) != expected or sha(ROOT / rel) != expected:
            raise AssertionError('Prior formal paper evidence changed: '+rel)
    for rel, expected in predecessor['files'].items():
        if new['files'].get(rel) != expected or sha(ROOT / rel) != expected:
            raise AssertionError('Immediate predecessor evidence changed: '+rel)
    for rel, expected in new['files'].items():
        path = (ROOT / rel).resolve(strict=True)
        if not path.is_relative_to(ROOT / 'outputs_e2e') or sha(path) != expected:
            raise AssertionError(f'Changed or unregistered input: {rel}')
        if '/inputs/' in rel or path.suffix == '.npz':
            raise AssertionError('Raw IQ was consumed')
    if sha(HERE / 'references.bib') != sha(BEFORE / 'references.bib'):
        raise AssertionError('Unapproved bibliography change')
    for name in ('model.tex', 'methods.tex', 'appendix.tex'):
        pattern = r'\\begin\{(?:equation|align)\}.*?\\end\{(?:equation|align)\}'
        previous = re.findall(pattern, (BEFORE / 'sections' / name).read_text(encoding='utf-8'), re.S)
        current = re.findall(pattern, (HERE / 'sections' / name).read_text(encoding='utf-8'), re.S)
        if current != previous:
            raise AssertionError('Unapproved mathematical change: '+name)
    assets = json.loads((HERE / 'qa/formal_asset_checks.json').read_text(encoding='utf-8'))
    if (not assets['passed'] or assets['row_metrics_replayed'] != 16560
            or assets['paired_rows_replayed'] != 4800 or not assets['paired_common_candidates']
            or assets['profile_rows_replayed'] != 2400 or not assets['profile_same_success_set']
            or not assets['profile_all_native_stable'] or not assets['geometry_partition_reconstructs_total']):
        raise AssertionError('Incomplete formal result replay')
    sections = '\n'.join(p.read_text(encoding='utf-8') for p in (HERE / 'sections').glob('*.tex'))
    if any(token in sections for token in (r'\begin{proposition}', r'\begin{proof}', r'\ref{prop:')):
        raise AssertionError('Obsolete theorem environment/reference remains')
    if any(x in sections for x in ('扩大比较正在运行', '未截断RMSE', '预留布局分组')):
        raise AssertionError('Obsolete/disallowed prose remains')
    exp = (HERE / 'sections/experiments.tex').read_text(encoding='utf-8')
    if re.search(r'\\pending\{', exp):
        raise AssertionError('Completed PROFILE content still reserved')
    structure_before = HERE / 'qa/comparison_structure_revision_before'
    section_order = ['sec:baselines', 'sec:commonformal', 'sec:coreformal',
                     'sec:apresults', 'sec:mechanismresults', 'sec:timecases']
    section_offsets = [exp.index(r'\label{'+label+'}') for label in section_order]
    if section_offsets != sorted(section_offsets):
        raise AssertionError('Common comparison / expanded validation order changed')
    if exp.index(r'\label{fig:formallayouts}') < section_offsets[-1]:
        raise AssertionError('Detailed geometry diagnosis must remain in the final subsection')
    preserved_statistics = [
        'formal_summary.csv', 'formal_auxiliary_metrics.csv', 'formal_geometry_groups.csv',
        'formal_pair_summary.csv', 'formal_profile_common.csv', 'rmse_attribution.csv',
        'formal_operation_counts.csv', 'method_parameters.csv', 'formal_pair_report.json',
        'formal_profile_report.json', 'formal_statistical_report.json',
    ]
    for name in preserved_statistics:
        if sha(HERE / 'sources' / name) != sha(structure_before / 'sources' / name):
            raise AssertionError('Statistical evidence changed during chapter reorganization: '+name)
    equation_pattern = r'\\begin\{(?:equation|align)\}.*?\\end\{(?:equation|align)\}'
    old_exp = (structure_before / 'sections/experiments.tex').read_text(encoding='utf-8')
    if re.findall(equation_pattern, exp, re.S) != re.findall(equation_pattern, old_exp, re.S):
        raise AssertionError('Evaluation equations changed during chapter reorganization')
    report = dict(passed=True, preserved_draft_pdf_sha256=sha(BEFORE / 'main.pdf'),
                  old_evidence_files=len(old['files']), formal_evidence_files=len(new['files']),
                  prior_formal_evidence_files=len(prior['files']),
                  predecessor_evidence_files=len(predecessor['files']),
                  chapter_order=section_order, preserved_statistics=preserved_statistics,
                  all_samples_retained=True, evaluation_equations_unchanged=True,
                  formal_metrics=assets, bibliography_unchanged=True, mathematical_derivations_unchanged=True,
                  scope='Approved common-240 comparison then expanded-2400 validation, geometry diagnosis consolidation; no new localization, exclusion or tuning.')
    (HERE / 'qa/formal_revision_checks.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
