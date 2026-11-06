"""One-time, mechanical migration of the preserved manuscript to the CAS layout.

The source archive is immutable. Later 研究方向\\强弱源混叠DPD\\论文\\AP_NOMP双语稿 edits must not rerun this migration.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
ARCHIVE = PAPER / 'archive/pre_cas_20261018'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = PAPER / 'qa/migration_manifest.json'
    if manifest.exists():
        raise RuntimeError('Migration already recorded; edit the active 研究方向\\强弱源混叠DPD\\论文\\AP_NOMP双语稿 instead.')
    transformations = {}
    for src in sorted((ARCHIVE / 'sections').glob('*.tex')):
        folder = 'tables' if src.name.startswith('generated_') else 'sections'
        dst = PAPER / folder / src.name
        text = src.read_text(encoding='utf-8')
        text = text.replace(r'\input{sections/generated_', r'\input{tables/generated_')
        text = text.replace(r'\begin{table}[H]', r'\begin{table}[pos=H]')
        text = text.replace(r'\begin{table}[htbp]', r'\begin{table}[pos=htbp]')
        text = text.replace(r'\begin{figure}[htbp]', r'\begin{figure}[pos=htbp]')
        if src.name == 'abstract.tex':
            text = text.replace(r'\begin{abstract}', r'\begin{abstract}[摘要]')
            prefix = r'\noindent\textbf{关键词：}'
            before, keywords = text.split(prefix, 1)
            text = before + '\\begin{keywords}[关键词]\n'
            text += keywords.strip().replace('；', r' \sep ') + '\n\\end{keywords}\n'
        dst.write_text(text, encoding='utf-8')
        transformations[str(dst.relative_to(PAPER))] = dict(
            source=str(src.relative_to(PAPER)), source_sha256=sha(src), migrated_sha256=sha(dst))
    # Preserve the offline asset pipeline, adapting storage paths only.
    for name in ('prepare_assets.py', 'prepare_formal_assets.py'):
        dst = PAPER / 'scripts' / name
        text = dst.read_text(encoding='utf-8')
        text = text.replace('HERE = Path(__file__).resolve().parent\nROOT = HERE.parents[2]',
                            'HERE = Path(__file__).resolve().parents[1]\nROOT = HERE.parent')
        text = text.replace("'sources/", "'data/").replace("'sources'", "'data'")
        text = text.replace("'references.bib'", "'references/references.bib'")
        text = text.replace("'sections/generated_", "'tables/generated_")
        text = text.replace("HERE / 'sections' / (name+'.tex')", "HERE / 'tables' / (name+'.tex')")
        text = text.replace(r'\\begin{table}[H]', r'\\begin{table}[pos=H]')
        text = text.replace(r'\\begin{table}[htbp]', r'\\begin{table}[pos=htbp]')
        dst.write_text(text, encoding='utf-8')
    # The requested original path now loads the one maintained manuscript.
    entry = ROOT / 'els-cas-templates/cas-sc-sample.tex'
    entry.write_text('% !TeX program = xelatex\n'
                     '% The maintained manuscript and all assets are in ../研究方向\\强弱源混叠DPD\\论文\\AP_NOMP双语稿/.\n'
                     '\\def\\PaperRoot{../研究方向\\强弱源混叠DPD\\论文\\AP_NOMP双语稿/}\n'
                     '\\input{../研究方向\\强弱源混叠DPD\\论文\\AP_NOMP双语稿/cas-sc-sample.tex}\n', encoding='utf-8')
    archive_files = {str(p.relative_to(ARCHIVE)): sha(p)
                     for p in sorted(ARCHIVE.rglob('*')) if p.is_file()}
    payload = dict(date='2026-10-18', original_location='研究方向/强弱源混叠DPD/论文/前期稿件/wideband_multisource_localization',
                   active_entry='研究方向\\强弱源混叠DPD\\论文\\AP_NOMP双语稿/cas-sc-sample.tex', template_entry='els-cas-templates/cas-sc-sample.tex',
                   transformations=transformations, archive_files=archive_files,
                   original_template_sha256=sha(PAPER / 'archive/template_original/cas-sc-sample.tex'))
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f'CAS migration complete: {len(transformations)} TeX files; previous manuscript archived.')


if __name__ == '__main__':
    main()
