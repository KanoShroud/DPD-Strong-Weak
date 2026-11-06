# English report

## Current edition: unknown complex gains (2026-10-13)

The current report has 31 pages. Its main text assumes unknown complex gains shared across frequencies. Appendices A--C retain the known-relative-gain derivations, all existing experiments, and their discussion. New experimental results are pending. Both languages contain 43 numbered equations, 11 figures, 14 tables, and 14 IEEE-style references.

- `document.tex` is the build entry point and contains the abstract, introduction, references, and appendices. It loads `unknown_gain_main.tex`, which contains Sections 2--4. Edit these LaTeX files directly; do not regenerate them from old Markdown or translation scripts.
- Figure 2.1 is an editable TikZ flow diagram with orthogonal return arrows. The ten PDFs in `figures/` are unchanged and now appear as appendix figures.
- Current validation: `../qa_unknown_gain_final/validation.json`. Legacy preservation: `../unknown_gain_migration.json`. The old validators assume the previous section and equation counts and are historical checks only.
- The experiment handoff is `../未知复增益DPD_功率差扫描执行方案.md`. The bilingual Word talk still describes the old model and will be updated after the new experiments.

Build with `& './build.ps1'`. From the repository root, use the project Python to run `研究方向/强弱源混叠DPD/研究记录/专题报告/kc_ho_20261012/validate_unknown_gain.py --output NEW_QA_DIRECTORY`. This checks formulas, references, preservation, algebraic identities, and page geometry, and renders the pages for visual review. Do not rerun the one-time migration script to apply later edits.

The following maintenance notes describe the 2026-10-11 edition. Their page counts, section numbers, and validation paths are historical.

## Historical edition notes

This is the English counterpart of the user-approved Chinese final edition in `../Latex PDF/`. The Chinese source, PDF, and ten figures are registered in `../chinese_final_manifest.json` and were not changed during translation.

Following the user's bilingual synchronization request, the Chinese edition now incorporates the English edition's explicit wording for spectral overlap, deadline outputs, percentage-point differences, overall RMSE, and successful versus newly recovered records. The English manuscript and PDF are unchanged. Current correspondence checks are in `qa_en_sync_01/validation.json`; the synchronization record is `../bilingual_sync.json`.

## Deliverables

- `document.tex`: the editable English manuscript. It contains all five main sections, 25 numbered equations, 10 figures, 11 tables, and 12 IEEE-style references from the Chinese edition.
- `document.pdf`: the compiled 23-page English report, without a date or table of contents.
- `figures/`: ten English PDF figures. All labels, legends, axes, and diagram text are in English. Numerical data, markers, and hatching retain the Chinese edition's meaning. Supplementary alternative charts remain excluded.
- `build.ps1`: three-pass XeLaTeX build using the existing TeX Live installation.
- `qa_en_final/validation.json`: final correspondence, input-identity, algebra, rendering, and manual-review results.
- `review_notes.md`: the scope and findings of the scientific and language review.

Build from this directory:

```powershell
& './build.ps1'
```

The figures use relative paths. The English bundle can be compiled independently of the project data. Use XeLaTeX; the manuscript uses `fontspec` and no Chinese font package.

## Correspondence and maintenance

Section, equation, table, and figure labels correspond to the Chinese final edition. The 25 equation environments and 12 bibliography entries are unchanged. Numeric values were checked row by row across all tables. The 20 m success radius remains distinct from the 100 m GOSPA cutoff. The fixed-case single-source update comparison remains separate from the complete solver on new layouts.

`../english_text.py` and `../create_english.py` record the initial translation of all 221 Chinese-bearing source lines. The creation script refuses to overwrite this directory. **Edit the current English LaTeX directly** for subsequent changes; do not regenerate it from the translation mapping or an older Markdown report.

The final figures come from `../analysis_english_final/`. That directory also contains PNG, SVG, grayscale previews, and a source manifest. `../english_figures.py` reads frozen results and saved plotting arrays; it performs no localization or training. It checks numerical artist coordinates before and after translation, checks the original cost/condition arrays, and verifies 5720 source-file identities.

The final document passed 5934 checks, including identical equations, reference correspondence, row-level numeric agreement, source identities, and English-only PDF and figure text. Twelve small algebra checks passed for the projector, concentrated LS, incremental score, derivatives, and equivalent Fisher information. The 24 existing regional-bound numerical checks were read back; no formal experiment was rerun.

To rerun document checks from the repository root, choose a QA directory that does not yet exist:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8:replace'
& 'D:/Software/anaconda3/envs/PyTorch/python.exe' '研究方向/强弱源混叠DPD/研究记录/专题报告/kc_ho_20261012/validate_english.py' --figures analysis_english_final --qa qa_en_recheck_01
```

Recompilation can change PDF metadata and its byte hash without changing the manuscript. A content revision requires a new review; do not silently replace the registered Chinese final edition.
