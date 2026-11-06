"""Check the initial LaTeX migration and render its PDF for visual inspection.

Markdown comparisons certify this migration only. After the LaTeX manuscript
is edited independently, use its compiler and new task-specific checks.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import fitz
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
LATEX = HERE/"Latex PDF"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(s):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s))


def math_tokens(s):
    s = re.sub(r"\\(?:tag|label)\{[^}]+\}", "", s)
    s = re.sub(r"\\(?:begin|end)\{aligned\}", "", s)
    s = re.sub(r"\\(?:q?quad)\b|\\\\|&", "", s)
    return re.sub(r"\s+", "", s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", required=True)
    args = parser.parse_args()
    out = (LATEX/args.qa).resolve()
    if out.parent != LATEX.resolve() or out.exists():
        raise ValueError("Use a new QA directory inside Latex PDF")
    out.mkdir()
    md = (HERE/"学术交流报告_中文.md").read_text(encoding="utf-8")
    tex = (LATEX/"document.tex").read_text(encoding="utf-8")
    log = (LATEX/"document.log").read_text(encoding="utf-8", errors="strict")
    aux = (LATEX/"document.aux").read_text(encoding="utf-8")
    checks = {}
    equations_md = {re.search(r"\\tag\{([^}]+)\}", b)[1]: math_tokens(b)
                    for b in re.findall(r"\$\$(.*?)\$\$", md, re.S)}
    equations_tex = {re.search(r"\\label\{eq:([^}]+)\}", b)[1]: math_tokens(b)
                     for b in re.findall(r"\\begin\{equation\}(.*?)\\end\{equation\}", tex, re.S)}
    checks["all_22_equations_preserved"] = len(equations_tex) == 22 and equations_tex == equations_md
    md_inline = re.sub(r"\$\$.*?\$\$", "", md, flags=re.S)
    tex_inline = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", "", tex, flags=re.S)
    checks["all_101_inline_math_occurrences_preserved"] = (
        Counter(re.findall(r"\$([^$\n]+)\$", md_inline))
        == Counter(re.findall(r"\$([^$\n]+)\$", tex_inline)))
    checks["five_sections"] = len(re.findall(r"\\section\{", tex)) == 5
    checks["eight_figures_nine_tables"] = (tex.count(r"\begin{figure}") == 8
                                          and tex.count(r"\begin{table}") == 9)
    checks["twelve_bibliographic_entries"] = tex.count(r"\bibitem{") == 12
    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    checks["cross_references_resolve"] = set(re.findall(r"\\ref\{([^}]+)\}", tex)) <= labels
    aux_labels = dict(re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]+)\}", aux))
    checks["all_numbers_match_markdown"] = all(aux_labels.get(k) == k.split(":")[1] for k in labels)
    bibs = dict(re.findall(r"\\bibcite\{([^}]+)\}\{([^}]+)\}", aux))
    checks["citations_numbered_1_to_12"] = bibs == {f"ref{n:02}": str(n) for n in range(1, 13)}
    cited = {key for group in re.findall(r"\\cite\{([^}]+)\}", tex) for key in group.split(",")}
    checks["all_citations_resolve"] = cited <= bibs.keys()
    serious = re.findall(r"^.*(?:Overfull|Missing character|Undefined|undefined|LaTeX Warning|LaTeX Error).*$",
                         log, re.M)
    checks["no_overflow_missing_glyphs_or_references"] = not serious
    checks["captions_centered"] = "justification=centering,singlelinecheck=false" in tex
    checks["no_absolute_or_parent_figure_dependencies"] = not any(
        ":" in p or ".." in p for p in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex))
    manifest = json.loads((LATEX/"migration_validation.json").read_text(encoding="utf-8"))
    assets = []
    for f in manifest["figures"]:
        src = HERE/("analysis_v8_latex/figures/fig_5_4_cost.pdf" if f["number"] == "3.5" else f["source"])
        dest = LATEX/f["destination"]
        assets.append(dict(number=f["number"], source=str(src.relative_to(HERE)),
                           destination=f["destination"], sha256=sha(dest), match=sha(src) == sha(dest)))
    checks["all_vector_assets_match_sources"] = all(a["match"] for a in assets)
    with np.load(HERE/"analysis_v6_print/figure_arrays.npz") as a, np.load(
            HERE/"analysis_v8_latex/figure_arrays.npz") as b:
        checks["all_16_plot_arrays_unchanged"] = len(a.files) == 16 and a.files == b.files and all(
            np.array_equal(a[k], b[k]) for k in a.files)
    rows = json.loads((HERE/"analysis_v8_latex/cost_plot_data.json").read_text(encoding="utf-8"))
    source = HERE.parents[2]/"outputs_e2e/unified/e2e_g9/formal_20261011_repaired/comparison_report.json"
    frozen = json.loads(source.read_text(encoding="utf-8"))["summary"]
    checks["all_18_cost_success_pairs_match_frozen_summary"] = len(rows) == 18 and all(
        row["seconds_mean"] == frozen[f"{row['N']}_{row['method']}_{row['factor']}"]["seconds_mean"]
        and row["weak_100_percent"] == 100*frozen[f"{row['N']}_{row['method']}_{row['factor']}"]["weak_100"]
        for row in rows)
    measured_times = {}
    for n in [4096, 16384]:
        for method in ["A", "B", "C"]:
            raw = [json.loads(p.read_text(encoding="utf-8"))
                   for p in (source.parent/"rows").glob(f"*_{n}_mixed_{method}.json")]
            assert len(raw) == 256
            measured_times[f"{n}_{method}"] = float(np.mean([r["seconds"] for r in raw]))
    checks["table_3_8_actual_times_preserved"] = all(
        " & ".join(f"{measured_times[f'{n}_{m}']:.3f}" for m in ["A", "B", "C"]) in tex
        for n in [4096, 16384])
    doc = fitz.open(LATEX/"document.pdf")
    text = "\n".join(p.get_text() for p in doc)
    (out/"pdf_text.txt").write_text(text, encoding="utf-8")
    normalized_pdf = compact(text)
    # Image alt text is replaced by the explicit printed caption, not body prose.
    printed_md = re.sub(r"!\[[^]]*\]\([^)]*\)", "", md)
    phrases = sorted(set(re.findall(r"[\u4e00-\u9fff，。、；：！？“”‘’]{10,}", printed_md)))
    missing = [s for s in phrases if compact(s) not in normalized_pdf]
    checks["all_long_chinese_text_spans_present_in_pdf"] = not missing
    checks["a4_pages"] = all(abs(p.rect.width-595.28) < .1 and abs(p.rect.height-841.89) < .1 for p in doc)
    outside = []
    for i, page in enumerate(doc):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    x0, y0, x1, y1 = span["bbox"]
                    # xeCJK compresses punctuation advance widths, but PDF text
                    # extraction reports the original full-width glyph box.
                    # Allow one em for those metrics; compiler checks actual boxes.
                    if x0 < 60 or x1 > page.rect.width-60 or y0 < 65 or y1 > page.rect.height-35:
                        outside.append(dict(page=i+1, text=span["text"], bbox=span["bbox"]))
        page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False).save(out/f"page_{i+1:02}.png")
    checks["all_text_within_margins_allowing_cjk_glyph_metrics"] = not outside
    for first in range(0, len(doc), 6):
        sheet = Image.new("RGB", (1260, 1240), "#CCCCCC")
        for j in range(min(6, len(doc)-first)):
            im = Image.open(out/f"page_{first+j+1:02}.png").convert("RGB")
            im.thumbnail((400, 580))
            x, y = (j % 3)*420+10, (j//3)*620+25
            sheet.paste(im, (x, y))
            ImageDraw.Draw(sheet).text((x, y-18), f"Page {first+j+1}", fill="black")
        sheet.save(out/f"contact_{first//6+1}.jpg")
    result = dict(status="PASS" if all(checks.values()) else "FAIL", checks=checks,
                  pages=len(doc), checked_chinese_spans=len(phrases), missing_spans=missing,
                  overflows=outside, serious_log_messages=serious, figures=assets,
                  full_run_seconds_mean=measured_times,
                  source_sha256=sha(HERE/"学术交流报告_中文.md"), tex_sha256=sha(LATEX/"document.tex"),
                  pdf_sha256=sha(LATEX/"document.pdf"), comparison_report_sha256=sha(source))
    (out/"validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["status", "pages", "checked_chinese_spans", "missing_spans"]},
                     ensure_ascii=False))
    print("Failed checks:", [k for k, v in checks.items() if not v])
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
