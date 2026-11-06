"""Validate Markdown structure, cross-references, evidence, and input identity."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="markdown_validation_v1.json")
    args = parser.parse_args()
    out = (HERE/args.output).resolve()
    if out.parent != HERE or out.exists():
        raise ValueError("Use a new validation filename in the report directory.")
    source = HERE/"学术交流报告_中文.md"
    text = source.read_text(encoding="utf-8")
    checks = {}
    blocks = re.findall(r"\$\$\s*(.*?)\s*\$\$", text, re.S)
    tags = re.findall(r"\\tag\{(\d+\.\d+)\}", text)
    migration = json.loads((HERE/"five_part_migration.json").read_text(encoding="utf-8"))
    checks["five_part_structure"] = re.findall(r"^## (.+)$", text, re.M) == migration["chapters"]
    checks["all_22_equations_numbered_once"] = (
        len(blocks) == len(tags) == len(set(tags)) == 22)
    chapters = {2: 19, 3: 2, 4: 1}
    checks["equation_numbering_contiguous"] = tags == [
        f"{c}.{i}" for c, n in chapters.items() for i in range(1, n+1)]
    for block, tag in zip(blocks, tags):
        depth = 0
        for bracket in re.findall(r"(?<!\\)[{}]", block):
            depth += 1 if bracket == "{" else -1
            if depth < 0:
                raise ValueError(f"Unbalanced braces in {tag}")
        checks[f"tex_braces_{tag}"] = depth == 0
        checks[f"tex_delimiters_{tag}"] = block.count(r"\left") == block.count(r"\right")
        checks[f"tex_environments_{tag}"] = (
            re.findall(r"\\begin\{([^}]+)\}", block)
            == re.findall(r"\\end\{([^}]+)\}", block))
    refs = re.findall(r"(?:式[（(]|—（)(\d+\.\d+)[）)]", text)
    checks["equation_references_resolve"] = set(refs).issubset(tags)
    fig_ids = re.findall(r"^\*\*图(\d+\.\d+)\s", text, re.M)
    table_ids = re.findall(r"^\*\*表(\d+\.\d+)\s", text, re.M)
    checks["eight_unique_figure_captions"] = len(fig_ids) == len(set(fig_ids)) == 8
    checks["nine_unique_table_captions"] = len(table_ids) == len(set(table_ids)) == 9
    checks["figure_references_resolve"] = set(re.findall(r"图(\d+\.\d+)", text)).issubset(fig_ids)
    checks["table_references_resolve"] = set(re.findall(r"表(\d+\.\d+)", text)).issubset(table_ids)
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    checks["eight_images_exist"] = len(images) == 8 and all((HERE/p).is_file() for p in images)
    for p in images:
        with Image.open(HERE/p) as img:
            img.verify()
    checks["images_decode"] = True
    links = re.findall(r"(?<!!)\[[^\]]*\]\(([^)]+)\)", text)
    anchors = re.findall(r'<a id="([^"]+)"></a>', text)
    checks["local_links_exist"] = all(
        (p[1:] in anchors if p.startswith("#") else (HERE/p).is_file())
        for p in links if not p.startswith(("https://", "http://")))
    bibliography = re.findall(r"^\[(\d+)\] ", text, re.M)
    checks["twelve_bibliography_entries"] = bibliography == [str(i) for i in range(1, 13)]
    checks["no_defensive_stock_phrases"] = not any(
        s in text for s in ["不能宣称", "不代表", "免责声明", "不能证明我们", "防御性"])
    checks["no_replacement_glyph"] = "\ufffd" not in text
    checks["latex_entry_exists"] = (HERE/"Latex PDF/document.tex").is_file()
    checks["portable_math_delimiters"] = r"\(" not in text and r"\)" not in text
    checks["no_unsupported_boldsymbol"] = r"\boldsymbol" not in text
    checks["all_17_captions_centered"] = len(re.findall(
        r'<div align="center" class="report-caption">\s*\*\*[图表]\d+\.\d+ .*?\*\*\s*</div>',
        text, re.S)) == 17
    checks["no_geometry_selection_commentary"] = "凸包" not in text and "构型" not in text
    before_path = HERE/"history/before_five_part_revision/学术交流报告_中文.md"
    before = before_path.read_text(encoding="utf-8")
    checks["revision_baseline_identity"] = hashlib.sha256(before_path.read_bytes()).hexdigest() == migration["source_sha256"]
    def math_map(content):
        return {re.search(r"\\tag\{([^}]+)\}", block)[1]:
                re.sub(r"\\tag\{[^}]+\}", "", block).strip()
                for block in re.findall(r"\$\$\s*(.*?)\s*\$\$", content, re.S)}
    old_math, new_math = math_map(before), math_map(text)
    checks["retained_equations_unchanged"] = all(
        old_math[old_id] == new_math[new_id] for old_id, new_id in migration["equation_map"].items())
    def table_map(content):
        return dict(re.findall(r"\*\*表(\d+\.\d+) .*?\*\*\s*</div>\s*((?:\|[^\n]*\n)+)", content, re.S))
    old_tables, new_tables = table_map(before), table_map(text)
    def renumber_equation_reference(content):
        return re.sub(r"式（(\d+\.\d+)）", lambda m: "式（"+migration["equation_map"][m[1]]+"）", content)
    checks["retained_table_data_unchanged"] = all(
        renumber_equation_reference(old_tables[old_id]) == new_tables[new_id]
        for old_id, new_id in migration["table_map"].items())
    def figure_files(content):
        return dict(re.findall(r"!\[图(\d+\.\d+) [^\]]*\]\(([^)]+)\)", content))
    old_figures, new_figures = figure_files(before), figure_files(text)
    redraws = {"2.2": "analysis_v7_latex", "3.5": "analysis_v8_latex"}
    checks["retained_figures_or_authorized_redraws"] = all(
        new_figures[new_id] == (old_figures[old_id].replace("analysis_v6_print", redraws[new_id])
                               if new_id in redraws else old_figures[old_id])
        for old_id, new_id in migration["figure_map"].items())
    old_refs = dict(re.findall(r"^\[(\d+)\] (.+)$", before, re.M))
    new_refs = dict(re.findall(r"^\[(\d+)\] (.+)$", text, re.M))
    checks["bibliography_preserved_and_renumbered"] = all(
        old_refs[old_id] == new_refs[str(new_id)] for old_id, new_id in migration["reference_map"].items())
    body = text.split("## 5. 参考文献")[0]
    cite_ids = [int(n) for group in re.findall(r"\[(\d+(?:,\d+)*)\]", body) for n in group.split(",")]
    checks["citations_resolve_in_first_use_order"] = list(dict.fromkeys(cite_ids)) == list(range(1, 13))
    discussion = text.split("## 4. 讨论")[1].split("## 5. 参考文献")[0]
    checks["discussion_targets_current_algorithms"] = all(
        word not in discussion for word in ["模型扩展", "源数扩展", "参数失配", "希望形成的决定", "已有基础"])
    with np.load(HERE/"analysis_v4_compat/figure_arrays.npz") as old_arrays, np.load(
            HERE/"analysis_v6_print/figure_arrays.npz") as new_arrays:
        checks["all_plot_data_preserved"] = old_arrays.files == new_arrays.files and all(
            np.array_equal(old_arrays[key], new_arrays[key]) for key in old_arrays.files)
    gray_files = list((HERE/"analysis_v6_print/grayscale").glob("*.png"))
    checks["nine_grayscale_figures"] = len(gray_files) == 9
    for file in gray_files:
        with Image.open(file) as img:
            checks[f"grayscale_{file.stem}"] = img.mode == "L"
    checks["explicit_font_macro_arguments"] = not re.search(
        r"\\(?:boldsymbol|mathbf|mathcal|mathbb)(?!\{)", text)
    checks["no_internal_project_references"] = not re.search(
        r"G[89](?:-R\d+)?|outputs_e2e|mixed_B|来源索引|端到端|学习模型", text)
    checks["no_private_document_links"] = all(
        p.startswith(("#", "https://", "http://")) for p in links)
    checks["no_multiline_display_body"] = all("\n" not in block for block in blocks)
    checks["finite_window_error_retained"] = r"+\mathbf{e}_m+\mathbf{n}_m" in text
    # Use the same dollar-only convention as the delivered Markdown.
    pandoc = Path("D:/Software/anaconda3/Library/bin/pandoc.exe")
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8:replace")
    result = subprocess.run(
        [str(pandoc), str(source), "-f", "markdown+tex_math_dollars",
         "-t", "json"], capture_output=True, text=True, encoding="utf-8",
        errors="replace", env=env, check=True)
    ast = json.loads(result.stdout)
    math = []
    parsed_chapters = []
    def walk(node):
        if isinstance(node, dict):
            if node.get("t") == "Header" and node["c"][0] == 2:
                parsed_chapters.append(node["c"])
            if node.get("t") == "Math":
                math.append(node["c"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(ast)
    checks["pandoc_recognizes_five_chapter_headings"] = len(parsed_chapters) == 5
    checks["pandoc_recognizes_22_display_equations"] = sum(
        m[0]["t"] == "DisplayMath" for m in math) == 22
    checks["pandoc_no_warnings"] = not result.stderr.strip()
    # Check numeric claims that are not covered by the figure summary validator.
    g9 = ROOT/"outputs_e2e/unified/e2e_g9/formal_20261011_repaired"
    rows = json.loads((g9/"evaluation_rows.json").read_text(encoding="utf-8"))
    key = lambda r: tuple(r[k] for k in ["group", "realization", "overlap", "gap", "N"])
    b = {key(r): r for r in rows if r["method"] == "B" and r["factor"] == 1.
         and r["kind"] == "mixed"}
    c = {key(r): r for r in rows if r["method"] == "C" and r["factor"] == 1.
         and r["kind"] == "mixed"}
    rescued = sum(not b[k]["metrics"]["success"]["100"][1]
                  and c[k]["metrics"]["success"]["100"][1] for k in b)
    lost = sum(b[k]["metrics"]["success"]["100"][1]
               and not c[k]["metrics"]["success"]["100"][1] for k in b)
    checks["C_B_rescued31_lost30"] = rescued == 31 and lost == 30
    r1 = ROOT/"outputs_e2e/unified/e2e_g9/r1_failure_20261011_01"
    sr = [json.loads(p.read_text(encoding="utf-8")) for p in (r1/"score_rows").glob("*.json")]
    coupled = [r for r in sr if not r["baseline_metrics"]["success"]["100"][1]
               and not r["estimated"]["truth_higher"]]
    small = [r["strong_error_m"] for r in coupled if r["strong_error_m"] <= 100]
    checks["coupled21_and_small5"] = len(coupled) == 21 and len(small) == 5
    checks["small_anchor_range"] = round(min(small), 2) == 13.40 and round(max(small), 2) == 27.99
    dense = [json.loads(p.read_text(encoding="utf-8")) for p in (r1/"dense_rows").glob("*.json")]
    checks["selected_four_layouts"] = len({r["tag"].split("_")[0] for r in dense}) == 4
    checks["dense_recovery_counts"] = [
        sum(r[k]["error_m"] <= 100 for r in dense if r["selection"] == "failure")
        for k in ["estimated", "true_strong"]] == [14, 15]
    narrow = next(r for r in dense if r["tag"] == "4_0_1.0_12_4096_mixed_B")["true_strong"]
    checks["narrow_case_values"] = np.allclose(
        [narrow["nearest_truth_grid_score"], narrow["score"],
         narrow["oracle"]["truth_score"], narrow["oracle"]["oracle_score"]],
        [.08159913, .08704823, .08989170, .08992482], rtol=0, atol=5.1e-9)
    numerical = json.loads((HERE/"analysis_v6_print/verification.json").read_text(encoding="utf-8"))
    checks["numerical_checks_all_passed"] = all(r["passed"] for r in numerical["checks"])
    checks["protected_input_hashes_unchanged"] = all(
        hashlib.sha256(Path(r["path"]).read_bytes()).hexdigest() == r["sha256"]
        for r in numerical["inputs"])
    # Make final source and figure identities explicit.
    deliveries = [source, HERE/"来源与核验索引.md", HERE/"analysis_and_figures.py", HERE/"five_part_migration.json",
                  Path(__file__), *[HERE/p for p in images]]
    output = dict(status="PASS" if all(checks.values()) else "FAIL",
                  characters=len(text), equations=len(tags), figures=len(fig_ids),
                  tables=len(table_ids), references=len(bibliography),
                  inline_math=sum(m[0]["t"] == "InlineMath" for m in math),
                  checks=checks, files=[dict(path=str(p), sha256=hashlib.sha256(
                      p.read_bytes()).hexdigest()) for p in deliveries])
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: output[k] for k in
                     ["status", "characters", "equations", "figures", "tables", "references",
                      "inline_math"]}, ensure_ascii=False))
    print("Failed:", [k for k, v in checks.items() if not v])
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
