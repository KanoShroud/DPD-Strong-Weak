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
    checks["all_48_equations_numbered_once"] = (
        len(blocks) == len(tags) == len(set(tags)) == 48)
    chapters = {2: 10, 3: 14, 4: 21, 5: 2, 6: 1}
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
    refs = re.findall(r"式[（(](\d+\.\d+)[）)]", text)
    checks["equation_references_resolve"] = set(refs).issubset(tags)
    fig_ids = re.findall(r"^\*\*图(\d+\.\d+)\s", text, re.M)
    table_ids = re.findall(r"^\*\*表(\d+\.\d+)\s", text, re.M)
    checks["nine_unique_figure_captions"] = len(fig_ids) == len(set(fig_ids)) == 9
    checks["thirteen_unique_table_captions"] = len(table_ids) == len(set(table_ids)) == 13
    checks["figure_references_resolve"] = set(re.findall(r"图(\d+\.\d+)", text)).issubset(fig_ids)
    checks["table_references_resolve"] = set(re.findall(r"表(\d+\.\d+)", text)).issubset(table_ids)
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    checks["nine_images_exist"] = len(images) == 9 and all((HERE/p).is_file() for p in images)
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
    checks["no_pdf_created"] = not list(HERE.rglob("*.pdf"))
    checks["portable_math_delimiters"] = r"\(" not in text and r"\)" not in text
    checks["explicit_font_macro_arguments"] = not re.search(
        r"\\(?:boldsymbol|mathbf|mathcal|mathbb)(?!\{)", text)
    checks["no_internal_project_references"] = not re.search(
        r"G[89](?:-R\d+)?|outputs_e2e|mixed_B|来源索引|端到端|学习模型", text)
    checks["no_private_document_links"] = all(
        p.startswith(("#", "https://", "http://")) for p in links)
    checks["no_multiline_display_body"] = all("\n" not in block for block in blocks)
    checks["finite_window_noise_addition"] = r"\,du +\mathcal{W}" in text
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
    def walk(node):
        if isinstance(node, dict):
            if node.get("t") == "Math":
                math.append(node["c"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(ast)
    checks["pandoc_recognizes_48_display_equations"] = sum(
        m[0]["t"] == "DisplayMath" for m in math) == 48
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
    numerical = json.loads((HERE/"analysis_v3_external/verification.json").read_text(encoding="utf-8"))
    checks["numerical_checks_all_passed"] = all(r["passed"] for r in numerical["checks"])
    checks["protected_input_hashes_unchanged"] = all(
        hashlib.sha256(Path(r["path"]).read_bytes()).hexdigest() == r["sha256"]
        for r in numerical["inputs"])
    # Make final source and figure identities explicit.
    deliveries = [source, HERE/"来源与核验索引.md", HERE/"analysis_and_figures.py",
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
