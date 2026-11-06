"""Validate current independent LaTeX revision against frozen-result readback."""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import re

import fitz
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
LATEX = HERE / "Latex PDF"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--r3", required=True)
    parser.add_argument("--qa", required=True)
    args = parser.parse_args()
    out = (LATEX / args.qa).resolve()
    assert out.parent == LATEX and not out.exists()
    data = json.loads((HERE / args.analysis / "report_data.json").read_text(encoding="utf-8"))
    verification = json.loads((HERE / args.analysis / "verification.json").read_text(encoding="utf-8"))
    r3 = json.loads((HERE / args.r3 / "readback.json").read_text(encoding="utf-8"))
    tex = (LATEX / "document.tex").read_text(encoding="utf-8")
    log = (LATEX / "document.log").read_text(encoding="utf-8")
    checks = {}
    def check(name, passed):
        checks[name] = bool(passed)
        if not passed:
            raise AssertionError(name)
    check("analysis_pass", verification["status"] == "PASS")
    check("r3_readback_pass", r3["status"] == "PASS" and r3["row_count"] == 24)
    check("r3_initial_controls_6_of_8", r3["initial_controls20"] == 6 and "初始只有6条满足20 m" in tex)
    check("empty_date_no_toc", r"\date{}" in tex and r"\tableofcontents" not in tex)
    check("no_incompatible_math", r"\boldsymbol" not in tex and "$$" not in tex)
    check("finite_window_only_model_assumption", tex.count("观测窗边缘效应") == 1 and r"\mathbf{e}_m" not in tex)
    check("no_internal_project_labels", not re.search(r"G[89]-R\d|outputs_e2e|\.json|\.py|Ho报告|test封存", tex))
    check("no_stale_threshold_claims", not re.search(r"100\s*m成功|半径100|主半径为100|250条|229条|21条失败|约一半", tex))
    labels = re.findall(r"\\label\{([^}]+)\}", tex)
    refs = re.findall(r"\\ref\{([^}]+)\}", tex)
    check("unique_labels", len(labels) == len(set(labels)))
    check("resolved_references", set(refs) <= set(labels))
    cites = [k for x in re.findall(r"\\cite\{([^}]+)\}", tex) for k in x.split(",")]
    bibliography = re.findall(r"\\bibitem\{([^}]+)\}", tex)
    check("12_references_resolve", len(bibliography) == 12 and set(cites) == set(bibliography))
    check("ieee_citation_order", list(dict.fromkeys(cites)) == bibliography)
    bib_entries = re.split(r"\\bibitem\{[^}]+\}", tex)[1:]
    check("ieee_reference_fields", all("``" in b and "''" in b and "pp. " in b and "doi: " in b for b in bib_entries))
    check("no_author_manuscript_links", "作者稿" not in tex and "arxiv.org" not in tex)
    check("methods_defined_before_use", tex.index(r"\label{tab:methods}") < tex.index(r"\label{fig:2.2}") < tex.index(r"\label{sec:3}"))
    check("region_bound_before_experiment", tex.index(r"\label{eq:cell-upper}") < tex.index(r"\label{tab:region-search}"))
    env_begin = Counter(re.findall(r"\\begin\{([^}]+)\}", tex))
    env_end = Counter(re.findall(r"\\end\{([^}]+)\}", tex))
    check("balanced_environments", env_begin == env_end)
    check("25_numbered_equations", env_begin["equation"] == 25)
    check("10_figures_11_tables", env_begin["figure"] == 10 and env_begin["table"] == 11)
    check("alternatives_removed", "_alt.pdf" not in tex and "备选样式" not in tex)
    for ref in re.findall(r"\\includegraphics\[[^]]+\]\{([^}]+)\}", tex):
        check("asset_"+ref, (LATEX / ref).is_file())
        original = HERE / args.analysis / ref
        if original.exists():
            check("asset_identity_"+ref, (LATEX / ref).read_bytes() == original.read_bytes())
    check("no_packaged_alternatives", not list((LATEX / "figures").glob("*_alt.pdf")))
    for m, label in [("DPD", "普通DPD"), ("A", "A"), ("B", "B"), ("C", "C")]:
        s = data["formal"][f"mixed/{m}/1.0"]
        cells = [label] + [f"{100*s[k]['rate']:.2f}\\%" for k in ("weak_20", "both_20", "strong_20")] + [f"{s[k]:.2f}" for k in ("gospa_m", "rmse_m", "p95_m")]
        check("formal_table_"+m, " & ".join(cells) in tex)
    for n in (4096, 16384):
        ss = [data["formal"][f"weak/B/1.0/N{n}"]["single_20"]] + [data["formal"][f"mixed/{m}/1.0/N{n}"]["weak_20"] for m in ("B", "C")]
        cells = [f"{n}点"] + [f"{s['successes']}/{s['total']}（{Decimal(str(100*s['rate'])).quantize(Decimal('.001'), rounding=ROUND_HALF_UP)}\\%）" for s in ss]
        check(f"isolated_mixed_table_{n}", " & ".join(cells) in tex)
        a = data["attribution"][str(n)]
        cells = [f"{n}点"] + [str(a[k]) for k in ("failures", "estimated_truth_higher", "true_strong_truth_higher", "joint_truth_higher")]
        check(f"attribution_table_{n}", " & ".join(cells) in tex)
    for m in ("B", "D"):
        s = data["r2_fresh"][m]
        cells = [m] + [f"{s[k]['successes']}/{s[k]['total']}" for k in ("weak_20", "both_20", "strong_20")] + [f"{s[k]:.2f}" for k in ("gospa_m", "rmse_m", "p95_m", "seconds_mean")]
        check("fresh_table_"+m, " & ".join(cells) in tex)
    for path, digest in verification["inputs"].items():
        check("input_identity_"+path, hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest)
    for method, name in (("D", "5 m均匀网格"), ("shift10", "10 m错位网格"),
                         ("S", "得分优先细分"), ("E", "上界优先细分")):
        s = r3["summary"][method]
        points = f"{s['grid_points_mean']:.1f}" if method in ("D", "shift10") else f"{s['grid_points_mean']:.0f}"
        cells = [name, f"{s['failure_recovered20']}/16", f"{s['controls20']}/8",
                 points, f"{s['seconds_mean']:.4f}"]
        check("region_search_table_"+method, " & ".join(cells) in tex)
    check("region_no_pruning", r3["summary"]["E"]["pruned_cells_mean"] == 0 and "剪枝数均为零" in tex)
    for path, digest in r3["inputs"].items():
        check("r3_input_identity_"+path, hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest)
    severe = re.findall(r"(?m)^.*(?:Overfull|Undefined control|Missing character|undefined references|undefined citations|multiply defined|LaTeX Error|Package .* Error).*$", log)
    check("no_severe_compiler_diagnostics", not severe)
    out.mkdir()
    doc = fitz.open(LATEX / "document.pdf")
    page_text = [page.get_text() for page in doc]
    check("no_date_on_title_page", not re.search(r"October|2026年|2026", page_text[0]))
    check("no_replacement_glyphs", all("\ufffd" not in s for s in page_text))
    alternative_pages = [i+1 for i, s in enumerate(page_text) if "备选样式" in s]
    check("no_alternative_pages", len(alternative_pages) == 0)
    check("method_table_in_section_2", "表2.2" in "".join(page_text).replace(" ", ""))
    bounds = []
    for i, page in enumerate(doc):
        # Screen punctuation may extend slightly beyond typographic text bounds;
        # only flag actual page overflow, not the inner margin.
        bad = [b[:4] for b in page.get_text("blocks") if b[0] < -1 or b[1] < -1 or b[2] > page.rect.width+1 or b[3] > page.rect.height+1]
        check(f"page_{i+1}_within_sheet", not bad)
        bounds.append({"page": i+1, "blocks": len(page.get_text("blocks"))})
        page.get_pixmap(matrix=fitz.Matrix(1.4, 1.4), alpha=False).save(out / f"page_{i+1:02}.png")
    for start in range(0, len(doc), 6):
        contact = Image.new("RGB", (1020, 1500), "#d4d4d4")
        draw = ImageDraw.Draw(contact)
        for j, index in enumerate(range(start, min(start+6, len(doc)))):
            with Image.open(out / f"page_{index+1:02}.png") as im:
                im.thumbnail((480, 680))
                x, y = (j % 2)*510+15, (j//2)*500+30
                # Three rows, preserving A4 aspect at a readable contact scale.
                im.thumbnail((315, 445))
                contact.paste(im, (x+75, y))
                draw.text((x+75, y-18), f"Page {index+1}", fill="black")
        contact.save(out / f"contact_{start//6+1}.png")
    result = {"status": "PASS", "pages": len(doc), "checks": len(checks),
        "alternative_pages": alternative_pages, "compiler_severe_diagnostics": severe,
        "source_sha256": hashlib.sha256((LATEX / "document.tex").read_bytes()).hexdigest(),
        "pdf_sha256": hashlib.sha256((LATEX / "document.pdf").read_bytes()).hexdigest(),
        "figures": env_begin["figure"], "tables": env_begin["table"], "equations": env_begin["equation"],
        "all_checks": checks, "page_bounds": bounds,
        "visual_review": "pending separate visual inspection"}
    (out / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("all_checks", "page_bounds")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
