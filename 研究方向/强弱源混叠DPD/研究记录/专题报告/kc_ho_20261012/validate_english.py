"""Cross-language integrity, frozen inputs, algebra, and rendered-PDF checks."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import fitz
from PIL import Image, ImageDraw

import analysis_and_figures as algebra

HERE = Path(__file__).resolve().parent
ENGLISH = HERE / "Latex PDF English"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tables(text):
    return re.findall(r"\\begin\{table\}.*?\\end\{table\}", text, re.S)


def numbers(text):
    # Normalize the original literal unit m² and the English TeX unit m$^2$.
    return Counter(re.findall(r"\d+(?:\.\d+)?", text.replace("²", "2")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures", required=True)
    parser.add_argument("--qa", required=True)
    args = parser.parse_args()
    out = (ENGLISH / args.qa).resolve()
    assert out.parent == ENGLISH and not out.exists()
    checks = {}

    def check(name, passed):
        checks[name] = bool(passed)
        if not passed:
            raise AssertionError(name)

    manifest = json.loads((HERE / "chinese_final_manifest.json").read_text(encoding="utf-8"))
    for path, expected in manifest["files"].items():
        check("chinese_final_"+path, digest(HERE / path) == expected)
    zh = (HERE / "Latex PDF/document.tex").read_text(encoding="utf-8")
    en = (ENGLISH / "document.tex").read_text(encoding="utf-8")
    check("no_chinese_in_english_source", not re.search(r"[\u4e00-\u9fff]", en))
    check("no_date_or_contents", r"\date{}" in en and r"\tableofcontents" not in en)
    check("no_untranslated_punctuation", not re.search(r"[，。；：（）]", en))
    check("no_internal_project_references", not re.search(r"G[89]-R\d|outputs_e2e|\.json|\.py|作者稿", en))
    for name, pattern in (("equations", r"\\begin\{equation\}.*?\\end\{equation\}"),
                          ("labels", r"\\label\{[^}]+\}"),
                          ("citations", r"\\cite\{[^}]+\}"),
                          ("bibliography", r"\\bibitem\{.*?\\end\{thebibliography\}")):
        check("identical_"+name, re.findall(pattern, zh, re.S) == re.findall(pattern, en, re.S))
    labels = re.findall(r"\\label\{([^}]+)\}", en)
    refs = re.findall(r"\\ref\{([^}]+)\}", en)
    check("labels_unique_and_refs_resolved", len(labels) == len(set(labels)) and set(refs) <= set(labels))
    check("section_structure_preserved", re.findall(r"\\(sub)*section\*?", zh) == re.findall(r"\\(sub)*section\*?", en))
    begin = Counter(re.findall(r"\\begin\{([^}]+)\}", en))
    end = Counter(re.findall(r"\\end\{([^}]+)\}", en))
    check("balanced_environments", begin == end)
    check("25_equations_10_figures_11_tables", begin["equation"] == 25 and begin["figure"] == 10 and begin["table"] == 11)
    check("methods_defined_before_use", en.index(r"\label{tab:methods}") < en.index(r"\label{fig:2.2}") < en.index(r"\label{sec:3}"))
    for i, (source, target) in enumerate(zip(tables(zh), tables(en)), 1):
        # A row-by-row comparison catches a transferred denominator or runtime.
        a, b = (s.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0] for s in (source, target))
        ar, br = [r.strip() for r in a.split(r"\\") if r.strip()], [r.strip() for r in b.split(r"\\") if r.strip()]
        check(f"table_{i}_row_count", len(ar) == len(br))
        for j, (x, y) in enumerate(zip(ar, br), 1):
            check(f"table_{i}_row_{j}_numeric_identity", numbers(x) == numbers(y))
    check("r3_scope_explicit", "single weak-source update" in en and "not population success rates" in en)
    check("thresholds_separate", "20 m success radius" in en and "GOSPA cutoff" in en)
    check("window_approximation_once", en.count("observation-window edge effects") == 1)

    fv = json.loads((HERE / args.figures / "verification.json").read_text(encoding="utf-8"))
    check("figure_verification_pass", fv["status"] == "PASS" and fv["figure_count"] == 10)
    for path, expected in fv["inputs"].items():
        check("frozen_input_"+path, digest(Path(path)) == expected)
    for ref in re.findall(r"\\includegraphics\[[^]]+\]\{([^}]+)\}", en):
        p = ENGLISH / ref
        check("figure_identity_"+ref, p.read_bytes() == (HERE / args.figures / ref).read_bytes())
        with fitz.open(p) as figure:
            check("english_figure_text_"+ref, not re.search(r"[\u4e00-\u9fff\ufffd]", figure[0].get_text()))
    check("no_alternative_assets", not list((ENGLISH / "figures").glob("*_alt.pdf")))

    algebra.CHECKS.clear()
    algebra.equations()
    check("algebra_checks_pass", bool(algebra.CHECKS) and all(c["passed"] for c in algebra.CHECKS))
    r3 = json.loads((HERE / "analysis_r3_readback_02/readback.json").read_text(encoding="utf-8"))
    check("frozen_region_bound_checks_pass", r3["status"] == "PASS" and r3["numerical_records_verified"] == 24)
    for path, expected in r3["inputs"].items():
        check("r3_input_"+path, digest(Path(path)) == expected)
    log = (ENGLISH / "document.log").read_text(encoding="utf-8")
    severe = re.findall(r"(?m)^.*(?:Overfull|Undefined control|Missing character|undefined references|undefined citations|multiply defined|LaTeX Error|Package .* Error).*$", log)
    check("no_severe_compiler_diagnostics", not severe)

    out.mkdir()
    doc = fitz.open(ENGLISH / "document.pdf")
    all_text = []
    for i, page in enumerate(doc):
        text = page.get_text()
        all_text.append(text)
        check(f"page_{i+1}_english_only", not re.search(r"[\u4e00-\u9fff\ufffd]", text))
        bad = [b[:4] for b in page.get_text("blocks") if b[0] < -1 or b[1] < -1 or b[2] > page.rect.width+1 or b[3] > page.rect.height+1]
        check(f"page_{i+1}_within_sheet", not bad)
        page.get_pixmap(matrix=fitz.Matrix(1.4, 1.4), alpha=False).save(out / f"page_{i+1:02}.png")
    (out / "extracted_text.txt").write_text("\n\n".join(all_text), encoding="utf-8")
    for start in range(0, len(doc), 6):
        contact = Image.new("RGB", (1020, 1500), "#d4d4d4")
        draw = ImageDraw.Draw(contact)
        for j, index in enumerate(range(start, min(start+6, len(doc)))):
            with Image.open(out / f"page_{index+1:02}.png") as im:
                im.thumbnail((315, 445))
                x, y = (j % 2)*510+90, (j//2)*500+30
                contact.paste(im, (x, y))
                draw.text((x, y-18), f"Page {index+1}", fill="black")
        contact.save(out / f"contact_{start//6+1}.png")
    result = {"status": "PASS", "pages": len(doc), "checks": len(checks), "all_checks": checks,
              "source_sha256": digest(ENGLISH / "document.tex"), "pdf_sha256": digest(ENGLISH / "document.pdf"),
              "equations": 25, "figures": 10, "tables": 11, "references": 12,
              "algebra_checks": algebra.CHECKS, "compiler_severe_diagnostics": severe,
              "visual_review": "pending", "scope": "translation, redraw, algebra checks, and frozen-result readback; no localization or training"}
    (out / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "pages": len(doc), "checks": len(checks), "algebra_checks": len(algebra.CHECKS)}))


if __name__ == "__main__":
    main()
