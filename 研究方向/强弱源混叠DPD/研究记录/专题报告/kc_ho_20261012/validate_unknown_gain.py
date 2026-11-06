"""Document and algebra QA; never generates localization-performance evidence."""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import re

import fitz
import numpy as np
from PIL import Image, ImageDraw

from restructure_unknown_gain import scientific_blocks

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equations(text):
    return {re.search(r"\\label\{([^}]+)\}", block)[1]: re.sub(r"\s+", "", block)
            for block in re.findall(r"\\begin\{equation\}.*?\\end\{equation\}", text, re.S)}


def algebra_checks():
    rng = np.random.default_rng(20261013)
    residuals = []
    for _ in range(12):
        n, m = 4, 29
        x = rng.normal(size=(n, m)) + 1j * rng.normal(size=(n, m))
        phase = np.exp(-1j * rng.normal(size=(n, m)))
        y = phase.conj() * x
        values, vectors = np.linalg.eigh(y @ y.conj().T)
        b = vectors[:, -1]
        s = b.conj() @ y
        exact = np.linalg.norm(x - phase * b[:, None] * s)**2
        eigen = np.linalg.norm(x)**2 - values[-1]
        residuals.append(float(abs(exact - eigen)))
        # The same identity applies to a block residual after subtracting fitted sources.
        a = rng.normal(size=(n, 2)) + 1j * rng.normal(size=(n, 2))
        z = x[:, 0]
        direct = np.linalg.norm(z - a @ (np.linalg.pinv(a) @ z))**2
        proj = np.eye(n) - a @ np.linalg.pinv(a)
        residuals.append(float(abs(direct - np.vdot(z, proj @ z).real)))
        p = np.eye(n) - np.outer(a[:, 0], a[:, 0].conj()) / np.vdot(a[:, 0], a[:, 0])
        u = p @ a[:, 1]
        reduction = abs(np.vdot(u, p @ z))**2 / np.vdot(u, u).real
        residuals.append(float(abs(np.linalg.norm(p @ z)**2 - direct - reduction)))
        # Complementary MUSIC projectors have the stated eigenvalue relation.
        q, _ = np.linalg.qr(a)
        ps = q @ q.conj().T
        d = np.diag(phase[:, 0])
        noise = d.conj().T @ (np.eye(n) - ps) @ d
        signal = d.conj().T @ ps @ d
        residuals.append(float(abs(np.linalg.eigvalsh(noise)[0] - (1-np.linalg.eigvalsh(signal)[-1]))))
    return {"kind": "algebraic identity checks only", "count": len(residuals),
            "max_absolute_error": max(residuals), "passed": max(residuals) < 1e-10}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="qa_unknown_gain_01")
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(exist_ok=True)
    migration = json.loads((ROOT / "unknown_gain_migration.json").read_text(encoding="utf-8"))
    checks = []
    editions = {}
    new_eqs = {}

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})

    for lang, dirname in [("zh", "Latex PDF"), ("en", "Latex PDF English")]:
        folder = ROOT / dirname
        tex = (folder / "document.tex").read_text(encoding="utf-8")
        body = (folder / "unknown_gain_main.tex").read_text(encoding="utf-8")
        expanded = tex.replace(r"\input{unknown_gain_main.tex}", body)
        legacy = tex.split(r"\appendix", 1)[1]
        check(lang+"_legacy_scientific_blocks", scientific_blocks(legacy) == migration["editions"][lang]["legacy_scientific_blocks"])
        for name, expected in migration["editions"][lang]["figure_assets"].items():
            check(lang+"_asset_"+Path(name).name, sha(ROOT/name) == expected)
        labels = re.findall(r"\\label\{([^}]+)\}", expanded)
        refs = re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", expanded)
        check(lang+"_unique_labels", len(set(labels)) == len(labels))
        check(lang+"_resolved_refs", set(refs) <= set(labels))
        check(lang+"_balanced_environments", Counter(re.findall(r"\\begin\{([^}]+)\}", expanded)) == Counter(re.findall(r"\\end\{([^}]+)\}", expanded)))
        bib = re.findall(r"\\bibitem\{([^}]+)\}", expanded)
        cites = []
        for citation in re.findall(r"\\cite\{([^}]+)\}", expanded):
            for key in citation.split(","):
                if key not in cites:
                    cites.append(key)
        check(lang+"_ieee_citation_order", cites == bib and len(bib) == 14)
        check(lang+"_empty_date_no_contents", r"\date{}" in expanded and r"\tableofcontents" not in expanded)
        check(lang+"_no_internal_stage_in_report", not re.search(r"G[89]-R[123456]|outputs_e2e|作者稿|author.s manuscript|codex-annotation", re.sub(r"(?m)^%.*$", "", expanded)))
        check(lang+"_no_experimental_result_in_new_main", "8.04" not in body and "40/48" not in body and "83.33" not in body)
        if lang == "en":
            check("english_no_cjk", not re.search(r"[\u4e00-\u9fff]", expanded))
        new_eqs[lang] = equations(body)
        log = (folder/"document.log").read_text(encoding="utf-8", errors="replace")
        bad = re.findall(r"(?:Overfull[^\n]*|LaTeX Warning:[^\n]*|Missing character:[^\n]*|Package tabularx Warning:[^\n]*)", log)
        check(lang+"_clean_compile", not bad)
        pdf = fitz.open(folder/"document.pdf")
        target = out/lang
        target.mkdir(exist_ok=True)
        geometry = []
        thumbs = []
        for num, page in enumerate(pdf, 1):
            pix = page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2), alpha=False)
            pix.save(target/f"page_{num:02d}.png")
            thumb = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            thumb.thumbnail((298, 422))
            card = Image.new("RGB", (310, 450), "#e1e3e5")
            card.paste(thumb, ((310-thumb.width)//2, 18))
            ImageDraw.Draw(card).text((8, 433), f"{lang.upper()} / {num}", fill="black")
            thumbs.append(card)
            blocks = [b for b in page.get_text("blocks") if b[6] == 0 and not re.fullmatch(r"\s*\d+\s*", b[4])]
            geometry.append({"page": num, "text_bottom": max((b[3] for b in blocks), default=0),
                             "text_top": min((b[1] for b in blocks), default=0),
                             "outside_page": [list(b[:4]) for b in blocks if b[0] < 0 or b[1] < 0 or b[2] > page.rect.width+1 or b[3] > page.rect.height+1]})
        contacts = []
        for start in range(0, len(thumbs), 12):
            sheet = Image.new("RGB", (310*3, 450*4), "white")
            for j, im in enumerate(thumbs[start:start+12]):
                sheet.paste(im, ((j%3)*310, (j//3)*450))
            dest = target/f"contact_{start//12+1}.jpg"
            sheet.save(dest, quality=90)
            contacts.append(str(dest))
        check(lang+"_page_geometry", all(not p["outside_page"] for p in geometry))
        editions[lang] = {"pages": len(pdf), "numbered_equations": len(equations(expanded)),
                          "figures": expanded.count(r"\begin{figure}"), "tables": expanded.count(r"\begin{table}"),
                          "labels": labels, "geometry": geometry, "compile_issues": bad, "contact_sheets": contacts,
                          "files": {str(p.relative_to(ROOT)): sha(p) for p in [folder/"document.tex", folder/"unknown_gain_main.tex", folder/"document.pdf"]}}
    check("bilingual_numbered_formulas_identical", new_eqs["zh"] == new_eqs["en"])
    check("bilingual_labels_identical", editions["zh"]["labels"] == editions["en"]["labels"])
    algebra = algebra_checks()
    check("algebra_identities", algebra["passed"])
    result = {"scope": "LaTeX, preservation, bilingual consistency, algebra and rendering only; no localization experiment",
              "checks": checks, "algebra": algebra, "editions": editions,
              "passed": all(c["passed"] for c in checks), "visual_review": "pending"}
    (out/"validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "checks": len(checks), "failed": [c["name"] for c in checks if not c["passed"]], "pages": {k:v["pages"] for k,v in editions.items()}, "algebra": algebra}, ensure_ascii=False))


if __name__ == "__main__":
    main()
