"""Preserve local reading evidence; no localization or external service writes."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import fitz


ROOT = Path(__file__).resolve().parent
PREFLIGHT = Path(
    "C:/Users/Administrator/.codex/skills/academic-research-suite/ars/scripts/pdf_read_preflight.py"
)
SOURCES = [
    ("ma2020", "LH5XMWUT", "736HDRY5", None),
    ("ding2026", "XM4DFU6E", "MU2TQ947", [1, 2, 3, 4, 5]),
]


def main():
    records = []
    for label, item, attachment, selected in SOURCES:
        paths = list((Path("F:/Zotero/storage") / attachment).glob("*.pdf"))
        if len(paths) != 1:
            raise ValueError(f"Ambiguous PDF {attachment}: {len(paths)}")
        source = paths[0]
        folder = ROOT / "admission_sources" / label
        folder.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [sys.executable, "-X", "utf8", str(PREFLIGHT), str(source)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
            env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        )
        # The upstream --output path needs POSIX dirfd. On Windows use its
        # supported stdout mode and preserve the generated structural report.
        integrity = json.loads(result.stdout)
        (folder / "pdf_integrity.json").write_text(
            json.dumps(integrity, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with fitz.open(source) as doc:
            pages = selected or list(range(1, len(doc) + 1))
            text = []
            for number in pages:
                page = doc[number - 1]
                text.append(f"\n--- PDF PAGE {number} ---\n{page.get_text(sort=True)}")
                page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6)).save(folder / f"page_{number}.png")
            (folder / "reading_text.txt").write_text("".join(text), encoding="utf-8")
            records.append({
                "slug": label, "item_key": item, "attachment_key": attachment,
                "path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "total_pages": len(doc), "extracted_pages": pages,
                "scope": "full text and selected equation/figure checks" if selected is None
                else "model/method sections only",
                "preflight_stdout": result.stdout.strip(),
            })
    (ROOT / "admission_source_identity.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
