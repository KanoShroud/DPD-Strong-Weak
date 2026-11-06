"""Extract and render selected model pages; never change Zotero or run solvers."""

import hashlib
import json
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parent
SOURCES = [
    ("weiss2005", "ID3FDGNH", "VZAL7S46", [2, 3]),
    ("weiss2011", "EVWD6GVH", "WQ22JB95", [1, 2, 3]),
    ("zhu2026", "VMJIU7GR", "SGRDEA47", [2, 3]),
    ("liang2026", "ZVZG57W7", "JYBBF8BS", [4, 5, 6]),
    ("hou2026", "7K4ELLJH", "RCHXV7WG", [1, 2]),
]


def main():
    records = []
    for name, item, attachment, pages in SOURCES:
        matches = list((Path("F:/Zotero/storage") / attachment).glob("*.pdf"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one PDF: {attachment}; got {len(matches)}")
        source = matches[0]
        folder = ROOT / "sources" / name
        folder.mkdir(parents=True, exist_ok=True)
        with fitz.open(source) as doc:
            text_parts = []
            for number in pages:
                page = doc[number - 1]
                text_parts.append(f"\n--- PDF PAGE {number} ---\n{page.get_text(sort=True)}")
                page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(
                    folder / f"page_{number}.png"
                )
            (folder / "model_pages.txt").write_text("".join(text_parts), encoding="utf-8")
            records.append({
                "label": name,
                "item_key": item,
                "attachment_key": attachment,
                "path": str(source),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "total_pages": len(doc),
                "extracted_pdf_pages": pages,
                "scope": "selected model pages, not a full-paper mathematical audit",
            })
    (ROOT / "sources.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
