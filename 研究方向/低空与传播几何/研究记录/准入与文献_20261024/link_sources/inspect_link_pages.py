"""Read-only source audit: selected PDF model pages, no localization runs."""
import hashlib
import json
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent
SOURCES = [
    ("weiss2005", "ID3FDGNH", "VZAL7S46", [2, 3]),
    ("you2026", "PHHWLZDF", "HNQIFEEA", [3, 6, 7]),
    ("chai2025", "CX82UIQA", "7LV7BYNC", [3, 5, 8]),
    ("hao2021", "2ENUNLHX", "ZUEMTL5U", [1, 2]),
    ("jin2025", "368F6YF3", "PI66HFU4", [3, 4, 5]),
    ("lin2026", "4KD3RU7N", "DEDESABG", [3]),
]


def main():
    manifest = []
    for label, item, attachment, pages in SOURCES:
        files = list((Path("F:/Zotero/storage") / attachment).glob("*.pdf"))
        if len(files) != 1:
            raise RuntimeError(f"Expected exactly one PDF: {attachment}")
        source = files[0]
        target = ROOT / label
        target.mkdir(parents=True, exist_ok=True)
        with fitz.open(source) as doc:
            blocks = []
            for number in pages:
                page = doc[number - 1]
                blocks.append(f"\n--- PDF PAGE {number} ---\n{page.get_text(sort=True)}")
                page.get_pixmap(matrix=fitz.Matrix(1.4, 1.4)).save(target / f"page_{number}.png")
            (target / "model_pages.txt").write_text("".join(blocks), encoding="utf-8")
            manifest.append({"label": label, "item_key": item, "attachment_key": attachment,
                             "source": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                             "total_pages": len(doc), "selected_pages": pages,
                             "scope": "model and method clauses only; not full proof replication"})
    (ROOT / "sources.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
