# -*- coding: utf-8 -*-
"""将 37 篇核心文献 PDF 逐页渲染为 PNG，供图像级精读。

输入映射：_rmap.json（R编号 -> Zotero 附件 key）
输出：_pdfrender/<R编号>/page_XX.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import fitz

DPI = 170
BASE = Path(__file__).resolve().parent
STORAGE = Path(r"F:\Zotero\storage")


def main() -> int:
    rmap = json.loads((BASE / "_rmap.json").read_text(encoding="utf-8"))
    missing: list[str] = []
    total_pages = 0
    total_bytes = 0

    for r in sorted(rmap, key=lambda x: int(x[1:])):
        src_dir = STORAGE / rmap[r]
        pdfs = sorted(src_dir.glob("*.pdf")) if src_dir.is_dir() else []
        if not pdfs:
            missing.append(r)
            continue
        out_dir = BASE / r
        out_dir.mkdir(parents=True, exist_ok=True)
        if list(out_dir.glob("page_*.png")):
            continue
        with fitz.open(pdfs[0]) as doc:
            for i, page in enumerate(doc, start=1):
                png = out_dir / f"page_{i:02d}.png"
                page.get_pixmap(dpi=DPI).save(png)
                total_pages += 1
                total_bytes += png.stat().st_size

    print(f"rendered pages: {total_pages}")
    print(f"rendered MB   : {total_bytes / 1024 / 1024:.1f}")
    print(f"missing       : {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
