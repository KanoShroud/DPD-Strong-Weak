# -*- coding: utf-8 -*-
"""按综述 2.4 节的推荐精读顺序重排 文献精读结果.csv 的行。

只重排行，不改列（列名、列序、列数均保持现状）。
保留当前文件的编码（UTF-8 with BOM）与引号风格。
"""
from __future__ import annotations

import csv
import io
import shutil
import sys
from pathlib import Path

CSV = Path(r"F:\SourceCount_DPD_E2E\文献精读结果.csv")
BACKUP = CSV.with_name("文献精读结果_重排前备份.csv")

# 综述 2.4 节「精读顺序」1—37
ORDER = [
    # 2.4.1 最相关：先判断它们是否已覆盖我们要解决的问题
    "R02", "R07", "R33", "R32", "R27", "R04", "R30", "R19",
    # 2.4.2 比较相关：场景不同，但核心思想不能重复主张
    "R14", "R17", "R08", "R37", "R11", "R25", "R18", "R21", "R09", "R22",
    "R34", "R03", "R01", "R28", "R23", "R20", "R31", "R12", "R05", "R16",
    "R10", "R24",
    # 2.4.3 相关性较低：了解边界后按需拓展
    "R29", "R35", "R36", "R06", "R15", "R26", "R13",
]


def detect_quoteall(raw_text: str, fieldnames: list[str]) -> bool:
    """若表头每个字段都被引号包裹，则沿用 QUOTE_ALL。"""
    header = raw_text.splitlines()[0]
    return all(f'"{f}"' in header for f in fieldnames)


def main() -> int:
    assert len(ORDER) == 37 and len(set(ORDER)) == 37, "顺序表必须覆盖 37 篇且不重复"

    raw = CSV.read_bytes()
    has_bom = raw[:3] == b"\xef\xbb\xbf"
    text = raw.decode("utf-8-sig")

    rows = list(csv.DictReader(io.StringIO(text)))
    cols = list(rows[0].keys())
    by_id = {r["编号"]: r for r in rows}
    missing = [r for r in ORDER if r not in by_id]
    if missing:
        print("缺少编号:", missing)
        return 1

    quoteall = detect_quoteall(text, cols)

    shutil.copy2(CSV, BACKUP)
    print("已备份 ->", BACKUP.name)

    out = io.StringIO()
    writer = csv.DictWriter(
        out, fieldnames=cols,
        quoting=csv.QUOTE_ALL if quoteall else csv.QUOTE_MINIMAL,
    )
    writer.writeheader()
    for rid in ORDER:
        writer.writerows([by_id[rid]])

    CSV.write_text(out.getvalue(), encoding="utf-8-sig" if has_bom else "utf-8", newline="")

    print("引号风格 :", "QUOTE_ALL" if quoteall else "QUOTE_MINIMAL")
    print("BOM      :", "保留" if has_bom else "无")
    print("列       :", len(cols), "列，列名与顺序未改动")
    print("行顺序   :", " ".join(ORDER))
    return 0


if __name__ == "__main__":
    sys.exit(main())
