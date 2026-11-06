# -*- coding: utf-8 -*-
"""把各精读批次产出的 fragment 合并为「文献精读结果.csv」。

输入：_frag/R##.md（字段行以 @@字段名: 开头）
输出：F:\\SourceCount_DPD_E2E\\文献精读结果.csv（UTF-8 with BOM / utf-8-sig）

编码说明：Excel 在中文 Windows 下按系统 ANSI 代码页（GBK）解析无 BOM 的
UTF-8 文件，会导致中文乱码，故显式写入 BOM。Python 侧如需读取请用
encoding="utf-8-sig"。
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
# 用法：merge_csv.py [fragment目录名] [输出文件名]
#   默认 _frag_slim -> 文献精读结果.csv（白话精简版）
#   merge_csv.py _frag 文献精读结果_详版.csv 可重建详细版
FRAG_DIR = sys.argv[1] if len(sys.argv) > 1 else "_frag_slim"
OUT_NAME = sys.argv[2] if len(sys.argv) > 2 else "文献精读结果.csv"
FRAG = BASE / FRAG_DIR
OUT = Path(r"F:\SourceCount_DPD_E2E") / OUT_NAME

COLUMNS = [
    "编号", "标题", "作者年份", "一句话概括", "研究对象与场景",
    "信号与观测模型", "是否已知源数", "核心方法", "对混合IQ的适用性",
    "实验条件", "主要结果", "可复现性", "论文结论与支持边界",
    "主要贡献", "主要局限", "总体评价", "与本研究的关系", "证据定位",
]

FIELD_RE = re.compile(r"^@@([^:：]+)[:：]\s?(.*)$")


def parse_fragment(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    cur: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        m = FIELD_RE.match(line)
        if m:
            if cur is not None:
                fields[cur] = "\n".join(buf).strip()
            cur, first = m.group(1).strip(), m.group(2)
            buf = [first] if first else []
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        fields[cur] = "\n".join(buf).strip()
    return fields


def main() -> int:
    expected = [f"R{i:02d}" for i in range(1, 38)]
    rows: list[dict[str, str]] = []
    problems: list[str] = []

    for r in expected:
        path = FRAG / f"{r}.md"
        if not path.is_file():
            problems.append(f"{r}: fragment 文件缺失")
            continue
        fields = parse_fragment(path.read_text(encoding="utf-8"))
        missing = [c for c in COLUMNS if c not in fields or not fields[c]]
        if missing:
            problems.append(f"{r}: 空字段 {missing}")
        row = {c: fields.get(c, "") for c in COLUMNS}
        row["编号"] = r
        for c in COLUMNS:
            row[c] = row[c].replace("\r\n", "\n").replace("\r", "\n").strip()
        rows.append(row)

    rows.sort(key=lambda x: int(x["编号"][1:]))

    with OUT.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    print(f"rows written : {len(rows)} / 37")
    print(f"output       : {OUT}")
    print(f"size         : {OUT.stat().st_size / 1024:.1f} KB")
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print("  -", p)
    else:
        print("PROBLEMS     : none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
