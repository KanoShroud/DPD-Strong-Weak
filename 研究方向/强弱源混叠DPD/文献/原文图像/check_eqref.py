# -*- coding: utf-8 -*-
"""定位「式(N)」引用出现在哪些字段，并统计全/半角括号两种写法。"""
from __future__ import annotations

import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
SLIM = BASE / "_frag_slim"

FIELD_RE = re.compile(r"^@@([^:：]+)[:：]\s?(.*)$")
HALF = re.compile(r"式\(\d+\)")        # 式(12)
FULL = re.compile(r"式（\d+）")        # 式（12）

where: dict[str, set[str]] = {}
half_total = full_total = 0

for p in sorted(SLIM.glob("R*.md")):
    cur: str | None = None
    for line in p.read_text(encoding="utf-8").splitlines():
        m = FIELD_RE.match(line)
        if m:
            cur, body = m.group(1).strip(), m.group(2)
        else:
            body = line
        if cur is None:
            continue
        nh, nf = len(HALF.findall(body)), len(FULL.findall(body))
        half_total += nh
        full_total += nf
        if nh or nf:
            where.setdefault(cur, set()).add(p.stem)

print(f"half-width 式(N)  : {half_total}")
print(f"full-width 式（N）: {full_total}")
print()
print("按字段统计：")
if not where:
    print("  （无）")
for k in sorted(where, key=lambda x: -len(where[x])):
    print(f"  {k:<16} {len(where[k]):>2}篇: " + ", ".join(sorted(where[k])))
