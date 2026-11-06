# -*- coding: utf-8 -*-
"""扫描白话精简版中的公式残留与篇幅分布。"""
from __future__ import annotations

import re
import statistics as st
from pathlib import Path

BASE = Path(__file__).resolve().parent
SLIM = BASE / "_frag_slim"

PATTERNS = [
    (r"\$[^$\n]{2,}\$", "LaTeX 行内公式"),
    (r"\^\{?[H*T]\b", "上标算子 ^H / ^* / ^T"),
    (r"arg\s*(?:min|max)", "arg min / arg max"),
    (r"[Σ∑∫√]", "数学算子符号"),
    (r"式\(\d+\)", "公式编号引用"),
    (r"\b[A-Za-zΑ-ω][A-Za-z0-9]*_\{[A-Za-z0-9]+\}", "带下标变量"),
    (r"\\frac|\\sum|\\arg", "LaTeX 命令"),
]

FIELD_RE = re.compile(r"^@@([^:：]+)[:：]\s?(.*)$")


def main() -> int:
    files = sorted(p for p in SLIM.glob("R*.md"))
    print(f"slim fragments: {len(files)}/37")
    print()

    hits: dict[str, list[tuple[str, int]]] = {}
    for p in files:
        t = p.read_text(encoding="utf-8")
        for pat, name in PATTERNS:
            n = len(re.findall(pat, t))
            if n:
                hits.setdefault(name, []).append((p.stem, n))

    if hits:
        print("公式残留：")
        for name, v in sorted(hits.items(), key=lambda kv: -sum(n for _, n in kv[1])):
            tot = sum(n for _, n in v)
            print(f"  {name:<18} 共{tot:>4}次 / {len(v):>2}篇  " +
                  ", ".join(f"{r}({n})" for r, n in sorted(v)[:10]))
    else:
        print("公式残留：未发现")

    print()
    print("篇幅：")
    tot_lens = []
    for p in files:
        n = len(p.read_text(encoding="utf-8"))
        tot_lens.append(n)
        if n > 5500:
            print(f"  偏长 {p.stem}: {n} 字")
    print(f"  min {min(tot_lens)} | median {int(st.median(tot_lens))} | max {max(tot_lens)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
