# -*- coding: utf-8 -*-
"""最终 CSV 全面校验：编码、结构、完整、篇幅、公式残留、身份。"""
from __future__ import annotations

import csv
import re
import statistics as st
from pathlib import Path

CSV = Path(r"F:\SourceCount_DPD_E2E\文献精读结果.csv")

FORMULA = [
    (r"\$[^$\n]{2,}\$", "LaTeX 公式"),
    (r"\^\{?[H*T]\b", "上标算子"),
    (r"arg\s*(?:min|max)", "arg min/max"),
    (r"[Σ∑∫√⊗]", "数学算子"),
    (r"式\(\d+\)", "式(N) 编号引用"),
    (r"\\[a-zA-Z]{2,}", "LaTeX 命令"),
]

EXPECT_TITLE = {
    "R01": "multiple radio signals", "R02": "high resolution direct position",
    "R03": "frequency-hopping", "R04": "cyclic cross-spectral",
    "R05": "without prior knowledge of source number",
    "R06": "rotating single-baseline interferometer", "R07": "parametric dictionary learning",
    "R08": "generalized correlation method", "R09": "data association",
    "R10": "clutter", "R11": "ghost discrimination", "R12": "neural network computation",
    "R13": "off-grid", "R14": "variable numbers", "R15": "radio map",
    "R16": "shift equivariant", "R17": "subspacenet", "R18": "envelope separation",
    "R19": "multiple-frequency fusion", "R20": "parallel factor analysis",
    "R21": "optimal transport", "R22": "pairwise neural cross-correlation",
    "R23": "sparse spectrum", "R24": "riemannian manifold", "R25": "source splitting",
    "R26": "transfer learning", "R27": "mmse-based", "R28": "bandpass signal design",
    "R29": "propagation loss", "R30": "multi-bernoulli", "R31": "parallel factor analysis",
    "R32": "efficiency-improved", "R33": "multiple-frequency function fusion",
    "R34": "sparse bayesian learning", "R35": "importance sampling",
    "R36": "without time synchronization", "R37": "uav cluster",
}


def main() -> int:
    raw = CSV.read_bytes()
    print("=" * 58)
    print("文件 :", CSV)
    print("大小 : %.1f KB" % (len(raw) / 1024))
    print("BOM  :", "有 (EF BB BF) -> Excel 可正确识别 UTF-8" if raw[:3] == b"\xef\xbb\xbf" else "无 -> Excel 会乱码")

    rows = list(csv.DictReader(CSV.open(encoding="utf-8-sig", newline="")))
    cols = list(rows[0].keys())
    print("结构 : %d 行 x %d 列" % (len(rows), len(cols)))

    ids = [r["编号"] for r in rows]
    print("编号 :", "R01–R37 连续" if ids == [f"R{i:02d}" for i in range(1, 38)] else f"异常 {ids}")

    empty = [(r["编号"], c) for r in rows for c in cols if not r[c].strip()]
    print("空值 :", "无" if not empty else empty)

    bad_title = [r["编号"] for r in rows if EXPECT_TITLE[r["编号"]] not in r["标题"].lower()]
    print("身份 :", "37/37 标题与论文一致" if not bad_title else f"不匹配 {bad_title}")

    # 公式残留（按列统计）
    hits: dict[str, list[str]] = {}
    for r in rows:
        for c in cols:
            for pat, name in FORMULA:
                if re.search(pat, r[c]):
                    hits.setdefault(name, []).append(f'{r["编号"]}:{c}')
    print()
    print("公式残留：")
    if not hits:
        print("  无")
    for name, v in hits.items():
        locs = sorted(set(x.split(":")[1] for x in v))
        print(f"  {name:<16} {len(v):>3}处，涉及列: {', '.join(locs)}")

    # 篇幅（整行总字符数 = 各字段之和 + 字段名开销）
    lens = sorted(sum(len(r[c]) for c in cols) for r in rows)
    print()
    print("篇幅（整行总字符数）：")
    print("  min %d | P25 %d | median %d | P75 %d | max %d"
          % (lens[0], lens[9], int(st.median(lens)), lens[27], lens[-1]))
    over = [r["编号"] for r in rows if sum(len(r[c]) for c in cols) > 2500]
    under = [r["编号"] for r in rows if sum(len(r[c]) for c in cols) < 1800]
    print("  超 2500:", over if over else "无")
    print("  低 1800:", under if under else "无")

    print()
    print("各列字符数（min/median/max）：")
    for c in cols:
        L = sorted(len(r[c]) for r in rows)
        print(f"  {c:<18} {L[0]:>5} {L[18]:>6} {L[-1]:>6}")
    print("=" * 58)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
