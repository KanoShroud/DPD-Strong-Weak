"""Read frozen estimates/statistics and redraw the editable LaTeX report.

No signal generation, solver calls, training or formal evaluation. Output must
be a new report-analysis directory; --install replaces only the report PDFs.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import itertools
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "outputs_e2e/unified/e2e_g9"
EVAL = BASE / "eval_2pctR_20261011_01"
FORMAL = BASE / "formal_20261011_repaired"
R1 = BASE / "r1_failure_20261011_01"
FRESH = BASE / "r2_coverage_fresh_20261011_01"
INPUTS = {}
REGISTERED = {}
COLORS = ["#AEBEC8", "#BCC9BD", "#DDCCB4", "#C9C3CE"]
INK = ["#405E72", "#46634F", "#86663E", "#68576E"]
HATCHES = ["/", ".", "\\", ""]
MARKERS = ["o", "s", "^", "D"]
plt.rcParams.update({"font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
    "axes.unicode_minus": False, "font.size": 10, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "svg.fonttype": "none", "hatch.linewidth": .45,
    "savefig.facecolor": "white"})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def track(path):
    path = path.resolve(strict=True)
    assert path.is_relative_to(ROOT / "outputs_e2e/unified"), path
    digest = sha(path)
    if path in REGISTERED:
        assert digest == REGISTERED[path], f"Frozen identity mismatch: {path}"
    INPUTS[str(path)] = digest
    return path


def read(path):
    return json.loads(track(path).read_text(encoding="utf-8"))


def flags(truth, pred, radius=20):
    """Maximum-cardinality one-to-one radius matching, then squared distance."""
    a, b = np.asarray(truth).reshape(-1, 2), np.asarray(pred).reshape(-1, 2)
    d = np.linalg.norm(a[:, None] - b[None], axis=-1)
    matches = [()]
    for k in range(1, min(len(a), len(b)) + 1):
        for ii in itertools.combinations(range(len(a)), k):
            for jj in itertools.permutations(range(len(b)), k):
                pairs = tuple(zip(ii, jj))
                if all(d[i, j] <= radius for i, j in pairs):
                    matches.append(pairs)
    best = min(matches, key=lambda p: (-len(p), sum(d[i, j]**2 for i, j in p), p))
    return [any(i == x for x, _ in best) for i in range(len(a))]


def load_data():
    manifest = read(EVAL / "input_manifest.json")
    REGISTERED.update({(ROOT / p).resolve(): h for p, h in manifest.items()})
    for directory in [FORMAL, R1, FRESH, EVAL]:
        audit = read(directory / "final_audit_report.json")
        assert audit["status"] == "PASS"
        if "report" in audit:
            REGISTERED[Path(audit["report"]["path"]).resolve()] = audit["report"]["sha256"]
    report = read(EVAL / "report.json")
    old_report = read(FORMAL / "comparison_report.json")
    old_rows = read(FORMAL / "evaluation_rows.json")
    compact = read(EVAL / "formal_rows.json")
    fields = ("group", "realization", "overlap", "gap", "N", "kind", "method", "factor")
    index = {tuple(r[k] for k in fields): r for r in old_rows}
    assert len(index) == len(old_rows) == len(compact) == 15360
    # The v2 exporter iterated sorted source files and (.5, 1, 2) snapshots.
    # Reattach condition fields using those complete case keys, not the
    # ambiguous (layout, realization, length) triple in compact records.
    rows = []
    cursor = 0
    for p in sorted((FORMAL / "rows").glob("*.json")):
        raw = read(p)
        for factor in (.5, 1., 2.):
            v2 = compact[cursor]
            cursor += 1
            key = tuple(raw[k] if k != "factor" else factor for k in fields)
            old = index[key]
            for k in ("group", "realization", "N", "kind", "method", "factor"):
                assert old[k] == v2[k], (p, k)
            assert old["metrics"]["errors_m"] == v2["metrics"]["errors_m"], p
            assert abs(old["metrics"]["gospa_m"] - v2["metrics"]["gospa_m"]) < 1e-9
            rows.append({**old, "metrics": v2["metrics"]})
    assert cursor == len(compact)
    scores = [read(p) for p in sorted((R1 / "score_rows").glob("*.json"))]
    dense = [read(p) for p in sorted((R1 / "dense_rows").glob("*.json"))]
    failed = [r for r in scores if not flags(r["truth"], r["predicted"])[1]]
    def counts(ss):
        return {"failures": len(ss),
            "estimated_truth_higher": sum(r["estimated"]["truth_higher"] for r in ss),
            "true_strong_truth_higher": sum(r["true_strong"]["truth_higher"] for r in ss),
            "joint_truth_higher": sum(r["truth_pair_score"] > r["returned_pair_score"] + 1e-10 for r in ss),
            "estimated_good_oracle_higher": sum(r["estimated"]["oracle_higher"] and r["estimated"]["oracle_error_m"] <= 20 for r in ss)}
    attribution = {str(n): counts([r for r in failed if r["N"] == n]) for n in (4096, 16384)}
    attribution["all"] = counts(failed)
    assert len(failed) == report["r1_all"]["failures20"]
    for k in ("estimated_truth_higher", "true_strong_truth_higher"):
        assert attribution["all"][k] == report["r1_all"][k]
    rescued_anchor = [r for r in failed if not r["estimated"]["truth_higher"] and r["true_strong"]["truth_higher"]]
    attribution["anchor_rescued"] = len(rescued_anchor)
    attribution["anchor_rescued_with_strong20"] = sum(r["strong_error_m"] <= 20 for r in rescued_anchor)
    fresh = [read(p) for p in sorted((FRESH / "rows").glob("*.json"))]
    read(FRESH / "report.json")
    config = read(FRESH / "contract.json")["config"]
    # Fresh rows have fixed labels, and both methods' strong errors < 1 m.
    # With source separation >= 300 m the 20 m truth balls are disjoint;
    # verify both the geometry and optimal error assignment before flags.
    rx = np.stack([500*np.cos(np.arange(4)*np.pi/2), 500*np.sin(np.arange(4)*np.pi/2)], 1).astype(np.float32)
    for r in fresh:
        rng = np.random.default_rng(np.random.SeedSequence([2026101009, 1, r["group"], 0]))
        while True:
            truth = rng.uniform(-1000, 1000, (2, 2))
            if np.linalg.norm(truth[:, None]-rx, axis=-1).min() >= 150 and np.linalg.norm(truth[0]-truth[1]) >= 300:
                break
        for m in ("B", "D"):
            distances = np.linalg.norm(truth[:, None]-np.asarray(r[m]["positions"])[None], axis=-1)
            perm = min(itertools.permutations(range(2)), key=lambda q: sum(distances[i, q[i]]**2 for i in range(2)))
            assert np.allclose([distances[i, perm[i]] for i in range(2)], r[m]["metrics"]["errors_m"], atol=1e-9, rtol=0)
            r[m]["flags20"] = flags(truth, r[m]["positions"])
    layout = [{"layout": i+1, "source_group": g,
               **{m: sum(r[m]["flags20"][1] for r in fresh if r["group"] == g) for m in ("B", "D")}}
              for i, g in enumerate(config["groups"])]
    failures = [r for r in fresh if not r["D"]["flags20"][1]]
    fresh_detail = {"layouts": layout,
        "rescued": sum(not r["B"]["flags20"][1] and r["D"]["flags20"][1] for r in fresh),
        "lost": sum(r["B"]["flags20"][1] and not r["D"]["flags20"][1] for r in fresh),
        "remaining_errors_m": [r["D"]["metrics"]["errors_m"][1] for r in failures]}
    for m in ("B", "D"):
        assert sum(r[m]["flags20"][1] for r in fresh) == report["r2_fresh"][m]["weak_20"]["successes"]
        assert abs(np.mean([r[m]["seconds"] for r in fresh]) - report["r2_fresh"][m]["seconds_mean"]) < 1e-9
    dev = read(BASE / "r2_coverage_dev_20261011_01/report.json")
    shortlist = read(BASE / "r2_coverage_shortlist_dev_20261011_01/report.json")
    return report, old_report, rows, dense, attribution, fresh_detail, dev, shortlist


def save(fig, out, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / "figures" / f"{name}.{ext}", dpi=240, bbox_inches="tight")
    with Image.open(out / "figures" / f"{name}.png") as im:
        im.convert("L").save(out / "grayscale" / f"{name}.png")
    plt.close(fig)


def rounded(value, places=2):
    return str(Decimal(str(float(value))).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def bars(ax, x, values, i, label=None, width=.23, fmt="%.2f"):
    b = ax.bar(x, values, width, color=COLORS[i], edgecolor="#58616A",
               linewidth=.6, hatch=HATCHES[i], label=label, zorder=3)
    if fmt == "%.2f":
        ax.bar_label(b, labels=[rounded(v) for v in values], fontsize=8, padding=3)
    else:
        ax.bar_label(b, fmt=fmt, fontsize=8, padding=3)
    ax.grid(axis="y", color=".9", lw=.7, zorder=0)
    return b


def dots(ax, values, y, i, label=None, counts=None):
    ax.plot(values, y, ls="none", marker=MARKERS[i], ms=6.5, mfc="white",
            mec=INK[i], mew=1.3, label=label, zorder=3, clip_on=False)
    for j, (v, yy) in enumerate(zip(values, y)):
        txt = rounded(v) if counts is None else f"{rounded(v)}  ({counts[j]})"
        ax.annotate(txt, (v, yy), xytext=(7, 0), textcoords="offset points", va="center", fontsize=8)
    ax.grid(axis="x", color=".9", lw=.7)


def plot_all(out, report, old, rows, dense, attribution, fresh_detail):
    formal = report["formal"]
    def stat(method, n=None, factor=1.):
        return formal[f"mixed/{method}/{factor}" + (f"/N{n}" if n else "")]
    isolated = [formal[f"weak/B/1.0/N{n}"]["single_20"] for n in (4096, 16384)]
    vals = [[100*v["rate"] for v in isolated]] + [[100*stat(m, n)["weak_20"]["rate"] for n in (4096, 16384)] for m in ("B", "C")]
    fig, ax = plt.subplots(figsize=(9, 3.5), layout="constrained")
    labels = ["孤立弱源", "混合弱源 B", "混合弱源 C"]
    for i in range(3):
        bars(ax, np.arange(2)+(i-1)*.24, vals[i], i, labels[i])
    ax.set(xticks=[0, 1], xticklabels=["4096 点 / 40.96 μs", "16384 点 / 163.84 μs"], ylim=(0, 117), ylabel="弱源成功率（20 m）/ %")
    ax.legend(ncol=3, loc="upper center", frameon=False, fontsize=9)
    save(fig, out, "fig_3_1")
    fig, axs = plt.subplots(1, 2, figsize=(9, 2.6), layout="constrained")
    for col, n in enumerate((4096, 16384)):
        ax = axs[col]
        ax.plot([v[col] for v in vals], [2, 1, 0], color=".7", lw=.9)
        for i in range(3):
            count = isolated[col] if i == 0 else stat(("B", "C")[i-1], n)["weak_20"]
            dots(ax, [vals[i][col]], [2-i], i, counts=[f"{count['successes']}/{count['total']}"])
        ax.set(yticks=[2, 1, 0], yticklabels=labels, xlim=(0, 132), xticks=[0, 25, 50, 75, 100], ylim=(-.5, 2.5), xlabel="弱源成功率 / %", title=f"N = {n}")
    save(fig, out, "fig_3_1_alt")

    methods = ["DPD", "A", "B", "C"]
    names = ["普通 DPD", "A", "B", "C"]
    metrics = [("strong_20", "强源"), ("weak_20", "弱源"), ("both_20", "双源同时")]
    fig, ax = plt.subplots(figsize=(9, 3.5), layout="constrained")
    for i, (k, label) in enumerate(metrics):
        bars(ax, np.arange(4)+(i-1)*.24, [stat(m)[k]["rate"]*100 for m in methods], i, label)
    ax.set(xticks=range(4), xticklabels=names, ylim=(0, 117), ylabel="成功率（20 m）/ %")
    ax.legend(ncol=3, loc="upper center", frameon=False)
    save(fig, out, "fig_3_2")
    fig, axs = plt.subplots(1, 3, figsize=(9, 2.8), layout="constrained", sharey=True)
    for ax, (k, label) in zip(axs, metrics):
        for j, m in enumerate(methods):
            dots(ax, [stat(m)[k]["rate"]*100], [3-j], j)
        ax.set(yticks=range(4), yticklabels=names[::-1], xlim=(0, 117), xticks=[0, 25, 50, 75, 100], ylim=(-.5, 3.5), xlabel="成功率 / %", title=label)
    save(fig, out, "fig_3_2_alt")

    r6 = read(ROOT / "outputs_e2e/unified/e2e_g8/r6_failure_20261008_01/rows.json")
    example = next(r for r in r6 if r["group"] == 19 and r["overlap"] == .5)
    identity = example["A"]["map_identity"]
    path = track(Path(identity["path"]))
    assert sha(path) == identity["sha256"]
    with np.load(path) as z:
        zmap = next(z[k] for k in z.files if z[k].size == 201*201)
    fig, axs = plt.subplots(2, 2, figsize=(9, 6.5), layout="constrained")
    ax = axs[0, 0]
    im = ax.imshow(zmap.reshape(201, 201)/zmap.max(), origin="lower", extent=(-1000, 1000, -1000, 1000), cmap="cividis")
    ax.scatter(*example["truth"][1], edgecolors="white", marker="o", facecolors="none", s=100, label="真实弱源")
    ax.scatter(*example["A"]["position"], c="white", marker="x", s=50, label="密搜输出")
    ax.set(title="(a) 长记录示例：10 m 评分图", xlabel="x / m", ylabel="y / m")
    ax.legend(fontsize=8, facecolor=".2", labelcolor="white", framealpha=1)
    fig.colorbar(im, ax=ax, shrink=.75, label="评分 / 最大评分")
    aa = attribution["all"]
    for i, v in enumerate((aa["estimated_truth_higher"], aa["true_strong_truth_higher"])):
        bars(axs[0, 1], [i], [v], i, width=.55, fmt="%d")
    axs[0, 1].set(xticks=[0, 1], xticklabels=["固定原强源估计", "固定真实强源"], ylim=(0, 370), ylabel="真弱源评分更高的记录数", title=f"(b) 全部 {aa['failures']} 条弱源失败")
    selected = [r for r in dense if r["selection"] == "failure"]
    for i, key in enumerate(("estimated", "true_strong")):
        bars(axs[1, 0], [i], [sum(r[key]["error_m"] <= 20 for r in selected)], i, width=.55, fmt="%d")
    axs[1, 0].set(xticks=[0, 1], xticklabels=["固定原强源估计", "固定真实强源"], ylim=(0, 18), ylabel="恢复数 / 16 条选定失败", title="(c) 10 m 网格与前5个候选")
    narrow = next(r for r in dense if r["tag"] == "4_0_1.0_12_4096_mixed_B")["true_strong"]
    scores = [narrow["nearest_truth_grid_score"], narrow["score"], narrow["oracle"]["truth_score"], narrow["oracle"]["oracle_score"]]
    score_labels = ["最近格点", "密搜返回", "真实位置", "真值附近\n局部解"]
    for i, v in enumerate(scores):
        bars(axs[1, 1], [i], [v], i, width=.58, fmt="%.5f")
    axs[1, 1].set(xticks=range(4), xticklabels=score_labels, ylim=(0, .108), ylabel="归一化增量评分", title="(d) 窄峰案例的候选评分")
    save(fig, out, "fig_3_3")
    fig, ax = plt.subplots(figsize=(9, 2.2), layout="constrained")
    for i, v in enumerate(scores):
        ax.plot(v, 3-i, marker=MARKERS[i], mfc="white", color=INK[i], ms=7)
        ax.annotate(f"{v:.8f}", (v, 3-i), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9)
    ax.set(yticks=range(4), yticklabels=[s.replace("\n", "") for s in score_labels[::-1]], xlim=(.080, .094), ylim=(-.5, 3.5), xlabel="归一化增量评分", title="图3.3(d)备选：横向点图")
    ax.grid(axis="x", color=".9")
    save(fig, out, "fig_3_3_alt")

    cost = []
    fig, axs = plt.subplots(2, 2, figsize=(9, 5.7), layout="constrained")
    for col, n in enumerate((4096, 16384)):
        for i, m in enumerate(("A", "B", "C")):
            ss = [old["summary"][f"{n}_{m}_{factor}"]["seconds_mean"] for factor in (.5, 1., 2.)]
            rr = [stat(m, n, factor)["weak_20"]["rate"]*100 for factor in (.5, 1., 2.)]
            for row, values in enumerate((rr, ss)):
                bars(axs[row, col], np.arange(3)+(i-1)*.25, values, i, m, width=.23)
            cost.extend(dict(N=n, method=m, factor=f, seconds=s, rate=r) for f, s, r in zip((.5, 1., 2.), ss, rr))
        for row in range(2):
            axs[row, col].set(xticks=range(3), xticklabels=[r"$0.5T_0$", r"$T_0$", r"$2T_0$"], xlabel="计算预算", title=f"N = {n}")
        axs[0, col].set(ylim=(0, 100), ylabel="弱源成功率（20 m）/ %")
        axs[0, col].legend(ncol=3, frameon=False)
        axs[1, col].set(ylim=(0, 9 if col == 0 else 27), ylabel="预算内平均在线时间 / s")
    save(fig, out, "fig_3_5")
    fig, axs = plt.subplots(1, 4, figsize=(10, 4.5), layout="constrained")
    for col, n in enumerate((4096, 16384)):
        subset = [r for r in cost if r["N"] == n]
        labels = []
        for j, r in enumerate(subset):
            i = ("A", "B", "C").index(r["method"])
            labels.append(f"{r['method']} · {r['factor']:g}" + r"$T_0$")
            dots(axs[2*col], [r["rate"]], [8-j], i)
            dots(axs[2*col+1], [r["seconds"]], [8-j], i)
        for j in (2*col, 2*col+1):
            axs[j].set(yticks=range(9), yticklabels=labels[::-1] if j % 2 == 0 else [""]*9, ylim=(-.6, 8.6))
            for y in (2.5, 5.5):
                axs[j].axhline(y, color=".8", lw=.6)
        axs[2*col].set(xlim=(0, 56), xticks=[0, 20, 40], xlabel="弱源成功率 / %", title=f"N = {n}")
        axs[2*col+1].set(xlim=(0, 11 if col == 0 else 31), xlabel="在线时间 / s", title=f"N = {n}")
    save(fig, out, "fig_3_5_alt")

    fig, axs = plt.subplots(2, 2, figsize=(9, 6.2), layout="constrained")
    strata = {}
    for col, n in enumerate((4096, 16384)):
        subset = [r for r in rows if r["N"] == n and r["kind"] == "mixed" and r["factor"] == 1.]
        for i, m in enumerate(methods):
            err = sorted(r["metrics"]["errors_m"][1] for r in subset if r["method"] == m)
            axs[0, col].step(err, np.arange(1, len(err)+1)/len(err), where="post", label=m,
                color=INK[i], ls=[":", "-", "--", "-."][i], marker=MARKERS[i], markevery=(i*6, 28), mfc="white", ms=4)
        axs[0, col].set(title=f"N = {n}", xlabel="弱源位置误差 / m", ylabel="经验CDF", ylim=(0, 1))
        axs[0, col].legend(fontsize=8)
        axs[0, col].grid(alpha=.2)
        delta = np.array([[100*(np.mean([r["metrics"]["success"]["20"][1] for r in subset if r["method"] == "C" and r["overlap"] == ov and r["gap"] == gap])-np.mean([r["metrics"]["success"]["20"][1] for r in subset if r["method"] == "B" and r["overlap"] == ov and r["gap"] == gap])) for gap in (6, 12)] for ov in (.5, 1.)])
        strata[str(n)] = delta.tolist()
        axs[1, col].imshow(delta, vmin=-8, vmax=8, cmap="Greys", aspect="auto")
        for i in range(2):
            for j in range(2):
                label = ("+" if delta[i,j] >= 0 else "") + rounded(delta[i,j])
                axs[1, col].text(j, i, label, ha="center", va="center", fontsize=14, bbox=dict(fc="white", ec="none", pad=3))
        axs[1, col].set(xticks=[0, 1], xticklabels=["6 dB", "12 dB"], yticks=[0, 1], yticklabels=["IoU 0.5", "IoU 1.0"], title="C − B 弱源成功率差 / 百分点")
    save(fig, out, "fig_3_6")

    fig, ax = plt.subplots(figsize=(9, 3.3), layout="constrained")
    for r in fresh_detail["layouts"]:
        y = 7-r["layout"]
        ax.plot([r["B"], r["D"]], [y, y], color=".65", lw=1.3)
        for i, m in enumerate(("B", "D")):
            ax.plot(r[m], y+(.12 if m == "B" else -.12), marker=("o", "s")[i], color=INK[i], mfc="white", ms=7, label=m if r["layout"] == 1 else None)
        ax.text(8.55, y, f"{r['B']}/8 → {r['D']}/8", va="center", fontsize=9)
    ax.set(yticks=range(1, 7), yticklabels=[f"布局 {i}" for i in range(6, 0, -1)], xlim=(-.4, 10.2), xticks=range(9), ylim=(.4, 6.8), xlabel="弱源成功数（20 m，每布局8个条件）")
    ax.legend(loc="upper center", ncol=2, frameon=False)
    ax.grid(axis="x", alpha=.2)
    save(fig, out, "fig_3_7")
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.1), layout="constrained")
    for ax, n in zip(axs, (4096, 16384)):
        for i, m in enumerate(("B", "D")):
            s = report["r2_fresh"][f"{m}/N{n}"]
            x, y = s["seconds_mean"], 100*s["weak_20"]["rate"]
            ax.plot(x, y, marker=("o", "s")[i], mfc="white", color=INK[i], ms=8)
            ax.annotate(f"{m}：{s['weak_20']['successes']}/24\n{x:.2f} s", (x, y), xytext=(8, -7), textcoords="offset points", fontsize=9)
        ax.set(title=f"N = {n}", xlim=(0, 19 if n == 4096 else 70), ylim=(0, 100), xlabel="平均完整在线时间 / s", ylabel="弱源成功率（20 m）/ %")
        ax.grid(alpha=.2)
    save(fig, out, "fig_3_8")
    return {"cost": cost, "strata": strata}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    out = (HERE / args.run).resolve()
    assert out.parent == HERE and not out.exists(), "Use a new report-analysis directory"
    data = load_data()
    report, old, rows, dense, attribution, fresh, dev, shortlist = data
    out.mkdir()
    (out / "figures").mkdir()
    (out / "grayscale").mkdir()
    plotted = plot_all(out, report, old, rows, dense, attribution, fresh)
    def key(r):
        return tuple(r[k] for k in ("group", "realization", "overlap", "gap", "N"))
    pair = {m: {key(r): r["metrics"]["success"]["20"][1] for r in rows if r["method"] == m and r["kind"] == "mixed" and r["factor"] == 1.} for m in ("B", "C")}
    summary = {"formal": {k: v for k, v in report["formal"].items() if "/1.0" in k and not "/HR/" in k},
        "paired_C_B": {"rescued": sum(pair["C"][k] and not v for k, v in pair["B"].items()), "lost": sum(v and not pair["C"][k] for k, v in pair["B"].items())},
        "attribution": attribution, "fresh_detail": fresh, "r2_fresh": report["r2_fresh"],
        "bootstrap": report["formal_weak20_paired_bootstrap"],
        "selected": report["fixed_historical_selection"],
        "development_cost": dev["summary"], "top32_cost": shortlist["summary"], **plotted}
    (out / "report_data.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for path, digest in INPUTS.items():
        assert sha(Path(path)) == digest, f"Input changed: {path}"
    verification = {"status": "PASS", "mode": "frozen-result readback and replot only",
        "protocol": report["protocol"], "radius_m": report["radius_m"],
        "complete_case_key_join_checked": len(rows), "figure_count": len(list((out / "figures").glob("*.pdf"))),
        "inputs": INPUTS, "no_training": True, "no_inference": True, "no_test_access": True}
    (out / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.install:
        for p in (out / "figures").glob("*.pdf"):
            # The manuscript keeps the original charts; alternatives are archival only.
            if not p.stem.endswith("_alt"):
                shutil.copyfile(p, HERE / "Latex PDF/figures" / p.name)
    print(json.dumps({k: v for k, v in verification.items() if k != "inputs"}, ensure_ascii=False))
    print(json.dumps({"attribution": attribution, "fresh": fresh, "C_B": summary["paired_C_B"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
