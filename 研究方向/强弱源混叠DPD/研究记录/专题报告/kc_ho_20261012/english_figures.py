"""English report figures from the same frozen data and saved plot arrays.

No signal generation, localization, or training. Numerical artist coordinates
are checked before and after translation, and the Chinese final stays intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MplPath
import numpy as np

import refresh_figures_20m as original

HERE = Path(__file__).resolve().parent
ENGLISH = HERE / "Latex PDF English"
TRANSLATIONS = {
    "孤立弱源": "Isolated weak source",
    "混合弱源 B": "Mixture: B",
    "混合弱源 C": "Mixture: C",
    "4096 点 / 40.96 μs": "4096 samples / 40.96 μs",
    "16384 点 / 163.84 μs": "16384 samples / 163.84 μs",
    "弱源成功率（20 m）/ %": "Weak-source success (20 m) / %",
    "普通 DPD": "Conventional DPD",
    "强源": "Strong source",
    "弱源": "Weak source",
    "双源同时": "Both sources",
    "成功率（20 m）/ %": "Success rate (20 m) / %",
    "真实弱源": "True weak source",
    "密搜输出": "Dense-search output",
    "(a) 长记录示例：10 m 评分图": "(a) Long-record example: 10 m score map",
    "评分 / 最大评分": "Score / maximum score",
    "固定原强源估计": "Estimated strong\nposition fixed",
    "固定真实强源": "True strong\nposition fixed",
    "真弱源评分更高的记录数": "Records with higher score\nat the true weak position",
    "(b) 全部 327 条弱源失败": "(b) All 327 weak-source failures",
    "恢复数 / 16 条选定失败": "Recovered / 16 selected failures",
    "(c) 10 m 网格与前5个候选": "(c) 10 m grid and top 5 candidates",
    "最近格点": "Nearest\ngrid point",
    "密搜返回": "Dense-search\noutput",
    "真实位置": "True\nposition",
    "真值附近\n局部解": "Local solution\nnear truth",
    "归一化增量评分": "Normalized incremental score",
    "(d) 窄峰案例的候选评分": "(d) Candidate scores near a narrow peak",
    "计算预算": "Time budget",
    "预算内平均在线时间 / s": "Mean time within budget / s",
    "弱源位置误差 / m": "Weak-source position error / m",
    "经验CDF": "Empirical CDF",
    "C − B 弱源成功率差 / 百分点": "C minus B: weak-source success\n(percentage points)",
    "弱源成功数（20 m，每布局8个条件）": "Weak-source successes (20 m; 8 conditions per layout)",
    "平均完整在线时间 / s": "Mean total processing time / s",
}


def translate(text):
    result = TRANSLATIONS.get(text, text)
    result = re.sub(r"布局 (\d+)", r"Layout \1", result).replace("：", ": ")
    assert not re.search(r"[\u4e00-\u9fff]", result), f"Untranslated figure text: {text}"
    return result


def numeric_signature(fig):
    values = []
    for ax in fig.axes:
        values.extend([np.asarray(ax.get_xlim()), np.asarray(ax.get_ylim())])
        for line in ax.lines:
            values.extend([np.asarray(line.get_xdata(), dtype=float), np.asarray(line.get_ydata(), dtype=float)])
        for item in ax.collections:
            values.append(np.asarray(item.get_offsets(), dtype=float))
            if item.get_array() is not None:
                values.append(np.asarray(item.get_array(), dtype=float))
        for item in ax.images:
            values.append(np.asarray(item.get_array(), dtype=float))
        for patch in ax.patches:
            values.append(np.asarray(patch.get_path().vertices, dtype=float))
            if hasattr(patch, "get_height"):
                values.append(np.asarray([patch.get_x(), patch.get_y(), patch.get_width(), patch.get_height()]))
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value.shape).encode("ascii"))
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def static_figures(out, save):
    path = HERE / "analysis_v7_latex/figure_arrays.npz"
    with np.load(path) as loaded:
        arrays = {k: loaded[k] for k in loaded.files}
    rx = np.stack([500*np.cos(np.arange(4)*np.pi/2), 500*np.sin(np.arange(4)*np.pi/2)], axis=1).astype(np.float32).astype(float)
    fig = plt.figure(figsize=(11.2, 6.4), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.2])
    ax = fig.add_subplot(gs[:, 0])
    ax.scatter(rx[:, 0], rx[:, 1], marker="^", s=85, c="#247BA0", label="Receiver")
    for p, label, color, marker in zip(arrays["scene_positions_m"], ["Strong source", "Weak source"],
                                     ["#D77D32", "#348753"], ["*", "D"]):
        ax.scatter(*p, marker=marker, s=130, c=color, edgecolors="black", label=label)
        ax.annotate(label, p, xytext=(8, 6), textcoords="offset points", color=color)
    for i, p in enumerate(rx):
        ax.annotate(f"Rx {i+1}", p, xytext=(6, 5), textcoords="offset points")
    ax.set(xlim=(-610, 640), ylim=(-610, 630), aspect="equal", title="(a) Four-receiver geometry", xlabel="x / m", ylabel="y / m")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=.2)
    for row, overlap in enumerate([.5, 1.]):
        ax = fig.add_subplot(gs[row, 1])
        freq = arrays["scene_frequency_mhz"]
        densities = arrays[f"scene_psd_{overlap}"]
        for density, label, color, style in zip(densities, ["Strong: 6 dBm", "Weak: 0 dBm"],
                                              ["#D77D32", "#348753"], ["-", "--"]):
            ax.plot(freq, 10*np.log10(np.maximum(density, 1e-7)), color=color, lw=1.7, ls=style, label=label)
        width = 12.5
        shift = width*(1-overlap)/(1+overlap)
        for w, y, label in [(width-shift, -43, "Intersection"), (width+shift, -56, "Union")]:
            ax.annotate("", xy=(w/2, y), xytext=(-w/2, y), arrowprops=dict(arrowstyle="|-|", lw=1.2, color="#334E68"))
            ax.text(0, y+1.6, f"{label}: {w:.3f} MHz", ha="center", fontsize=8.5, bbox=dict(fc="white", ec="none", pad=.1))
        ax.set(xlim=(-11, 11), ylim=(-63, 5), title=f"({'b' if row == 0 else 'c'}) Ideal spectra, IoU = {overlap:g}",
               xlabel="Relative frequency / MHz", ylabel="PSD / (dBm/MHz)")
        ax.grid(alpha=.2)
        if row == 0:
            ax.legend(loc="upper right", fontsize=8)
    save(fig, out, "fig_2_1")

    fig, ax = plt.subplots(figsize=(10, 3.5), layout="constrained")
    ax.set(xlim=(0, 10), ylim=(0, 3.6))
    ax.axis("off")
    boxes = [(0.1, "Mixed IQ\nJoint model at\neach frequency"),
             (2.7, "Candidate search\nInitial 2-D\npositions"),
             (5.3, "Continuous\nrefinement\nUpdate one position"),
             (7.9, "Joint fitting\nRefit all source\ncoefficients")]
    for x, text in boxes:
        ax.add_patch(FancyBboxPatch((x, 1.7), 1.85, 1., boxstyle="round,pad=.09", fc="#EAF2F5", ec="#247BA0"))
        ax.text(x+.925, 2.2, text, ha="center", va="center", fontsize=12.5)
    for x in [2.1, 4.7, 7.3]:
        ax.annotate("", xy=(x+.48, 2.2), xytext=(x-.08, 2.2), arrowprops=dict(arrowstyle="->", color="#334E68", lw=1.6))
    path_return = MplPath([(8.825, 1.60), (8.825, .66), (3.625, .66), (3.625, 1.60)],
                         [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO])
    ax.add_patch(FancyArrowPatch(path=path_return, arrowstyle="->", mutation_scale=13, lw=1.5, color="#334E68"))
    ax.text(6.225, .92, "Revisit other sources;\nrecompute the conditional projection", ha="center", va="bottom", fontsize=12)
    ax.text(5, 3.25, "Common objective: minimize the joint residual of the original mixture", ha="center", weight="bold", fontsize=11)
    save(fig, out, "fig_2_2")

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axs[0].plot(arrays["toy_frequencies_hz"]/1e6, arrays["eta"], color="#348753")
    axs[0].set(ylim=(0, 1.05), title="(a) Weak-source energy retained", xlabel="Frequency / MHz", ylabel=r"$\eta_m$")
    axs[1].loglog(arrays["offsets_m"], arrays["leakage_ratio"], label="Exact projection", color="#247BA0", marker="o", markevery=12, mfc="white", ms=4)
    axs[1].loglog(arrays["offsets_m"], arrays["leakage_quadratic"], "--", label="First-order response approximation", color="#D77D32")
    axs[1].axhline(1, color=".5", lw=1)
    axs[1].set(title="(b) Leakage relative to retained weak energy", xlabel="Strong-source position offset along x / m",
               ylabel="Leakage / retained weak-source energy")
    axs[1].legend(fontsize=8)
    for ax in axs:
        ax.grid(alpha=.2)
    save(fig, out, "fig_3_4")
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    out = (HERE / args.run).resolve()
    assert out.parent == HERE and not out.exists()
    out.mkdir()
    (out / "figures").mkdir()
    (out / "grayscale").mkdir()
    plt.rcParams.update({"font.sans-serif": ["DejaVu Sans"], "font.size": 10, "axes.titlesize": 10})
    provenance = {}
    original_save = original.save

    def save(fig, destination, name):
        if name.endswith("_alt"):
            plt.close(fig)
            return
        before = numeric_signature(fig)
        if name == "fig_3_5":
            # Stagger annotations, not data, when adjacent bars have near-equal values.
            for ax in fig.axes:
                for i, item in enumerate(ax.texts):
                    item.set_position((0, 3+9*(i//3)))
        # Fixed tick formatters restore their original labels during drawing.
        # Replace their labels through the public axis API before editing Text.
        for ax in fig.axes:
            for axis_name in ("x", "y"):
                labels = [v.get_text() for v in getattr(ax, f"get_{axis_name}ticklabels")()]
                if any(re.search(r"[\u4e00-\u9fff]", s) for s in labels):
                    getattr(ax, f"set_{axis_name}ticks")(
                        getattr(ax, f"get_{axis_name}ticks")(), [translate(s) for s in labels])
        texts = []
        for item in fig.findobj(Text):
            translated = translate(item.get_text())
            item.set_text(translated)
            if translated:
                texts.append(translated)
        assert numeric_signature(fig) == before, name
        fig.canvas.draw()
        assert all(not re.search(r"[\u4e00-\u9fff]", t.get_text()) for t in fig.findobj(Text)), name
        provenance[name] = {"numerical_artist_sha256": before, "text": texts, "data_unchanged_by_translation": True}
        original_save(fig, destination, name)

    original.save = save
    report, old, rows, dense, attribution, fresh, _, _ = original.load_data()
    plotted = original.plot_all(out, report, old, rows, dense, attribution, fresh)
    previous = json.loads((HERE / "analysis_v11_20m/report_data.json").read_text(encoding="utf-8"))
    assert plotted["cost"] == previous["cost"] and plotted["strata"] == previous["strata"]
    static_inputs = static_figures(out, save)
    assert len(provenance) == 10
    for path, digest in original.INPUTS.items():
        assert original.sha(Path(path)) == digest
    manifest = json.loads((HERE / "chinese_final_manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        assert original.sha(HERE / name) == digest, f"Chinese final changed: {name}"
    for p in (out / "figures").glob("*.pdf"):
        shutil.copyfile(p, ENGLISH / "figures" / p.name)
    verification = {"status": "PASS", "figure_count": 10, "figures": provenance,
                    "inputs": {**original.INPUTS, **static_inputs}, "cost_and_strata_match_chinese": True,
                    "chinese_final_unchanged": True, "no_signal_generation_or_inference": True}
    (out / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "figure_count": 10, "input_count": len(verification["inputs"])}))


if __name__ == "__main__":
    main()
