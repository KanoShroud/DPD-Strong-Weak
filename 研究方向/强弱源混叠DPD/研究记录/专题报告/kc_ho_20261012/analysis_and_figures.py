"""Generate nine report figures and verify equations, using frozen results only.

Run with the project Python interpreter. Each run uses a new output directory.
No IQ regeneration, training, formal evaluation, or test access is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MplPath
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
G9 = ROOT / "outputs_e2e/unified/e2e_g9/formal_20261011_repaired"
R1 = ROOT / "outputs_e2e/unified/e2e_g9/r1_failure_20261011_01"
R6 = ROOT / "outputs_e2e/unified/e2e_g8/r6_failure_20261008_01"
C0 = 299792458.0
RX = np.stack([500*np.cos(np.arange(4)*np.pi/2),
               500*np.sin(np.arange(4)*np.pi/2)], axis=1).astype(np.float32).astype(float)
SEED = 2026101201
INPUTS: dict[str, str] = {}
CHECKS: list[dict] = []
COLORS = {"A": "#247BA0", "B": "#348753", "C": "#D77D32", "DPD": "#8B6BB1"}
LINESTYLES = {"A": "-", "B": "--", "C": "-.", "DPD": ":"}
MARKERS = {"A": "o", "B": "s", "C": "^", "DPD": "D"}
HATCHES = ["///", "...", "xxx", "\\\\"]
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
    "axes.unicode_minus": False, "font.size": 10, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.facecolor": "white",
})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def track(path):
    p = Path(path).resolve(strict=True)
    if not p.is_relative_to(ROOT / "outputs_e2e/unified"):
        raise ValueError(f"Unexpected input: {p}")
    INPUTS[str(p)] = sha(p)
    return p


def read(path):
    return json.loads(track(path).read_text(encoding="utf-8"))


def check(name, value, tolerance):
    passed = bool(np.isfinite(value) and value <= tolerance)
    CHECKS.append(dict(name=name, value=float(value), tolerance=tolerance, passed=passed))
    if not passed:
        raise AssertionError(CHECKS[-1])


def rel(a, b):
    return np.linalg.norm(a-b) / max(np.linalg.norm(a), np.linalg.norm(b), 1e-30)


def response(p, f, rx=RX):
    """a: [F,L], da: [F,L,2]; exact derivative of exp(-j2pi f d/c)/d."""
    delta = np.asarray(p)[None, :] - rx
    d = np.linalg.norm(delta, axis=1)
    a = np.exp(-2j*np.pi*np.asarray(f)[:, None]*d[None, :]/C0)/d[None, :]
    da = a[:, :, None]*(-1/d[None, :, None]
                        - 2j*np.pi*np.asarray(f)[:, None, None]/C0)
    da = da * (delta/d[:, None])[None, :, :]
    return a, da


def projector(a):
    return np.eye(a.shape[0], dtype=complex) - a @ np.linalg.pinv(a, rcond=1e-12)


def information(pos, f, s, rx=RX):
    # s: [F,K], unit variance proper-complex noise; all complex spectra unknown.
    pairs = [response(p, f, rx) for p in pos]
    k = len(pos)
    info = np.zeros((2*k, 2*k))
    for j in range(len(f)):
        a = np.stack([v[0][j] for v in pairs], axis=1)
        g = np.concatenate([v[1][j]*s[j, q] for q, v in enumerate(pairs)], axis=1)
        info += 2*(g.conj().T @ projector(a) @ g).real
    return (info+info.T)/2


def equations():
    rng = np.random.default_rng(SEED)
    f = np.linspace(-11e6, 9e6, 7)
    pos = np.array([[680., -260.], [-710., -620.]])
    pairs = [response(p, f) for p in pos]
    s = (rng.normal(size=(7, 2))+1j*rng.normal(size=(7, 2)))*350
    projection_error = least_squares_error = gain_error = 0.
    for j in range(len(f)):
        a = np.stack([v[0][j] for v in pairs], axis=1)
        x = a@s[j] + rng.normal(size=4) + 1j*rng.normal(size=4)
        pi = projector(a)
        projection_error = max(projection_error, np.linalg.norm(pi@pi-pi),
                               np.linalg.norm(pi-pi.conj().T), np.linalg.norm(pi@a))
        fitted = np.linalg.lstsq(a, x, rcond=1e-12)[0]
        least_squares_error = max(least_squares_error, rel(x-a@fitted, pi@x))
        p1 = projector(a[:, :1])
        u, r = p1@a[:, 1], p1@x
        q = abs(np.vdot(u, r))**2/np.vdot(u, u).real
        drop = np.vdot(r, r).real-np.vdot(pi@x, pi@x).real
        gain_error = max(gain_error, abs(q-drop)/max(1., abs(q)))
    check("projector_hermitian_idempotent_orthogonal", projection_error, 1e-10)
    check("concentrated_LS_matches_direct_lstsq", least_squares_error, 1e-10)
    check("incremental_score_equals_residual_drop", gain_error, 1e-10)
    h = 1e-3
    a, da = pairs[0]
    numerical = np.stack([(response(pos[0]+h*np.eye(2)[d], f)[0]
                           - response(pos[0]-h*np.eye(2)[d], f)[0])/(2*h)
                          for d in range(2)], axis=-1)
    check("position_response_derivative_central_difference", rel(da, numerical), 2e-7)
    # Full real-parameter complex-normal FIM, followed by an explicit Schur complement.
    jac = np.zeros((len(f)*4, 4+len(f)*4), complex)
    for j in range(len(f)):
        a = np.stack([v[0][j] for v in pairs], axis=1)
        jac[4*j:4*j+4, :4] = np.concatenate(
            [v[1][j]*s[j, q] for q, v in enumerate(pairs)], axis=1)
        jac[4*j:4*j+4, 4+4*j:8+4*j] = np.column_stack([a, 1j*a])
    full = 2*(jac.conj().T@jac).real
    schur = full[:4, :4] - full[:4, 4:] @ np.linalg.solve(full[4:, 4:], full[4:, :4])
    eff = information(pos, f, s)
    check("EFIM_projected_derivatives_equals_real_Schur", rel(eff, schur), 1e-10)
    check("EFIM_symmetry", rel(eff, eff.T), 1e-12)
    check("EFIM_positive_semidefinite", max(0., -np.linalg.eigvalsh(eff).min()), 1e-10)
    deg = information(pos[:1], f, s[:, :1], np.tile([[500., 0.]], (4, 1)))
    check("coincident_receivers_zero_position_information",
          np.linalg.norm(deg)/np.linalg.norm(information(pos[:1], f, s[:, :1])), 1e-10)
    # One-source information is invariant to a common frequency offset if complex
    # coefficients are unconstrained only in the spatially common phase direction;
    # no such invariance is assumed for relative-delay carrier phases.
    p1 = projector(pairs[0][0][2, :, None])
    aw = pairs[1][0][2]
    aa = pairs[0][0][2]
    eta = np.linalg.norm(p1@aw)**2/np.linalg.norm(aw)**2
    rho = abs(np.vdot(aa, aw))**2/(np.linalg.norm(aa)**2*np.linalg.norm(aw)**2)
    check("energy_retention_identity", abs(eta-(1-rho)), 1e-12)
    # Fixed-anchor score gradient, including denominator derivative.
    anchor = [projector(response(pos[0], f)[0][j, :, None]) for j in range(len(f))]
    x = np.stack([np.stack([v[0][j] for v in pairs], axis=1)@s[j]
                  for j in range(len(f))])
    def score(p, gradient=False):
        a, da = response(p, f)
        q, dq = 0., np.zeros(2)
        for j, pi in enumerate(anchor):
            u, du, r = pi@a[j], pi@da[j], pi@x[j]
            v, den = np.vdot(u, r), np.vdot(u, u).real
            dv = du.conj().T@r
            dd = 2*(u.conj()@du).real
            q += abs(v)**2/den
            dq += 2*(v.conjugate()*dv).real/den - abs(v)**2*dd/den**2
        return (q, dq) if gradient else q
    p = pos[1]+[8., -11.]
    numeric = np.array([(score(p+h*np.eye(2)[d])-score(p-h*np.eye(2)[d]))/(2*h)
                        for d in range(2)])
    check("incremental_score_gradient", rel(score(p, True)[1], numeric), 2e-7)
    def joint(theta, gradient=False):
        pair = [response(t, f) for t in np.asarray(theta).reshape(2, 2)]
        total, grad = 0., np.zeros(4)
        for j in range(len(f)):
            af = np.stack([v[0][j] for v in pair], axis=1)
            shat = np.linalg.lstsq(af, x[j], rcond=1e-12)[0]
            residual = x[j]-af@shat
            total += np.vdot(residual, residual).real
            df = np.concatenate([v[1][j]*shat[q] for q, v in enumerate(pair)], axis=1)
            grad += -2*(residual.conj()@df).real
        return (total, grad) if gradient else total
    theta = (pos+[[1., -2.], [8., -11.]]).ravel()
    jnumeric = np.array([(joint(theta+h*np.eye(4)[d])-joint(theta-h*np.eye(4)[d]))/(2*h)
                         for d in range(4)])
    check("joint_variable_projection_gradient", rel(joint(theta, True)[1], jnumeric), 2e-7)
    cov = p1@p1.conj().T
    check("projected_noise_covariance_equals_projector", rel(cov, p1), 1e-12)
    return dict(seed=SEED, frequencies_hz=f.tolist(), positions_m=pos.tolist())


def summary(rows):
    errors = [e for r in rows for e in r["metrics"]["errors_m"] if e is not None]
    return {
        "n": len(rows),
        "weak": np.mean([r["metrics"]["success"]["100"][1] for r in rows]),
        "strong": np.mean([r["metrics"]["success"]["100"][0] for r in rows]),
        "both": np.mean([all(r["metrics"]["success"]["100"]) for r in rows]),
        "rmse_m": float(np.sqrt(np.mean(np.square(errors)))),
        "p95_m": float(np.quantile(errors, .95)),
        "gospa_m": np.mean([r["metrics"]["gospa_m"] for r in rows]),
    }


def save(fig, out, name):
    fig.savefig(out/"figures"/(name+".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out/"figures"/(name+".svg"), bbox_inches="tight")
    fig.savefig(out/"figures"/(name+".pdf"), bbox_inches="tight")
    gray_dir = out/"grayscale"
    gray_dir.mkdir(exist_ok=True)
    with Image.open(out/"figures"/(name+".png")) as img:
        img.convert("L").save(gray_dir/(name+".png"), dpi=(300, 300))
    plt.close(fig)


def decorate(ax, xlabel=None, ylabel=None):
    ax.grid(alpha=.2)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def figures(out, report, rows, score_rows, dense_rows):
    use = lambda method, n=None: [r for r in rows if r["kind"] == "mixed"
          and r["factor"] == 1. and r["method"] == method
          and (n is None or r["N"] == n)]
    stats = {m: summary(use(m)) for m in ["A", "B", "C", "DPD", "HR"]}
    arrays = {}
    # Read the plotted positions directly from an existing frozen record.
    fig = plt.figure(figsize=(11.2, 6.4), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.2])
    ax = fig.add_subplot(gs[:, 0])
    ax.scatter(RX[:, 0], RX[:, 1], marker="^", s=85, c="#247BA0", label="接收站")
    scene_record = next(r for r in score_rows if r["group"] == 5
                        and r["realization"] == 0 and r["overlap"] == .5
                        and r["gap"] == 6 and r["N"] == 4096)
    schematic = np.asarray(scene_record["truth"], dtype=float)
    for p, label, color, marker in zip(schematic, ["强源", "弱源"],
                                     ["#D77D32", "#348753"], ["*", "D"]):
        ax.scatter(*p, marker=marker, s=130, c=color, edgecolors="black", label=label)
        ax.annotate(label, p, xytext=(8, 6), textcoords="offset points", color=color)
    for i, p in enumerate(RX):
        ax.annotate(f"站{i+1}", p, xytext=(6, 5), textcoords="offset points")
    ax.set(xlim=(-610, 640), ylim=(-610, 630), aspect="equal", title="(a) 四站强弱源定位场景")
    decorate(ax, "x / m", "y / m")
    ax.legend(loc="upper right", fontsize=8)
    freq = np.linspace(-12, 12, 12001)  # MHz; integration gives mW.
    rate, alpha = 10., .25
    bw = rate*(1+alpha)
    def psd(center, power_dbm):
        af = abs(freq-center)
        shape = np.zeros_like(freq)
        shape[af <= rate*(1-alpha)/2] = 1.
        edge = (af > rate*(1-alpha)/2) & (af <= bw/2)
        shape[edge] = .5*(1+np.cos(np.pi/(alpha*rate)*(af[edge]-rate*(1-alpha)/2)))
        return 10**(power_dbm/10)*shape/rate
    for row, overlap in enumerate([.5, 1.]):
        ax = fig.add_subplot(gs[row, 1])
        shift = bw*(1-overlap)/(1+overlap)
        inter, union = bw-shift, bw+shift
        for center, dbm, color, label in [(-shift/2, 6., "#D77D32", "强源：6 dBm"),
                                          (shift/2, 0., "#348753", "弱源：0 dBm")]:
            density = psd(center, dbm)
            ax.plot(freq, 10*np.log10(np.maximum(density, 1e-7)), color=color,
                    lw=1.7, ls="-" if dbm == 6 else "--", label=label)
            check(f"spectrum_power_integral_{overlap}_{dbm}",
                  abs(np.trapezoid(density, freq)-10**(dbm/10)), 1e-6)
        for width, y, label in [(inter, -43, "交集"), (union, -56, "并集")]:
            ax.annotate("", xy=(width/2, y), xytext=(-width/2, y),
                        arrowprops=dict(arrowstyle="|-|", lw=1.2, color="#334E68"))
            ax.text(0, y+1.6, f"{label}：{width:.3f} MHz", ha="center", fontsize=8.5,
                    bbox=dict(fc="white", ec="none", pad=.1))
        ax.set(xlim=(-11, 11), ylim=(-63, 5),
               title=f"({'b' if row == 0 else 'c'}) 理想功率谱，IoU = {overlap:g}")
        decorate(ax, "相对频率 / MHz", "PSD / (dBm/MHz)")
        if row == 0:
            ax.legend(loc="upper right", fontsize=8)
        check(f"spectrum_support_IoU_{overlap}", abs(inter/union-overlap), 1e-12)
        arrays[f"scene_psd_{overlap}"] = np.stack([psd(-shift/2, 6), psd(shift/2, 0)])
    check("scene_weak_source_inside_receiver_hull", max(0., np.abs(schematic[1]).sum()-499), 0)
    baseline_distance = abs(schematic[0].sum()-500)/np.sqrt(2)
    check("scene_strong_source_away_from_station12_line", max(0., 100-baseline_distance), 0)
    check("schematic_source_spacing_at_least_300m",
          max(0., 300-np.linalg.norm(schematic[0]-schematic[1])), 0)
    arrays.update(scene_positions_m=schematic, scene_frequency_mhz=freq,
                  scene_frozen_record=scene_record["tag"])
    save(fig, out, "fig_1_1_scene")
    # Figure 3.1: algorithm diagram.
    fig, ax = plt.subplots(figsize=(10, 3.5), layout="constrained")
    ax.set(xlim=(0, 10), ylim=(0, 3.6)); ax.axis("off")
    boxes = [(0.1, 1.7, "混合 IQ\n逐频率联合模型"),
             (2.7, 1.7, "候选发现\n二维位置初值"),
             (5.3, 1.7, "连续精修\n更新一个位置"),
             (7.9, 1.7, "联合拟合\n重估所有源系数")]
    for x, y, text in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 1.85, 1., boxstyle="round,pad=.09",
                                   fc="#EAF2F5", ec="#247BA0"))
        ax.text(x+.925, y+.5, text, ha="center", va="center")
    for x in [2.1, 4.7, 7.3]:
        ax.annotate("", xy=(x+.48, 2.2), xytext=(x-.08, 2.2),
                    arrowprops=dict(arrowstyle="->", color="#334E68", lw=1.6))
    # Return from the joint fit to candidate discovery with square corners.
    return_path = MplPath([(8.825, 1.60), (8.825, .66), (3.625, .66), (3.625, 1.60)],
                          [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO])
    ax.add_patch(FancyArrowPatch(path=return_path, arrowstyle="->",
                                mutation_scale=13, lw=1.5, color="#334E68"))
    ax.text(6.225, .92, "回访其他源，重新构造条件投影", ha="center", va="bottom")
    ax.text(5, 3.25, "共同目标：最小化原始混合观测的联合拟合残差", ha="center", weight="bold")
    save(fig, out, "fig_3_1_solver")
    # Figure 4.1: conditional CRLB example with an explicit synthetic spectrum.
    f = np.linspace(-6.25e6, 6.25e6, 65)
    ps, pw = np.array([650., -300.]), np.array([-800., -700.])
    rng = np.random.default_rng(SEED+1)
    positions = np.array([ps, pw])
    a = [response(p, f)[0] for p in positions]
    s = np.column_stack([np.sqrt(10**(pdb/10)/np.mean(abs(aa[0])**2))
                         *np.exp(1j*rng.uniform(-np.pi, np.pi, len(f)))
                         for pdb, aa in zip([15., 3.], a)])
    j = information(positions, f, s)
    jsingle = information([pw], f, s[:, 1:])
    jc = j[2:, 2:]
    ju = jc-j[2:, :2]@np.linalg.solve(j[:2, :2], j[:2, 2:])
    matrices = [np.linalg.inv(jsingle), np.linalg.inv(jc), np.linalg.inv(ju)]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.3), layout="constrained")
    for cov, label, color, style in zip(matrices, ["孤立弱源", "强源位置已知", "两源位置均未知"],
                                      ["#247BA0", "#348753", "#D77D32"], ["-", "--", "-."]):
        val, vec = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(vec[1, 1], vec[0, 1]))
        axs[0].add_patch(Ellipse((0, 0), 2*np.sqrt(val[1]), 2*np.sqrt(val[0]),
                                angle=angle, fill=False, ec=color, lw=2, ls=style, label=label))
    limit = 1.2*max(np.sqrt(np.linalg.eigvalsh(v).max()) for v in matrices)
    axs[0].set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal",
               title="(a) 单位 Mahalanobis 半径的 CRLB 椭圆")
    decorate(axs[0], "位置偏移 x / m", "位置偏移 y / m")
    axs[0].legend(fontsize=8)
    eigs = np.array([np.linalg.eigvalsh(x) for x in [jsingle, jc, ju]])
    xx = np.arange(3)
    axs[1].bar(xx-.17, eigs[:, 0], width=.34, label="最小特征值",
               color="#C9DCE5", edgecolor="black", hatch="///")
    axs[1].bar(xx+.17, eigs[:, 1], width=.34, label="最大特征值",
               color="#F2D2B6", edgecolor="black", hatch="...")
    axs[1].set(xticks=xx, xticklabels=["孤立弱源", "强源位置已知", "两源位置未知"],
               yscale="log", ylim=(eigs.min()*.75, eigs.max()*4),
               title="(b) 弱源等效信息的特征值")
    decorate(axs[1], ylabel=r"信息 / $\mathrm{m}^{-2}$"); axs[1].legend(fontsize=8)
    save(fig, out, "fig_4_1_information")
    arrays["information_eigenvalues"] = eigs
    arrays["crlb_covariances"] = np.asarray(matrices)
    # Figure 4.2: normalized coherence and anchor-location leakage.
    aas, aws = a
    eta = np.array([np.linalg.norm(projector(aas[j, :, None])@aws[j])**2
                    /np.linalg.norm(aws[j])**2 for j in range(len(f))])
    offsets = np.geomspace(.01, 50., 120)
    leakage = []
    retained = np.sum(abs(s[:, 1])**2*np.sum(abs(aws)**2, axis=1)*eta)
    for delta in offsets:
        ah = response(ps+[delta, 0], f)[0]
        leakage.append(sum(abs(s[jj, 0])**2*np.linalg.norm(
            projector(ah[jj, :, None])@aas[jj])**2 for jj in range(len(f)))/retained)
    da = response(ps, f)[1][:, :, 0]
    quadratic = sum(abs(s[jj, 0])**2*np.linalg.norm(
        projector(aas[jj, :, None])@da[jj])**2 for jj in range(len(f)))/retained
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axs[0].plot(f/1e6, eta, color="#348753")
    axs[0].set(ylim=(0, 1.05), title="(a) 已知强源位置时的弱源能量保留")
    decorate(axs[0], "频率 / MHz", r"$\eta_f$")
    axs[1].loglog(offsets, leakage, label="精确投影", color="#247BA0",
                  marker="o", markevery=12, mfc="white", ms=4)
    axs[1].loglog(offsets, quadratic*offsets**2, "--", label="位置误差的一阶响应近似",
                  color="#D77D32")
    axs[1].axhline(1, color=".5", lw=1)
    axs[1].set_title("(b) 强源残留相对保留弱源的能量")
    decorate(axs[1], "强源 x 方向位置偏差 / m", "残留能量 / 保留弱源能量")
    axs[1].legend(fontsize=8)
    save(fig, out, "fig_4_2_projection")
    arrays.update(eta=eta, offsets_m=offsets, leakage_ratio=np.asarray(leakage),
                  leakage_quadratic=quadratic*offsets**2, toy_frequencies_hz=f,
                  toy_positions_m=positions, toy_coefficients=s)
    # Figure 5.1: matched isolated versus mixed records.
    fig, ax = plt.subplots(figsize=(8.3, 4), layout="constrained")
    xx = np.arange(2)
    for i, (label, method, isolated) in enumerate(
            [("孤立弱源（共同参照）", "B", True), ("混合弱源：B", "B", False),
             ("混合弱源：C", "C", False)]):
        vals = [(report["isolated"][f"{n}_{method}_weak"]["recall100"] if isolated
                 else summary(use(method, n))["weak"])*100 for n in [4096, 16384]]
        rects = ax.bar(xx+(i-1)*.24, vals, .24, label=label,
                       color=["#D2D8DD", "#CEE3D3", "#F2D2B6"][i],
                       edgecolor="black", linewidth=.8, hatch=HATCHES[i])
        ax.bar_label(rects, fmt="%.2f", fontsize=9, padding=3)
    ax.set(xticks=xx, xticklabels=["4096 点 / 40.96 μs", "16384 点 / 163.84 μs"],
           ylim=(0, 112), ylabel="弱源成功率 @100 m / %")
    ax.legend(loc="upper right", fontsize=8); ax.grid(axis="y", alpha=.2)
    save(fig, out, "fig_5_1_isolated_mixed")
    # Figure 5.2: actual complete-method comparison.
    fig, ax = plt.subplots(figsize=(8.5, 4), layout="constrained")
    methods = ["DPD", "A", "B", "C"]; xx = np.arange(4)
    for i, (key, label, color) in enumerate([
            ("strong", "强源", "#8199A8"), ("weak", "弱源", "#348753"),
            ("both", "双源同时", "#D77D32")]):
        vals = [100*stats[m][key] for m in methods]
        bars = ax.bar(xx+(i-1)*.24, vals, .24, label=label,
                      color=["#D2D8DD", "#CEE3D3", "#F2D2B6"][i],
                      edgecolor="black", linewidth=.8, hatch=HATCHES[i])
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
    ax.set(xticks=xx, xticklabels=["普通 DPD", "多初值 AP（A）", "区域搜索（B）", "新增排序（C）"],
           ylim=(0, 112), ylabel="成功率 @100 m / %")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.2)
    save(fig, out, "fig_5_2_recovery")
    # Figure 5.3: existing map plus new frozen attribution, no scoring rerun.
    r6 = read(R6/"rows.json")
    old = next(r for r in r6 if r["group"] == 19 and r["overlap"] == .5)
    identity = old["A"]["map_identity"]
    path = track(Path(identity["path"]))
    if sha(path) != identity["sha256"]:
        raise ValueError("R6 map SHA mismatch")
    with np.load(path) as z:
        arrays_in = {k: z[k] for k in z.files}
    # Map file schema is inspected, not inferred from PNG pixels.
    map_values = next(v for v in arrays_in.values() if v.size == 201*201)
    fig, axs = plt.subplots(2, 2, figsize=(10, 8.2), layout="constrained")
    im = axs[0, 0].imshow(map_values.reshape(201, 201)/map_values.max(), origin="lower",
                          extent=(-1000, 1000, -1000, 1000), cmap="viridis")
    axs[0, 0].scatter(*old["truth"][1], edgecolors="white", marker="o", facecolors="none", s=100,
                      label="真实弱源")
    axs[0, 0].scatter(*old["A"]["position"], c="white", marker="x", s=55, label="密搜输出")
    axs[0, 0].set(title="(a) 辅助长记录示例：10 m 评分图", xlabel="x / m", ylabel="y / m")
    axs[0, 0].legend(fontsize=8, facecolor=".2", labelcolor="white", framealpha=1)
    fig.colorbar(im, ax=axs[0, 0], shrink=.8, label="评分 / 图内最大评分")
    failed = [r for r in score_rows if not r["baseline_metrics"]["success"]["100"][1]]
    good = sum(r["estimated"]["truth_higher"] for r in failed)
    bars = axs[0, 1].bar(["原强源估计下\n真弱源评分更高", "替换真实强源后\n真弱源评分更高"],
                         [good, len(failed)], color=["#CEE3D3", "#D2D8DD"], edgecolor="black")
    for bar, hatch in zip(bars, HATCHES):
        bar.set_hatch(hatch)
    axs[0, 1].bar_label(bars, padding=3)
    axs[0, 1].set(ylim=(0, 285), ylabel="失败记录数", title="(b) 250 条弱源失败的条件评分")
    selected = [r for r in dense_rows if r["selection"] == "failure"]
    vals = [sum(r[k]["error_m"] <= 100 for r in selected) for k in ["estimated", "true_strong"]]
    bars = axs[1, 0].bar(["固定原强源", "固定真实强源"], vals,
                        color=["#CEE3D3", "#D2D8DD"], edgecolor="black")
    for bar, hatch in zip(bars, HATCHES):
        bar.set_hatch(hatch)
    axs[1, 0].bar_label(bars, padding=3)
    axs[1, 0].set(ylim=(0, 18), ylabel="恢复条数 / 16 个选定失败",
                  title="(c) 16 条选定失败的加密搜索")
    narrow = next(r for r in dense_rows if r["tag"] == "4_0_1.0_12_4096_mixed_B")["true_strong"]
    vals = [narrow["nearest_truth_grid_score"], narrow["score"], narrow["oracle"]["truth_score"],
            narrow["oracle"]["oracle_score"]]
    bars = axs[1, 1].bar(["最近格点", "密搜返回", "真实位置", "真值附近局部解"], vals,
                         color=["#D2D8DD", "#F2D2B6", "#CEE3D3", "#C9DCE5"], edgecolor="black")
    for bar, hatch in zip(bars, HATCHES):
        bar.set_hatch(hatch)
    axs[1, 1].bar_label(bars, fmt="%.5f", fontsize=8, padding=3)
    axs[1, 1].set(ylim=(0, .105), ylabel="归一化增量评分",
                  title="(d) 案例Ⅱ：10 m 网格遗漏的窄峰")
    save(fig, out, "fig_5_3_search")
    # Figure 5.4 (report Figure 3.5): paired metric panels avoid coincident curves.
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.7), layout="constrained")
    cost_rows = []
    for col, n in enumerate([4096, 16384]):
        x = np.arange(3)
        for j, m in enumerate(["A", "B", "C"]):
            keys = [f"{n}_{m}_{factor}" for factor in [.5, 1., 2.]]
            sx = [report["summary"][k]["seconds_mean"] for k in keys]
            sy = [100*report["summary"][k]["weak_100"] for k in keys]
            for row, values in enumerate([sy, sx]):
                ax = axs[row, col]
                bars = ax.bar(x+(j-1)*.26, values, width=.24, label=m,
                              color=["#E0E8ED", "#E0EADF", "#F2E4D5"][j],
                              edgecolor="black", linewidth=.75, hatch=HATCHES[j], zorder=3)
                ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
            cost_rows.extend(dict(N=n, method=m, factor=factor,
                                  seconds_mean=t, weak_100_percent=s)
                             for factor, t, s in zip([.5, 1., 2.], sx, sy))
        for row in range(2):
            ax = axs[row, col]
            ax.set(xticks=x, xticklabels=[r"$0.5T_0$", r"$T_0$", r"$2T_0$"],
                   xlim=(-.55, 2.55), xlabel="计算预算")
            ax.grid(axis="y", alpha=.2, zorder=0)
        axs[0, col].set(title=f"(a{col+1}) N = {n}：弱源成功率", ylim=(0, 100),
                        ylabel="弱源成功率 @100 m / %")
        axs[0, col].legend(ncol=3, loc="upper center", fontsize=9, frameon=False)
        axs[1, col].set(title=f"(b{col+1}) N = {n}：预算内耗时", ylim=(0, 9 if col == 0 else 27),
                        ylabel="平均计入在线时间 / s")
    (out/"cost_plot_data.json").write_text(json.dumps(cost_rows, indent=2), encoding="utf-8")
    save(fig, out, "fig_5_4_cost")
    # Figure 5.5: untruncated weak errors and eight fixed-condition comparisons.
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.6), layout="constrained")
    for col, n in enumerate([4096, 16384]):
        for m in ["DPD", "A", "B", "C"]:
            err = sorted(r["metrics"]["errors_m"][1] for r in use(m, n))
            axs[0, col].step(err, np.arange(1, len(err)+1)/len(err), where="post",
                             label=m, color=COLORS[m], ls=LINESTYLES[m],
                             marker=MARKERS[m], markevery=(6*list(COLORS).index(m), 28),
                             mfc="white", ms=4, lw=1.5)
        max_error = max(r["metrics"]["errors_m"][1]
                        for m in ["DPD", "A", "B", "C"] for r in use(m, n))
        axs[0, col].set(title=f"(a{col+1}) N = {n}，平方距离匹配误差",
                        xlim=(0, 100*np.ceil(max_error/100)), ylim=(0, 1))
        decorate(axs[0, col], "弱源位置误差 / m", "经验 CDF")
        axs[0, col].legend(fontsize=8)
        delta = np.zeros((2, 2))
        for i, overlap in enumerate([.5, 1.]):
            for j, gap in enumerate([6, 12]):
                means = [np.mean([r["metrics"]["success"]["100"][1] for r in use(m, n)
                                  if r["gap"] == gap and r["overlap"] == overlap])
                         for m in ["B", "C"]]
                delta[i, j] = 100*(means[1]-means[0])
        im = axs[1, col].imshow(delta, vmin=-8, vmax=8, cmap="Greys", aspect="auto")
        for i in range(2):
            for j in range(2):
                axs[1, col].text(j, i, f"{delta[i,j]:+.2f}", ha="center", va="center", fontsize=14,
                                 bbox=dict(facecolor="white", edgecolor="none", pad=3))
        axs[1, col].set(xticks=[0, 1], xticklabels=["6 dB", "12 dB"],
                        yticks=[0, 1], yticklabels=["IoU 0.5", "IoU 1.0"],
                        title=f"(b{col+1}) C − B 弱源成功率 / 百分点")
        fig.colorbar(im, ax=axs[1, col], shrink=.7)
        arrays[f"strata_difference_{n}"] = delta
    save(fig, out, "fig_5_5_errors")
    np.savez_compressed(out/"figure_arrays.npz", **arrays)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="analysis_v1")
    args = parser.parse_args()
    out = (HERE/args.run).resolve()
    if out.parent != HERE or out.exists():
        raise ValueError("Use a new, direct child output directory.")
    # Resolve and verify every report with a registered SHA before any computation.
    t0 = time.monotonic()
    g9_audit = read(G9/"final_audit_report.json")
    r1_audit = read(R1/"final_audit_report.json")
    for audit in [g9_audit, r1_audit]:
        if audit["status"] != "PASS":
            raise ValueError("Input audit not PASS")
        p = Path(audit["report"]["path"]).resolve(strict=True)
        if sha(p) != audit["report"]["sha256"]:
            raise ValueError(f"Report identity mismatch: {p}")
    report = read(G9/"comparison_report.json")
    rows = read(G9/"evaluation_rows.json")
    r1_report = read(R1/"report.json")
    score_rows = [read(p) for p in sorted((R1/"score_rows").glob("*.json"))]
    dense_rows = [read(p) for p in sorted((R1/"dense_rows").glob("*.json"))]
    if len(rows) != 15360 or len(score_rows) != 512 or len(dense_rows) != 24:
        raise ValueError("Unexpected frozen result count")
    for n in [4096, 16384]:
        for m in ["A", "B", "C", "DPD", "HR"]:
            subset = [r for r in rows if r["N"] == n and r["kind"] == "mixed"
                      and r["factor"] == 1. and r["method"] == m]
            s = summary(subset)
            for key, field in [("weak", "weak_100"), ("strong", "strong_100"),
                               ("both", "both_100"), ("rmse_m", "rmse_m"),
                               ("p95_m", "p95_m"), ("gospa_m", "gospa_m")]:
                check(f"frozen_summary_{n}_{m}_{key}",
                      abs(s[key]-report["summary"][f"{n}_{m}_1.0"][field]), 1e-9)
    out.mkdir()
    (out/"figures").mkdir()
    try:
        spec = equations()
        stats = figures(out, report, rows, score_rows, dense_rows)
        # Verify raw attribution counts, independently of prose.
        for line in r1_report["score_summary"]:
            sub = [r for r in score_rows if r["N"] == line["N"]
                   and not r["baseline_metrics"]["success"]["100"][1]]
            check(f"R1_failed_count_{line['N']}", abs(len(sub)-line["weak_failures"]), 0)
            check(f"R1_truth_higher_{line['N']}",
                  abs(sum(r["estimated"]["truth_higher"] for r in sub)
                      - line["estimated_truth_higher"]), 0)
        for path, digest in INPUTS.items():
            if sha(path) != digest:
                raise ValueError(f"Input changed during generation: {path}")
        result = dict(status="PASS", seconds=time.monotonic()-t0, checks=CHECKS,
                      theoretical_example=spec, combined_summary=stats,
                      figure_groups=9, no_training=True, no_test_access=True,
                      no_IQ_regeneration=True, no_formal_evaluation=True,
                      inputs=[dict(path=p, sha256=h) for p, h in INPUTS.items()])
        (out/"verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
        print(json.dumps({k: result[k] for k in ["status", "seconds", "figure_groups"]}))
        print("Checks:", len(CHECKS), "Protected inputs:", len(INPUTS))
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    except Exception as exc:
        (out/"failure.json").write_text(json.dumps(
            dict(error=repr(exc), checks=CHECKS), ensure_ascii=False, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
