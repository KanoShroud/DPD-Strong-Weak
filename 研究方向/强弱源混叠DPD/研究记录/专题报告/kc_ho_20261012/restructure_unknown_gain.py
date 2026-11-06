"""One-time structural migration from the current LaTeX, preserving old evidence.

The manuscripts remain directly editable. This is not a Markdown converter.
"""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def scientific_blocks(text):
    result = {}
    for env in ("equation", "table", "figure"):
        for block in re.findall(r"\\begin\{" + env + r"\}.*?\\end\{" + env + r"\}", text, re.S):
            labels = re.findall(r"\\label\{([^}]+)\}", block)
            if labels:
                if env == "table":
                    block = re.search(r"\\begin\{tabularx\}.*?\\end\{tabularx\}", block, re.S).group()
                elif env == "figure":
                    block = "|".join(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", block))
                result[labels[0]] = sha(re.sub(r"\s+", "", block).encode("utf-8"))
    return result


ABSTRACT = {
"zh": r"""频谱重叠与接收功率差异使非合作多源定位中的时延提取和跨站关联更加困难。本文面向四个同步单天线站和两个源，采用直达传播及跨频共享的未知复增益模型，建立消去未知波形后的联合最小二乘目标。由单源LS逐步推导最大特征值空间谱，并说明它如何用于多源残差上的交替更新，以及它与正交投影条件谱、MUSIC型DPD和DPD-HR的区别。局部可辨识性与曲率分析同时考虑波形和增益干扰参数。仿真比较拟在固定弱源及噪声条件下，将接收功率差从0 dB增加至15 dB，评价强弱源精度、成功率、误差分布与完整处理时间。未知增益条件下的结果尚待获得。附录保留已知相对增益条件下的推导与实验，用于说明该条件下的候选遗漏及搜索成本，不作为新模型的性能证据。""",
"en": r"""Spectral overlap and unequal received powers complicate delay extraction and association across receivers in passive multisource localization. This report considers four synchronized single-antenna receivers and two sources under line-of-sight propagation, with unknown complex gains shared across frequencies. Eliminating the unknown waveforms gives a joint least-squares objective. A largest-eigenvalue spatial spectrum is derived from single-source LS and extended to alternating updates on multisource residuals. Its relationship to projected conditional spectra, MUSIC-type DPD, and DPD-HR is established. Local identifiability and curvature are discussed with both waveform and gain nuisance parameters included. A planned comparison holds weak-source power and noise fixed while increasing the received-power difference from 0 to 15 dB, and evaluates localization accuracy, success rates, error distributions, and total processing time for both sources. Results with unknown gains are pending. The appendices retain derivations and experiments with known relative gains as evidence about candidate omission and search cost under that assumption, not as performance evidence for the new model."""
}

RELATED = {
"zh": r"""\subsection{相关研究与本文问题}\label{sec:1.3}
Chen等从宽带接收信号建立最大似然定位模型，通过消去未知波形处理联合估计\cite{ref05}。未知增益单源模型可导出最大特征值评分；多源模型还需要区分联合拟合与逐点扫描。Weiss与Amar的多源DPD利用信号子空间构造位置评分\cite{ref04}，DPD-HR则结合直接定位与最小方差思想，通过逆协方差提高空间分辨能力\cite{ref06}。这些方法的统计观测要求须与短记录一致。

较新的DPD-PDL同时估计位置离网偏移和未知复衰减，适用于单天线多站的稀疏直接定位\cite{ref13}。另一些多源ML方法使用共同已知参考信号获得的CSI\cite{ref14}，其输入条件与未知独立波形的混合观测不同。方法选择应依据观测、未知参数及求解代价，而不能仅依据发表时间。

投影抑制、候选保留和循环精修也用于多源估计。Round-robin定位采用投影及位置回访\cite{ref07}，NOMP结合分量检测与连续精修\cite{ref08}，SRP与组稀疏方法保留多个方向候选\cite{ref09}。其搜索思想可以参考，但原有声学、正弦或阵列模型中的性能不能直接移用于本文。

本文首先明确接收模型、空间谱和联合目标的关系，再检验代表性方法随强弱接收功率比变化的定位表现。弱源功率通过独立单源校准后固定，只增加另一源的功率。是否需要进一步解释退化机制或设计新的求解方法，由比较结果决定。已知相对增益条件下的已有推导与结果保留在附录\ref{sec:2}--\ref{sec:4}，与正文未知增益模型分别解释。
""",
"en": r"""\subsection{Related work and research question}\label{sec:1.3}
Chen et al. formulate maximum-likelihood localization from wideband observations and eliminate unknown waveforms in joint estimation\cite{ref05}. A single-source model with unknown gains leads to a largest-eigenvalue score; multisource estimation requires a distinction between joint fitting and pointwise scanning. The multisource DPD of Weiss and Amar uses signal subspaces\cite{ref04}, whereas DPD-HR combines direct localization with a minimum-variance criterion and inverse covariance matrices\cite{ref06}. Their statistical observation requirements must be consistent with short records.

The more recent DPD-PDL estimates off-grid position offsets and unknown complex attenuation jointly in sparse direct localization with single-antenna receivers\cite{ref13}. Other multisource ML methods use CSI obtained from a common known reference signal\cite{ref14}, which differs from a mixture of unknown independent waveforms. Method selection therefore depends on observations, unknown parameters, and computational cost, not publication date alone.

Projection, candidate retention, and cyclic refinement also appear in multisource estimation. Round-robin localization uses projection and position revisits\cite{ref07}; NOMP combines component detection with continuous refinement\cite{ref08}; SRP and group-sparse methods retain multiple direction candidates\cite{ref09}. Their search ideas are relevant, but results from the original acoustic, sinusoidal, or array models do not transfer directly to the present setting.

This report first establishes the relationships between the observation model, spatial spectra, and joint objective, then specifies a comparison of representative methods as the received-power ratio changes. Weak-source power is calibrated independently and fixed; only the other source becomes stronger. The need for mechanism analysis or a new solver will be judged from those comparisons. Existing derivations and results with known relative gains are retained in Appendices~\ref{sec:2}--\ref{sec:4} and interpreted separately from the unknown-gain model.
"""
}

NEW_BIB = {
"ref13": r'''\bibitem{ref13} Z. Wang, D. Wang, F. Nie, and N. Zheng, ``TDOA-based direct position determination method with parametric dictionary learning,'' \emph{IEEE Sensors J.}, 2026, doi: \href{https://doi.org/10.1109/JSEN.2026.3693237}{\nolinkurl{10.1109/JSEN.2026.3693237}}.''',
"ref14": r'''\bibitem{ref14} Z. Wang, B. Tan, E.-S. Lohan, M. Valkama, and Q. Wan, ``Maximum likelihood direct position determination of multiple sources with channel state information,'' \emph{IEEE Trans. Veh. Technol.}, vol. 74, no. 6, pp. 10017--10022, 2025, doi: \href{https://doi.org/10.1109/TVT.2025.3538794}{\nolinkurl{10.1109/TVT.2025.3538794}}.'''
}


def main():
    provenance = {"scope": "theory, document structure, and experiment guidance only", "editions": {}}
    for lang, directory in [("zh", "Latex PDF"), ("en", "Latex PDF English")]:
        folder = ROOT / directory
        path = folder / "document.tex"
        original_bytes = path.read_bytes()
        text = original_bytes.decode("utf-8").replace("\r\n", "\n")
        if r"\input{unknown_gain_main.tex}" in text:
            raise RuntimeError("Migration already applied; edit the LaTeX directly.")
        sec2 = re.search(r"\\section\{[^\n]+\}\\label\{sec:2\}", text).start()
        sec5 = re.search(r"\\section\{[^\n]+\}\\label\{sec:5\}", text).start()
        old = text[sec2:sec5].strip()
        blocks_before = scientific_blocks(old)
        prefix = text[:sec2]
        prefix = re.sub(r"(% Source block 3\n).*?(?=% Source block 7)", lambda m: m[1] + ABSTRACT[lang] + "\n\n", prefix, count=1, flags=re.S)
        related_start = re.search(r"\\subsection\{[^\n]+\}\\label\{sec:1.3\}", prefix).start()
        prefix = prefix[:related_start] + RELATED[lang] + "\n"
        prefix = prefix.replace(r"\usepackage{amsmath,amssymb}", r"\usepackage{amsmath,amssymb}" + "\n" + r"\usepackage{tikz}" + "\n" + r"\usetikzlibrary{arrows.meta,positioning}")
        prefix = prefix.replace(r"\pretocmd{\subsection}{\FloatBarrier}{}{}" + "\n", "").replace(r"\pretocmd{\subsubsection}{\FloatBarrier}{}{}" + "\n", "")
        if lang == "en":
            prefix = prefix.replace(r"\setcounter{tocdepth}{1}", "\\setlength{\\intextsep}{8pt plus 2pt minus 2pt}\n\\setlength{\\textfloatsep}{10pt plus 2pt minus 2pt}\n\\setlength{\\floatsep}{8pt plus 2pt minus 2pt}\n\\setcounter{tocdepth}{1}")
        titles = ["已知相对增益模型及求解", "已知相对增益条件下的实验结果", "已知增益实验的讨论"] if lang == "zh" else ["Known-relative-gain model and solvers", "Experiments with known relative gains", "Discussion of the known-gain experiments"]
        notes = [
            r"本附录及附录\ref{sec:3}、\ref{sec:4}采用位置确定的$1/d$相对增益。以下推导、信息界和搜索界仅适用于该理想匹配条件；不作为正文未知复增益模型的推导。符号按本附录局部定义使用。",
            r"本附录全部数据均来自已知相对增益条件下的既有实验。原比较组、新布局比较组与固定诊断组分别保留，统计分母、成功判据和处理时间不变；不用于推断正文未知增益模型的性能。",
            r"以下讨论保留已知相对增益实验所提出的搜索与联合更新问题，其证据范围限于附录\ref{sec:3}。未知增益条件下是否出现同样现象，尚待正文实验检验。",
        ] if lang == "zh" else [
            r"This appendix and Appendices~\ref{sec:3} and~\ref{sec:4} assume relative gains determined by the position-dependent $1/d$ law. The derivations, information bounds, and search bounds below apply only to that ideally matched setting. They do not establish results for the unknown-complex-gain model in the main text. Notation is local to this appendix.",
            r"All data in this appendix are from the existing experiments with known relative gains. The original comparison, new-layout comparison, and selected diagnostics remain separate, with unchanged denominators, success criteria, and processing times. They do not establish performance for the unknown-gain model.",
            r"The following discussion retains the search and joint-update questions raised by the known-relative-gain experiments in Appendix~\ref{sec:3}. Whether the same behavior occurs with unknown gains remains to be tested in the main-text experiment.",
        ]
        for i, (title, note) in enumerate(zip(titles, notes), 2):
            pattern = r"\\section\{[^\n]+\}\\label\{sec:" + str(i) + r"\}"
            old = re.sub(pattern, lambda m: rf"\section{{{title}}}\label{{sec:{i}}}" + "\n\n" + note, old, count=1)
        if lang == "zh":
            old = old.replace("未知波形的自由度并不包含每站任意复增益，否则位置引起的相位变化可被增益吸收，失去相应定位约束。", "本附录不估计逐站增益。若允许每站每频点独立变化的任意复增益，时延相位可被吸收；正文采用的跨频共享增益不属于这种退化情形。")
            extra = "原始混合观测先经逐站RMS归一化产生最大特征值谱候选，随后A--C在原始观测的集中LS目标下执行多初值、逐源搜索、连续精修及全部波形重估。搜索不限于初始位置邻域，第3、5轮还有重新搜索；D进一步补充全区域扫描。初始化可能影响有限预算下的结果，但现有比较未单独量化这一影响。"
        else:
            old = old.replace("The unknown-waveform model does not include an arbitrary complex gain at each receiver: such gains could absorb position-dependent phase changes and remove the corresponding localization constraints.", "Gains are not estimated in this appendix. Arbitrary gains varying independently at every receiver and frequency could absorb delay phases; the frequency-shared gains in the main text do not constitute that degenerate case.")
            extra = "The mixed observations are first normalized by receiver RMS to generate largest-eigenvalue spectral candidates. A--C then use the concentrated LS objective on raw observations for multiple starts, source-wise searches, continuous refinement, and joint waveform refitting. Searches are not restricted to neighborhoods of the initial positions, and rounds 3 and 5 include renewed searches. D adds a further full-region scan. Initialization may affect finite-budget estimates, but its effect has not been isolated in the existing comparisons."
        marker = "% Source block 62"
        assert marker in old
        old = old.replace(marker, extra + "\n\n" + marker, 1)
        # Exact equation bodies, numeric table cells, and referenced figure assets are preserved.
        blocks_after = scientific_blocks(old)
        assert blocks_before == blocks_after
        main_text = (folder / "unknown_gain_main.tex").read_text(encoding="utf-8")
        main_text = re.sub(r"\\boldsymbol\\(mu|rho)", r"\\boldsymbol{\\\1}", main_text)
        (folder / "unknown_gain_main.tex").write_text(main_text, encoding="utf-8", newline="\r\n")
        entries = dict((m[0], m[1].strip()) for m in re.findall(r"\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\})", text[sec5:], re.S))
        entries = {key: r"\bibitem{" + key + "}" + re.sub(r"% Source block \d+\s*", "", value).strip() for key, value in entries.items()}
        entries.update(NEW_BIB)
        order = []
        for cited in re.findall(r"\\cite\{([^}]+)\}", prefix + main_text + old):
            for key in cited.split(","):
                if key not in order:
                    order.append(key)
        assert set(entries) == set(order)
        bib = (r"\section{参考文献}" if lang == "zh" else r"\section{References}") + r"\label{sec:5}" + "\n\\begin{thebibliography}{99}\n\\small\n" + "\n\n".join(entries[key] for key in order) + "\n\\end{thebibliography}\n"
        result = prefix + "\\input{unknown_gain_main.tex}\n\n\\clearpage\n" + bib + "\n\\clearpage\n\\appendix\n" + old.rstrip() + "\n\\end{document}\n"
        path.write_text(result, encoding="utf-8", newline="\r\n")
        provenance["editions"][lang] = {
            "original_source_sha256": sha(original_bytes),
            "legacy_scientific_blocks": blocks_before,
            "bibliography_order": order,
            "figure_assets": {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in (folder / "figures").glob("*.pdf")},
        }
    (ROOT / "unknown_gain_migration.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Migrated both current LaTeX reports; legacy scientific blocks preserved.")


if __name__ == "__main__":
    main()
