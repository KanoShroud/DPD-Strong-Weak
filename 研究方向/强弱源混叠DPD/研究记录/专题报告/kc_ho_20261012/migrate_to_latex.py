"""One-time, loss-checked Markdown to editable LaTeX migration.

After migration document.tex is authoritative. This script refuses to overwrite
it on subsequent runs; do not regenerate the manuscript over later edits.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
OUT = HERE / "Latex PDF"
PANDOC = "D:/Software/anaconda3/Library/bin/pandoc.exe"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def escape(s):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(c, c) for c in s)


def text_tex(s, refs=True):
    s = escape(s)
    if not refs:
        return s
    s = re.sub(r"\[(\d+(?:,\d+)*)\]", lambda m:
               r"\cite{" + ",".join(f"ref{int(n):02}" for n in m[1].split(",")) + "}", s)
    s = re.sub(r"式（(\d+\.\d+)）", lambda m: r"式（\ref{eq:"+m[1]+"}）", s)
    s = re.sub(r"(?<=—)（(\d+\.\d+)）", lambda m: r"（\ref{eq:"+m[1]+"}）", s)
    s = re.sub(r"([图表])(\d+\.\d+)", lambda m:
               m[1]+r"\ref{"+{"图": "fig:", "表": "tab:"}[m[1]]+m[2]+"}", s)
    s = re.sub(r"第(\d+\.\d+(?:\.\d+)?)节", lambda m:
               r"第\ref{sec:"+m[1]+"}节", s)
    return s


def inline(items, refs=True):
    parts = []
    for node in items:
        kind, c = node["t"], node.get("c")
        if kind == "Str":
            parts.append(text_tex(c, refs))
        elif kind in {"Space", "SoftBreak"}:
            parts.append(" ")
        elif kind == "LineBreak":
            parts.append("\\\\\n")
        elif kind == "Math":
            if c[0]["t"] != "InlineMath":
                raise ValueError("Display math must be processed as a block")
            parts.append("$"+c[1]+"$")
        elif kind in {"Strong", "Emph", "Superscript", "Subscript"}:
            cmd = {"Strong": "textbf", "Emph": "emph", "Superscript": "textsuperscript",
                   "Subscript": "textsubscript"}[kind]
            parts.append("\\"+cmd+"{"+inline(c, refs)+"}")
        elif kind == "Link":
            parts.append(r"\href{"+c[2][0]+"}{"+inline(c[1], refs)+"}")
        elif kind == "Code":
            parts.append(r"\texttt{"+escape(c[1])+"}")
        elif kind == "RawInline" and c[0] == "html":
            pass
        else:
            raise ValueError(f"Unsupported inline node: {node}")
    return "".join(parts)


def caption(block):
    items = copy.deepcopy(block["c"][1][0]["c"][0]["c"])
    prefix = items.pop(0)["c"]
    if items and items[0]["t"] == "Space":
        items.pop(0)
    return prefix[0], prefix[1:], inline(items)


def equation(body):
    tag = re.search(r"\\tag\{([^}]+)\}", body)[1]
    body = re.sub(r"\s*\\tag\{[^}]+\}", "", body).strip()
    # Break long equalities without changing their mathematical tokens.
    if tag == "2.11":
        body = body.replace(r", \quad", ", \\\\\n&", 1)
        body = r"\begin{aligned}"+"\n&"+body+"\n"+r"\end{aligned}"
    elif tag == "3.1":
        body = body.replace(r", \qquad", ", \\\\\n&", 1)
        body = r"\begin{aligned}"+"\n&"+body+"\n"+r"\end{aligned}"
    return "\\begin{equation}\\label{eq:"+tag+"}\n"+body+"\n\\end{equation}", tag


WIDTHS = {
    "2.1": [.85, 1.35, .8], "3.1": [.65, 1.35], "3.2": [.85, 1.05, 1.1],
    "3.3": [.6, 1.4, 1, 1], "3.4": [.95, 1.05, 1.15, 1.05, 1.1, .85, .85],
    "3.5": [.65, .8, 1.15, 1.2, 1.2], "3.6": [1, .6, 1.15, 1.1, 1.15],
    "3.7": [1.1, 1, .9], "3.8": [1, 1, 1, 1, 1],
}


def table(block, tag, title):
    c = block["c"]
    headings = c[3][1]
    rows = [r for body in c[4] for r in body[3]]
    assert len(headings) == 1 and not c[5][1]
    specs = "".join(f"Y{{{w}}}" for w in WIDTHS[tag])

    def row_tex(row, bold=False):
        cells = []
        for cell in row[1]:
            assert cell[2:4] == [1, 1]
            value = " ".join(inline(b["c"]) for b in cell[4])
            cells.append((r"\textbf{"+value+"}") if bold else value)
        return " & ".join(cells) + r" \\"

    return ("\\begin{table}[!htbp]\n\\centering\n\\caption{"+title+"}\\label{tab:"+tag+"}\n"
            "\\small\n\\setlength{\\tabcolsep}{4pt}\n\\renewcommand{\\arraystretch}{1.2}\n"
            "\\begin{tabularx}{\\linewidth}{"+specs+"}\n\\toprule\n"
            +row_tex(headings[0], True)+"\n\\midrule\n"
            +"\n".join(row_tex(r) for r in rows)+"\n\\bottomrule\n\\end{tabularx}\n\\end{table}")


PREAMBLE = r"""% !TeX program = xelatex
% !TeX encoding = UTF-8
% Main editable manuscript. Figures are packaged in ./figures/.
% The original format reference is preserved in ../history/before_latex_migration/.
\documentclass[11pt,a4paper]{article}
\usepackage[UTF8,scheme=plain,fontset=windows]{ctex}
\usepackage{amsmath,amssymb}
\usepackage{geometry}
\geometry{a4paper,margin=1in}
\usepackage{graphicx,booktabs,tabularx,array}
\usepackage{caption}
\usepackage[section]{placeins}
\usepackage{flafter}
\usepackage{cite,etoolbox}
\usepackage[unicode,hidelinks]{hyperref}
\hypersetup{pdftitle={短时强弱重叠多源被动定位},pdfsubject={中文学术交流稿}}
\numberwithin{equation}{section}
\numberwithin{figure}{section}
\numberwithin{table}{section}
\renewcommand{\figurename}{图}
\renewcommand{\tablename}{表}
\renewcommand{\contentsname}{目录}
\captionsetup{justification=centering,singlelinecheck=false,labelsep=quad,font=small,labelfont=bf}
\captionsetup[table]{position=top,skip=6pt}
\newcolumntype{Y}[1]{>{\hsize=#1\hsize\linewidth=\hsize\raggedright\arraybackslash}X}
\patchcmd{\thebibliography}{\section*{\refname}}{}{}{}
\setcounter{tocdepth}{1}
\setcounter{secnumdepth}{3}
\setlength{\parindent}{2em}
\setlength{\parskip}{.25em}
\setlength{\emergencystretch}{2em}
\linespread{1.18}
\renewcommand{\topfraction}{.9}
\renewcommand{\bottomfraction}{.8}
\renewcommand{\textfraction}{.08}
\renewcommand{\floatpagefraction}{.75}
\setcounter{topnumber}{3}
\setcounter{totalnumber}{5}
\raggedbottom
\title{\bfseries 短时强弱重叠多源被动定位}
\author{\normalsize 交流对象：K. C. Ho 教授}
\date{\normalsize 2026年10月12日\\[3pt]
\small 数据与分析截至：2026年10月11日\quad 版本：中文学术交流稿}
\begin{document}
\maketitle
"""


def main():
    manifest_path = OUT/"migration_validation.json"
    if manifest_path.exists():
        raise FileExistsError("LaTeX is now authoritative; edit document.tex directly.")
    source = HERE/"学术交流报告_中文.md"
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8:replace")
    result = subprocess.run([PANDOC, str(source), "-f", "markdown+tex_math_dollars", "-t", "json"],
                            capture_output=True, text=True, encoding="utf-8", errors="strict",
                            check=True, env=env)
    ast = json.loads(result.stdout)
    blocks = ast["blocks"]
    body, consumed, maps = [], [], {"figures": [], "tables": [], "equations": [], "references": []}
    # Title, metadata, and the five-entry contents list are handled in native LaTeX.
    excluded = {0, 1, 4, 5}
    i = 0
    in_bib = False
    while i < len(blocks):
        b = blocks[i]
        if i in excluded:
            consumed.append(i)
            i += 1
            continue
        start = i
        kind = b["t"]
        if kind == "Header":
            level, _, items = copy.deepcopy(b["c"])
            if inline(items) == "摘要":
                tex = r"\section*{摘要}"
            else:
                tag = items.pop(0)["c"].rstrip(".")
                if items[0]["t"] == "Space":
                    items.pop(0)
                title = inline(items)
                command = {2: "section", 3: "subsection", 4: "subsubsection"}[level]
                tex = "\\"+command+"{"+title+"}\\label{sec:"+tag+"}"
                if tag == "1":
                    tex = "\\tableofcontents\n\\clearpage\n"+tex
                if tag == "5":
                    tex += "\n\\begin{thebibliography}{99}\n\\small"
                    in_bib = True
            i += 1
        elif kind == "Figure":
            _, tag, title = caption(blocks[i+1])
            image = b["c"][2][0]["c"][0]["c"][2][0]
            name = "fig_"+tag.replace(".", "_")+".pdf"
            source_pdf = HERE/"analysis_v7_latex/figures"/(Path(image).stem+".pdf")
            (OUT/"figures").mkdir(exist_ok=True)
            shutil.copy2(source_pdf, OUT/"figures"/name)
            note = inline(blocks[i+2]["c"])
            tex = ("\\begin{figure}[!htbp]\n\\centering\n"
                   "\\includegraphics[width=\\linewidth]{figures/"+name+"}\n"
                   "\\caption{"+title+"}\\label{fig:"+tag+"}\n"
                   "\\begin{minipage}{\\linewidth}\\small\n"+note+"\n\\end{minipage}\n\\end{figure}")
            maps["figures"].append(dict(number=tag, source=str(source_pdf.relative_to(HERE)),
                                       destination="figures/"+name, sha256=digest(source_pdf)))
            i += 3
        elif kind == "Div":
            cap_kind, tag, title = caption(b)
            assert cap_kind == "表" and blocks[i+1]["t"] == "Table"
            tex = table(blocks[i+1], tag, title)
            maps["tables"].append(tag)
            i += 2
        elif kind in {"Para", "Plain"}:
            items = copy.deepcopy(b["c"])
            if len(items) == 1 and items[0]["t"] == "Math" and items[0]["c"][0]["t"] == "DisplayMath":
                tex, tag = equation(items[0]["c"][1])
                maps["equations"].append(tag)
            elif in_bib:
                tag = int(re.search(r"\[(\d+)\]", items[0]["c"])[1])
                items.pop(0)
                if items[0]["t"] == "Space":
                    items.pop(0)
                tex = f"\\bibitem{{ref{tag:02}}} "+inline(items, refs=False)
                maps["references"].append(tag)
            else:
                tex = inline(items)
            i += 1
        else:
            raise ValueError(f"Unhandled block {i}: {kind}")
        consumed.extend(range(start, i))
        if tex.strip():
            body.append(f"% Source block {start}"+"\n"+tex)
    assert consumed == list(range(len(blocks))), "Missing or duplicated source blocks"
    assert [len(maps[k]) for k in ["equations", "figures", "tables", "references"]] == [22, 8, 9, 12]
    text = PREAMBLE+"\n\n".join(body)+"\n\\end{thebibliography}\n\\end{document}\n"
    (OUT/"document.tex").write_text(text, encoding="utf-8", newline="\n")
    maps.update(status="PASS", source_sha256=digest(source), tex_sha256=digest(OUT/"document.tex"),
                source_blocks=len(blocks), all_source_blocks_consumed=True,
                future_edit_source="document.tex", conversion="one-time; no regeneration after manual edits")
    manifest_path.write_text(json.dumps(maps, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "source_blocks": len(blocks),
                      "equations": 22, "figures": 8, "tables": 9, "references": 12}))


if __name__ == "__main__":
    main()
