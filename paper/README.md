# 论文独立工作目录

2026-10-18已完成贡献表述、既有机制证据整理及有限分段核验；最新状态和检查结果见[修订与核验](qa/evidence_revision/修订与核验.md)。

今后直接修改 [paper_draft.tex](paper_draft.tex)。摘要、正文、公式、表格和附录已全部写入这一文件，不再通过 `\input` 引入章节或表格。阅读版为 [paper_draft.pdf](paper_draft.pdf)。

## 独立编译

将本文件夹复制到项目之外仍可编译，不需要项目Python、实验结果、原模板目录或任何父目录文件。需要安装XeLaTeX、BibTeX及常用TeX包；当前中文字体使用Windows字体配置。

在本目录执行：

```powershell
& .\build.ps1
```

脚本优先查找PATH中的TeX工具，也支持 `-TexBin '你的TeX工具目录'`。生成 `paper_draft.pdf`，中间文件放在 `build/`。也可用编辑器直接打开 `paper_draft.tex`，按XeLaTeX → BibTeX → XeLaTeX → XeLaTeX编译。

最小编译包只需要 `paper_draft.tex`、`build.ps1`、`figures/`、`references/`、`template/`。图片和BibTeX文献仍使用本文件夹内的相对路径，这是LaTeX正常素材组织，不依赖原项目。

## 素材分类和维护

| 位置 | 用途 |
| --- | --- |
| `paper_draft.tex` | 唯一正文编辑入口，包含完整正文和表格 |
| `figures/` | 图的PDF、PNG与绘图CSV |
| `references/` | BibTeX文献库 |
| `template/` | CAS模板类、样式与说明 |
| `data/` | 已导出统计数据、图注及来源清单 |
| `build/` | 可重新生成的编译中间文件 |
| `qa/` | 核验记录和渲染页面 |
| `archive/` | 历史稿与展开前主文件备份 |
| `sections/`、`tables/` | 上版拆分源文件，保留追溯，不参与编译，也不作为后续正文编辑入口 |
| `scripts/` | 研究结果重绘和历史核验工具，不是独立编译依赖 |

`scripts/`中的实验来源核验、重绘工具仍可能需要原项目结果，只在需要重新分析数据时使用；普通编译不调用它们。修改主文档后，不要用历史拆分文件覆盖它。旧 `paper.pdf` 为历史输出，当前阅读与编译输出统一为 `paper_draft.pdf`。

本次完整展开保持文字、公式和统计不变；独立编译验证见 [qa/独立编译核验.md](qa/独立编译核验.md)。
