# 论文独立工作目录

2026-10-19按用户最新决定，本论文迁回 `F:/SourceCount_DPD_E2E/paper`，这里是唯一正文维护入口，不再另设 `DPD_WeakSource_Localization`。前次分离的统计重放见`data/migration_replay_20261019/`，历史验证见`qa/migration_20261019/`与`qa/migration_build_20261019/`；本次迁回检查见`qa/paper_home_20261019/`与[交付记录](../outputs_e2e/maintenance/paper_home_20261019/交付记录.md)。当前`prepare_assignment_cost.py`、`analyze_contribution_closeout.py`及对应两个验证脚本通过`relocated_evidence.py`兼容两个历史根并解析本地证据，不改写旧合同。其他历史脚本不承诺自动迁移续跑兼容。

2026-10-19已完成贡献论证收尾：补齐谱方法及停止核验结论，解释恢复次数与RMSE来源，保留原公式、结果表及图。[修订与25页核验](qa/contribution_closeout_20261019/修订与核验.md)。本轮仅整理保存结果，不运行定位；前一版归档在`archive/contribution_closeout_20261019/`。

2026-10-19已完成统一评价匹配、AP候选数—VP共同计时及CDF/耗时展示，见[最终核验](qa/assignment_cost_20261019/修订与核验.md)。新评价器不覆盖冻结v2证据；当前结果重绘入口为`scripts/prepare_assignment_cost.py`，新版数据及图表分别写入`data/assignment_cost_20261019/`和`figures/assignment_cost_20261019/`。它只读取已完成结果，不运行定位。历史`prepare_formal_assets.py`及拆分文件保留追溯，不用于覆盖当前内联正文。修订前稿件保存在`archive/assignment_cost_20261019/`。

2026-10-18已完成独立审稿后获批的内容修订：补充LAOLS近邻、明确局部与联合消元及参照实现、补报既有区间并修正轨迹时间。最新检查见[审稿后修订核验](qa/review_followup_20261018/修订与核验.md)。此前的[机制证据与有限分段核验](qa/evidence_revision/修订与核验.md)继续保留。

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
