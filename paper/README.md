# 论文独立工作目录

2026-10-20投稿整理：中文审读稿、英文投稿稿及各自补充材料分开维护。正文集中呈现两项方法及正式证据，详细基线公式、诊断和辅助图表移入补充材料；34组原理论公式在中文正文及补充材料中完整保留。英文投稿稿采用A4、12磅、单栏双倍行距。详见[当前核验](qa/review_supplement_20261020/核验说明.md)。

2026-10-20按投稿图形要求更新：现行四份文稿引用的13份不同图片全部使用原绘图代码重绘的矢量PDF，字体嵌入，空间谱及色条也不栅格化。PNG仅供预览，CSV保留追溯；不是将PNG封装为PDF。核对51份既有CSV的哈希不变，不运行定位或改变统计口径。此条替代本日早先“只保留PNG”的维护约定，历史压缩归档保持不变。

2026-10-19按用户最新决定，本论文现位于 `F:/DPD_Strong_Weak/paper`，这里是唯一正文维护入口，不再另设 `DPD_WeakSource_Localization`。前次分离的统计重放见`data/migration_replay_20261019/`，历史验证见`qa/migration_20261019/`与`qa/migration_build_20261019/`；本次迁回检查见`qa/paper_home_20261019/`与[交付记录](../outputs_e2e/maintenance/paper_home_20261019/交付记录.md)。当前`prepare_assignment_cost.py`、`analyze_contribution_closeout.py`及对应两个验证脚本通过`relocated_evidence.py`兼容两个历史根并解析本地证据，不改写旧合同。其他历史脚本不承诺自动迁移续跑兼容。

2026-10-19已完成贡献论证收尾：补齐谱方法及停止核验结论，解释恢复次数与RMSE来源，保留原公式、结果表及图。[修订与25页核验](qa/contribution_closeout_20261019/修订与核验.md)。本轮仅整理保存结果，不运行定位；前一版归档在`archive/contribution_closeout_20261019/`。

2026-10-19已完成统一评价匹配、AP候选数—VP共同计时及CDF/耗时展示，见[最终核验](qa/assignment_cost_20261019/修订与核验.md)。新评价器不覆盖冻结v2证据；当前结果重绘入口为`scripts/prepare_assignment_cost.py`，新版数据及图表分别写入`data/assignment_cost_20261019/`和`figures/assignment_cost_20261019/`。它只读取已完成结果，不运行定位。历史`prepare_formal_assets.py`及拆分文件保留追溯，不用于覆盖当前内联正文。修订前稿件保存在`archive/assignment_cost_20261019/`。

2026-10-18已完成独立审稿后获批的内容修订：补充LAOLS近邻、明确局部与联合消元及参照实现、补报既有区间并修正轨迹时间。最新检查见[审稿后修订核验](qa/review_followup_20261018/修订与核验.md)。此前的[机制证据与有限分段核验](qa/evidence_revision/修订与核验.md)继续保留。

中文审读稿修改 [paper_draft.tex](paper_draft.tex)，中文补充材料修改 [paper_supplement_zh.tex](paper_supplement_zh.tex)；英文投稿稿修改 [paper_submission.tex](paper_submission.tex)，英文补充材料修改 [paper_supplement.tex](paper_supplement.tex)。各文件内联完整正文，不通过 `\input` 引入章节或表格。作者、单位、资助与声明须由作者据实补全后再投稿。

## 独立编译

将本文件夹复制到项目之外仍可编译，不需要项目Python、实验结果、原模板目录或任何父目录文件。需要安装XeLaTeX、BibTeX及常用TeX包；当前中文字体使用Windows字体配置。

在本目录执行：

```powershell
& .\build.ps1 -Document All
```

脚本优先查找PATH中的TeX工具，也支持 `-TexBin '你的TeX工具目录'`。`-Document Chinese`（默认）生成中文主稿和补充材料，`English`生成英文两份，`All`生成全部四份；中间文件放在`build/`。中文两份使用文件夹内部的交叉引用，应通过脚本一起编译。

最小编译包需要对应的`.tex`文件、`build.ps1`、`figures/`、`references/`、`template/`。图片和BibTeX文献使用本文件夹内的相对路径，不依赖原项目；补充材料引用的全精度坐标表位于`data/submission/new_layouts.csv`。

## 素材分类和维护

| 位置 | 用途 |
| --- | --- |
| `paper_draft.tex`、`paper_supplement_zh.tex` | 中文主稿与补充材料 |
| `paper_submission.tex`、`paper_supplement.tex` | 英文投稿主稿与补充材料 |
| `figures/` | 投稿矢量PDF、PNG预览、绘图CSV及图注／来源文件 |
| `references/` | BibTeX文献库 |
| `template/` | CAS模板类、样式与说明 |
| `data/` | 已导出统计数据、图注及来源清单 |
| `build/` | 可重新生成的编译中间文件 |
| `qa/` | 核验记录和渲染页面 |
| `archive/` | 历史稿与展开前主文件备份 |
| `sections/`、`tables/` | 上版拆分源文件，保留追溯，不参与编译，也不作为后续正文编辑入口 |
| `scripts/` | 研究结果重绘和历史核验工具，不是独立编译依赖 |

`scripts/`中的来源核验和重绘工具只读取本项目已完成的证据，普通编译不调用它们。`prepare_assignment_cost.py`重绘CDF和成本图，追加`--review-supplement`重绘新布局CDF；`prepare_formal_assets.py --vector-only`重绘原正式图但不重写旧表格或来源清单；`validate_assignment_cost.py --submission`检查数值、来源、四稿图片及英文格式；`render_review.py`用于全页排版核验。`flatten_manuscript.py`和`migrate_cas_layout.py`是历史迁移工具，不要再次运行覆盖当前正文。

本次完整展开保持文字、公式和统计不变；独立编译验证见 [qa/独立编译核验.md](qa/独立编译核验.md)。
