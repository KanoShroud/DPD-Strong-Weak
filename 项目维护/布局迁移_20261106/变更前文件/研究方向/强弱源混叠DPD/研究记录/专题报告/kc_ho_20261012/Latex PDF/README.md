# LaTeX 报告维护入口

## 当前版本：确定性相对响应与正式比较（2026-10-14）

当前中文版为29页，正文已经按G10代码、实现文档和正式保存结果更新。编译入口为 `document.tex`，当前第2—4节为 `known_response_main.tex`，汇总表为 `g10_summary.tex`、`g10_cost.tex`。旧实验完整保留附录，共同LS和信息关系移至正文。英文版与双语Word讲稿未同步，本轮保持原版。

绘图与核验入口分别为 `../g10_report_assets.py` 和 `../validate_g10_report.py`，数据及文稿检查见 `../qa_g10_report/validation.json`。编译继续运行 `build.ps1`。完整来源、输出身份和版本关系见 [维护说明](../G10中文报告维护说明.md) 及 `../chinese_final_manifest.json`。以下未知增益说明属于历史版本，其中旧核验脚本与章节统计不再适用于当前中文稿。

## 历史版本：未知复增益模型（2026-10-13）

当前报告25页，正文采用未知且跨频共享的复增益模型，附录A—C保留已知相对增益推导、全部实验和原讨论。新实验尚未执行，第3节给出协议并预留结果。中英文各含43个编号公式、11幅图、14张表和14篇IEEE格式参考文献。

- `document.tex`是编译主入口，直接维护摘要、引言、参考文献和附录；通过 `\input{unknown_gain_main.tex}` 载入新的第2—4节。两份TeX都属于当前可编辑源文件，不从旧Markdown生成。
- `unknown_gain_main.tex`维护新模型、LS与特征值/子空间推导、实验协议及讨论；图2.1用TikZ直接绘制，直角循环箭头、文字和节点可编辑。
- `figures/`十幅旧图原字节保留，现显示为附录图A.1—A.2、B.1—B.8。已否定的备选图未恢复。
- 当前文稿核验为 `../qa_unknown_gain_final/validation.json`；旧模型保全索引为 `../unknown_gain_migration.json`。旧验证脚本固定了旧章节数和公式数，不再作为当前整篇报告的入口。
- [实验执行指导](../未知复增益DPD_功率差扫描执行方案.md)供另一对话承接；原双语Word讲稿尚未同步新模型，待实验完成后更新。

在本目录运行 `& './build.ps1'`，连续三遍XeLaTeX。仓库根目录使用项目Python运行 `outputs_e2e/reports/kc_ho_20261012/validate_unknown_gain.py --output 新QA目录`，核对双语公式、引用、旧结果保全并渲染。页面视觉复查仍需人工完成。`restructure_unknown_gain.py`仅记录本次结构迁移，禁止用其覆盖后续编辑；它会拒绝重复迁移。

下文为2026-10-11历史维护说明，其中页数、章节号及QA入口均属于旧版。

## 2026-10-11 历史维护说明

**当前状态：中文版定稿，已按用户要求同步英文版的最新表述。** 本次明确频谱重叠、预算截止前输出、百分点单位、总体RMSE，以及成功数与新增恢复数的区别；实验数据、公式和结论不变。正文、PDF和十幅配图的身份见 `../chinese_final_manifest.json`，本次核验见 `qa_bilingual_sync_01/validation.json`。对应英文版位于 `../Latex PDF English/`，两版均直接维护各自LaTeX。

后续正文修改以 **document.tex** 为准。上一级 Markdown 保留本次迁移时的对照稿，不再作为 LaTeX 的生成源。

- `document.tex`：全部正文、25个编号公式、10幅编号图、11张表及12篇IEEE格式参考文献。章节、公式、图表和引文使用自动编号；修改时保留 `label`、`ref`、`cite` 与 `bibitem` 的对应关系。
- `document.pdf`：已编译的18页中文报告，无日期和目录。图2.1采用横向三子图并进入第2页底部；当前排版核验见 `qa_layout_final/validation.json`。
- `figures/`：正文使用10幅PDF配图，文字、线条和柱形为矢量，空间评分图的热图部分为嵌入图像；全部采用本目录内的相对路径。四幅备选图已从当前文档包移除。
- `build.ps1`：调用现有TeX Live中的XeLaTeX，连续编译三次，更新交叉引用。使用A4、1英寸页边距及Windows中文字体。

在此目录执行：

```powershell
& './build.ps1'
```

也可在 TeX 编辑器中打开 `document.tex`，选择 **XeLaTeX** 编译。文件首行已指定编译引擎；中文稿不使用原英文参考文件的 pdfLaTeX 配置。

当前主成功半径为20 m，GOSPA截断参数为100 m。图3.1、3.2、3.3、3.5保留柱状图及纹理，不再附加备选图。图3.7、3.8是完整B与补充搜索D的新布局结果图，继续保留。图3.5使用预算内计入耗时，图3.8使用完成预定流程的实测时间，两者分别解释。

方法说明已前移至第2.4节表2.2，在首次使用A/B/C/D之前定义。第2.5.4节给出区域上界，第3.10节及表3.9纳入R3单次弱源更新比较，并同步摘要、引言、结论与讨论。此处5 m单次扫描与完整方法D分开命名；历史8条对照初始只有6条满足20 m，全文已按这一事实解释。

图形入口为 `../refresh_figures_20m.py --run 新分析目录 --install`，已用资产仍来自 `../analysis_v11_20m/`，图2.1、2.2、3.4沿用此前资产。本轮未重绘；安装逻辑已排除备选图，旧分析目录中的备选文件仅作历史保留。

R3只读核验入口为 `../readback_r3_report.py --run 新回读目录`，最终输出为 `../analysis_r3_readback_02/readback.json`。文稿核验入口为 `../validate_report_revision.py --analysis analysis_v11_20m --r3 analysis_r3_readback_02 --qa 新QA目录`，最终检查为 `qa_r3_v2/validation.json`：5852项文稿、数据及输入身份检查通过；三遍XeLaTeX、三份修改脚本的py_compile与Ruff F/E9通过。已查看20页PDF，并放大检查方法表、新公式、R3表、文献及灰度配图。

原格式参考和迁移前文件仍在 `../history/before_latex_migration/`。本轮直接编辑当前文件，未新建备份。`../migrate_to_latex.py`、旧 `validate_latex.py` 及迁移核验仅记录初次迁移，不应用其覆盖当前正文。当前身份以 `qa_r3_v2/validation.json` 为准。所有研究数据来自冻结结果，未训练、未生成IQ或重跑定位算法。
