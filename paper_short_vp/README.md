# VP短文

本目录维护聚焦变量投影曲率改进的Signal Processing短文，作为长文之外的投稿备选，最终仅选择一篇投稿。英文主文件为`main.tex`，中文为`main_zh.tex`；两者内容对应，不设补充材料。正文围绕局部曲率改进、机制推导及240条受控配对展开，并简要报告480条新源位置观测的完整流程表现。

## 文件与编译

| 位置 | 用途 |
|---|---|
| `main.tex`、`main.pdf` | 英文短文 |
| `main_zh.tex`、`main_zh.pdf` | 内容对应的中文审读稿 |
| `template/` | 项目原Elsevier CAS 2.4类文件与样式；未修改 |
| `references/` | 当前使用的7篇文献 |
| `figures/` | 两幅矢量PDF、600 dpi PNG预览及CDF绘图CSV |
| `data/` | 已保存结果、36个源位置构型坐标、冻结参数及来源身份 |
| `scripts/prepare_assets.py` | 从本目录保存结果重绘和核验，不运行定位 |
| `build/` | 可再生的编译、检查与渲染文件，不参与科学结论 |

在本目录运行：

```powershell
& .\build.ps1 -Document All
```

可用`-Document English`或`Chinese`只编译一种语言，`-TexBin`可指定TeX目录。编译仅需要对应`.tex`、`template/`、`references/`、`figures/`及XeLaTeX/BibTeX，不依赖项目Python、实验输出或父目录。中文需Windows字体。复制整个短文目录到其他位置仍可独立编译。

英文采用`cas-sc`的原生`12pt,review`设置。类文件内部固定了192×262 mm纸张，因此正文仅用`geometry{a4paper}`落实已批准的A4要求，保留类文件原有页边距、字体、图注、公式对齐、页眉页脚。中文增加字体及名称翻译支持，篇幅不限。两份稿件均不通过章节引用依赖长文。

## 证据与复现

`data/per_case.csv`和`summary.csv`是已完成补证的原导出文件，复制时逐字节核对；评价版本为`EVAL-20261019-v3-SINGLE-SQUARED-ASSIGNMENT`。1920次求解包含240条曲率配对的480次求解及480条新观测的1440次求解，不能把求解数当独立观测数。

- FC–VP：固定系数曲率＋波形消元接受目标；VP–VP：消元曲率＋同一接受目标。两者使用相同发现规则、系数重估时序、保护和外层停止。它们是场景适配及机制对照，不冒称原论文逐行复现。
- `VP_VP`与`PROFILE_DIAG`是本次机制和新构型两批中的同一冻结VP流程，统计与时间分别来自各自批次。
- 83.77%来自240条共同配对的平均完整时间，不能与旧79.92%或其他批次时间混算。置信区间按固定布局内随机实现配对重采样，六档功率差保持关联。
- 数据中的`updates`计数为Newton试探方向数，包括未接受方向；`cycles`累计所有源添加阶段。
- `data/source_manifest.json`登记输入来源、SHA256、配置、随机实现与种子；`source_positions.csv`保存两个实验的36个构型坐标。长文在其他对话中并行维护，其当次版本仅作为写作参考，不是短文编译或数值核验依赖。

普通重绘和检查：

```powershell
& 'D:\Software\anaconda3\envs\PyTorch\python.exe' scripts\prepare_assets.py
& 'D:\Software\anaconda3\envs\PyTorch\python.exe' scripts\prepare_assets.py --check-documents
```

`--import-from`仅用于明确需要刷新写作参考与证据清单时；正常编译和重绘不使用。已导入的原始结果不因参考长文更新而自动改变。

近邻依据：NOMP的固定增益Newton方向及重估增益接受条件（原文§II-A）、Yang等多频二维BNOMP的§4.2及附录II、QNOMP的§III和Golub–Pereyra变量投影。谱方法和AP候选消融不构成这篇短文的第二主线。

## 核验与投稿待填项

本次使用`research-writing-skill`组织论证、`scientific-visualization`重绘，按`humanizer-zh`及`anti-defensive-writing-en`检查表达。7组显示公式环境中的9个编号公式、中英文5行结果数据和引用对应；英文摘要142词。来源核对、数据重算、PDF页数及图件矢量性检查见`build/data_check.json`与`build/document_check.json`。独立临时目录编译的文字及逐页像素一致性见`build/independent_check.json`。

最终英文PDF为8页（包括图表、声明及参考文献），中文版为7页；已逐页审读全部15页。两份稿件复制到项目外的临时目录后重新编译，文字和逐页渲染结果均与本目录一致。

两张图均为直接从数据绘制的矢量PDF，字体嵌入，无位图对象，蓝／红及虚／实线提供重复区分。图形保留全部样本和误差尾部。未发现中文缺字、未解析引用、正文溢出或图表裁切。

CAS 2.4在原生前置信息排版处产生一条129.88684 pt的内部盒子Overfull警告，另有未填作者信息对应的空锚点提示。已核对页面文字边界及首图，实际内容未裁切；未修改模板或隐藏警告。作者信息留空时原生模板会显示空ORCID和作者页脚位置。

投稿前由作者补全并确认姓名、单位、通讯作者、资助、利益冲突、数据／代码可用性。AI声明已说明实际辅助用途，保留作者审阅确认位置。当前不填写虚构身份、仓储链接或审阅承诺。后续补全这些内容后应再次检查英文10页上限。
