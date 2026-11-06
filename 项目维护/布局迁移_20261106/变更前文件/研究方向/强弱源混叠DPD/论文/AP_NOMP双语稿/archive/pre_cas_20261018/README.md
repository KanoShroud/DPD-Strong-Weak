# 中文论文初稿

题目：未知宽带强弱重叠源的高效直接定位：多候选联合精修与波形消元更新。

## 阅读与编辑

- 阅读：[main.pdf](main.pdf)。当前为独立中文单栏初稿，作者与单位暂留空。
- 编辑：[main.tex](main.tex)；各节在 `sections/`，参考文献在 `references.bib`。
- 图表：`figures/` 含矢量PDF、600 dpi PNG及对应CSV；双语图注、筛选条件和文件身份在 `sources/source_manifest.json`。
- 核验：[来源与核验索引.md](来源与核验索引.md)，数值、引用及排版检查在 `qa/`。

正文已纳入R4正式结果、五候选完整配对和PROFILE完整2400条验证。论文VP-BNOMP统一采用`PROFILE_DIAG`的消元Newton实现，正文公式、算法2、参数与结果列一致；共同240条的耗时仍用原同批PROFILE结果，不将新批墙钟拼成严格配对加速比。

实验章按“共同240条全面比较→2400条核心方法扩展验证→方法贡献验证→求解轨迹与困难构型解释”组织。240条覆盖同样的20布局、每布局2次随机实现；2400条增加到每布局20次，并非另一批独立验证。比较表内方法使用相同观测，不跨样本范围排名。

八幅图继续使用NPG风格配色。全部2400条的总体成绩与CDF保留，功率差图改为两列：其余19布局与诊断困难布局4；布局失败分布及候选案例集中在章末解释。分组CSV为`sources/formal_geometry_groups.csv`，总体可由分组重建，不删除或替换困难观测。几何图不显示数字；消融柱图仍为单行四面板；时间图仍采用上行对数、下行前三秒线性放大。图表重排后编号自动更新，各主要图表下方保留分析，不增加重复表。本轮前版保存在`qa/comparison_structure_revision_before/`，更早备份保留。

## 重新生成PDF

在此目录打开PowerShell，执行：

```powershell
& .\build.ps1
```

先从现有结果重绘，再编译：

```powershell
& .\build.ps1 -RefreshAssets
```

`build.ps1` 调用本机TeX Live的XeLaTeX、BibTeX和项目Python。`-RefreshAssets` 使用已缓存并核验的文献元数据，不联网、不生成IQ、不启动定位，只读已完成的登记结果。正文编辑若有意改变公式、数字或引文，需审查后更新 `validate_draft.py --snapshot-polish` 的基准；纯文字润色应继续通过 `--check-polish`。

`prepare_formal_assets.py`读取已完成R4逐例JSON，核对审计字节身份、记录封印、2400条输入配对，并独立重算位置误差、成功标记及GOSPA；汇总对照原正式报告，时间曲线对照保存事件。它不加载IQ、不导入求解器、不调用GPU。`validate_revision.py`保护旧证据及新来源身份，允许本次已批准的正式统计替换旧表；不再套用“新旧数据必须相同”的旧修订规则。

渲染逐页检查图：

```powershell
& 'D:\Software\anaconda3\envs\PyTorch\python.exe' .\render_review.py
```

## 本版写作与验证

使用 `research-writing-skill` 组织论证，`citation-management` 核对文献元数据，`scientific-visualization` 生成图表，PDF技能指导编译与逐页检查，最后按 `humanizer-zh` 完成语言润色。保留模型条件、证明前提和真实实验范围，删改重复铺垫及防御性表述。润色前后对作者编写各节的数学块、行内数学、数字和引用标记作指纹比较。

交付PDF由本机XeLaTeX实际编译；不以内置预览是否返回作为编译通过依据。原Ho报告、实验求解代码、冻结输入与旧结果均保持原样。

## 本版证据状态

PROFILE最终版本和五候选完整配对均已回填，不再保留占位。PROFILE全部2400条逐结果重放通过；图4.2的构型分组可重建原总体样本数、成功数和平方误差。参数从冻结合同导出，未修改定位代码或配置。

本轮不启动新实验、不重跑定位、不修改旧Ho报告。旧图表保留在文件夹及前版归档中，当前正文只引用`formal_`前缀的八幅新图。
