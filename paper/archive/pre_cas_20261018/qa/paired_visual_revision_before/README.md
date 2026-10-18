# 中文论文初稿

题目：未知宽带强弱重叠源的高效直接定位：多候选联合精修与波形消元更新。

## 阅读与编辑

- 阅读：[main.pdf](main.pdf)。当前为独立中文单栏初稿，作者与单位暂留空。
- 编辑：[main.tex](main.tex)；各节在 `sections/`，参考文献在 `references.bib`。
- 图表：`figures/` 含矢量PDF、600 dpi PNG及对应CSV；双语图注、筛选条件和文件身份在 `sources/source_manifest.json`。
- 核验：[来源与核验索引.md](来源与核验索引.md)，数值、引用及排版检查在 `qa/`。

正文已更新为R4正式结果版本：2400条核心比较统一呈现，共同240条用于八类方法比较；不再单列960条，也不以旧120条实验承担主结论。AP和NOMP分别成节，模型与方法推导保留。4.4的五候选交替／联合完整配对、4.5的最终PROFILE对应内容按用户要求留空，不读取或提前填入其结果。

本轮生成八幅正式结果图，采用区分明确的NPG风格分类色与线型／标记，替换原灰蓝配色；CSV、双语图注及来源清单同步。传播速度按实际代码明确为299792458 m/s，候选提取写清3×3邻域极大值与30 m间距筛选。前版完整保存在`qa/formal_revision_before/`，更早的解释性修订备份仍在`qa/revision1_before/`；未修改算法、IQ或实验结果。

## 重新生成PDF

在此目录打开PowerShell，执行：

```powershell
& .\build.ps1
```

先从现有结果重绘，再编译：

```powershell
& .\build.ps1 -RefreshAssets
```

`build.ps1` 调用本机TeX Live的XeLaTeX、BibTeX和项目Python。`-RefreshAssets` 使用已缓存并核验的文献元数据，不联网、不生成IQ、不启动定位，也不接触正在运行的实验结果。正文编辑若有意改变公式、数字或引文，需审查后更新 `validate_draft.py --snapshot-polish` 的基准；纯文字润色应继续通过 `--check-polish`。

`prepare_formal_assets.py`读取已完成R4逐例JSON，核对审计字节身份、记录封印、2400条输入配对，并独立重算位置误差、成功标记及GOSPA；汇总对照原正式报告，时间曲线对照保存事件。它不加载IQ、不导入求解器、不调用GPU。`validate_revision.py`保护旧证据及新来源身份，允许本次已批准的正式统计替换旧表；不再套用“新旧数据必须相同”的旧修订规则。

渲染逐页检查图：

```powershell
& 'D:\Software\anaconda3\envs\PyTorch\python.exe' .\render_review.py
```

## 本版写作与验证

使用 `research-writing-skill` 组织论证，`citation-management` 核对文献元数据，`scientific-visualization` 生成图表，PDF技能指导编译与逐页检查，最后按 `humanizer-zh` 完成语言润色。保留模型条件、证明前提和真实实验范围，删改重复铺垫及防御性表述。润色前后对作者编写各节的数学块、行内数学、数字和引用标记作指纹比较。

交付PDF由本机XeLaTeX实际编译；不以内置预览是否返回作为编译通过依据。原Ho报告、实验求解代码、冻结输入与旧结果均保持原样。

## 本版保留的两处待补

1. 五候选交替／联合配对：保留表、图与分析位置，不用短测或中途成绩替代。
2. PROFILE最终版本：待另行确定实现及对应验证后，同步方法、伪代码和正式统计；本轮不提前替换当前NOMP。

本轮不启动新实验、不重跑定位、不修改旧Ho报告。旧图表保留在文件夹及前版归档中，当前正文只引用`formal_`前缀的八幅新图。
