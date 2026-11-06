# 逐源链路受损、多径与盲多源DPD：最近邻准入审查

2026-10-25补读更新：本记录中TMC2024和TWC2024两篇“待全文”现已关闭；两篇均通读并核对关键页。它们是单源阵列/CSI及共同支持方法，未直接覆盖盲单天线部分LOS多源；详细公式对应和准入修订见[两篇补读](F:/DPD_Strong_Weak/研究方向/低空与传播几何/研究记录/准入与文献_20261024/两篇结构近邻精读与准入修订.md)。下文保留2026-10-24的原始检索与阅读范围。

2026-10-24。子任务范围：核对地面分布式单天线、严格未知数据波形、多源定位的链路问题近邻。本文是文献证据调查，不是算法立项，也没有运行定位实验。没有修改Zotero、根文档、旧结果或定位器。

## 1. 结论先行

**“不同源在不同站上弱或缺失”单独不足以成为新问题；“多径导致源相关的错误时延，且没有干净逐源参考”更值得继续核验，但尚未达到低风险创新立项。**

原因不是题目是否写了无人机，而是已有模型已覆盖的未知量和已有解法：

- 经典多源DPD已有逐源—逐站未知复增益，明确提到信号在部分站弱或缺失；纯零增益/衰减不能伪装成新模型。
- 多径处理中的“共同LOS结构＋站特异NLOS结构”已经有原子范数、分组稀疏、二元结构稀疏和低秩工作。我们即使不使用阵列，也必须说明新观测条件迫使数学问题发生了什么变化。
- 有单天线、未知波形的多径DPD近邻，但部分算法在推导中使用了**无多径参考信号**，摘要不易看出来。
- 也有单传感器、未知波形、未知路径复增益、三维和全LOS遮挡的TSP工作，但借助已知环境反射几何。这种跨领域方法不能因“水声不是无人机”而排除。
- 不加任何可辨识约束的未知逐链路时延/信道可以吸收位置变化。需要首先确认足够LOS、结构化反射几何或其他信息，不能从“盲＋NLOS”直接跳到优化。

本轮支持继续进行**问题可辨识性与常规方法适用性核验**，不支持现在声称已发现无人做过的鲁棒多源DPD算法。

## 2. 检索过程与范围

使用Zotero本地库（API已启用、运行正常），按题名/全文线索搜索 `multipath`、`contaminated`、`semi-blind`、`High Precision`、`Binary Structured`、`Unknown Signals in the Presence`、`Cross-Spectra`，再按关键条目读取PDF全文提取。`multipath`返回31条；其中大量是RSS、阵列或已知波形问题，没有按标题直接归入可比较基线。

网上实际检索式包括：

1. `"direct position determination" "unknown" "NLOS" multipath multiple sources`
2. `"Robust direct position determination with contaminated measurements"`
3. `"direct localization" "source" "LOS" "unknown waveforms"`
4. `"direct position determination" "multipath" "unknown" single antenna`
5. `"direct localization" "unknown signals" "multipath" sparse`
6. `"direct position" "partial" "NLOS" sources`
7. `"semi-blind" "localization" "unknown" "2022" Weiss`
8. `"direct localization" "unknown" "partially" "line-of-sight"`
9. `"DPD" "unknown signals" "non-line-of-sight" "single"`
10. `"Direct localization" "LOS" "sparse" Garcia Wymeersch`
11. `"A binary structured sparsity approach for multi-anchor direct localization" waveform`
12. `"direct position determination" "multipath" "unknown waveforms" robust single antenna 2024 2025 2026`

沿SaDPD的参考文献追到Hao–Wan 2021、Garcia 2017、Akbari–Valaee 2024；沿经典DPD检查未知增益和跨站共享波形；跨领域检索发现TSP 2022 semi-blind方法。主要使用出版商、作者机构PDF和arXiv原文；ResearchGate/索引仅用于发现。没有Scopus/WoS专用接口，不宣称两库穷尽检索或完整引用网络遍历。PMC一处后续打开触发验证页，未据此补写未见原文。

## 3. 核心近邻的原文核对

### L1. Weiss–Amar：纯缺失不构成新建模

**Direct Position Determination of Multiple Radio Signals**，EURASIP JASP 2005，37–49。DOI：[10.1155/ASP.2005.37](https://doi.org/10.1155/ASP.2005.37)。Zotero：`ID3FDGNH`；PDF：`VZAL7S46`。

阅读范围：引言、§2模型、§3.1未知波形方法与相关注释；没有把全文取回等同于全篇证明复核。

- p.38引言明确说目标在部分站可能很弱或不存在。
- p.38式(1)有逐源逐站复增益 (b_{\ell q}\)、未知源波形、位置相关时延；允许D=2/3。p.39式(6)跨站共享同一个源波形。
- p.39 Remarks(2)指出波形与复增益的尺度/相位不唯一，并作规范化。
- p.39§3.1未知波形采用子空间路线避免完整ML多维搜索，不是已知波形的匹配滤波。

**对本项目的判断：**自由复增益可以表示某源某站为零/很小。实际性能是否因有限数据和弱可辨识下降仍可研究，但“添加visibility变量”必须胜过这一常规模型并指出额外作用。原文有阵列；M=1时不再有AOA，需要保留足够跨频时延信息及辨识条件，不能说论文所有MUSIC推导在任何单频M=1设置都直接有效。

### L2. You等2026：鲁棒未知波形DPD已存在，但处理的是回传数据污染

**Robust Direct Position Determination with Contaminated Measurements**，IEEE TSP accepted author version。DOI：[10.1109/TSP.2026.3743837](https://doi.org/10.1109/TSP.2026.3743837)。Zotero：`PHHWLZDF`；PDF：`HNQIFEEA`。

阅读范围：摘要、引言贡献、§II–IV模型和两类方法、未知波形算法；仿真表仅核对其比较任务，不独立复现数值/逐式证明。

- p.3§II式(1)–(20)：**单个静止源、运动的单天线站**，三维位置、时延与多普勒、未知站增益、未知波形。
- p.4式(21)：传感器到融合中心的误码进入复合误差。不是源到站的反射路径产生错误几何时延。
- p.4–5式(22)–(24)：Student-t散布模型和似然；p.6式(29)–(32)对大残差样本截断。
- p.6–7§IV-B式(33)–(40)：投影估计未知波形、残差截断、噪声尺度及位置迭代。
- 摘要明确LOS-dominated narrowband，强多径不存在或不可分辨。

**判断：**“盲DPD＋鲁棒损失/截断”本身已有近邻。它不能直接证明当前源相关NLOS问题已经解决；也不能直接套用其单源闭式关系到多源并称原版复现。若进一步实现，须核查等方差LS投影与Student-t目标更新的一致性，不能仅因发表就免于数值审计。

### L3. Chai等2025：真正单天线、未知波形的多径近邻，含干净参考先验

**Direct Position Determination of Wideband Source over Multipath Environment: Combining Taylor Expansion and Subspace Data Fusion in the Cross-Spectrum Domain**，Sensors 25,4967。DOI：[10.3390/s25164967](https://doi.org/10.3390/s25164967)。Zotero：`CX82UIQA`；PDF：`7LV7BYNC`。

阅读范围：§2–4模型、归一化、平滑、子空间、Taylor精修和算法步骤；§5设置与结果说明；未逐式独立证明。

- p.3§2：单个未知宽带源、单传感器站、时间同步、**固定高度平面定位**。
- p.4–5式(3)–(6)：任意站对cross-spectrum含所有路径对差时延。
- **p.5式(7)–(12)：明确假设另选reference signal没有multipath，用其自谱归一化源谱；p.8算法步骤1仍要求按prior选reference。** 任意站对并不意味着完全没有干净参考。
- p.6–8：频率平滑恢复秩、噪声子空间谱、候选位置Taylor补偿。
- 原文式(19)用“largest eigenvalue”构造信号子空间，但多径路径对模型可产生多方向。后续若作为强基线实施，必须核实维数约定和公式/代码，而不是机械照录。

**判断：**不能向盲多源算法免费提供每个源的干净参考，也不能强行让不满足先验的原方法参加排名然后宣称优越。可作“参考可得”诊断上界/适用分支；需研究的是如何在真实未知混合中形成足够关联，而非把参考源谱替换成真值。

前身：Zhu等2022，**A Direct Position Determination Method Based on Subspace Orthogonality in Cross-Spectra under Multipath Environments**，[10.3390/s22197245](https://doi.org/10.3390/s22197245)。在线模型§2–3核对；单源、参考站、多径数视为已知以简化推导。库中题名搜索未找到；本轮没有独立完整PDF阅读。

### L4. Hao–Wan2021：LOS组结构/NLOS站特异结构已经存在

**Sparse Bayesian Inference-Based Direct Off-Grid Position Determination in Multipath Environments**，IEEE WCL 10(6),1148–1152。DOI：[10.1109/LWC.2021.3057502](https://doi.org/10.1109/LWC.2021.3057502)。Zotero：`2ENUNLHX`；PDF：`ZUEMTL5U`。

阅读范围：§II–III模型和主要更新、引言与方法来源；未完成性能复现。

- p.1148§II明确**known signal waveform**、单个发射源、混合阵列多个panels，观测是CFR。
- p.1149式(2)–(13)：NLOS虚拟源只作用于对应站；LOS在各站对应同一位置块，NLOS呈单站稀疏。
- p.1149–1150式(14)–(28)：离网字典线性化、SBL层级先验、EM及GD网格更新。

**判断：**共同/私有稀疏结构不是新概念。其AOA+CFR不满足本项目严格未知波形单天线数据权限，只能借鉴结构，不能让本项目凭空获得CFR。

### L5. Jin等2025 SaDPD：联合行/元素稀疏与低秩已有系统做法

**An Indoor Direct Localization Method Utilizing Structural Sparsity and Low-Rankness With Grid Refinement**，IEEE IoTJ 12(13),25407–25423。DOI：[10.1109/JIOT.2025.3559125](https://doi.org/10.1109/JIOT.2025.3559125)。Zotero：`368F6YF3`；PDF：`PI66HFU4`。

阅读范围：§II、§III-A/B结构与目标、参考文献链及附录相关条件；不是全17页推导复核。

- p.25409§II：一个目标、二维、各站线阵；主要定位信息为AOA。
- p.25410–25411§III：将源位置权重拆成W与H，W表达跨站一致LOS，H表达不一致NLOS；式(9)后采用行稀疏、元素稀疏与trace norm，ADMM与网格精修。
- 附录讨论所有站LOS一致性及正则参数条件，不能默认其保证自动覆盖任意源相关部分LOS。

**判断：**“一起估计可靠链路并用稀疏性分开NLOS”必须与此类结构比较。M=1时其AOA字典会丢掉关键位置辨识信息，不能直接删阵元就称公平适配。

沿参考文献发现两篇重要待全文项：Akbari–Valaee，**A Binary Structured Sparsity Approach for Multi-Anchor Direct Localization**，TMC2024，[10.1109/TMC.2024.3439099](https://doi.org/10.1109/TMC.2024.3439099)；**Decimeter Level Cooperative Direct Localization With Ising Model Approach**，TWC2024，[IEEE题名检索](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=Decimeter%20Level%20Cooperative%20Direct%20Localization%20With%20Ising%20Model%20Approach)。目前仅参考表与摘要发现，不能据此判定已解决/未解决盲多源问题；不是本轮全文证据。

### L6. Garcia等2015：已知波形的多源单天线多径，是结构近邻

**High Precision TOA-based Direct Localization of Multiple Sources in Multipath**，[arXiv:1505.03193](https://arxiv.org/abs/1505.03193)，[作者稿PDF](https://arxiv.org/pdf/1505.03193)。Zotero题名搜索未找到；本轮未核定对应正式期刊DOI，不擅写为某一期刊版。

阅读范围：作者稿§II模型、§III–IV概要及摘要；恢复定理未逐式核验。

- p.2式(1)–(6)：已知各源波形和源数；每源每站可有未知个数NLOS、未知复幅和时延。
- 方法利用LOS跨站同位置一致性与NLOS站局部性，先去卷积再原子范数定位。
- 不能使用其已知波形消卷积当作本项目可用公共前端。

这是比“多径论文标题不同”更重要的近邻：几何一致性/局部污染思想已有，可能新增处只能是未知混合波形下的可辨识约束与有效估计，而非换一个正则项名字。

### L7. Weiss等2022：跨领域的强原理近邻

**A Semi-Blind Method for Localization of Underwater Acoustic Sources**，IEEE TSP70,3090–3106。DOI：[10.1109/TSP.2022.3173731](https://doi.org/10.1109/TSP.2022.3173731)；[作者机构全文](https://arl.nus.edu.sg/wp-content/uploads/2022/07/A_Semi-Blind_Method_IEEETransactionsOnSignalProcessing2022.pdf)。Zotero `semi-blind` 搜索未找到。

阅读范围：p.3090–3094引言、§II–IV模型及算法、Proposition1假设；未完成后续证明/全部实验精读。

单源、三维、各站一个全向水听器、未知波形、每路径未知复增益；已知水面/水底几何产生三条位置相关路径，即使LOS完全遮挡仍可定位。式(15)–(16)联合拟合位置、波形和路径系数；式(17)–(22)低维矩阵计算。闭式结论的Proposition1使用spectrally flat waveform；一般非平谱另有近似分析，不能把任意未知谱与精确闭式等同。

**判断：**如果以后做地面反射/两径辅助盲定位，这篇必须作为数学近邻。不能仅把水声速度换成电磁速度、三径换两径就主张新理论。它也提示：可辨识性可以来自已知反射几何，而不是假定所有路径自由。

### L8. Lin等2026：多源、单天线、非合作，但不是本项目同一种观测

**Variational Bayesian Multi-Source Localization in Complex Multipath Environments**，IEEE TCOM74,7091–7105。[10.1109/TCOMM.2026.3679265](https://doi.org/10.1109/TCOMM.2026.3679265)。Zotero：`4KD3RU7N`；PDF：`DEDESABG`。

阅读范围：摘要、引言、§II–III环境/观测模型。其输入为大量空间位置上的混合RSS样本，用环境划分网络和分组后VB定位。虽包含单天线、多源、非合作和逐源传播差异，但不是少数固定站的短时同步IQ盲DPD。不能直接与短记录IQ方法同列而不解释数据量和传播先验；作为宏观场景/模型近邻保留。

## 4. 常规强基线应如何建立

真正进入小实验时，先选最小可辨识模型，再决定基线，不能先列八个算法名字。

1. **纯LOS衰减/缺失：**逐源逐站未知复增益的多源条件ML/子空间DPD；AP/VP只是求解器。模型本身应先覆盖零增益，不添加真值visibility。
2. **少量受损链路：**基于相同未知波形模型的常规鲁棒损失、可恢复的站子集/一致性搜索。要区分样本异常、整站异常和逐源路径异常，避免拿容易失配的整站删减作为唯一对手。
3. **有少量已知反射几何：**位置相关多径＋未知系数的联合LS/ML，并与semi-blind/matched-field原理对应；不免费给真值反射系数。
4. **没有干净参考：**参考依赖cross-spectrum方法须标注先验不满足；若用真实源谱只可作诊断上界。拟议无参考方法必须先证明不是把更少信息问题强行套进不可辨识模型。

对已发表算法，先审计其假设和数值目标，再决定实施：论文原版与本场景适配版分开登记。相同模型/信息/预算下比较，不将已知波形、天线阵列、环境地图或额外时段悄悄送给其中一方。

## 5. 最需要先回答的三个问题

**第一：什么是受损？** 纯衰减、LOS消失但仍有反射、样本误码是不同问题。不能统称bad link并用同一独立噪声模型代替。

**第二：位置是否还可辨识？** 若每源每站的自由附加时延可以任意调整，那么改变候选位置后，用附加时延抵消几何时延变化即可产生同样观测；位置不能仅由拟合残差唯一确定。非负附加时延也不自动消除所有等价位置。需要足够LOS约束、已知/受限反射几何、可信多时段结构等具体信息。此处是模型推理，不是已完成的多源可辨识定理。

**第三：为什么需要逐源处理？** 一个站对源A为LOS、对源B为NLOS时，整站删减会同时丢掉A信息。这是一个可检验机制，但不是新颖性证明；共同/私有稀疏、联合信道源估计已有工作必须公平比较。

**最小可否定验证应先包含：**无噪声/充足数据下能否辨识；纯零增益是否被传统自由增益模型正常覆盖；单源与混合数据权限相同；逐源受损与整站受损的差别；在同样先验下常规鲁棒/子集方法是否已经解决。只有这些留下明确剩余损失，再设计新规则。

## 6. 完成情况与限制

- 已完成：Zotero与网上交叉发现；8组原文/模型证据核对；从SaDPD追溯结构稀疏来源；识别干净参考、known waveform/CFR、阵列AOA、地图几何和RSS空间采样等隐藏权限。
- 没有完成：Scopus/WoS穷尽检索、所有论文全篇证明重推、最近邻算法复现、三维盲多源NLOS可辨识性证明、实际性能实验、创新性最终确认。
- 特别提醒：本报告引用的网页全文读取与本地PDF提取用于限定模型；没有将“接口返回完整文本”记作“全篇逐式精读完成”。
- 建议准入状态：**有条件进入最小理论/机制核验，不直接进入新算法大规模开发。** 若用户目标是尽快短文，必须把“多源＋未知波形＋任意未知NLOS＋三维＋速度”拆开，不能一次打包。

## 7. 可追溯附件与检查

本地6篇的来源路径、SHA256、PDF总页数与选读页登记在[link_sources/sources.json](F:/DPD_Strong_Weak/研究方向/低空与传播几何/研究记录/准入与文献_20261024/link_sources/sources.json)。选定模型/方法页的文本和渲染图按论文分目录保存；本轮实际视觉复核了Chai2025第5页、Hao2021第1页、You2026第3页，确认干净参考、已知波形/混合阵列、单源运动接收机等关键表述并非抽取误读。其余渲染图是供后续追溯的附件，不把已生成图片等同于已逐页视觉核查。

证据生成脚本为[inspect_link_pages.py](F:/DPD_Strong_Weak/研究方向/低空与传播几何/研究记录/准入与文献_20261024/link_sources/inspect_link_pages.py)，只读取Zotero PDF，写本子目录。已通过项目Python `py_compile` 和Ruff `F,E9`；没有运行定位求解器。网上来源保持上述原始链接；本轮未把开放全文下载进Zotero，也未修改分类。
