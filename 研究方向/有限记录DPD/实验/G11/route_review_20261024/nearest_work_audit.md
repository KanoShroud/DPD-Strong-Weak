# G11 路线终审：近邻方法与创新边界独立核查

> 全文补读状态更新（2026-10-24）：Isbi–Weiss 2008、Yeredor 2012已从用户Zotero读取，9页全文及公式图表已核。下文“未取得全文”描述初次终审时的状态，现已由[全文补充报告](F:/DPD_Strong_Weak/研究方向/有限记录DPD/实验/G11/route_review_20261024/fulltext_followup/两篇边界误差论文精读与终审补充.md)补齐。确认2008加窗/位置驱动移窗先例、2012自相关相关MSE及窗外样本假设；不将它们扩大为当前双源问题已全部解决。其余论文阅读状态不变。

日期：2026-10-24。任务仅含网络检索、原文核验和研究判断；未运行定位仿真，未改变算法或长期记录。此报告为终审的一项输入，不单独替代需求和实验审计。

## 1. 结论

**当前已经完成的有限记录表示、公共边界复用和 P7 分阶段串接，还不足以确认独立的方法贡献；不建议继续通过更换串接顺序、增加分支或调节模型精度来维持主线。**

同时，不能据此宣布“短记录定位已经没有研究问题”。本轮重新核验了一个真实且更具体的文献边界：R025 的作者不仅做已知方向波形恢复，还研究未知 DOA；但其多传感器快速搜索采用可分离时空近似，明确指出非可分离候选子空间的快速投影仍需研究。这个边界有理论意义，却不是 P7 当前首源更新或残差候选串接已经解决的问题。

若继续，必须转成一个新、明确、可证伪的数学问题，而不是把现有模块命名为这个空白的解法。最有利于继续的主张见第 5 节；它目前只有研究假设资格，没有已完成贡献资格。

## 2. 检索范围与执行记录

没有局限于已有 13 篇。采用 Web 搜索发现、出版社/作者机构原文、Crossref DOI 核对和引用链回溯，覆盖：

1. 有限观测与 DFT 边缘误差的 DPD/TDOA 原理；
2. Slepian 宽带未知源定位与非可分时空表示；
3. 模型失配到参数 MSE 的理论；
4. 变量消元、非精确内层与多保真模型管理；
5. 近期有限测量 DPD 和上述方法的新版本。

主要检索式（2026-10-24）：

- `"direct position determination" "finite" "observation" waveform`
- `"direct position determination" "short" "record"`
- `"direct position determination" "finite" "window"`
- `"direct position determination" "spectral leakage"`
- `"direct position determination" "truncation"`
- `"time delay estimation" "edge effects"`
- `"time delay estimation" "finite observation" bias window`
- `"time delay estimation" "windowing" "bias" IEEE Signal Processing`
- `"Slepian" "localization" unknown waveform`
- `"broadband" "source localization" "Slepian"`
- `"non-separable" "Slepian" localization 2025 2026`
- `"Slepian" "source localization" 2024 2025 2026 delay`
- `"direct position determination" "model mismatch" "bias" 2024 2025`
- `"variable projection" "model error" parameter estimation bias`
- `"Inexact variable projection" inverse problems`
- `"model reduction" "parameter estimation" "goal-oriented" nonlinear least squares`
- `"multi-fidelity" "source localization"`
- 相关准确题名检索及作者主页、参考文献回溯。

检索跨期覆盖是有意安排：近年论文不能替代早期已确立的边界误差和变量消元理论。没有以命中数证明穷尽性，也没有因检索未命中宣称某项不存在。搜索结果中没有取得原文的条目不用于公式级覆盖判断。

## 3. 最影响判断的原文与来源

### 3.1 R025：已有未知 DOA 定位，不应与 Slepian Beamforming 混为一谈

**Iterative Broadband Source Localization**，IEEE JSAIT 4, 453–469, 2023，DOI [10.1109/JSAIT.2023.3317094](https://doi.org/10.1109/JSAIT.2023.3317094)。本轮直接获取作者 [arXiv PDF](https://arxiv.org/pdf/2210.11669)，核对 §7–8，尤其 PDF pp.22–24；论文版本与期刊版排版可能不同。

另核对 Coleman DeLude 博士论文 **Slepian Array Processing: Concepts and Practical Considerations**，Georgia Tech，2024 年 12 月，[机构页面](https://repository.gatech.edu/items/4e15c314-d402-40a3-adc8-9536138a8b95)、[机构全文](https://repository.gatech.edu/server/api/core/bitstreams/2d7c903e-3522-41d4-bf49-dcbe877737cb/content)。阅读 §3.4–3.8 和 §4 引言，重点为印刷 pp.53–57（PDF pp.76–80）。

确认事实：

- 多传感器部分确实从有限采样估计未知 DOA 和频谱支持；不需要把方向当作已知量。
- §3.7 明确说明，相同频谱支持的多个源可利用不同空间方向区分。
- 快速识别采用时域/空域子空间分别投影；作者指出一般情形不可分离，带宽增加时耦合更明显。
- 可分模型维数按乘积增长，而物理信号自由度可能更接近两维贡献之和；快速非可分投影被明确列为后续问题。
- 多传感器实验为 32 元 ULA、5 个等功率源、平面波 DOA。前面的高动态范围实验主要是频谱定位与相互分离的频带，不能合并宣称其已完整验证“同频高动态双源四站近场定位”。

因此，未知波形、有限记录、Slepian、逐源发现、高动态范围等单独要素已有充分先例；但我们的近场距离时延、四站、统一带限观测与联合位置消元矩阵也不能直接等同于其可分离 DOA 模型。

### 3.2 DFT 边界误差影响 DPD/TDOA，不是新发现

- **DFT model errors for finite length observations with spatially distributed sensors**，Yaniv Isbi、Anthony J. Weiss，IEEEI 2008, pp.80–84，[DOI 10.1109/EEEI.2008.4736661](https://doi.org/10.1109/EEEI.2008.4736661)。Crossref 核对题名、作者、日期；[会议目录](https://www.proceedings.com/content/004/004496webtoc.pdf)核对页码；目前仅取得书目、摘要和后续论文引用，**未取得全文**。摘要明确研究时延相位表达的边缘项，量化 SIR，并以小误差分析和 Monte Carlo 研究 DPD 定位影响。不能据此判定它已覆盖本项目所有多源公式，但已经足以否定“首次发现有限 DFT 边界损害 DPD”这一宽泛主张。
- **Analysis of the edge-effects in frequency-domain TDOA estimation**，Arie Yeredor，ICASSP 2012, pp.3521–3524，[DOI 10.1109/ICASSP.2012.6288676](https://doi.org/10.1109/ICASSP.2012.6288676)，[作者机构书目及摘要](https://cris.tau.ac.il/en/publications/analysis-of-the-edge-effects-in-frequency-domain-tdoa-estimation/)。本轮未取得全文。其摘要确认研究循环相关替代线性相关的 TDOA MSE，指出影响不一定仅由边缘长度比例决定。
- **On Frequency Domain Models for TDOA Estimation**，Jensen 等，ICASSP 2015, pp.11–15，[DOI 10.1109/ICASSP.2015.7177922](https://doi.org/10.1109/ICASSP.2015.7177922)，[机构全文](https://vbn.aau.dk/ws/files/226082971/20150213_ICASSP15.pdf)。本轮读完整正文与参考文献。§1 指出固定 N 周期模型与边缘效应；适当零填充会改变噪声协方差并产生秩亏。§2 从未知周期谐波模型构造 NLS，并消去线性系数，§3 用合成及人工延迟语音验证。它不是独立随机宽带双源的完整 DPD 解法，但证明“比固定 DFT 模型更准确的表示＋系数消元＋位置/时延优化”有长期方法先例。

**纠偏：** 不能因为这些工作较早就认为问题不重要；也不能因为我们在 RF 而对方在声学，就忽略相同有限窗数学机制。

### 3.3 短记录精确时延 ML 已有经典理论

Champagne、Eizenman、Pasupathy，**Exact maximum likelihood time delay estimation for short observation intervals**，IEEE TSP 39(6),1245–1257,1991，[DOI 10.1109/78.136531](https://doi.org/10.1109/78.136531)。作者 [期刊 PDF](https://www.ece.mcgill.ca/~bchamp/Papers/Jounal/IEEE-SP1991.pdf)可下载，但为扫描件，本轮没有完成全文 OCR/逐页核验。

本轮可读的是其作者发布的 [ICASSP 1989 前身全文](https://www.ece.mcgill.ca/~bchamp/Papers/Conference/ICASSP89.pdf)，读到任意观测时长模型、有限时间波束形成、边缘项、积分特征问题及实验描述。模型是双接收端同一 Gaussian 随机源、已知有理 PSD 和白噪声；不是我们两个独立未知确定波形的直接位置估计。它证明有限观测影响本身不冷门到无人研究，也表明必须明确统计先验差别，不能拿它当无改造的排名基线。

### 3.4 参数误差机制可以有贡献，但一般扰动公式不够

- **On the accuracy limits of misspecified delay-Doppler estimation**，McPhee 等，Signal Processing 205,108872,2023，[DOI 10.1016/j.sigpro.2022.108872](https://doi.org/10.1016/j.sigpro.2022.108872)。本轮核对出版社引言、方法节说明和开放稿可访问部分，未逐式复证附录。它针对忽略加速度的模型，建立伪真参数、失配 CRB 与 MSE 对比，展示简单模型某些条件下因方差更低反而更好。不能据此断言完整有限模型在所有 RMSE 条件下必优。
- **Parameter bounds on estimation accuracy under model misspecification**，Richmond、Horowitz，IEEE TSP63(9),2263–2278,2015，[DOI 10.1109/TSP.2015.2411222](https://doi.org/10.1109/TSP.2015.2411222)，[作者机构元数据/摘要](https://scholars.duke.edu/publication/1645372)。本轮未取得完整原文，作为已有理论框架而非细节证明。

这里最有参考价值的不是新增一个通用 MCRB 公式，而是给出当前边界误差、源间功率和位置敏感方向的具体可检验结论。

### 3.5 分阶段使用高精度模型和不精确内层也已有方法体系

- **Local Convergence Analysis of a Variable Projection Method for Regularized Separable Nonlinear Inverse Problems**，Español、Jeronimo，SIAM J. Matrix Anal. Appl.46(2),858–878,2025，[DOI 10.1137/24M1639087](https://doi.org/10.1137/24M1639087)。出版社元数据/摘要已核；直接读 [作者预印本 2402.08568v1](https://arxiv.org/pdf/2402.08568) 的 §§1–3、Algorithm2.2、LSQR 停止条件和结论。它分析内层近似对 residual/Jacobian 的影响并给局部收敛条件；不是我们的代码复现，却足以说明“先粗解波形、逐步精解、仍维持外层精度”不是新概念。
- **Survey of Multifidelity Methods in Uncertainty Propagation, Inference, and Optimization**，Peherstorfer、Willcox、Gunzburger，SIAM Review2018，[DOI 10.1137/16M1082469](https://doi.org/10.1137/16M1082469)，[作者全文](https://kiwi.oden.utexas.edu/papers/multi-fidelity-survey-peherstorfer-willcox-gunzburger.pdf)。本轮核对出版社摘要、框架和引用链，不声称全文逐节精读。低成本模型筛选/搜索、高精模型校验的思想已有系统研究。它是方法先例，不是已解决 RF 双源问题的证据。
- **Goal-Oriented Inference: Approach, Linear Theory, and Application to Advection Diffusion**，Lieberman、Willcox，[SIAM Review版本 DOI10.1137/130913110](https://doi.org/10.1137/130913110)。仅核对摘要及论文首页；研究为预测目标保留相关子空间。它提示“优先保留关心参数的信息”也有邻域先例，不能只换成位置变量就主张原创。

### 3.6 近期有限测量 DPD 仍在研究，但与边界模型不应混称

**Importance sampling based direct maximum likelihood position determination of multiple emitters using finite measurements**，Hao、Wan，Signal Processing186,108111,2021，[DOI 10.1016/j.sigpro.2021.108111](https://doi.org/10.1016/j.sigpro.2021.108111)。本轮核对出版社摘要/引言可见部分，未取得全文。它以有限样本和初值敏感为动机，用重要性采样求联合 ML；“finite measurements”主要指有限统计样本，并不能由题名推断它已修正 DFT 截断边界。

## 4. 当前改动与常规适配的边界

| 当前要素 | 已有覆盖 | 当前可成立的研究陈述 |
|---|---|---|
| 有限记录比周期近似更准确 | 短记录精确 ML、DFT edge、Slepian 表示 | 项目观测证据可保留；不是首次机制发现 |
| 未知波形消元后优化位置 | R001、一般 VarPro、多源 LS/AP | 合理且必要的求解框架，不独立构成新理论 |
| 公共基、FFT主项和边界/低秩修正 | Fast Slepian、ROAST、FE/AZ 等已有阅读 | 只有新结构/误差/复杂度结论超出通用代入，才可能升级贡献 |
| 便宜候选发现后用有限模型精修 | 多保真管理、初始化后精估是通用策略 | 可以是好工程流程；需新增可验证的场景相关选用准则或保证 |
| 有限模型先修首源，再做周期残差发现 | 本项目 P7 的具体串接 | 目前没有稳定独立收益，不能靠继续组合来自动取得贡献 |
| 非可分时空模型的低成本候选投影 | R025 明确提出未解决方向，后续 Slepian Beamforming已有给定方向重建 | 有具体差别，但还没有本项目相应的新定理与强基线收益 |

这不是要求任何工具都原创。方法可以使用已有 Slepian、VarPro、Newton 和低秩工具；需要新增的是具体场景中的结构结果、可验证机制或可靠精度—成本优势。

## 5. 最强支持继续的可证伪主张，以及反证条件

比“再调整阶段安排”更有原则性的主张是：

> 在同一有效频带、有限记录、未知宽带波形的多站定位中，利用非可分位置相关子空间结构，可以只保留对位置梯度有显著影响的表示误差方向，获得对完整有限模型的位置更新误差的可计算控制，并以显著低于常规 Slepian/结构化 VarPro 的成本达到同等 RMSE。

这不是已验证结论。它至少要新增并连接三件事：

1. **具体结构：** 明确截断误差和候选位置导数的关系，不只写通用 Jacobian 伪逆乘 residual。
2. **可实施控制：** 无真值地估计误差或筛选所需方向；不能先解完整高精模型才知道该保留什么。
3. **强参照收益：** 与常规有限表示、合理非精确 VarPro、已有快速表示比较，准备及检查成本全计。

最强反证/常规替代：

- 若合理常规 finite/Slepian + 原有搜索已经满足实际精度和延迟要求，研究需求关闭。
- 若所谓新准则只是一般 inexact-VarPro 的 residual/Jacobian 容差代入，且没有新的特殊结构或性能结论，方法贡献不成立。
- 若梯度相关误差没有低维可利用结构，或计算它已接近完整求解成本，该方向失去成本价值。
- 若差异仅在候选覆盖，普通多初值/候选合并已解决，不应继续包装边界模型精修贡献。
- 若只有平均小残差，不能推出全局候选不漏或位置 RMSE 优势；局部结论必须与搜索保证分开。

**终审建议：** 这项新主张尚未形成可直接运行的小测试所需的完整理论定义，也没有应用约束证明它非做不可。当前更可靠的决策是停止原 P7 串接主线，保存有限模型、物理解释和实现为资产。若主审从需求及数学核验中另有充分证据支持上述具体问题，可单独立项；不把“文献中还有一个开放问题”自动当作批准继续现有路线。

## 6. 尚未取得全文的关键项与可靠性限制

- Isbi–Weiss2008 与 Yeredor2012 的准确全文仍值得补齐；因此不能宣称它们已涵盖本项目全部多源耦合结论。
- Champagne1991 已有可用扫描 PDF，本轮未完成 OCR；1989 前身可读，但不冒称完全代替1991期刊版。
- 尚未给出新非可分候选投影方法的完整排他性查重；本报告没有证明所有可能新方法都已存在。
- 新颖性风险与当前实验不支持独立收益是两条不同证据。即使未来取得上述全文并发现存在细分空白，也不会把 P7 的无增益结果变成新方法证据。

本报告没有修改 Zotero 分类/条目，也没有启动新实验。
