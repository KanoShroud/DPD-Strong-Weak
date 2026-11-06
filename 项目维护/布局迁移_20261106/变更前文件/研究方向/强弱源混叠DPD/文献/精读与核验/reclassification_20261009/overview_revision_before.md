# 多源 DPD／TDOA／被动定位：文献对照与可验证的创新方案

> **当前主线：在少量同步单天线站点的短时混合IQ中，保留强弱重叠源的有效位置候选，通过联合连续精修改善弱源恢复与计算成本。** R6支持当前搜索存在损失；相似方法是基线和方法来源。贡献可以是相同可靠性下更省计算，或相同预算下更少漏弱源，不要求全局最优证明。

## 1. 研究范围与阅读方式

本文面向少量同步单天线站点、未知发射波形的多源RF被动定位，先研究已知两源的候选发现与连续求解，再扩展源数、频谱和位置对应。文献整理截至2026-10-09；前期完成Zotero相关分类及引用链初筛，本次结合新增离网、连续求解和快速搜索全文更新。

编号表共116项（含1项专利），其中核心18篇、条件补充14篇、历史归档84项。原10条延伸线索已纳入编号，未完成全文评估的状态保持不变。R026仍仅有摘要；R005可核对作者公开稿，但当前Zotero条目没有PDF。阅读版本、正文与补充材料的取得情况分别记录。

逐篇证据见[文献精读结果.csv](文献精读结果.csv)，[原37篇详版](文献精读结果_详版.csv)保留历史笔记。效果栏转述作者结果，不代表本项目已复现。

### 1.1 如何判断一篇论文是否适合比较

核心文献按“候选发现／连续精修／联合更新／计算成本”组织，同时保留接收配置、实际输入和迁移条件。该流程不是方法分类全集；还需区分联合似然、贝叶斯联合估计、协方差拟合、贪心消除、区域覆盖及整图加速，避免只沿NOMP引用链选择对照。其他文献按用途压缩或归档：

- **直接基线候选**：观测模型接近，仍须核对源数、信道与计算预算。
- **条件适用**：满足原有条件时可比较，例如可分循环频率、单源或运动历史。
- **需重推适配**：思想可借鉴，但观测接口或可辨条件需要重新推导。
- **仅机制**：主要借鉴方法设计，原论文的精度不用于同条件排名。

研究用途与适用等级分开标记，同一篇文献可有多个用途：

| 研究用途 | 阅读时回答的问题 |
| --- | --- |
| 任务近邻 | 已有任务或联合输出与我们接近到什么程度，还缺哪项实际能力 |
| 性能对照 | 哪些方法适合原样使用或做合理适配，作为系统或模块比较的代表 |
| 方法来源 | 哪些表示、物理接口、训练或关联设计可以借鉴 |
| 场景扩展 | 增加运动、传播失配、局部污染等条件时，哪些工作值得继续使用 |

三级精读等级表示研究优先级，不表示阅读完成度；摘要近邻也可能需要优先补全文。逐篇用途见第2章与CSV的“研究用途”列。用途允许交叉；“性能对照”包含条件适用和机制适配候选，具体选择见§6，不等于首轮全部实现。

K表示信源数；DPD直接利用接收信号估计位置；TDOA是到达时间差；DOA只估方向；RSS是接收功率。这些量及相干IQ不是等价输入。标题链接指向原论文或专利，最后一列保留全文版本、证据位置和可用的Zotero检索key。R036为出版商完整网页，R113补读在线授权文本，R069已核读正式期刊版；有全文不等于库内一定有PDF。

### 1.2 阵列方法能否迁移到多站单天线

声学与RF标签不能代替模型比较。先检查方法真正使用的信息：单麦的跨站时差通常可借鉴，多阵元角度、语音先验、已知导频和运动历史则需要单独处理。实值声音的半谱公式迁移到复IQ时，应重写双边频谱和幅相关系；R025阵列部分本身使用光速，不能笼统归为声学方法。

判断的第一步，是检查位置变量出现在模型的哪一部分：

$$
\mathbf x_l(f)=\sum_k g_{lk}\,\mathbf a_l(\theta_{lk})\,S_k(f)e^{-j2\pi f\tau_l(\mathbf p_k)}+\mathbf n_l(f).
$$

令每站阵元数为1，站内角度响应可能变为常数，但跨站时延项不一定消失。**只依赖角度响应的方法不能原样迁移；保留跨站延迟的方法可以继续推导，但仍需验证可辨性。** 还应检查跨站相关是否被丢弃、同步与带宽是否足够、逐频任意增益是否吸收了时延，以及快拍、秩和源数条件是否成立。不能把“公式还含位置”当作“多个源一定能分开”。

“单个阵列”不一定是每站多天线。R001把空间分离的全向标量传感器组成网络，其球面时延模型可与多个单天线站点对应；全网的near-field也不等于单根RF天线的反应近场。R021/R001的AP机制应与仅靠站内角度响应的方法分开判断。

迁移难度按实际模型判断。有的方法需要重建物理接口，有的只需替换导向模型、调整输入并重新训练。二者都可以提供有用的适配，但应如实说明改动规模及作用。

### 1.3 本研究采用的创新判断标准

按“具体场景—关键假设—迁移障碍—针对性改动—验证结果”判断贡献。场景适配、任务扩展和系统整合都可以成立，不要求每个部件或定位原理全新。应用需求只能说明问题值得研究；还需说明针对性适配怎样改善明确场景下的可靠性、精度或成本。已有方法能工作，不等于这些取舍已经没有改善空间。

| 判断项 | 需要说明什么 |
| --- | --- |
| 现实需求 | 为什么需要少量单天线站点、有限观测和逐源频谱—位置对应 |
| 已有能力 | 最接近文献已经完成什么，在我们的观测和任务条件下还缺什么 |
| 适配或扩展 | 哪些改动保留时延信息、改善源身份关联，或增加有实际用途的输出 |
| 实际收益 | 联合质量、弱源可靠性、观测需求或在线成本中，至少有一项值得采用的收益 |

场景名称和输出头数量本身不说明价值。新增输出若解决了实际缺失，并与原任务可靠对应，可以构成任务扩展；已有思想通过必要适配解决新场景问题，也可以形成贡献。相似文献用于确定可借鉴内容和比较对象，不据此直接否定路线。具体贡献以验证结果为准。

### 1.4 迁移判断与证据强度

研究用途说明“这篇文献怎样帮助我们”，迁移判断说明“在候选场景中怎样使用”。二者分开记录。

| 迁移判断 | 含义 |
| --- | --- |
| 可直接使用 | 定位核心或指定组件的模型适用，主要需校准参数；不表示已完成全部联合任务 |
| 常规适配即可 | 替换导向模型、调整输入、重新训练或接入常规前后端，可先建立合理代表 |
| 需针对性改造 | 原方法依赖的信息缺失，或关键假设确实不成立，需要改变表示、统计量或求解方式 |
| 尚待核验 | 缺少模型推导或同条件实验，当前不能确定常规适配是否足够 |

判断范围和条件见逐篇说明。原文没有测试某个条件，不等于方法不能迁移；已有传播模型过于简化，也不等于网络在新模型上重训后仍无效。模型适用与实际质量达标分别核验。CSV的“迁移判断”“关键障碍与依据”对应本候选场景，文献原始事实仍以模型、实验及证据定位列为准。

判断时区分四层证据：现实需求说明问题值得做；原文未验证说明适用性尚待核验；普通适配不足需要同条件实验；针对性改动的收益通过同等可靠性或同等预算比较确认。不要求已有方法完全失效，也不要求所有部件全新。

## 2. 文献表与阅读顺序

共116项，按核心18篇、条件补充14篇、历史归档84项连续编号。编号采用三位数字R001—R116；新旧编号和Zotero条目对应见[编号对照](参考文献编号对照.csv)。历史实验与阶段文档保留原编号，查阅时使用该对照。精读记录和迁移依据见[文献精读结果](文献精读结果.csv)。

### 2.1 01 核心文献：R001—R018

用于当前问题定位、关键方法设计和主要对照选择。核心不等于全部需要复现。

| 编号 | 论文标题与入口 | 当前用途 | 阅读证据 |
| --- | --- | --- | --- |
| R001 | [Maximum-likelihood source localization and unknown sensor location estimation for wideband signals in the near-field](https://doi.org/10.1109/TSP.2002.800420) | 未知宽带源的逐频最小二乘、波形解析消元和AP已经形成完整定位框架。多源推导依赖匹配的链路增益；原文明确指出多源增益估错会造成位置偏差。先用它建立同评分求解参照，不能将当前投影公式本身列为创新。 | 见CSV精读记录 |
| R002 | [TDOA-Based Direct Position Determination Method with Parametric Dictionary Learning](https://doi.org/10.1109/JSEN.2026.3693237) | 原生单天线、多未知源、离网位置及未知复衰减联合估计，是最直接的贝叶斯对照。Algorithm 1输入已知源数；p.8 Remark 2只更新当前前N个峰的字典参数，但全网格后验仍可重排，不等于永久删除其他候选。正文转置、复参数维度与公式交叉编号存在需核验之处… | 见CSV精读记录 |
| R003 | [High Resolution Direct Position Determination of Radio Frequency Sources](https://doi.org/10.1109/LSP.2015.2503921) | 单天线多源HR评分可作为谱法参照，构谱不需要源数；未知源数的可靠检测仍需额外阈值校准。有限总记录下分段长度、可用协方差秩和加载条件必须一致，不能给它额外快拍，也不能用退化统计量故意削弱。 | 见CSV精读记录 |
| R004 | [Source cancellation in cross-correlation functions for broadband multisource DOA estimation](https://doi.org/10.1016/j.sigpro.2019.107442) | 完整互相关消除可以处理频谱完全重叠及相关源，不应按声学论文直接排除。原始相关消除与将PHAT权重代入的实现不是同一严格等价关系。先对齐时延和链路响应，用完整公式区分抑制机制收益与搜索收益；窄自相关下仍对强源位置误差敏感。 | 见CSV精读记录 |
| R005 | [Round-robin multiple-source localization](https://doi.org/10.1121/1.4835795) | 作者稿式(10)与(11)已区分未投影和投影响应的归一化，明确讨论后者可能产生尖锐峰；因此位置相关分母不是新的问题发现。宽带相干版本利用已知的跨频复幅度关系，不能用于未知IQ；应参考逐频未知幅度版本。37接收器的稳健区域抑制自由度不能直接搬到四站。 | 作者公开稿；库内无PDF |
| R006 | [Multi-Source Direction-of-Arrival Estimation Using Steered Response Power and Group-Sparse Optimization](https://doi.org/10.1109/TASLP.2024.3419417) | 以位置/方向相关PSD的组稀疏模型解释SRP或相关量，提供不同于贪心逐源消除的联合拟合路线。它不依赖单源占优时频单元，但模型采用源间不相关的统计关系；短IQ的样本交叉项是否可忽略需验证。最终选J个峰用已知源数，不能直接宣传为完整盲计数。 | 见CSV精读记录 |
| R007 | [Robustness Analysis and New Hybrid Algorithm of Wideband Source Localization for Acoustic Sensor Networks](https://doi.org/10.1109/TWC.2010.06.090515) | AP粗初始化加EM细化已经研究定位精度与计算量折中。原文EM仍有位置网格搜索，不能与新R008的单源闭式位置更新混为一谈。其复用收益主要是混合求解和初值敏感性，不能把声学小样本仿真直接当成当前有限截取RF记录的验证。 | 见CSV精读记录 |
| R008 | [Low-complexity TDOA and FDOA localization: a compromise between two-step and DPD methods](https://doi.org/10.1016/j.dsp.2019.102600) | 未知波形、标量站，含静态TDOA分支。位置约束局部时延搜索，再闭式更新位置，可作非NOMP候选/局部求解参照。原文单源且会局部收敛；3σ置信域和最大相关峰在强弱混合时可能错归属，需要保留多假设后由联合拟合筛选。这是待验证迁移判断。 | 全文已读 |
| R009 | [Newtonized Orthogonal Matching Pursuit: Frequency Estimation over the Continuum](https://doi.org/10.1109/TSP.2016.2580523) | 发现新分量、连续精修、循环修正旧分量和联合幅度重估均已有明确先例；不能作为新贡献单独列出。原频率模型每个源对应一个复幅度，迁移到未知宽带IQ必须处理逐频系数块和位置时延字典。必须与充分过采样、合理循环次数的版本比较。 | 见CSV精读记录 |
| R010 | [Block Newtonised Orthogonal Matching Pursuit for Off-Grid DOA Estimation in the Presence of Unknown Mutual Coupling](https://doi.org/10.1049/sil2.12068) | 块NOMP可借鉴其矩阵原子和块Newton更新，但原块结构来自阵列互耦，MMV观测共享阵列字典。频率组在本项目具有不同的时延响应，不能直接当作共享同一字典的快拍；须建立逐频块算子。 | 见CSV精读记录 |
| R011 | [High dynamic range gridless DOA estimation for single-snapshot sparse automotive MIMO radar](https://doi.org/10.1109/GeMiC71240.2026.11516433) | 强源离网残余掩盖弱源、连续修正、RELAX回访和BIC拆分已有非常接近的思想。实验使用2352虚拟阵元，每目标单复系数；其动态范围结果不能外推到四站。保留它是为界定机制先例，而不是要求照搬汽车阵列系统。 | 见CSV精读记录 |
| R012 | [A block-sparse bayesian learning algorithm with dictionary parameter estimation for multi-sensor data fusion](https://doi.org/10.23919/FUSION65864.2025.11124160) | 多字典SBL支持连续位置、活动对象更新与停用、重新发现对象及噪声更新，已经具有候选回访能力。把频率作为观测组有借鉴价值，但共享稀疏先验、频谱功率不均衡及大量频率组成本需重核。不能把一般的增删候选或重激活作为我们独有机制。 | 见CSV精读记录 |
| R013 | [Theory and practice of light-weight sequential SBL algorithm: an alternative to OMP](https://doi.org/10.1109/TSP.2025.3600492) | 低成本序贯SBL提供与OMP不同的选择准则；其基本算法偏向前向选择，连续版本仍先估支撑再局部细化。属于低成本贝叶斯工具来源，不是现成的多站盲宽带联合定位器。与新R012互补，但首轮不要求两者都完整实现。 | 见CSV精读记录 |
| R014 | [Efficient Direct Localization Based on Sparsity-Gradient Hybrid Optimization](https://doi.org/10.1049/ell2.70447) | 四站未知单源的相关域稀疏重建、样条插值与Armijo连续更新，已覆盖普通的粗细搜索改进。收敛至驻点不意味着找到全局最优。混合相关交叉项和逐源增益需要多源适配，适合作为便宜初始化或局部更新参照。 | 见CSV精读记录 |
| R015 | [A computationally efficient direct position determination algorithm based on OFDM system](https://doi.org/10.1109/LCOMM.2022.3231548) | 单天线未知波形分支利用站间频域乘积的FFT预计算，按候选时差构造小矩阵。应先实现其普通加速思路，再判断是否需要复杂区域搜索；FFT插值仍有误差，原单源评分不等同投影归一化的逐源增益。 | 见CSV精读记录 |
| R016 | [Grid hopping in sensor networks: acceleration strategies for single-step estimation algorithms](https://doi.org/10.1109/TSP.2024.3465842) | Grid Hopping通过局部参数网格相关量与插值计算位置域评分，可和连续精修组合。原边界与成本分析依赖具体相关形式，不能直接套用到逐频含投影分母的目标；仍是完整位置图加速的重要参照。 | 见CSV精读记录 |
| R017 | [Scalable-Complexity Steered Response Power Based on Low-Rank and Sparse Interpolation](https://doi.org/10.1109/TASLP.2024.3496317) | 先采样相关函数，再用低秩或稀疏插值计算整张细SRP图；少传感器大孔径时稀疏插值有价值。它加速的是线性SRP映射，不证明多源投影目标同样低秩或可作相同插值。比较时记录预计算、每轮更新、整图和最终定位成本。 | 见CSV精读记录 |
| R018 | [A Steered-Response Power Algorithm Employing Hierarchical Search for Acoustic Source Localization Using Microphone Arrays](https://doi.org/10.1109/TSP.2014.2336636) | 区域边界与分层搜索已有先例，且边界计算本身会昂贵，论文讨论跳过高成本初始层。迁移到当前条件评分须重新验证区域界；若只使用中心梯度作为变化量估计，应称搜索优先级，不能称严格上界或全局保证。 | 见CSV精读记录 |

### 2.2 02 条件补充：R019—R032

用于基础引文、特定先验、备选求解及后续扩展；满足相应条件后启用。

| 编号 | 论文标题与入口 | 当前用途 | 阅读证据 |
| --- | --- | --- | --- |
| R019 | [Direct Position Determination of Multiple Radio Signals](https://doi.org/10.1155/ASP.2005.37) | 保留为RF-DPD联合建模基础。理论含跨站时延及多频表示，不能再概括为完全没有频率维；多数数值示例为阵列、J=1。短窗误差与未知波形消元值得引用，但不能把阵列实验精度作为… | 见CSV精读记录 |
| R020 | [The Generalized Correlation Method for Estimation of Time Delay](https://doi.org/10.1109/TASSP.1976.1162830) | GCC统一多种加权，PHAT并非所有条件下最优。低能量频带相位噪声会被放大，已知信号/噪声谱的ML权重与估计权重也不同。作为候选生成组件，不能直接承担强弱混合源的全部分离… | 见CSV精读记录 |
| R021 | [Maximum likelihood localization of multiple sources by alternating projection](https://doi.org/10.1109/29.7543) | AP交替极大化和投影矩阵分解属于基础方法。原文承认全局收敛无一般保证、初始化关键；单快拍和相干源可用不等于任意四站宽带模型均可辨。当前具体基线按新R001建立，避免重复实… | 见CSV精读记录 |
| R022 | [TDOA Based Direct Positioning of Co-Channel Signals Using Cyclic Cross-Spectral Functions](https://doi.org/10.1109/LCOMM.2023.3347620) | 未知同频RF源可借助不同循环频率从观测中分离，再直接定位；不能默认同频就无法传统分源。主实验必须检查循环特征是否稳定可分，能够分开的条件用本方法作参照。原循环频率相异和记… | 见CSV精读记录 |
| R023 | [An Efficiency-Improved TDOA-Based Direct Position Determination Method for Multiple Sources](https://doi.org/10.1109/ICASSP.2019.8683421) | 原生多站单传感器，主要降低每个候选处的谱计算量；仿真使用大量分段，短记录下先核对统计秩。行列式与特征值核的实际速度应在同一优化环境测量，不能直接照文中阶数表断言固定加速比… | 见CSV精读记录 |
| R024 | [Importance sampling based direct maximum likelihood position determination of multiple emitters using finite measurements](https://doi.org/10.1016/j.sigpro.2021.108111) | 重要抽样给出不同于局部贪心搜索的联合优化路线，但原波形已知，并借助对应的可计算目标和提议分布。可借鉴搜索工具，不能把原算法的性能与盲IQ模型直接排名；有限样本的交叉项问题… | 见CSV精读记录 |
| R025 | [Iterative Broadband Source Localization](https://doi.org/10.1109/JSAIT.2023.3317094) | Slepian基和带限子空间减少强源表示残差，不能把换基或循环剥离当新贡献。原一维频带恢复与阵列DOA实验应区分，频谱完全相同但空间可分的情形不能简单排除；四站未知链路及… | 见CSV精读记录 |
| R026 | [An iteratively reweighted steered response power approach to multisource localization using a distributed microphone network](https://doi.org/10.1121/10.0024764) | 目前仅能重读作者摘要：按相干性和SNR选时频单元，检测主导源后移除其贡献单元，再找其余源。全文仍不可获得，不能断言必需时频稀疏，也不能认定它已解决或无法迁移到完全重叠RF… | 待全文核验 |
| R027 | [Newtonized Orthogonal Matching Pursuit for High-Resolution Target Detection in Sparse OFDM ISAC Systems](https://doi.org/10.1109/TVT.2025.3573165) | 单天线双基地OFDM强弱回波、联合Newton和实测都值得参考；接收端知道发送资源和参考信号，不是多个独立未知波形发射源。其动态范围和速度优势需连同参考信号先验一起理解。 | 见CSV精读记录 |
| R028 | [Efficient Position Determination Using Low-Rank Matrix Completion](https://doi.org/10.1109/ICSIDP62679.2024.10868130) | 随机计算少量位置统计量后低秩补全整图，属于直接RF加速参照。论文主要验证单源图；小整体重建误差不保证微弱窄峰被保留。启用时以弱峰命中和最终定位成功率评价，而非只看矩阵误差… | 见CSV精读记录 |
| R029 | [CFAR based NOMP for line spectral estimation and detection](https://doi.org/10.1109/TAES.2023.3282610) | NOMP-CFAR已有活动分量增加和删除，并处理未知噪声方差。多源自适应搜索改变最大统计量和相关性，原门限不能直接转成位置域CFAR。已知K阶段先评定位成功率；扩展未知K… | 见CSV精读记录 |
| R030 | [Efficient 2-D direction of arrival estimation using adaptive grid integrated dictionaries](https://doi.org/10.1109/JSEN.2025.3564909) | 区域积分字典可减少初始网格，但低于估计噪声功率的块会被丢弃，不能据此保证弱源覆盖。原主要实验为36阵元、等功率源；两维角域到位置域要重构算子。保留为区域候选备选，不直接照… | 见CSV精读记录 |
| R031 | [A fast direct locator for radiation source based on composite convolution neural network](https://doi.org/10.1049/ell2.13271) | 三站单天线、原始接收信号到位置的CNN已覆盖直接RF学习定位。源数已知且两源要重新训练，未系统验证大强弱差。可用作后续学习对照；随机复增益扰动实验不能等同解决显式多径传播… | 见CSV精读记录 |
| R032 | [Contextual direct position determination for path loss informed localization](https://doi.org/10.1109/LSP.2025.3550047) | 路径损耗先验约束可用于未知链路扩展，但MM是在固定候选位置下更新增益，位置仍需搜索。若加入该先验，应给适用基线同等信息，并扫先验失配；当前不把它作为搜索加速主线。 | 全文已读 |

### 2.3 03 历史归档：R033—R116

归档不表示论文质量低，也不表示它不影响未来主张。以下只保留题录，完整阅读记录保存在CSV；原10条延伸线索已纳入编号，但未读全文的状态保持不变。

<details><summary>01 传统方法与变体（17项，R033—R049）</summary>

| 编号 | 论文标题与入口 |
| --- | --- |
| R033 | [MMSE-based Distributed Direct Position Determination in Single-hop Sensor Networks](https://doi.org/10.1109/TVT.2026.3654660) |
| R034 | [An Improved TDOA-based DPD Method via Multiple-frequency Function Fusion](https://doi.org/10.1109/ICSIP55141.2022.9886640) |
| R035 | [Direct Localization of Emitters Based on Sparse Bayesian Learning](https://doi.org/10.1109/TVT.2019.2910831) |
| R036 | [A decoupled algorithm for geolocation of multiple emitters](https://doi.org/10.1016/j.sigpro.2007.03.008) |
| R037 | [Robust Expectation-Maximization Algorithm for Multiple Wideband Acoustic Source Localization in the Presence of Nonuniform Noise Variances](https://doi.org/10.1109/JSEN.2010.2050311) |
| R038 | [Direct Positioning With Efficient Search for a Passive Multistatic Radar](https://doi.org/10.1109/RadarConf2663773.2026.11675331) |
| R039 | [A Sparse Representation Direct Position Determination Method Based on Iterative Local Search](https://doi.org/10.1007/s00034-024-02937-2) |
| R040 | [An Analytical Subspace-Based Robust Sparse Bayesian Inference Estimator for Off-Grid TDOA Localization](https://doi.org/10.1016/j.dsp.2017.06.012) |
| R041 | [Fast and Fault-Tolerant Passive Hyperbolic Localization Using Sensor Consensus](https://doi.org/10.3390/s24092891) |
| R042 | [Resolution enhanced newtonized orthogonal matching pursuit solver for compressive beamforming](https://doi.org/10.1016/j.apacoust.2022.108884) |
| R043 | [Two-dimensional multi-snapshot newtonized orthogonal matching pursuit for DOA estimation](https://doi.org/10.1016/j.dsp.2021.103313) |
| R044 | [Efficient direct position determination using frobenius norm approximation](https://doi.org/10.1049/ell2.12479) |
| R045 | [Newtonalized orthogonal matching pursuit for mixed far-field and near-field source localization](https://doi.org/10.1109/ICASSP48485.2024.10448001) |
| R046 | [Direct position determination of multiple coherent sources using an iterative adaptive approach](https://doi.org/10.1016/j.sigpro.2019.03.023) |
| R047 | [Direct position determination by covariance-fitting on the riemannian manifold of hermitian positive definite matrices](https://doi.org/10.1109/ICASSP48485.2024.10445866) |
| R048 | [Expectation-maximization algorithm for direct position determination](https://doi.org/10.1016/j.sigpro.2016.10.015) |
| R049 | [Lightweight and computationally efficient direct localization based on subspace reconstruction](https://doi.org/10.1109/TVT.2025.3636667) |

</details>

<details><summary>02 学习与联合任务（26项，R050—R075）</summary>

| 编号 | 论文标题与入口 |
| --- | --- |
| R050 | [A Direct Position-Determination Approach for Multiple Sources Based on Neural Network Computation](https://doi.org/10.3390/s18061925) |
| R051 | [A Neural Network-Based DPD Algorithm For Low SNR And Off-Grid](https://doi.org/10.1109/SAM68950.2026.11687963) |
| R052 | [An End-to-End Deep Neural Network for Passive Positioning of Emitters and Receivers With Variable Numbers](https://doi.org/10.1109/TCCN.2025.3564488) |
| R053 | [A Lightweight End-to-End Multi-Source Localization Framework in Urban Environments via Local Radio Map Construction](https://doi.org/10.1109/TCCN.2026.3727490) |
| R054 | [Extending GCC-PHAT using Shift Equivariant Neural Networks](https://doi.org/10.21437/Interspeech.2022-524) |
| R055 | [SubspaceNet: Deep Learning-Aided Subspace Methods for DoA Estimation](https://doi.org/10.1109/TVT.2024.3496119) |
| R056 | [An envelope separation aided multi-task learning model for blind source counting and localization](https://doi.org/10.1109/ICASSP55912.2026.11464467) |
| R057 | [Joint Spectrogram Separation and TDOA Estimation using Optimal Transport](https://ieeexplore.ieee.org/document/11226045) |
| R058 | [Pairwise Neural Cross-Correlation for Accurate Estimates in Low SNR](https://doi.org/10.1109/MILCOM64451.2025.11310550) |
| R059 | [Deep learning based multi-source localization with source splitting and its effectiveness in multi-talker speech recognition](https://doi.org/10.1016/j.csl.2022.101360) |
| R060 | [A TDOA Localization Method of HF Source Based on Dual-branch CNN and Transfer Learning](https://doi.org/10.1109/LWC.2025.3623182) |
| R061 | [DNN-Aided Distributed Direct Position Determination for UAV Cluster](https://doi.org/10.1109/TVT.2023.3295411) |
| R062 | [Spatially Distributed Task-Oriented Compression for Multi-Emitter Localization and Characterization with Spectral Overlap](https://doi.org/10.48550/arXiv.2606.01446) |
| R063 | [End-to-End Signal-Aware Direction-of-Arrival Estimation Using Weighted Steered-Response Power](https://doi.org/10.23919/EUSIPCO55093.2022.9909962) |
| R064 | [Robust Speaker Localization Guided by Deep Learning-Based Time-Frequency Masking](https://doi.org/10.1109/TASLP.2018.2876169) |
| R065 | [A DNN Based Normalized Time-Frequency Weighted Criterion for Robust Wideband DoA Estimation](https://doi.org/10.1109/ICASSP49357.2023.10096809) |
| R066 | [Exploiting CNNs for Improving Acoustic Source Localization in Noisy and Reverberant Conditions](https://doi.org/10.1109/TETCI.2017.2775237) |
| R067 | [NeuralMUSIC: A Hybrid Neural-Subspace Framework for Robot Sound Source Localization](https://doi.org/10.48550/arXiv.2606.18664) |
| R068 | [MSDET: Multitask Speaker Separation and Direction-of-Arrival Estimation Training](https://doi.org/10.21437/Interspeech.2024-2537) |
| R069 | [Fast Direct Localization for Millimeter Wave MIMO Systems via Deep ADMM Unfolding](https://doi.org/10.1109/LWC.2023.3243157) |
| R070 | [SepLocNet: Multi-speaker localization with separation-guided TDOA estimation in wireless acoustic sensor networks](https://doi.org/10.1016/j.apacoust.2024.110488) |
| R071 | [Enhancing Localization Accuracy via Score-Based Interference Suppression](https://doi.org/10.1109/LWC.2026.3708142) |
| R072 | [Deep Tensor Completion for Fast Direct Position Determination](https://doi.org/10.1109/ICASSP55912.2026.11462366) |
| R073 | [ToA and TDoA estimation using artificial neural networks for high-accuracy ranging](https://doi.org/10.1109/JSAC.2023.3322803) |
| R074 | [High-frequency skywave source geolocation using deep learning-based TDOA estimation and bias-regularized semidefinite programming with field evaluation](https://doi.org/10.3390/s26092755) |
| R075 | [An improved deep learning-assisted TDOA estimation method for high-frequency skywave signals in ionospheric channels](https://doi.org/10.1109/ICSPS66615.2025.11347663) |

</details>

<details><summary>03 特殊观测与场景（27项，R076—R102）</summary>

| 编号 | 论文标题与入口 |
| --- | --- |
| R076 | [Direct TDOA geolocation of multiple frequency-hopping emitters in flat fading channels](https://doi.org/10.1049/iet-spr.2016.0299) |
| R077 | [Direct position determination based on rotating single-baseline interferometer for multiple time–frequency aliased sources](https://doi.org/10.1016/j.cja.2025.103767) |
| R078 | [Multiple Emitters Direct Tracking With Distributed Single-Sensors: Improved Particle Filter Based on Multiple-Frequency Fusion](https://doi.org/10.1109/TAES.2025.3555214) |
| R079 | [Direct Position Determination Method Based on Sparse Spectrum Data](https://doi.org/10.1109/ICASSP55912.2026.11462936) |
| R080 | [A Riemannian Manifold-Based Spatio-Temporal Joint Interference Mitigation Method for Direct Position Determination](https://doi.org/10.1109/LWC.2026.3732731) |
| R081 | [Direct Position Determination in Presence of Uncertain Propagation Loss](https://doi.org/10.1109/TVT.2025.3609715) |
| R082 | [Direct Multi-Emitter Tracking Based on the Generalized Labeled Multi-Bernoulli Filter](https://doi.org/10.1109/TSP.2025.3595842) |
| R083 | [TDOA-Based Passive Localization for Multidisjoint Sources Without Time Synchronization](https://doi.org/10.1109/JSEN.2025.3619052) |
| R084 | [Riemannian Covariance Enhancement for Robust Direct Position Determination Under Intermittent Interference](https://doi.org/10.1109/OJSP.2026.3733589) |
| R085 | [Multi-Source Weighted Localization Based on Cascaded DOA-TDOA](https://doi.org/10.3390/s26051614) |
| R086 | [Successive interference cancellation-based direct position determination for multi-interference localization via single LEO satellite](https://doi.org/10.1038/s44459-026-00078-3) |
| R087 | [Direct Localization With Direct-Path Interference for Reference Channel-Free Distributed Passive Radars](https://doi.org/10.1109/TSP.2025.3595854) |
| R088 | [Direct Localization of Strong and Weak Sources with a Moving Array](https://doi.org/10.1109/ICTC61510.2024.10602269) |
| R089 | [Maximum Likelihood Direct Position Determination of Multiple Sources With Channel State Information](https://doi.org/10.1109/TVT.2025.3538794) |
| R090 | [Off-grid sparse Bayesian learning for direct localization of cyclostationary sources with adaptive grid refinement](https://doi.org/10.1109/TWC.2026.3738889) |
| R091 | [Sparse Bayesian Inference-Based Direct Off-Grid Position Determination in Multipath Environments](https://doi.org/10.1109/LWC.2021.3057502) |
| R092 | [Direct Position Determination of Wideband Source over Multipath Environment: Combining Taylor Expansion and Subspace Data Fusion in the Cross-Spectrum Domain](https://doi.org/10.3390/s25164967) |
| R093 | [An Enhanced Direct Position Determination of Non-Circular Sources via Sparse Bayesian Inference and Grid Refinement Strategy](https://doi.org/10.1016/j.dsp.2024.104564) |
| R094 | [Direct Localization of High-Order QAM Sources With Multiple Anchors: Dual Atomic Norm Minimization Framework](https://doi.org/10.1109/TWC.2025.3599530) |
| R095 | [Gridless Maximum Likelihood One-Bit Direct Position Determination](https://doi.org/10.1109/LSP.2024.3491020) |
| R096 | [Self-calibrated direct position determination based on global-local hybrid optimization](https://doi.org/10.1109/LCOMM.2026.3675651) |
| R097 | [Direct localization of wideband sources using distributed arrays: a subspace focusing and dimension reduction approach](https://doi.org/10.1016/j.sigpro.2025.109902) |
| R098 | [An indoor direct localization method utilizing structural sparsity and low-rankness with grid refinement](https://doi.org/10.1109/JIOT.2025.3559125) |
| R099 | [Iterative strong targets cancellation for multi-target detection in JRC systems](https://doi.org/10.1109/LCOMM.2025.3542371) |
| R100 | [Sensing with unknown signals: ISAC enabled distributed passive sensing for multi-target detection and localization](https://doi.org/10.1109/TNSE.2026.3670825) |
| R101 | [Passive moving emitter localization and tracking via robust PCA](https://doi.org/10.1109/RadarConf2663773.2026.11674976) |
| R102 | [Enhanced direct position determination of multiple sources based on 1-bit sparse bayesian learning](https://doi.org/10.1109/JSEN.2025.3605149) |

</details>

<details><summary>04 检测关联与基础理论（14项，R103—R116）</summary>

| 编号 | 论文标题与入口 |
| --- | --- |
| R103 | [Direct Position Determination for Multiple Signals Without Prior Knowledge of Source Number](https://doi.org/10.1109/LWC.2025.3576170) |
| R104 | [Multi-Source Localization and Data Association for Time-Difference of Arrival Measurements](https://doi.org/10.23919/EUSIPCO63174.2024.10715317) |
| R105 | [Multi-Sensor Multi-Source Localization under Clutter and Uncertainty](https://doi.org/10.1109/TSP.2026.3685137) |
| R106 | [Ghost Discrimination Method for Broadband Direct Position Determination Based on Frequency Coloring Technology](https://doi.org/10.3390/jmse12122182) |
| R107 | [High Accuracy Source Localization Based on Parallel Factor Analysis of TDOA in the Cross Correlation Domain](https://doi.org/10.1109/TCOMM.2025.3534582) |
| R108 | [Bandpass Signal Design for Passive Time Delay Estimation](https://doi.org/10.1109/ACSSC.2016.7869537) |
| R109 | [Multi-TDOA Estimation and Source Direct Position Determination Based on Parallel Factor Analysis](https://doi.org/10.1109/JIOT.2022.3228810) |
| R110 | [Multiple Sound Source Counting and Localization Based on TF-Wise Spatial Spectrum Clustering](https://doi.org/10.1109/TASLP.2019.2915785) |
| R111 | [Model-Based Expectation-Maximization Source Separation and Localization](https://doi.org/10.1109/TASL.2009.2029711) |
| R112 | [Direct detection and position determination of multiple sources with intermittent emission](https://doi.org/10.1016/j.sigpro.2010.05.010) |
| R113 | [一种强弱信号的直接定位与信源数估计方法](https://kns.cnki.net/kcms2/article/abstract?v=xiNUFhx2xxb9IyWwbG4BOeZgJc8hGs1gdHBsx6TooTujuSPH9iVb8qKRNtrgs1ZFkf-ZmPPzgttjriGJW_FfoahnJnkAcfoS8lbmcvZUceBfJH5QTboWp0hsrkhP8Cx63cKWWMfNaIN6HF-NPZvXKuxNMCaTQZ6D0husp91Z6sQSw0lysOEycg==) |
| R114 | [Fundamental limitations in passive time delay estimation–part I: narrow-band systems](https://doi.org/10.1109/TASSP.1983.1164061) |
| R115 | [SNR-adaptive ranging waveform design based on ziv-zakai bound optimization](https://doi.org/10.1109/LSP.2023.3320893) |
| R116 | [Fundamental limitations in passive time-delay estimation–part II: wide-band systems](https://doi.org/10.1109/TASSP.1984.1164429) |

</details>

## 3. 已有方法的能力、条件与迁移判断

### 3.1 候选发现：离网误差不等于候选遗漏

粗网格确实可能造成off-grid问题，但应把三种现象分开。它们可能同时出现，处理方式却不同。

| 现象 | 具体含义 | 已有方法与当前关注点 |
| --- | --- | --- |
| 网格量化／字典失配 | 已找到正确区域，但真位置不在网格上 | R002、R009/R010、R092/R095等用连续修正减小偏差；不必一直加密全域网格 |
| 候选区域遗漏 | 粗网格没有采到窄峰，或正确区域被Top-K筛掉 | 局部精修没有合适初值；R018/R038处理区域覆盖，R090/R093采用候选细化，仍需检查低分区域是否被保留 |
| 强源拟合残余 | 强源位置、波形或链路估计不准，残余遮住弱源 | R021/R001、R005/R004、R009、R010、R027通过联合重估、循环修正或消除处理；搜索与消除需要分开归因 |

R009的NOMP已将“粗网格发现—连续精修—循环反馈—联合幅度重估”连起来，且测试了不同分量强度。R027进一步研究强车辆与弱行人RF回波，其方法仍有功率差和初始化边界。已有方法能够改善问题，并不表示问题已在所有观测和成本条件下解决。

R090与R093说明自适应网格和局部候选更新已有先例。R093的Remark 4要求初始网格足够密；§5.3将某离网方法的退化推测为仅更新前Q个峰的影响。这里既不能把作者推测写成已证实原因，也不能把对比算法的退化写成该文增强方法已在强弱场景失败。

本项目重点检查：**正确区域是否进入候选、进入后能否连续精修、强源修正后能否重新发现弱源。** 当前归一化分母随位置变化，需要适配导数和搜索规则，但这一点本身不构成新贡献。

R011已经研究强源离网误差遮蔽弱源，并结合候选拆分和循环精修；R042修复合并主瓣导致的初始化偏差。R030以区域积分响应代替孤立格点，提供另一种候选覆盖思路，但门限筛选仍可能删弱源。三者分别代表候选拆分、初始化修复和区域表示，不等于已经解决本项目的短窗盲多源问题。

### 3.2 连续精修：优先建立充分的非神经参照

| 方法族 | 已有能力 | 在本项目中的使用方式 |
| --- | --- | --- |
| AP／AML：R021/R001，混合求解R007 | 消去未知波形，固定其他源更新一个位置；粗初始化接连续优化 | 作为当前物理模型的首要参照；核对初值、停止、增益和全搜索区 |
| NOMP／Block-NOMP：R009/R010 | 新分量发现、局部Newton、循环反馈和联合系数重估；R010处理未知系数块 | 将角度／频率参数改为位置，把每源未知量改为逐频源谱块 |
| 参数字典与离网SBL：R002、R035、R090/R093 | 稀疏支撑、链路或网格偏移联合更新 | R002为单天线场景的重要近邻；阵列版本只借鉴适用组件 |
| 连续局部更新：R014/R092/R095 | 梯度、Taylor或MM减少迭代内细网格计算 | 用作简单且充分优化的精修代表；原单源、参考信号或量化似然另行对齐 |
| 支撑替换／联合抽样：R039、R024/R089 | 替换早期错误候选，或用重要抽样求解联合目标 | R039原文只换九个离散邻点；R024/R089保留优化价值，但重建盲多源似然 |

本项目的每个位置对应多个未知频率系数，普通NOMP的单复幅度原子不能原样使用。R010先消去系数块再精修的结构更接近当前问题。需要推导位置梯度、病态条件和接受准则；这属于必要适配，适配后是否值得采用由性能和成本决定。

连续算法仍依赖候选。更好的局部优化器不能自动找回从未进入搜索的区域。反过来，若扩大初值或普通循环更新已取得同等效果，就不把该效果归到新的候选机制。

### 3.3 联合更新：减少强源残余并保持弱源信息

R021/R001已覆盖未知波形消元和逐源AP，R036进一步估计链路与源谱；R009、R010、R027也强调发现新分量后重新更新已有分量。联合更新已有充分方法来源，本研究可围绕其在短时盲多源条件下的使用方式形成改进。

| 路线 | 可以借鉴什么 | 需要保留的条件 |
| --- | --- | --- |
| R005稳健轮流定位 | 用位置邻域响应构造有限秩零陷，平衡强源残余与弱源损伤 | 原传播场已知；少站下扩大投影秩可能消耗弱源信息 |
| R004宽带互相关消除 | 将相关矩阵投影写成时延组合，不要求时频稀疏 | 适配复IQ、跨站幅相和球面时延；完整式、截断式和PHAT分别核对 |
| R027强弱RF回波NOMP | 连续修正强分量、联合更新幅度，再检测弱分量 | 已知共同OFDM信号；不能直接用于多个独立未知波形 |
| R006组稀疏联合拟合 | 估计与方向／位置关联的频率系数 | 不相关源统计与短记录估计质量；最终取峰K另行处理 |
| R022循环特征定位 | 可分循环频率下分离同频RF源，含实测依据 | 循环特征可分时纳入比较，不把所有同频源都视为不可分 |

频率权重会同时作用于同一频点的多个分量；完全重叠时，单纯带限不能直接分源。传统联合拟合的系数也能承载源的频谱结构，原文只报告位置不等于无法增加频谱输出。

R003继续作为支持单天线、未知链路的高分辨谱法参照。必须对齐总记录、分段、加载和搜索；当前实现失效不能替代对该方法族的判断。

R012将多字典连续参数、稀疏强度和噪声联合更新，正式版包含多雷达四目标及距离相关路径损耗；不能概括为只验证单目标。可将频率作为观测组、共享位置并估计逐源频谱，但大量频率组下的更新成本和跨站增益约束需重新设计。R013提供轻量贝叶斯递推，适合比较硬消除与保留参数不确定性的处理；其相对普通OMP的归一化优势不能直接当作相对现有归一化AP评分的优势。

### 3.4 计算成本：分别比较少算网格与快速算完整图

| 路线 | 代表文献 | 当前应检查什么 |
| --- | --- | --- |
| 区域覆盖与粗细搜索 | R018、R038；R041为共识目标的分支定界 | 粗筛是否保留弱峰所在区域；区域覆盖、可靠检测和全局保证分别报告 |
| 完整细网格映射加速 | R017，及R023/R034的每点计算简化 | 预计算和在线成本；复IQ与投影目标适配后是否保持弱峰 |
| 低秩评分图补全 | R028 | 单源图的低秩性是否延续到多源投影图，随机采样会否漏掉窄峰 |
| 学习式张量补全 | R072 | 原单峰加低秩背景假设如何适应多峰；训练投入、未见几何及弱峰保留 |
| 连续求解与减少无效更新 | R014/R007、R009、R010、R027、R024/R089/R095 | 初值与联合更新成本，有限预算下的可靠性，失败与超时都计入 |

R017降低的是完整细网格映射成本，R028减少实际评分点数，R072增加了学习先验，三者不能只比较图重建MSE。低幅弱峰被抹掉时，全图平均误差仍可能很小。

R038利用参考通道的CAF和已知照射源几何；R041处理给定到时的共识计数。它们的区域界有借鉴价值，不能直接变成当前投影似然的保证。区域上界和停止误差可以增强论文，但不是研究成立的前提。

R015的未知波形分支通过站间频域乘积及FFT预计算相关响应，R016从低维参数域插值到完整位置图。二者应与R017一起提前核验，不能把OFDM标题解释为必须已知导频。R044用范数近似改变评分，其收益需和同评分加速分开归因；这些单源结果均不自动成为多源投影评分的性能保证。

### 3.5 观测条件与RF迁移

| 关键条件 | 相关来源 | 对当前方案的影响 |
| --- | --- | --- |
| 分布式标量、未知波形 | R001、R003、R002、R028 | 最接近当前信息条件；分别核对源数、增益及多源能力 |
| 已知参考／共同波形 | R027/R038、R024/R089/R091/R095 | 可以迁移优化思想，但不能把参考除法后的通道估计当作盲混合IQ |
| 站内阵列或非圆／高阶统计 | R010、R090/R093/R094 | 重建多站时延响应；不能保留原阵列自由度后宣称单天线同条件优势 |
| 已检测时差／到时 | R040/R041及关联文献 | 前端弱峰漏检要计入，不能输入真实时差 |
| 有限观测窗 | R001/R036、R025 | 先比较共同有效区间、裁剪、加窗和已有有限窗处理；模型方向按诊断启动 |
| 不同噪声和未知链路 | R037、R002/R036 | 噪声白化与链路知识分别对齐；未知增益必须有跨频约束 |
| 语音、运动和地图先验 | R056/R059/R068/R070、R052/R078/R082、拓展文献 | 作为对应模块或场景来源，不用于静止短时单天线的直接性能排名 |

逐站、逐源、逐频的任意复增益可能吸收时延相位。增益估计要有跨频共享或有依据的平滑结构，并固定尺度／相位基准。采样同步、接收机相位校准和传播增益也分别定义。

四站逐频自由源谱模型在四个候选响应列满秩时可拟合任意观测。未知K不能只靠残差持续下降来增加源数。这是当前模型的秩条件，不是所有方法统一最多只能定位三个源。

R096依赖移动阵列角度响应；R097按式(35)–(40)聚焦并剖面消去自由复增益，M=1时原定位准则退化；R098是单标签、多阵列、多径，不能把多任务理解为独立多发射源。R099使用共同已知OFDM参考，有限和消除关系不对任意离网时延严格成立。上述方法需要改变关键模型，不能只修改阵元数；R015保留跨站时延，迁移条件与它们不同。

### 3.6 计数、频谱对应和学习组件的后续用途

R052/R056/R062/R068提供可变源数、逐源表示和联合输出；R104/R106/R110/R111/R070提供关联或分离思路，R112区分间歇发射状态与总源集合。这些文献仍有价值，当前先完成已知两源的定位机制。

随后再加入未知K、空场、三源及频谱—位置对应，并保留“传统定位＋轻量关联”对照。属性完全相同的源按无序集合评价，不人为制造配对错误。

现有E2E系统作为基线。R055/R063、R064、R065、R066、R067等学习统计量或权重的工作在相应瓶颈出现时再使用。网络可以依靠先验或计算摊销带来收益，不需要先证明某个环节只能用深度学习。

## 4. 具体场景、关键困难与研究目标

### 4.1 实际需求与观测范围

同频干扰排查需要发现容易被强信号遮住的发射源。R022的同频RF实测、R112的间歇发射，以及ITU-R的[Spectrum monitoring evolution](https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-SM.2355-2-2023-PDF-E.pdf)提供需求背景；R027进一步说明强弱RF分量恢复有实际用途，但其回波模型不同。

当前研究固定为少量同步单天线站点、一次短时混合复IQ、独立未知波形和重叠频谱。先采用已知两源、匹配传播响应，随后逐项放宽。短截获可以对应间歇发射和共同可用记录受限，但项目的具体采样点数尚不是已落实的业务硬约束。没有明确处理时限时，以可靠性—耗时—内存曲线评价。

目前以仿真为主。实际性由需求来源、合理物理模型和独立条件验证支撑；实测能力不写入已经完成的结果。

### 4.2 当前主问题

> 少量同步单天线站点接收短时混合IQ时，在给定计算预算下，提高强弱重叠源的恢复可靠性，并降低位置联合精修的代价。

核心关注两个量：弱源候选是否被保留，以及最终恢复所需的计算。已有方法能成功定位时，仍可比较同等可靠性下能否更省计算；预算相同时，则比较能否少漏弱源。

| 目标 | 需要得到的结果 |
| --- | --- |
| 分清原因 | 将网格失配、候选遗漏、强源残余和信息不足分开 |
| 建立合理参照 | 充分AP／AML、R002、多字典SBL、一个块NOMP适配及整图加速代表 |
| 形成专用处理 | 根据失效机制选择候选保留、联合不确定性处理或评分加速中的一项改动 |
| 证明有用 | 相同可靠性下降低成本，或相同成本下提高弱源召回，并报告适用范围 |

不同站点由不同源主导仍作为分层条件，不预设它是主要失效原因。联合DPD并非各站独立取最强观测后拼接。

### 4.3 项目证据与待检验机制

R6保持新投影评分不变、更充分搜索后恢复了部分原失败，剩余错误位置也存在真位置评分更高的情况。这支持当前搜索有损失。R6使用挑选出的长窗诊断条件，尚不能替代完整检查集、原短记录或独立条件验证；R3针对旧DPD／HR谱的结论也不能直接转移到新评分。

强源误差与候选遗漏是否共同限制弱源，是下一步待检验机制。真位置得分高于算法输出，只说明有更好的位置未找到，不说明整个连续多源目标的全局最优一定在真位置。

精确实验事实见[实验结果记录](实验结果记录.md)，执行与授权见[当前任务](当前任务.md)，历史诊断方法见[实现文档6.18](统一端到端模型研究.md#618-g8-r6统一投影流程剩余失败诊断)。

### 4.4 研究gap的工作定义

当前候选缺口是：**在盲多源、短记录和明确计算预算下，常用候选选择与连续更新的组合仍有可以改善的弱源可靠性—成本取舍。**

NOMP、参数字典学习和自适应网格已经覆盖其中的重要部分。本研究不主张首次离网定位，也不要求已有方法完全失效。若针对性候选机制通过普通适配无法同样有效地获得，并在独立条件下带来有用改善，就可以形成贡献。收益大小和适用范围由比较决定。

## 5. 方法方案：候选发现、连续精修与联合更新

### 5.1 在同一物理模型上建立连续求解参照

用站点数$M$、源数$K$表示逐频观测：

$$
\mathbf x_f=A_f(\mathbf p,g)\mathbf s_f+\mathbf n_f,\qquad
[A_f]_{mk}=g_{mk}e^{-j2\pi f\tau_m(\mathbf p_k)}.
$$

给定响应和白噪声时，先按R021/R001建立：

$$
\min_{\mathbf p,\{\mathbf s_f\}}\sum_f\|\mathbf x_f-A_f(\mathbf p)\mathbf s_f\|_2^2,\qquad
\hat{\mathbf s}_f=A_f^\dagger\mathbf x_f.
$$

固定其余源响应$A_{-,f}$，令$P_{-,f}=I-A_{-,f}A_{-,f}^\dagger$，候选源解释的能量为：

$$
\Delta J(\mathbf p)=\sum_f
\frac{|(P_{-,f}\mathbf a_f(\mathbf p))^H(P_{-,f}\mathbf x_f)|^2}
{\|P_{-,f}\mathbf a_f(\mathbf p)\|_2^2}.
$$

按登记的数值规则处理病态与分母退化。匹配增益与未知增益分别评价，后者明确跨频约束。有限窗或非白噪声改变观测假设时，同时调整似然和基线。

实现充分AP更新，并借鉴R010建立块NOMP适配：从残差发现位置，连续精修，循环更新已有源，联合重估逐频系数。采用观测产生的多初值和目标改善接受准则；真位置不能进入部署初始化。先检查单个组件的一致性，再比较定位性能。

R012/R013作为不同于NOMP的贝叶斯参照，先核对多频共享位置、逐源系数块与增益约束。整图预计算按§5.4提前做成本筛查；若已足够便宜，不预设候选淘汰为核心方法。以下候选改动仅在诊断支持后选用。

### 5.2 针对候选遗漏，先选择一项改动

优先测试“根据几何、有效带宽和投影响应变化决定候选保留与局部采样”的机制。粗网格得分低不必立即永久删除该区域；根据残差变化重新检查仍可能包含有效响应的区域。

| 可选处理 | 作用 | 最简单的对照 |
| --- | --- | --- |
| 调整局部采样步长 | 减少窄峰被网格跨过 | 固定细网格、普通多层网格、增加初值 |
| 残差重新搜索并补充候选 | 强源修正后重新发现弱源 | 普通循环NOMP、同次数全域残差搜索 |
| 候选替换与恢复 | 纠正早期错误选择，避免Top-K一次截断决定全部结果 | 扩大Top-K、R039式邻域支撑替换、随机补点 |

先选最有效且成本清楚的一项，不把所有机制叠加后统称创新。候选规则只利用观测、几何和已估参数；真实弱源位置仅用于事后判定是否覆盖。

若采用几何或响应变化准则，应说明它怎样影响漏搜概率或实际覆盖。区域上界可以在后续补充；没有严格界时，如实作为启发式方法验证，不要求整个多源问题的全局最优证明。

### 5.3 将候选机制与联合连续精修连接起来

发现新源后，重新优化已有源的位置和逐频系数，再形成下一轮残差。检查强源位置变化、残余能量、弱源候选覆盖和重复候选是否同步改善。

至少拆开以下比较：

1. 普通候选＋普通连续精修。
2. 改进候选＋同一连续精修。
3. 普通候选＋联合循环更新。
4. 改进候选＋联合循环更新。

这些是模块归因，不代表四个独立新算法。与普通循环NOMP、增加初值和扩大Top-K分别比较；保持相同评分来判断搜索收益。研究R004消除机制时，则固定合理搜索策略与预算，区分消除收益。

### 5.4 提前检查完整位置图加速

优先适配R017的插值计算和R028的低秩补全；若采用学习，再加入R072。固定输入、候选分辨率与终止条件，检查弱峰保留、峰位置、排序、最终集合误差和完整耗时。

图重建误差与弱峰恢复分别报告。投影目标随已选源变化时，统计重复预计算和更新成本；几何变化需要重建映射时也计入。全图方法已足够快时，可以直接使用，并判断候选机制是否还带来额外收益。

### 5.5 先验证短记录，再扩展完整任务

先已知两源，返回原短记录，并设置长记录对照。固定弱源带内SNR改变强源功率，配对改变网格偏移、站点几何和频谱重叠。先回答短记录中是否仍能可靠保留并精修弱源。

随后加入未知K、空场和三源，校准新增源停止、重复候选与虚警。最后恢复频谱—位置对应，保留“传统定位＋轻量关联”对照。源谱系数、频带占用和定位权重分别定义；未知链路下不把未标定系数直接当作真实发射功率。

学习模块按需要进入：候选排序、观测可靠性或固定迭代步数的计算摊销都可以研究。与非学习替代比较，说明训练先验和离线成本；既有E2E模型继续作为系统基线。

### 5.6 备选方向与实现核验

只有搜索充分后仍出现可归因的短窗拟合偏差，才转向共享带限波形、传播时延和各站截取窗口的联合模型。先比较裁剪、加窗、补零以及R001/R036/R025的直接适配，再判断新的窗口处理是否必要。未知链路、多径和异步逐项增加，不同时改变主问题。

实现时核对时延符号、分数时延、双边复频谱、相位基准、投影幂等和数值病态。检查候选覆盖、选错候选和精修破坏三个环节；不使用真位置窗口限制搜索。实现一致性检查不作为定位性能证据。

## 6. 最小对照集合与公平比较

### 6.1 第一轮的强代表

| 代表 | 对应能力 | 实现范围 |
| --- | --- | --- |
| R021/R001的充分AP／AML | 当前未知波形联合目标的可达性能 | 合理初值、连续更新、联合系数重估；R007可提供混合求解思路 |
| R010块NOMP适配，结合R009/R027/R011 | 逐源发现、连续精修和循环反馈 | 使用本项目时延模型和逐频系数块，登记与原算法的差异 |
| R002参数字典学习；R012多字典SBL | 联合稀疏与连续参数估计 | 先选择最适用的贝叶斯代表；对齐源数、频率分组和增益先验；R013作为低成本扩展 |
| 一种代表性加速方法 | 有限成本下保留有效候选 | 先核验R015/R016/R017完整图加速，再按瓶颈选R028或R018/R038；不要求全部实现 |
| R004宽带消除 | 区分搜索与强源处理的收益 | 适配完整互相关消除，必要时比较低成本近似 |

R003保留为高分辨谱法背景参照，分段和加载要合理。R005仅在稳健投影成为主张时展开，R025仅在有限窗方向启动后展开。阅读了更多论文不等于必须全部复现。

### 6.2 按贡献补充对照

| 主张 | 需要排除的简单解释 |
| --- | --- |
| 候选保留更可靠 | 只是扩大Top-K、增加初值或加密网格 |
| 联合精修改善弱源 | 只是普通循环NOMP／AP或多算几轮 |
| 相同可靠性下降低成本 | 对照未优化，或未计预处理和预计算 |
| 学习补全有效 | 非神经插值／低秩补全已经同样有效 |
| 有限窗建模更准确 | 普通裁剪、加窗、补零已能解释改善 |

只要求覆盖核心能力的强代表。比较适用条件和实际收益，不要求所有文献方法都失败；未实现的方法不纳入性能胜出范围。

### 6.3 观测、先验与系统比较

| 层次 | 对齐内容 | 允许变化 |
| --- | --- | --- |
| 模块比较 | 同一IQ、评分及与该模块无关的后端 | 候选或精修等指定模块 |
| 完整系统 | 总观测、搜索区、可用先验、硬件与预算 | 各方法合理的原生初始化、计数与求解 |
| 真值辅助诊断 | 明示真实K、位置、频带或链路 | 仅用于分解原因，不参与部署排名 |

采样点数不能当作独立空间快拍数。已知精确增益、由观测估计增益和统计先验分别比较；共同参考波形是额外信息。R027/R089等不能借用参考信号后与盲IQ方法直接排名。

两步TDOA方法从同一混合IQ提取多时差，并计入漏检与关联。完整传统系统不必经过本项目候选模块；联合输出比较补齐轻量频谱—位置关联。

### 6.4 计数和搜索公平性

已知K定位与未知K检测分别报告。CFAR或残差阈值在开发数据校准，包含空场；自适应全域搜索后的最大评分分布不能直接套用固定单点检验。

所有方法有合理初始化、停止和调参机会。记录失败、超时、未收敛以及预算截断。小规模充分搜索作为参考解时说明精度范围，不直接称连续全局真解。搜索没找到好解与观测不支持弱源分别判断。

### 6.5 完整成本

计时包含IQ／FFT、统计量、候选生成、精修、联合系数重估、计数、关联和数据搬运。冷启动、缓存复用和离线预计算分开，报告时间与内存；观测时长与计算时延也分别报告。

核心结果用两种口径：同等弱源可靠性下比较成本，同等成本下比较弱源召回和集合误差。没有稳定优势的组件采用简单版本。学习方案另报数据生成、训练、调参成本及未见波形／几何表现。

## 7. 验证顺序与判断规则

### 7.1 执行顺序

| 顺序 | 要做什么 | 要回答什么 |
| --- | --- | --- |
| 1 | 建立充分AP、R002/多字典SBL和一个块NOMP代表；同步检查R015/R016/R017整图成本 | 不同原理的普通适配能改善多少，成本多大；同评分搜索与换评分比较分开 |
| 2 | 配对改变强源估计误差和候选筛选，真值仅作诊断 | 残余与候选遗漏是否分别或共同影响弱源 |
| 3 | 按诊断选择候选保留、联合不确定性处理或评分加速中的一项改动 | 相比对应方法的普通适配，改动是否有针对性收益 |
| 4 | 连接联合更新，复核完整成本与弱峰保留 | 是否提高可靠性—成本表现 |
| 5 | 原短记录、不同网格偏移、几何、功率差和重叠条件 | 收益是否超出长窗挑选案例 |
| 6 | 固定方案后独立验证，再扩展未知K与频谱对应 | 能否形成完整可用系统 |

这是研究设计。代码入口、数据清单、资源预算和正式运行方案仍需在执行文档中落实，不能把本次文档更新记为新实验。

强弱比扫描固定弱源带内SNR，并另设固定总功率对照。网格平移不改变真实场景；强源误差控制要与纯候选变化配对。分开部分／完全重叠、独立／相关波形、同一强源主导／站间主导源变化。

### 7.2 指标与独立验证

主报弱源召回、包含漏检与虚警的集合误差、完整耗时与内存。未知K时固定虚警水平；已知K诊断另报强弱源恢复率、尾部误差和预算内最优目标值。

过程指标包括：连续精修前后的弱源候选覆盖率、精修成功率、重复候选、强源残余以及最终弱源恢复。近似图另报弱峰位置、幅值与排序，不只统计整图MSE。失败样本不能从RMSE或成功率分母中删除。

先在开发数据固定最小有用改善、可接受退化和预算，再用未见几何、波形及随机实现确认。使用配对重复和按场景／布局重采样的置信区间；学习方案另报训练seed差异。封存test不参与方案选择。

### 7.3 继续、简化或转向

| 验证结果 | 决定 |
| --- | --- |
| 相同可靠性下明显更省计算，或相同预算下更少漏弱源 | 保留该贡献，明确条件和收益 |
| 已有方法也有效，但专用处理仍改善成本或稳定性 | 研究继续，贡献落在针对性适配和可验证改进 |
| 普通适配与新方法效果、成本相当 | 采用简单版本，停止该项改进主张 |
| 只有普通加密网格或增加初值带来改善 | 记为基线修复，继续判断是否还有值得研究的成本问题 |
| 搜索充分后仍有明确窗口误差 | 启动有限窗备选方向 |
| 观测信息不足 | 调整观测或目标范围，不能靠增加网络容量补回信息 |

### 7.4 论文证据

论文围绕一个具体问题、一项主要改动和相应验证展开。方法部件可以来自已有文献；需要清楚说明为何在当前条件下这样组合或修改，以及相较合理适配获得什么收益。

面向SCI二区及以上目标，重点补足问题机制、强代表比较、消融、独立条件和完整成本。区域上界、停止误差或理论分析按能够证明的范围提供，不把全局最优证明设为必需。当前以仿真支撑，不预先承诺期刊分区或录用。

## 8. 研究主张与路线

研究目标是**短时盲多源观测下，兼顾弱源候选保留、连续定位精度和计算成本**。核心文献围绕候选发现、连续精修、联合更新与计算成本组织；其他工作按计数关联、学习组件或场景扩展保留。

执行路线为：**不同原理的适用参照与整图成本筛查 → 分清主要失效机制 → 一项针对性改动及联合更新 → 短记录和独立条件 → 未知K及频谱—位置对应。** 不提前把NOMP或候选淘汰锁定为最终方法。

当前可检验的贡献表述是：针对少量同步单天线站的强弱重叠RF源，在明确预算和观测条件下，通过与失效机制对应的候选处理、联合估计或计算加速，改善弱源恢复的可靠性—成本取舍。R6提供搜索损失证据；具体方法、机制和改善幅度由后续实验确认。

相似工作是方法来源和比较对象。已有方法成功不否定研究价值；没有额外收益时采用已有方法。端到端网络、严格区域界和有限窗重建都作为按需工具，不预设为论文必须采用的路线。
