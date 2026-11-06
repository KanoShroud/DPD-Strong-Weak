# Round 3：量化 DPD 与已知阈值最近邻核验

日期：2026-10-22。执行范围：只读文献；没有实验、代码检查或实现、Zotero改动。当前验收目标为至少1个合格方向；本报告不保证期刊，不因该目标降低近邻核验标准。

## 结论先行

**这三篇并未共同证明“四个单天线站＋未知独立宽带波形＋同频强弱源”的完整方案已经解决，但它们足以否定若干简单创新主张。** 已知dither、非零/时变阈值恢复幅度、probit-ML、量化SBL、协方差恢复后再定位，均不是可直接宣称的新机制。

**本轮没有形成一个可认定为超出常规适配的合格候选机制。** 不把“论文没有恰好用四个单天线”当成创新空白。任意未知波形和时间相关样本仍有模型边界未解决，但本轮没有拿出超出现有通用量化似然、协方差恢复与常规实验设计的独立新解法，故不计入候选数。

## 1. 三篇最小对照

|条件|Li et al. 2024 gridless ML|Wang et al. 2025 OBSBL-DPD|Xiao et al. 2023 covariance recovery|
|---|---|---|---|
|定位对象/源数|单个静止OFDM源|正文说多源且Q未知；Algorithm 1却列Q为输入，接口矛盾未解|不直接估计源数；恢复任意有限维观测协方差|
|波形先验|已知QPSK导频、已知子载波|未知源样本；窄带，跨站包络相同；稀疏Gaussian-gamma层次先验|零均值多元Gaussian观测；无需知道发射样本|
|独立性|频域Gaussian噪声似然，已知导频结构|多快拍相互独立、共同空间支撑；条件源协方差为对角|N个iid向量样本；不能直接套在任意相关时间序列上|
|站/阵元|每站标量信道；例子3站|模型每站M阵元；例子5站×8阵元|维数通用，4复通道可以转8实维；不是定位几何定理|
|量化位置|DFT、导频消除之后的channel measurements|堆叠站内阵列接收数据直接取符号|对实/虚观测和已知阈值的差取符号|
|阈值|固定0|固定0|已知非零、时变阈值；精确值进入似然|
|幅度尺度|仅估计信道/噪声标准差之比|零阈值符号仍有整体尺度不变性；先验选择尺度不等于数据辨识尺度|恢复接收观测协方差对角与非对角；不自动分离发射功率与未知信道增益|
|校准|源与站时间/频率同步；站位置已知|使用位置决定的已知字典；无显式站钟偏/CFO/独立增益误差参数|阈值必须与待估信号在同一已知数值尺度；未知阈值误差不在本文模型|
|强弱证据|单源，不能验证多源强弱|已读§IV未见固定弱源、单独扫强源功率的实验|有接收变量方差不均衡与相关系数试验；不是同一物理几何的强弱源检测曲线|
|最近邻作用|probit-ML＋MM/gridless基线|零阈值多源稀疏定位近邻，需处理本文数学/接口疑点|“known dither恢复幅度”的直接反证近邻|

定位与来源详见下文；该表不表示将三篇拼接即可无条件解决本项目。

## 2. Gridless ML：不能把导频信息无偿移植给未知波形算法

Li, P., Li, J., Zhang, X., & Wu, Q. (2024). Gridless maximum likelihood one-bit direct position determination. *IEEE Signal Processing Letters, 31*, 3099–3103. [DOI:10.1109/LSP.2024.3491020](https://doi.org/10.1109/LSP.2024.3491020)。Zotero：62T6T2NB，附件VHKUPEZF。

### 读到的实际链路

§II，印刷页3100，式(1)–(3)：

r_tilde_l,n = b_tilde_l exp(-j 2π(n-1)Δf τ_l(p)) s_n + noise；
r_l,n = conj(s_n) r_tilde_l,n；
y_l,n = sign(Re r_l,n) + j sign(Im r_l,n)。

s_n是已知QPSK参考导频，且全文明确源与站在时间、频率上同步。**先做DFT和导频消除，后做one-bit量化**，不能把它等同于“对未知混合原始IQ直接由1-bit前端采集”。本轮不推断其具体模拟/数字硬件实现。

§III式(4)是Gaussian噪声下的probit likelihood；式(5)将b_tilde和未知噪声标准差合为归一化b。之后构造MM surrogate，式(18)–(19)闭式更新增益参数和位置。它没有提供未知的多个独立s_q[n]与各源位置的联合辨识结果。

§IV是单源、3个标量站、16或128导频的数值例子；作者明确承认128导频可能不切实际，并指出低SNR局部极值以及较粗surrogate的代价。没有独立读取补充文件Appendices A–D，因此不把定理细节视为已完全审定。

### 可迁移与不可宣称

可作为已知波形场景的量化ML/MM近邻；把阈值h引入Phi(y(mu-h)/sigma)是常规模型适配，不构成新颖性。将已知s_n替换为多源未知波形会产生高维、双线性 nuisance；本论文没有直接解决这一问题，但“多加未知参数然后联合优化”仍不是足够的新机制。

## 3. OBSBL-DPD：未知多源的表述有必要边界

Wang, Q., Liu, Q., Xie, J., & He, Y. (2025). Enhanced direct position determination of multiple sources based on 1-bit sparse Bayesian learning. *IEEE Sensors Journal, 25*(20), 38371–38382. [DOI:10.1109/JSEN.2025.3605149](https://doi.org/10.1109/JSEN.2025.3605149)。Zotero：TZKIAJWN，附件8FEBBSJP。出版存在性及摘要通过[IEEE记录](https://ieeexplore.ieee.org/document/11154896/)交叉确认。

### 明确假设

- §II.A式(1)–(5)使用Q个静止远场源和L个M阵元站；窄带、慢衰落，并直接令s_1(t)=…=s_L(t)。未建立未知宽带延迟波形s_q(t-tau_lq)的站间模型。
- §II.B式(10)–(12)使用给定几何字典，源恰好在网格上。
- §III.A式(15)–(19)使用共同Gaussian噪声精度与Gaussian-gamma稀疏层次先验；§III.B式(47)–(49)使用独立快拍、共同空间支撑。
- 量化式(9)、(13)、(42)全为固定零阈值csgn；没有可变阈值或已知dither方法。
- §IV例子5站、每站8阵元、512快拍。其“broadband reconnaissance”动机措辞不能覆盖§II写出的窄带模型。

### 两个必须保留的核对疑点

1. **源数接口**：§II说Q未知，但Algorithm 1第一行把Q列为输入。原版PDF已视觉确认。更新式未显式使用Q，因此可能是伪代码冗余/误写；本轮不据此武断宣布实现必需真Q，也不接受“源数完全无先验已验证”的强表述。未读代码，未验证其检峰/计数规则。
2. **量化似然与Gaussian posterior之间缺少说明**：式(22)给出符号数据的probit关系，式(29)–(34)却直接使用Gaussian posterior及Gaussian evidence，式(30)为mu=alpha0 Sigma A^H y；多快拍式(52)–(53)条件变量改用量化前R。原版印刷页38375–38376已核对，不是OCR替换。

第二项为reader-inferred数学疑点：在最简单实标量模型s~N(0,lambda)、y=sign(s+e)中，精确p(s|y=+1)正比于Phi(s/sigma)exp(-s²/(2lambda))，一般不是Gaussian。只有明确近似、latent unquantized-data增广或相应截断矩更新才可能建立所需桥梁。已读§III没有给出该桥梁；因此不把其Gaussian更新认作已证的精确one-bit EM。但这不证明其算法经验上无效，也不是不当行为指控。

### 四单天线问题

若a_l(p)采用普通站内相对阵列响应，M=1时为常数，位置敏感性消失；要保留定位信息，需要显式站间时延、已校准绝对相位或幅度模型。本文没有把这些条件写成“四单天线、未知宽带混合”的特例。本轮不擅自把未展开的generalized steering记号判为严格不可辨识；结论是**尚无可直接搬用的模型与验证证据**。

## 4. 非零阈值协方差恢复：最强的常规适配近邻

Xiao, Y.-H., Huang, L., Ramírez, D., Qian, C., & So, H. C. (2023). Covariance matrix recovery from one-bit data with non-zero quantization thresholds: Algorithm and performance analysis. *IEEE Transactions on Signal Processing, 71*, 4060–4076. [DOI:10.1109/TSP.2023.3325664](https://doi.org/10.1109/TSP.2023.3325664)。Zotero：6MNGLP5J，附件CCZ2K8DZ。[作者机构记录](https://scholars.cityu.edu.hk/en/publications/covariance-matrix-recovery-from-one-bit-data-with-non-zero-quanti/)及[作者组出版页](https://gts.tsc.uc3m.es/covariance-matrix-recovery-from-one-bit-data-with-non-zero-quantization-thresholds-algorithm-and-performance-analysis/)核对元数据；以本地出版版为式号依据。

### 实际覆盖

§II.A式(1)–(2)设零均值Gaussian向量y，取N个iid样本，x(t)=sign(y(t)-v(t))。v(t)是可获取的阈值向量；已知加性dither d(t)可对应v(t)=-d(t)，这是观测模型重写，不是新算法。

§IV.A式(43)直接使用每个样本的精确阈值来估计方差；§IV.B式(51)–(61)使用双变量正交域概率估计协方差；§IV.C式(62)–(65)可选联合ML。§IV.D式(66)–(67)把复数问题转为实虚堆叠后还原复协方差。

§IV.E式(68)–(70)给出阈值相关FIM；Remark 2已讨论利用当前协方差估计来调整后续阈值、降低估计MSE。§V.A验证“精确阈值值”比仅利用阈值统计更有信息，§V.D验证方差不均衡，§V.E研究阈值窗口与数量，§V.H已经接后端DOA估计。

所以以下说法均不成立：

- “首次用已知dither恢复one-bit幅度。”
- “首次同时恢复对角和非对角协方差。”
- “首次利用FIM选择非零阈值。”
- “首次恢复协方差后接定位算法。”

### 同频强弱源如何映射

若各源为独立零均值Gaussian，经过线性传播和Gaussian接收噪声后，接收向量仍为联合Gaussian。源的数目、具体波形样本及强弱关系可以隐含在总协方差中，前端协方差恢复不必知道Q；四个单天线复通道对应8个实变量，不因没有站内阵列就自动脱离该方法。

但要面向宽带延迟定位，还要构造跨站、跨时滞的统计量。固定时滞成对量化数据仍能使用相应阈值的双变量概率；**样本序列是否iid、是否有足够独立记录、阈值与数据的对应、统计平稳性**都要另行检查。现成iid FIM不能直接当成相关时间样本的真实FIM。

Gaussian已知阈值可以恢复接收电压域协方差尺度，但不会自动拆开源功率与未知信道增益的乘积，也不解决站间时间/频率偏差。这是物理参数规范自由度，不能靠增加dither消除。

### 作者已经注明的边界

§I末段采用N趋于无穷且维数/N趋于0；§IV.C Remark 1承认逐对恢复可能产生非半正定矩阵，尤其高维时需留意。把有限样本PSD投影或常规结构化covariance fitting补上，不可因此自动称为创新。本文Gaussian假设之外的任意波形、短突发及相关样本并未由这些论证保证；但这些未覆盖条件也不是本轮已确认的新解法。

## 5. 本轮的唯一候选判断

**合格候选机制：无。**

这不是宣布量化方向没有研究价值，而是当前能具体写出的路线仍属于“现有已知阈值恢复＋既有DPD/量化似然”的组合或常规适配。观察约束不同不等于原理不同。即使后续考虑弱源信息而非总协方差MSE，也须先排除现成 nuisance-aware FIM/CRB设计、主动one-bit感知及量化实验设计；本轮没有完成该核验，更没有一个已证明的不同机制，因此不包装为候选。

不建议仅为保住这一方向启动常规dither对零阈值的性能演示；那类实验最多验证已知机制在本项目的迁移有效性，不能填补当前创新机制缺口。当前有用的交付是强基线及已排除主张的清单。

## 6. 实际读取、质量与证据限制

|文献|读取范围|原版视觉核对|未执行/未取得|
|---|---|---|---|
|62T6T2NB|出版letter §I–V、参考文献|印刷页3100，模型、式(1)–(7)|补充文件Appendices A–D未读；代码未读；实验未运行|
|TZKIAJWN|正文§I–V、参考文献、作者简介|印刷页38373–38376，模型、式(1)–(41)、Algorithm 1|无独立实现验证；源数接口疑点未解决|
|6MNGLP5J|正文§I–VI，Appendices A–E至式(134)|印刷页4062、4065–4066；式(1)–(5)、(43)–(67)、Remark 1|完整参考文献列表未审计；代码/实测未核对|

三篇PDF结构预检均PASS；原PDF SHA256见round3_source_manifest.json。Poppler发出Symbol/ArialUnicode字体警告，但上述图像中的关键公式和文字已可视核对。未进行全页视觉排版审查，也未作完整撤稿、COI、出版者审计。

方法弱点固定清单（限定本轮scope，不作总分）：

|类型维度|62T6T2NB|TZKIAJWN|6MNGLP5J|
|---|---|---|---|
|概念定义精度|checked: none found|checked: found：Q接口矛盾|checked: none found|
|论证链完整性|not checked：补充证明未读|checked: found：probit到Gaussian posterior桥梁|checked: found：iid渐近论证不可直接转相关短记录，reader-inferred迁移边界|
|反论证处理|not checked|not checked|not checked|
|理论新颖性|not checked：不评价作者整体新颖性|not checked|not checked|
|可检验性|checked: none found|checked: found：量化前R的可用性需解释|checked: none found|
|假设清晰度|checked: none found|checked: found：宣称宽带与窄带模型需区分|checked: none found|
|变量定义|checked: none found|checked: found：Q、r/y条件变量|checked: none found|
|内在效度|not checked：未运行|not checked：未运行|not checked：未运行|
|外部效度|checked: found：已知单源导频不能无偿替换未知混合|checked: found：5×8阵列不覆盖4×1|checked: found：iid Gaussian与大样本条件|
|统计报告完整性|not checked|not checked|not checked|
|结论克制|checked: none found：作者自述局部极值/导频边界|not checked|checked: none found：作者自述PSD和高维边界|

片段和逐项式号见round3_evidence_locators.json；实际查询先写入round3_query_plan.json与round3_query_supplement_metadata.json，返回全文保存在round3_web_*.json。网页批量检索没有逐查询总命中量或分页，不冒称数据库穷尽检索。公开来源仅作元数据交叉核对；方法结论来自已读出版原文。

本报告由一个AI子代理完成，不是人类双评，也未完成方向放行所需的独立反方复核。全部新文件限于本BC目录；未触碰Zotero、论文、冻结代码和原实验。

