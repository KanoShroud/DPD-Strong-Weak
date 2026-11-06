# Weiss one-bit DPD：原文核对与单天线适用边界

日期：2026-10-22。状态：Phase 2 有边界证据提取；不是系统综述完成、低创新风险判定或复现实验。本报告由单个 AI 子代理完成，不构成两名人类筛选者或独立复核。

## 1. 结论

1. **HROBDPD 不要求已知源数或已知信道增益，但其推导用的是单源模型，多源效果由热图和仿真展示。** 原文 §II 式(1)令位置、复信道系数和波形未知；§II 末段明确算法无需源数先验，同时承认模型为单源。不能将“无需输入源数”改述为已解决多源计数、检出率或虚警控制。
2. **量化器是固定零阈值的 I/Q 符号量化，每个复样本两位。** HROBDPD 式(4)–(5)有四个复输出值，所谓 one-bit 是每个实分量一位，不是复样本总共一位。原文算法未实现可变阈值或 dither（已查 §II–V）。
3. **Gaussian 是 arcsine-law 统计恢复和一致性论证的条件，不是装饰。** HROBDPD §II 使用零均值 circular complex Gaussian、白源样本以及独立时空高斯噪声；§III.B 的 Corollary 1 对零均值联合 CCN 向量成立。§IV 的均匀非高斯信号热图属于有限仿真展示，不把该理论扩成任意未知宽带波形保证。
4. **one-bit 恢复的是归一化协方差，不恢复站间绝对能量尺度；但不能说强弱源功率比的所有信息都消失。** 2021 OBDPD §4 式(8)–(10)、§4.1 明确了这个归一化与“平行场景”的统计等价。
5. **最重要的项目适用边界：每站单天线时，HROBDPD 在其公式的可逆条件下退化为 OBDPD 的同一位置谱。** 这是本轮依据两篇论文公式作出的代数推论，不是作者原句；推导见第3节。2024 原文实验是4站、每站4阵元，不是4个单天线站。

## 2. 主文献与模型证据

### C1：Weiss (2024), HROBDPD

Weiss, A. (2024). High-resolution one-bit direct position determination of narrowband Gaussian emitters. In *2024 58th Asilomar Conference on Signals, Systems, and Computers* (pp. 1745–1749). DOI: [10.1109/IEEECONF60004.2024.10942891](https://doi.org/10.1109/IEEECONF60004.2024.10942891)。

Zotero item：VAVRZW5P；PDF attachment：6RID5AXS。原文全文已提取并阅读 §I–V 与参考文献；公式页另用原版 PDF 图像核对。题名、作者、会议、页码由[作者机构页面](https://cris.biu.ac.il/en/publications/high-resolution-one-bit-direct-position-determination-of-narrowba/)交叉确认；IEEE 页面直接打开遇到 JavaScript/robot 验证，已保留失败。

|项目|原文事实与定位|
|---|---|
|观测|L 个已校准站内阵列，每站 M 阵元；远场、窄带；r_l[n] = b_l a_l(p)s_l[n]+v_l[n]，§II 式(1)–(3)|
|已知|站/阵列几何所定义的响应 a_l(p)、采样周期、候选位置区域；§II|
|未知|位置 p、复信道 b_l、源波形、站噪声方差；§II；§III.A 以 ||b||²=1 固定波形—增益尺度规范，并非已知物理增益|
|源数|位置扫描不需要 K，但建立式(1)和理论时用单源；§II 末段|
|量化|Q(z)=[sign(Re z)+j sign(Im z)]/sqrt(2)，零阈值、四象限，§II 式(4)–(5)|
|统计结构|CCN 源与噪声、联合 Gaussian；arcsine 反演得到尺度归一化相关矩阵；§III.B Corollary 1、式(17)|
|定位计算|逐候选构造归一化协方差矩阵，求逆并按站阵列响应收缩，再取相应 L×L 矩阵最大特征值；§III.C 式(18)–(21)|
|理论边界|式(21)后用连续映射论证，并明确假设逆映射连续；不可把矩阵奇异或有限样本非正定问题自动当作已解决|
|实验|§IV：L=4、M=4、N=1024；双源；Gaussian 与均匀非 Gaussian 热图；另有250次试验的定位误差曲线。阅读范围内没有固定弱源而单独扫强源功率的动态范围试验|
|阈值/dither|§II–V 未给出变阈值/dither算法；§V 将新的量化策略列为后续工作，引用[21] extremum encoding|

实质判断：在已读正文中，“arcsine 相关恢复＋MVDR 位置谱”已由此文覆盖；“one-bit 多源定位”“不知道 K 的位置扫描”本身不能作为新颖性主张。本轮不判断任意新方法是否与其等价。

### C2：Weiss & Wornell (2021), OBDPD

Weiss, A., & Wornell, G. W. (2021). One-bit direct position determination of narrowband Gaussian signals. In *2021 IEEE Statistical Signal Processing Workshop (SSP)* (pp. 466–470). DOI: [10.1109/SSP49050.2021.9513768](https://doi.org/10.1109/SSP49050.2021.9513768)。

Zotero：T3ASKFN7；附件：TBV5KAZU。已读全文 §1–8；原版公式图像核对式(8)–(13)。[arXiv:2010.15640v2](https://arxiv.org/abs/2010.15640v2) 与[作者实验室手稿](https://sia.mit.edu/wp-content/uploads/2021/08/2021-weiss-wornell-ssp.pdf)确认版本关系；预印本与会议版不计为两项独立研究。

- §2 是单个未知源、未知复增益及未知噪声方差；源为平稳 circular Gaussian，**不要求时域 iid**。静止源/阵列排除 Doppler；与2024文的白源表述需分开。
- §4 式(9)：归一化相关矩阵为 R_bar^(i,j)[m]=R^(i,j)[m]/(rho_i rho_j)，rho_i²=|b_i|² R_s(0)/M+sigma_vi²。
- §4.1：若各站 rho_i 相同，则 OBDPD 与相应未量化 DPD 的渐近位置准则等价；一般情况下对应的是信道 b_l/rho_l、噪声方差 sigma_vl²/rho_l² 的“平行场景”，不是原始未量化数据完全等价。
- §4.1 的可辨识性结论继承该归一化平行场景，不是对任意多源、任意几何、任意波形的无条件可辨识保证。
- §4.1 明确站内 SNR 的比例关系保留、站间相对幅度信息丢失。§4 式(3)、§4 的算法仍是零阈值；没有实现 dither。参考文献[22]已经列出2019年的 time-varying sampling thresholds radar 工作，因此不能以 Weiss 本文未实现作为该技术不存在的证据。

## 3. 四单天线条件下的代数核对

以下属于 reader-inferred、未经数值运行的代数结论；前提必须保留：M=1，单个全向阵元响应取 a_l(p)=1，使用两文同一归一化协方差估计，矩阵可逆。记 H(p) 为2024文式(19)的 L×L 归一化协方差。

2024文式(20)把 H(p)^{-1} 分成 M×M 块。M=1 时每块就是标量。故由式(18)：

D_HR(p)^{-1}_{ij} = a_i(p)^* [H(p)^{-1}]_{ij} a_j(p)
                   = [H(p)^{-1}]_{ij}，

从而 D_HR(p)=H(p)，式(21)位置得分是 lambda_max(H(p))。

2021文式(11)–(12)给出：

D_OB(p)_{ij}=N a_i(p)^* H(p)_{ij} a_j(p)=N H(p)_{ij}，

因此 lambda_max(D_OB(p))=N lambda_max(D_HR(p))，峰位置相同。如果单阵元响应只有已知单位模相位，则是单位酉对角相似变换，特征值仍相同。

这个化简排除了“直接把 HROBDPD 搬到单天线就获得额外 MVDR 分辨率”的解释。它**不证明**所有 one-bit 方法都无法改善单天线多源定位，也不涉及另加结构、已知增益、约束权重、多记录或非零阈值后的不同模型。矩阵不可逆或采用不同 regularization 时不能无说明套用本等价式；regularization 本身不自动构成新颖贡献。

## 4. 强弱动态范围：可以与不可以说什么

原文支持：零阈值符号量化具有正尺度不变性，Q(c r)=Q(r)，c>0；arcsine 恢复的是归一化协方差。因此单凭这些符号观测，不能恢复每个站整个接收向量的绝对幅度尺度。依据：C1 §III.B，C2 §4–4.1。

本轮推论：两个独立 Gaussian 源的和仍为 Gaussian，**强弱混合本身不破坏 arcsine law 的 Gaussian 前提**。在其余条件成立时，源功率权重仍可通过归一化互相关结构影响观测分布。因此不能把“绝对能量尺度丢失”夸大为“强弱比完全不可辨识”或“弱源信息必然全部丢失”。极大功率比下弱源的归一化相关贡献可能很小，但其有限样本信息量、检测门槛和可辨识性需要针对确切模型推导；两篇文的上述实验不能替代该证明。

本轮未取得的结论：四个单天线、多个未知宽带源、未知站增益下，强弱比可恢复哪些组合参数；有限比特/clip阈值下的弱源有效信息；所需阈值数量及可获取先验。这些是未决问题，不是已确认创新空白。

## 5. 可变阈值与 dither 最近邻

### C3：Ni et al. (2024)——只作边界核对，不混同未知波形模型

Ni, L., Nyantakyi, I. O., Liu, N., & Wan, Q. (2024). Enhancing direct position determination in distributed base station systems through time-varying quantization design. *IEEE Sensors Journal, 24*(5), 6953–6963. [DOI:10.1109/JSEN.2024.3353804](https://doi.org/10.1109/JSEN.2024.3353804)。Zotero：MUQ9RWX6；附件：W5A4N9ZP。

已读 §II、§III.B、§IV、§V及§VI相关实验段落，未完成整篇独立质量审计。

- §II 式(1)–(12)为已知发射机—目标—接收节点的双基地距离模型；§III.B 明确 transmitted signal known。未知参数为反射复幅度与目标位置，不能直接当作本项目“全部发射波形未知”同信息基线。
- §IV.A 式(49)–(55)通过最小化 CRLB 设计阈值，最优阈值等于未知无噪样本的实/虚均值，并提供迭代估计方案。
- §IV.B 式(56)–(57)给出随机阈值方案，正文明确类比 dithering / stochastic resonance；简单“加随机阈值”已存在先例。
- §VI.A 明确仿真 OQ 使用真实最优阈值作为比较对象；这是 **oracle** 参考，不可在新实验中冒充不使用真值的可实现算法。
- §VI 含5个单阵元节点、已知 LFM 波形的仿真。它说明单天线阈值优化并非空白，但不证明本项目未知波形多源情形已被解决。

### 作者后续工作与未读近邻

HROBDPD参考文献[21]指向2024 ICASSP extremum encoding。作者官网同时列有2025 TSP期刊版、2025 ICASSP复基带扩展，以及2026通信受限估计/检测工作。本轮只核对公开元数据，不评其算法弱点。

[作者官网](https://weissamir.com/publication/)在2024 ICASSP条目下明确发布 Theorem 1 证明的勘误（定理保留、证明修正）；引用理论结果时必须读修正版或后续期刊版。2021 OBDPD的 Matlab code 链接也在该页，本轮未下载或运行。

初轮Zotero还返回：62T6T2NB（Gridless maximum likelihood one-bit DPD，2024）、TZKIAJWN（1-bit SBL多源DPD，2025）、P2YHUS7U（one-bit multiple targets，2023）、6MNGLP5J（非零阈值协方差恢复，2023）。这些没有在收窄后的本轮读完全文，均保留为关键未决近邻；不能宣称量化定位候选已低风险。

## 6. 证据质量与核对范围

本工程问题按已确认协议使用逐项质量状态，不机械套用临床层级或合成总分。

|条目|状态|
|---|---|
|模型、已知未知量、目标式|C1/C2已核对；C3仅上述章节|
|公式与文字提取一致性|C1式(17)–(21)、C2式(8)–(13)经原PDF图像核对|
|理论/概念清单：定义精度|checked: none found（当前核对范围未发现定义性问题）|
|理论/概念清单：论证链|checked: found（单源推导与多源实证不能升级成任意多源保证；reader-inferred，C1 §II、§III.C、§IV）|
|理论/概念清单：反论证处理、理论新颖性|not checked（未审完全领域）；不评价作者整体贡献|
|理论/概念清单：可检验性|checked: none found（目标式与情形明确；未运行验证）|
|经验研究清单：假设清晰度、变量定义|checked: none found（就本次读取的问题）|
|经验研究清单：内在效度、统计报告完整性|not checked（本轮未复现或审计全部实验）|
|经验研究清单：外部效度|checked: found（M=4阵列和特定信号实验不自动覆盖本项目M=1；reader-inferred，C1 §IV）|
|经验研究清单：结论克制|checked: none found（本报告保留单源理论与多源仿真的边界，不指控作者过度声称）|
|代码与数值一致性|未独立运行；单天线结果为代数核对|
|来源核实|C1/C2原PDF存在并记录SHA256；出版元数据交叉核对；C3 DOI与本地全文核对，IEEE实时页面受访问限制|
|利益冲突、撤稿与出版者审计|未作独立完整审计，不给“无冲突/无撤稿风险”证明|
|PDF预检|C2 PASS；C1 UNAVAILABLE：xref coverage warning，页数三路均5。该警告不升级成PASS；正文采用章节/公式定位，另保留公式图像|

## 7. 文件与执行说明

- 查询前保存：query_plan.json；收窄后两份 query_plan_supplement_*.json。
- 实际返回：zotero_*_offset0.json、web_batch*.json、web_weiss_*.json、web_primary_open.json；均保留返回原貌与日期。
- 首轮Zotero：8条查询共92次返回、74个唯一item key、18次重复返回，全部hasMore=false；这些是检索记录，不是最终纳入文献数。
- 正文：VAVRZW5P_extracted.txt、T3ASKFN7_extracted.txt、MUQ9RWX6_extracted.txt；metadata/fulltext JSON为工具原始回执。
- source_manifest.json 记录原PDF身份与限定公式范围的视觉核对；visual_verification_receipt.json 记录核对范围和工程降级；*_preflight_stdout.json 记录预检与Windows输出限制；T3ASKFN7_preflight_utf8.json 为中文路径编码修正后的成功回执；*_formula-*.png为只读渲染证据。
- 初轮B/C范围在收到收窄消息前已经实际执行。AKY65WBV、XM4DFU6E等仅作为先前检索/提取保留，不在此报告中完成B族评审。
- 未修改Zotero、论文、求解器或其他项目文件；未实验、未安装、未提交Git。所有新文件只在本子目录内。

