# D/F 分层：信号结构与检测边界的来源核验（初轮）

日期：2026-10-22。实际项目根：F:/DPD_Strong_Weak。仅 Phase 2 证据整理，不构成最终创新方案或实验放行。按已确认协议执行，未运行定位／训练实验，未改论文、求解器或 Zotero。ARS bibliography/source-verification 的作用是固定查询、保留原始结果及区分正文证据与推断；本次为同一 AI 代理的检索与提取，不是双人筛选。

## 1. 当前可确认的覆盖

- 非圆数据增广、严格非圆未知波形 ML-DPD、非圆多站 TDOA 的原子回访均已有明确近邻。仅把 BPSK 共轭增广加入现有定位器，不能据此建立新颖性。
- 循环平稳 DPD 已覆盖循环频率选择、虚拟阵列、离网 SBL、网格细化和有限样本交叉循环项校正。本项目的同符号率／同成形强弱独立源，不应默认能靠一个循环频率把强源消掉。
- 未知总源数的 DDPD、AP 加序列检验，以及 NOMP 加自适应 CFAR 已有先例。必须区分：全局源数未知、每批活跃源数已知、噪声方差已知，以及完全盲的检出任务。
- 当前仍有 2009 CDPD、2017 严格非圆 ML-DPD、2019 改进 DDPD 的关键全文未完整获取，不能宣布任一方向“创新风险低”。

以上是文献覆盖边界，不是“所有方法均适用于当前四单天线宽带模型”的结论。

## 2. 检索账本

先保存 query_plan.json，随后执行 8 个 Zotero 查询与 10 个网页查询；补充查询另存 query_supplement_01/02.json。Zotero 首轮共 73 次记录返回，按 itemKey 去重为 67 条，重复出现 6 次；各查询 limit=50，工具均报告 hasMore=false。补充三查询返回 1、0、0 条，唯一命中为首轮已有 DUZWQM3B。这些只是本地库返回量。

网页执行 IEEE、arXiv、ScienceDirect 定域及不限域检索，并保留完整工具返回文本。工具不提供可遍历数据库分页或可靠总量；多查询合并返回无法精确归属每条结果到某个子查询，不据此制作伪精确 PRISMA 数据库计数。中文“直接定位 未知源数”返回大量无关结果已原样保留，不能将其当领域不存在证据。第二轮从正文参考文献追溯 2009 CDPD、2017 非圆 ML、2019 DDPD；CFAR 的近期词检索只属前向线索发现，不是完整引用索引追踪。

日志保留说明：key_metadata_raw.json 与 web_broad_initial_raw.json 的出版者临时签名下载URL查询串已机械脱敏为 SIGNED_QUERY_REDACTED；题名、DOI、主URL、结果正文未变。21个JSON均以PowerShell ConvertFrom-Json解析通过。未对全文数学公式做自动重排。

zotero_candidate_register.json 保留全部 67 个唯一项与查询来源。not_prioritized_in_this_bounded_pass 不是最终排除决定；主流程仍需统一题摘判定。父代理交叉线索 ATV46JBA 单列来源，不计成本代理独立发现。默认未调用 S2/OpenAlex/Crossref API。

## 3. 关键来源与正文条件

### DF01：循环平稳离网 SBL（核心定位近邻）

Wang, Y., Zhang, Y., Han, C., Wang, L., Tellambura, C., Mumtaz, S., Di Renzo, M., & Debbah, M. (2026). Off-grid sparse Bayesian learning for direct localization of cyclostationary sources with adaptive grid refinement. IEEE Transactions on Wireless Communications. DOI: [10.1109/TWC.2026.3738889](https://doi.org/10.1109/TWC.2026.3738889)。Zotero DUZWQM3B / PDF 6Q3VDKDF；当前附件标明 accepted author version，非最终排版。

读到的正文：§III-A 式(7)–(19)，§V-B，§VI-A及§VI-B开头，相关参考文献。完整提取已保存，未把整篇下载等同整篇逐式复核。

- 原文模型：L 个站，每站 M 阵元；未知源信号，复衰落与延迟信号吸收为站内有效波形。利用站内循环相关、差共阵冗余平均、SVD 压缩、SBL 和离网／自适应网格更新。式(9)以循环不相关得到理想对角源循环相关矩阵，有限样本非对角项另作两阶段校正。
- 先验：循环频率是处理输入；仿真期望信号 1 MHz、干扰 0.8 MHz，四站各六阵元 MRA；BPSK/QPSK 使用 raised-cosine，roll-off 0.5，1200 原始快拍。未给已知符号序列，不等于不使用调制统计特征。
- 适用条件：§V-B 的校正版本 II 要物理阵列流形满列秩，因此作者将其定位能力限制在 M−1；版本 I 使用扩展差共阵。式(19)的 SVD 截断还被限制到 [1,min(N,3)]，这里 N 为源数；不能仅凭 SBL 标签宣称本实现完全不依赖源数。
- 质量边界：§VI 式(73)之后明说 RMSE 包含成功 Monte Carlo 次数；本项目不能直接拿它与全样本未截断 RMSE 作数值比较。
- 读者推断、未实验：其站内差共阵在 M=1 时不再提供阵列角度孔径，不能原样视作四单天线跨站 TDOA 算法；从跨站循环相位恢复几何是否可辨识需要另建映射。相同循环频率的强弱源不能按“不同循环频率干扰被滤除”的证据处理。
- 工程清单：模型／先验／主更新／秩条件已核对；原始仿真未复现，完整运行成本、失败样本分布和参数敏感性未独立核验。

### DF02：未知总源数 DDPD（核心定位近邻）

Oispuu, M., & Nickel, U. (2010). Direct detection and position determination of multiple sources with intermittent emission. Signal Processing, 90(12), 3056–3064. [DOI](https://doi.org/10.1016/j.sigpro.2010.05.010)。Zotero IMURV42Q / PDF CUL7569J。

已核对 §2.1–2.3、§3.3、结论与参考文献。

- 原文模型：移动的已校准 M 阵元阵列，多观测时隙，远场窄带，未知确定性波形（允许相干）；每批活跃源数 Q_n<M，流形列独立，空间时间白的圆复高斯噪声。
- §2.3 A2/A4、§3.3：噪声方差与每批有效源数 Q_n 必须已知。总源数 Q 在 DDPD 中未知，不代表所有计数均未知。
- 算法：假设源数逐步增加，每次用 ML/AP 重估位置；以拟合残差与相应卡方量比较，符合则停止。§3.3 的虚警表述带检验独立条件；本文未独立重推其阈值／拟合后自由度。
- 作者承认：§3.3 最后指出，已知 Q_n 导致低 SNR 仿真偏乐观。
- 读者推断：可作为“定位与源数检验耦合”的强历史基线；当前单记录四单天线宽带模型没有可直接照搬的批内阵列源数先验。除去已知 Q_n 并非未有人做：DF07 的 2019 后续已明确声称处理这一点，必须取得其正文。
- 工程清单：模型、假设、AP和序列检验规则已核对；准确虚警校准、多重检验依赖与宽带映射未核验；未复现性能。

### DF03：NOMP-CFAR（迁移近邻）

Xu, M., Zhu, J., Fang, J., Zhang, N., & Xu, Z. (2023). CFAR based NOMP for line spectral estimation and detection. IEEE Transactions on Aerospace and Electronic Systems, 59(5), 6971–6990. [DOI](https://doi.org/10.1109/TAES.2023.3282610)。Zotero XWI8HEAU / PDF 8TLKPLQ4。2022 [arXiv 2210.10302](https://arxiv.org/abs/2210.10302) 与期刊版关联，不计两项独立研究。

已核对 §II、§III-A 式(9)–(28)、§IV开头与参考文献。

- 原文模型：多维复正弦叠加，未知连续频率、幅度、分量数和白高斯噪声方差；扩展 MMV/压缩观测。观测不是任意未知宽带通信波形。
- 方法覆盖：NOMP 初始化、逐候选 CFAR、删除最不可信分量、Newton 重精修；从参考单元估计噪声并排除已拟合频点。添加 CFAR 或删添分量并不新。
- 阈值条件：DFT 在纯白噪声下的独立性是推导基础；因参考单元由选峰位置决定，§III-A 式(17)明确采用近似独立高斯，作者另做模拟验证。检测概率分析忽略分量间干扰。
- §IV 默认 16 分量，任意两频率相隔大于 2.5 DFT；这不是所有强弱近邻分量条件下的保证。
- 迁移映射：复正弦原子对应位置参数化的几何子空间，频率检验对应候选位置的新增源检验，噪声功率估计对应剖面残差尺度。但 DPD 中未知宽带波形使每个位置携带多维系数；扫描点相关、拟合后残差依赖和连续位置优化改变极值分布。原阈值不可未经证明直接引用为 DPD 的全域 CFAR。
- 工程清单：噪声／阈值／参考窗规则已核对；DPD迁移仅为结构映射而非有效性证明，未实验验证。

### DF04：NC-LMP 非圆多站 TDOA（核心定位近邻，部分正文）

Wang, Z., Wang, D., Yin, J., & Zheng, N. (2026). Loop matching pursuit multistation TDOA estimation method based on noncircular signal. IET Radar, Sonar & Navigation, 20(1), e70114. [出版者正文](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/rsn2.70114)。Zotero 6RY9NR6X 未附 PDF；网页确认 2026-01-19。

网页成功取得 §1–§4.1 式(1)–(16) 和检索返回的算法表及式(31)–(36)。后续按行请求两轮均超时，因此标记 partial_publisher_fulltext，不称完整阅读全文。

- 已核对：一主站与 M 辅站的单一未知确定性源；以主站含噪接收波形的时移构字典；最大非圆率 s=s0 exp(jφ)，共轭增广，支持集循环删添，残差阈值停止。
- 没有已知发射符号；但不是多个未知独立源混合下逐源字典的现成模型。
- 式(16)把共轭支路的增益合并为同一 λ，和任意未知复链路增益模型的兼容性应回读完整推导；本轮不自行补出作者未明确处理的相位条件。
- 不能从现有局部正文断言它解决源数检测、全域虚警或复杂链路相位。完整仿真与 CRB 未核对。

## 4. 其余最近邻与未获全文清单

|ID|文献／状态|本轮可用依据与剩余风险|
|---|---|---|
|DF05|Zhou, Yi & Kong (2019), coherent DPD-IAA；10.1016/j.sigpro.2019.03.023；Zotero PIX9ZHCJ|PDF已取得；读§2–3及§4开头。s_k=β_k s_0 是完全相干，不是独立BPSK；频域堆叠、WLS/IAA协方差与功率谱、连通域提峰已有。完整更新与性能未复核。|
|DF06|Reuven & Weiss (2009), Direct position determination of cyclostationary signals；10.1016/j.sigpro.2009.04.009|出版者摘要、引言和部分节预览；report_not_retrieved（完整原文）。明确循环子空间定位与不同循环频率干扰抑制已存在，不能依其未读细节作新颖性差异。|
|DF07|Hao & Wan (2019), High resolution direct detection and position determination of sources with intermittent emission；10.1109/ACCESS.2019.2907136|来源：DF01参考文献[31]、网页摘要元数据；report_not_retrieved。摘要声称无需每批有效源数，以K-means识别活跃时隙并重建MVDR谱。决定F方向的关键未决近邻。|
|DF08|Yin, Wang, Wu & Yao (2017), ML-based single-step estimation of the locations of strictly noncircular sources；10.1016/j.dsp.2017.06.024|出版者摘要／引言；report_not_retrieved。明确未知波形、未知复衰落、时域严格非圆ML和实值化迭代。不能把未知相位／增益剖面ML轻率认作缺口。|
|DF09|Li, Dong, Shi & Zhang (2024), enhanced non-circular DPD via sparse Bayesian inference/grid refinement；10.1016/j.dsp.2024.104564|仅出版者摘要／亮点；report_not_retrieved。DF01称其需非圆相位先验，该限制尚未从DF09原文独立证实，不能作为确定事实。|
|DF10|Napolitano (2026), Wide-band cyclic cross-correlation processing；10.1109/TSP.2026.3672729；Zotero ATV46JBA|父代理A族线索；全文已取得，仅查看§IX开头和仿真信号段。宽带尺度/延迟/频差及可估计循环频率已有；不宣称完成D族全文核查。|
|DF11|Tabassum & Ollila (2018), Simultaneous Signal Subspace Rank and Model Selection...；arXiv:1806.07320|预印本摘要线索，未取全文。c-LARS路径+GIC选择模型及单快拍DOA，属于计数与估计联合迁移近邻。|
|DF12|Cyclic weighted centroid localization for spectrally overlapped sources in cognitive radio networks；arXiv:1903.07655|仅预印本题摘线索，未核完元数据，不进入已核验参考表；证明arXiv搜索存在相关分支，不证明其适用本模型。|

## 5. 交给主流程的未决问题（不是创新建议）

1. 同循环频率、同脉冲成形、独立未知BPSK：非圆／循环信息能给什么可辨识增量，以及这些增量是否已被2017 ML与2009 CDPD吸收？关键正文尚欠，不能定低风险。
2. 未知复链路增益／非圆相位是否与每站延迟或符号时钟存在规范自由度？必须严格区分“结构未知参数要联合估计”和“把真符号／真循环参数给候选”。
3. 当候选由同一数据扫描产生、强源先被拟合时，全域弱源检验的虚警是否可保持？已知 NOMP-CFAR 和 DDPD 是先例，当前数据模型的阈值条件尚未证实；不能仅用通用GLRT/MDL名称充当贡献。
4. 独立源的强弱问题与完全相干源的秩亏问题不同。将两者混合会引入新的主要因素，应按协议分开判断。
5. D/F与A交叉：DF01已覆盖有限样本循环交叉项校正。若后续形成“交叉项消除+网格细化”，应优先逐式比较它，而不是仅比较传统MUSIC。

## 6. 证据质量与未做事项

采用工程模型／先验／公平比较／完整指标清单，不套用临床证据等级。DF01、02、03为原始方法论文且有可定位正文；DF04为出版者部分正文；其余按上表限制。不存在因此获得的“整篇已验证”或“方法正确证书”。

理论类五项检查：概念定义和可检验规则在DF01–03 checked: found；论证链仅核对所列节，完整证明 not checked；反例处理和理论新颖性 not checked（非本阶段最终判断）。仿真类的变量定义 checked: found；内部有效性、外部有效性、完整统计、结论范围仅做上文点状核对，整体 not checked。出版社/DOI/正文题名相合；未做完整撤稿、COPE、作者信誉或利益冲突外部审计，COI均未核验，不以“未发现”写成“没有”。

未下载额外付费全文、未注册协议、未运行算法、未安装依赖、未提交Git。保存的是资料与出处；原始长文属现有Zotero或出版者工具提取，本报告不重新分发PDF。网页链接使用出版者或arXiv，第三方聚合页仅作检索线索。

覆盖分布提示：本次重点为理论／仿真信号处理方法，非人群证据；该方法类型集中是限定任务导致，不能外推真实系统可靠性。本轮没有足够字段评估地理偏斜。

