# G11路线终审：应用需求外部证据

> 2026-10-24补读提示：2008边缘误差原文还明确讨论反应时间与运动引起的短观测动机，并指出移窗需要缓存或重新观测；2012线性相关对照需要边界外样本。这补充了需求及数据权限的解释，不把当前20.48/81.92μs自动绑定到某项实际任务。详见[两篇全文补充](fulltext_followup/两篇边界误差论文精读与终审补充.md)。以下应用检索原记录保留。

核验日期：2026-10-24。独立应用证据分工；只读项目，不运行定位、不更改基线或Zotero。检索范围不是仅限已有13篇文献。以下区分原文事实、对本项目的推断与未取得的证据；没有找到精确应用对应不等于证明这种需求不存在。

## 结论

**有限记录、多源混叠和接收功率差是有实际依据的问题，不能以“不是主流/没有真实需求”否定整个领域。但当前G11的特定任务——四个静态单天线站、单段20.48或81.92 μs、两条持续的未知BPSK波形、已知相对传播响应——尚没有完成外部应用约束到数值配置的闭合。**

尤其不能将论文中的“short capture”“short snapshot”、单跳驻留时间、一次chirp周期都等同于“定位器只能得到几十微秒”。本次三条应用链都存在合理常规处理：跨时段积累、利用跳频结构、同步截获并截取共同脉冲、TDOA/DPD联合估计。新方法需要解释为何这些处理不足，而不是只证明短记录仿真中周期模型有误差。

当前设置的物理量（由项目配置计算，不是外部论文数值）：50 MHz采样下1024/4096点对应20.48/81.92 μs；12.5 MHz名义带宽的BT约256/1024；1000 m最大基线对应不超过约3.34 μs的站间时延。因此边界时移可能占记录约16.3%/4.1%，说明它是可诊断的机制，但不证明现实系统必须接受这个记录长度。

## 外部证据矩阵

| 应用链及来源 | 已直接读取的证据 | 常规处理与G11边界 | 对当前立项的支持程度 |
|---|---|---|---|
| GNSS干扰多星无源定位；Clements、Ellis、Humphreys，IEEE/ION PLANS 2023 | 原作者12页稿中模型、实测与定位章节：真实多干扰源、重叠频带和接收功率差；双同步LEO站原始IF采集；无需已知发射波形。20 ms、100 ms、1 s积分；原采集为60 s。 | 延迟/多普勒随时间变化，位置域相关及非相干积累；功率差会加剧CAF关联困难，但现有直接定位已能处理部分困难。不是静态四站，也不是单段几十微秒。 | **强支持应用真实性和多源动态范围；不支持把20.48/81.92 μs定为硬限制。** |
| 非合作跳频发射源；Ouyang等，IET Signal Processing 2017 | 已读出版社正文§1–6（公式以图片显示，未据此审计推导）：未知波形、静态多站、多源；10 MHz跳频范围、10000 hop/s、10跳、BPSK。 | 驻留约100 μs，但使用10跳，即约1 ms；频域稀疏性和最大特征值搜索。还明确要求观测时长足以让信号到达最近和最远接收站。 | **支持有限驻留和未知多源；不支持“只能处理一跳”或当前两条持续同频波形。** |
| UAV/遥控器被动RF定位；Dickerson等，IEEE ICC Workshops 2025 | 已读作者预印本§I–V及AERPAW官方数据页：四个固定RF站、3.32 GHz UAV波形，1.25/2.5/5 MHz，真实五次飞行；重点是高度、NLOS、天线和构型。 | 数据为TDOA/位置结果而非本轮可直接拿来跑G11的原始多源IQ；单源channel sounding，定位器内部细节属商用系统。未给出几十微秒约束。 | **支持分布式被动定位的现实意义；当前最明显问题是传播/几何，不是已证实的有限窗强弱源问题。** |
| UAV RC突发截获补充（非学术主依据）；Keysight工程作者，Microwave Journal 2017 | 四同步站隔离1.2 ms遥控突发；需要带时戳IQ缓存，从扫频记录中回取同一片段。文中实验记录还比所需长3倍。 | 时间/频率隔离是常规输入处理；短突发不等于单段20 μs。来源是工程文章，只作现实需求旁证。 | **支持截获约束；不提供G11精确时长或理论创新证据。** |

## 主来源与阅读边界

1. Z. Clements, P. Ellis, T. E. Humphreys, *Dual-Satellite Geolocation of Terrestrial GNSS Jammers from Low Earth Orbit*, IEEE/ION PLANS, 2023. DOI [10.1109/PLANS53410.2023.10140058](https://doi.org/10.1109/PLANS53410.2023.10140058)；[作者稿](https://radionavlab.ae.utexas.edu/wp-content/uploads/clements-direct-geolocation.pdf)。已读§II、III-B、IV、V及引用链。p6记录2-bit 6.2 Msps原始IF，p10给20 ms/100 ms/1 s及多时段积累；不把原始IF采样率和文中下变频基带率混为一谈。该文已有实测和现成解法，不能只引导言证明“多源问题尚未解决”。
2. X. Ouyang et al., *Direct TDOA geolocation of multiple frequency-hopping emitters in flat fading channels*, IET Signal Processing 11(1),80–85,2017. DOI [10.1049/iet-spr.2016.0299](https://doi.org/10.1049/iet-spr.2016.0299)；[出版社全文](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-spr.2016.0299)。未知复通道系数、非相关源和频域稀疏是其建模条件，与G11已知相对幅相不同。数值例见§5。不能把10 MHz跳频跨度解释成单跳BPSK带宽。
3. C. Dickerson et al., *Impact of Altitude, Bandwidth, and NLOS Bias on TDOA-Based 3D UAV Localization: Experimental Results and CRLB Analysis*, IEEE ICC Workshops,2025. DOI [10.1109/ICCWorkshops67674.2025.11162218](https://doi.org/10.1109/ICCWorkshops67674.2025.11162218)；[作者全文](https://arxiv.org/html/2502.01771v1)；[AERPAW官方数据说明](https://aerpaw.org/dataset/aerpaw-rf-sensor-measurements-with-uav-july-2024/)。读取模型与实验，未把其LOS界或商用定位过程视为已核验的G11基线。
4. [Drone Detection and Location Systems](https://www.microwavejournal.com/articles/28459-drone-detection-and-location-systems),2017。只作工程来源；不能替代IEEE/Elsevier方法出处。

## 延伸核查：已有常规处理不可遗漏

| 来源 | 读取范围及用途 | 不应推断的内容 |
|---|---|---|
| J. S. Clements、Z. L. Clements, *GPU-accelerated Direct Geolocation of GNSS Interference*, [arXiv:2508.06672](https://arxiv.org/html/2508.06672v1),2025 | 读模型、仿真、计算章节。模拟58个50 ms捕获、5 MHz采样，以时变几何积累解决多干扰源结构；有20 μs chirp周期但这不是采集长度。 | 尚未核实正式期刊版本；不能将GPU并行当成有限模型理论方法，也不能把20 μs周期当G11观察窗依据。 |
| G. Pojani et al., *Low-Complexity Direct Geolocation of Terrestrial GNSS Jammers from Low Earth Orbit*, [arXiv:2606.22751](https://arxiv.org/abs/2606.22751),2026 | 本轮仅读作者摘要。真实OPS-SAT PRETTY/Jammertest2025，低SWaP、下行受限，QDG压缩与星上处理。 | 未精读全文，不采纳二手AI总结中的数值；适用波形限制不能忽略，不能作为当前未知宽带双源的直接对手。 |
| J. Li et al., *Coherent summation of multiple short-time signals for direct positioning of a wideband source based on delay and Doppler*, DSP48,58–70,2016；[DOI](https://doi.org/10.1016/j.dsp.2015.09.008) | 题录及2023原文引用链已核验，全文本轮未成功获取。它是“多个短段能否联合”的必须核对近邻。 | 不能据题目宣称已解决G11边界误差；也不能忽略多段积累这个自然替代办法。 |
| K. Hao、Q. Wan, *Importance sampling based direct maximum likelihood position determination of multiple emitters using finite measurements*, Signal Processing186,108111,2021；[出版社](https://www.sciencedirect.com/science/article/abs/pii/S0165168421001493)，[DOI](https://doi.org/10.1016/j.sigpro.2021.108111) | 已读摘要和导言：实时/平稳限制观测；有限样本下源独立近似与高维ML求解。 | 其“finite measurements”不自动等于当前非周期截窗模型；全局求解和边界表示是不同问题。摘要所称全局保证未核验其条件。 |
| M. Sabeti, *Alternating minimisation approach for direct position determination in passive localisation*, IET RSN17(7),1152–1159,2023；[DOI/全文](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/rsn2.12408) | 已读摘要/场景/模型：短突发，移动双天线，每时刻仅一个源活动。 | **排除为当前强弱重叠需求的直接支持**：非同时混叠，且靠跨时刻运动。 |
| *Passive Geolocation of Multiple Pulsed Emitters*, Kevin Joe, IEEE MILCOM2023 pp729–734；[DOI](https://doi.org/10.1109/MILCOM58377.2023.10356379) | 题录由会议目录和多个索引交叉确认；IEEE页访问失败、全文未获。 | 不凭二手摘要确定其脉宽、强弱比或IQ边界处理；保留为待获取条目。 |

## 对终审的判断建议

需求应分两层判定，不能“一刀切”：

- **一般层面通过**：未知发射波形、有限观测、多源/功率差、实际计算与通信预算在真实无源定位系统中存在，且并非冷门到没有应用。
- **当前研究实例尚未闭合**：现有证据没有给出为何必须是当前几十微秒、为何无法用更长/多段记录、为何已知响应静态双BPSK模型代表该应用主要困难，以及1–3 m改善或数十秒代价对具体系统的价值。

这不足以判定“有限记录题目不存在价值”，却足以阻止直接把G11现有流水线当成已立项的论文核心。要继续，必须有最近邻常规方法仍留下的明确问题和可检验的新机制，而不是将应用名称换成UAV或GNSS后保留全部原参数。本文件不建议立即扩展到多径、动态源或硬件校准；这些会改变任务而非补一句背景。

## 可复查检索日志

本轮使用 nature-academic-search 多源检索协议；专用academic MCP缺席，使用Web检索发现并优先打开IEEE/Elsevier/IET/作者机构或arXiv原文。未操作Zotero。跨链检索关键词：

1. `"direct position determination" "frequency hopping" short duration`
2. `"TDOA" "burst" "microseconds" emitter localization`
3. `"GNSS" jammer "geolocation" "short" observation satellite`
4. `"direct position determination" "frequency-hopping" signal`
5. `"TDOA" "frequency hopping" localization dwell time IEEE`
6. `"passive localization" "short-duration" "pulse" IEEE`
7. `"TDOA" "microseconds" "UAV" localization signal`
8. `"Passive Geolocation of Multiple Pulsed Emitters"`
9. `"TDOA" "UAV" "AERPAW" bandwidth localization IEEE`
10. `"Coherent summation of multiple short-time signals"`
11. `"Direct position determination" "finite" observation duration 2024 2025`
12. `"Direct position determination" "short-time" 2023 2024 2025`
13. `"Low-Complexity Direct Geolocation" samples capture duration`
14. `"burst" "direct position determination" signal observation length`

检索时间截止2026-10-24；不是穷尽性系统综述，不能从未检出推断不存在。排除结果包括已知通信导频定位、窄带DOA阵列、小快拍却长记录、单源脉冲分选、主动雷达回波以及仅有二手宣传的技术结论。访问问题：部分DOI/IEEE页不可达；原作者GNSS PDF通过requests成功读取，首次urllib因本地证书库异常失败，未关闭证书验证。上述缺全文条目不承担决定性否定结论。
