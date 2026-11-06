# 同信息强近邻：独立补检备忘

日期：2026-11-02。范围：二维、五同步单天线站、双未知同频重叠源、未知每链路多径条数与时延、无地图；不向算法提供真实波形或路径参数。仅检索与原文条件核验；未修改定位器、未运行实验、未写入Zotero。此文件是主综述的补充，不是“找全所有近邻”的声明。

## 结论先行

1. **Flood–Elvander 2024 已经在本项目完成作者固定K LP适配和48条共同后端比较，不是新增发现，也不应重新开工。** 复核旧报告后确认：它与当前前端结果几乎一致，已是可复用的同信息两步参照；但不能单独填满“已有多径IQ-DPD强参照”的空缺。
2. **Delay-locking（IEEE SPAWC 2020）是值得保留的盲多源多径思想近邻，不宜直接移植为五站定位器。** 其共址阵列、多径副本公共时延结构和时间白源条件不能从分布式站点自动继承。
3. 本次未获得一个可以未经实质修改就直接满足全部条件、且有公开完整实现的新增IQ多径DPD方法。这是本轮检索状态，不是不存在此类方法的证据。Xia、Friedlander等已有核心近邻仍必须保留并由主线核验。

## 1. 已完成、应复用的参照：OT关联定位

**Gabrielle Flood, Filip Elvander, Multi-Source Localization and Data Association for Time-Difference of Arrival Measurements, EUSIPCO 2024, pp.111–115.**

- [会议正式全文](https://eurasip.org/Proceedings/Eusipco/Eusipco2024/pdfs/0000111.pdf)
- [作者MATLAB代码](https://github.com/gabrielleflood/omt-based-toda-localization)
- 原文定位：p.112 §II 式(1)–(2)，§III–IV；p.113 式(4)–(6)；p.114 §VI。
- 已核对：先用最小多边定位子集生成候选，再作稀疏、熵正则化的非平衡最优传输关联；虚假测量进入虚拟源，容量约束容纳漏检；最终对已关联TDOA连续精修。
- 信息边界：原文输入是已估计TDOA，仿真也从真TDOA加扰动生成；不是从混合IQ验证多径。不能把模拟真TDOA作为我们基线输入。
- 本项目已完成：作者固定提交`da40de7df2d1af603ea7b547a4f13840eb288281`、固定K运输LP、二维RF距离尺度适配、实际IQ峰与共同后端配对。原熵版收敛不足的记录不用于证明候选优势。复用这些证据，不重新提出相同适配任务。
- 公平性重点：保留假峰和缺峰；统一距离差单位；正则与虚拟源代价只按原文或预登记开发规则设定，不用真值挑参数；报告完整峰提取、候选、关联、精修成本。
- 证据级别：本次核对正式全文关键建模与公开仓库；更深的已有源码审计与数值核验见[强近邻配对核验](../neighbor_review_20261029_235730_185004/强近邻配对核验.md)。

## 2. 思想近邻：Delay-locking

**Mohamed Salah Ibrahim, Nicholas D. Sidiropoulos, Delay-locking: Unraveling Multiple Unknown Signals in Unknown Multipath, IEEE SPAWC 2020. DOI 10.1109/SPAWC48557.2020.9154338.**

- [正式发表链接](https://doi.org/10.1109/SPAWC48557.2020.9154338)
- [NSF作者稿](https://par.nsf.gov/servlets/purl/10303809)
- [作者正式目录确认](https://nikossidiropoulos.github.io/publications.html)
- 原文索引内容定位：§I思想与假设；Algorithm 1/2；§IV实验。网页直接打开NSF正文多次502，搜索索引可返回这些原文段落；因此本轮**不标记完整精读**，不凭未读公式开发完整算法。
- 核心：对多天线观测尝试时间移位，用CCA识别对齐后共享的源子空间；不要求已知训练波形。
- 障碍：共址多天线、时间白信号、各用户不同相对路径时延；示例为四用户、六天线、每用户两路径、整数时延。当前RRC过采样源不是时间白样本，分布式站的路径时延也不同。需要新的模型推导，不是坐标参数替换。
- 分类：思想/可辨识条件对照，暂不进入同信息性能排名；不能拿恢复波形性能替代位置RMSE。

## 3. Elsevier盲信道路线：有价值但单源边界明确

**Madison L. Rose, Joseph L. Ipson, Todd K. Moon, Jacob H. Gunther, Sparse adaptive channel identification method using the cross-relation, Physical Communication 47 (2021), 101354.**

- [正式发表及作者可见正文片段](https://doi.org/10.1016/j.phycom.2021.101354)
- [配套学位论文（作者机构）](https://engineering.usu.edu/ece/files/pdfs/student-papers/thesis/2020/indoor-source-localization-of-radio-frequency-transmitters-using-blind-channel-identification-techniques-thesis.pdf)
- 已核对出版社“Complex cross relations methods”观测式：每个接收通道是**一个未知源**经不同未知通道卷积；有复数、稀疏和锚定约束改进。实验使用较长白噪声序列及20–60dB。
- 理论适配障碍：单源时不同通道的卷积交换性产生交叉关系；两个未知源叠加后同一简单消去式不成立。先用候选算法分离源、再给该法处理，不能称原文的独立双源参照。
- 分类：单源诊断/理论对照；若做完整双源改造，应单列新适配方法，成本不低。

## 4. 新近DPD与关联论文：避免标题误判

| 文献 | 已核对内容与具体障碍 | 分类 |
|---|---|---|
| Chai et al., Direct Position Determination of Wideband Source over Multipath Environment: Combining Taylor Expansion and Subspace Data Fusion in the Cross-Spectrum Domain, Sensors 2025, DOI 10.3390/s25164967 | §II式(1)只有一个未知宽带源；§III–IV全站对互谱、平滑、Taylor补偿。不能把两峰提取视为已证明双源版本。 | 单源原型及有限适配研究，不能直接双源排名 |
| Zhu et al., A Direct Position Determination Method Based on Subspace Orthogonality in Cross-Spectra under Multipath Environments, Sensors 2022, DOI 10.3390/s22197245 | §II式(1)单源；其后明确参考信号无多径/有先验信息；路径阶数称可AIC/MDL估计但推导视为已知。 | 额外参考信息隔离；不可免费提供干净参考 |
| Liu et al., An Iterative Implementation-Based Approach for Joint Source Localization and Association Under Multipath Propagation Environments, IEEE TSP 71 (2023), 121–135, DOI 10.1109/TSP.2023.3241776 | 作者上传全文引言明确目标是DOA；式(21)–(22)源—路径相干关联建立在传感器阵列方向响应。路径数可未知不等于能用五个异地单天线直接替换。 | 阵列信息隔离；原理关系仍重要 |
| Zhao et al., Direct positioning of multiple targets based on electromagnetic vector sensors array, Signal Processing 239 (2026),110292, DOI 10.1016/j.sigpro.2025.110292 | 出版社摘要与模型说明使用电磁矢量传感器电/磁分量、张量及多径去相关；本轮未获得全文。 | 硬件/观测信息隔离，不能冒充同信息 |
| Hao & Wan, Sparse Bayesian Inference-Based Direct Off-Grid Position Determination in Multipath Environments, IEEE WCL 10(6) (2021),1148–1152, DOI 10.1109/LWC.2021.3057502 | IEEE官方目录确认题名/出版信息；本轮未取得可核对全文。不能仅凭摘要将其认定未知宽带双源单天线可用。 | 待主线现有全文核验，未排除 |

来源：

- Chai正式全文：https://www.mdpi.com/1424-8220/25/16/4967 ；备份 https://pmc.ncbi.nlm.nih.gov/articles/PMC12390600/
- Zhu正式全文：https://pmc.ncbi.nlm.nih.gov/articles/PMC9571738/
- Liu作者上传全文：https://www.researchgate.net/publication/368230154_An_Iterative_Implementation-Based_Approach_for_Joint_Source_Localization_and_Association_Under_Multipath_Propagation_Environments （本轮仅使用作者上传原文，不使用平台自动相关摘要）
- Zhao出版社：https://www.sciencedirect.com/science/article/pii/S0165168425004062
- Hao正式链接：https://doi.org/10.1109/LWC.2021.3057502 ；IEEE目录 https://www.comsoc.org/system/files/2021-06/Publications_Contents_Digest_2021_Jun.pdf

## 5. 两条容易误归类的线索

**另补一条题名很近、实则需要地图的原文排除。** Jamali-Rad、Ramezani、Leus，Blind Sparsity-Aware Multi-Source Localization，EUSIPCO2013，[会议正式全文](https://eurasip.org/Proceedings/Eusipco/Eusipco2013/papers/1569738735.pdf)。p.2–3 §III式(3)–(5)把未知源功率谱与跨站信道乘积分开，但式(4)字典来自训练源在每个网格点测得的频域指纹。这里blind指不知源统计，不是未知环境、无指纹地图。对应扩展[Signal Processing2014作者稿](https://sps.ewi.tudelft.nl/pubs/hadi14esp.pdf)同样是RSS/相关指纹；不属于可直接实现的无地图强参照。

**Haimovich 2016 NJIT摘要。** [机构会议手册](https://research.njit.edu/cwcspr/sites/research.cwcspr/files/Symposium%20Program%203_22_16lores.pdf)确有“Localization of Unknown Signals Over Multipath Channels”，描述未知波形、多源、协方差、SOCP。本轮题名反向检索未定位完整正式方法论文。不能用它补造算法，也不能把Garcia 2015已知波形DLM当作该未知波形版本。

**Jamali-Rad & Leus IEEE TSP 2013。** DOI [10.1109/TSP.2013.2272288](https://doi.org/10.1109/TSP.2013.2272288)，[作者学位论文第5章](https://sps.ewi.tudelft.nl/pubs/hadi15phd.pdf)为公开对应材料；原文§II明确单径平坦衰落。可作多源TDOA关联参照，不能称显式多径DPD。Flood作者仓库附有其第三方实现，需标明不是Jamali-Rad作者官方源码。

## 6. 本轮检索路径与未覆盖边界

使用nature-academic-search的multi-source-search流程；专用academic-search MCP未发现，因此以网页检索发现线索、出版社/作者/机构原文核对。没有安装依赖、下载新论文到用户库或更新Zotero。

查询族：

1. `direct position determination multipath unknown multiple sources`
2. `direct localization unknown waveforms multipath covariance Haimovich`
3. `multiple emitters multipath unknown waveforms`
4. `blind multipath direct position determination`
5. `source localization blind channel identification RF multipath journal`
6. `time delay estimation multiple sources multipath IEEE`
7. `direct position determination multipath 2024 unknown signals single`
8. `multipath multi-source direct position 2026`
9. Delay-locking、Sparse adaptive cross-relation、Joint Localization and Source Association、Sparse Bayesian off-grid等题名定向与引用追踪。

前后向链：2025 Chai→2022 Zhu/2023 PARAFAC；2024 Flood→2013 SMTL/2021最小多边定位；2021 Rose→1995盲交叉关系/作者学位论文；2026 Signal Processing EMVS用于检查近期同题但不同硬件。聚合站仅作检索线索，未用其摘要代替技术原文。

## 7. 建议的最小实现顺序

1. 保留主线正在核验的原始IQ近邻，不因实现难而降级为仅LOS比较。
2. OT两步参照复用10月29日已完成的作者固定K LP与适配证据，不重新审计/运行同一任务。其良好表现不能作为新发现，也不能把更弱的原熵未充分收敛结果拿来排名。
3. Delay-locking仅列候选理论对照；没有完成其观测结构适配证明前，不投入完整定位工程。Rose单源CR也同理。
4. 后续排名必须区分“原文复现”“二维/单天线有限适配”“新增联合方法”；同信息不要求同内部模型，但禁止真实路径数、干净波形、地图或真初值专供其中一方。
