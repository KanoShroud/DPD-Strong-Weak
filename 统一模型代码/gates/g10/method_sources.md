# G10方法出处、公式与适配登记

本表与源码一同冻结。下列都是当前四站、已知相对响应、未知波形问题的实现，不将“已知响应适配版”冒称为原论文完整复现。共同响应为 `q_f(p)=a_f(p)/||a_f(p)||`，`a_l=exp(-j2π f d_l/c)/d_l`；单位列尺度吸收到逐频未知源系数，不按站归一化观测。`FFT/N`对应噪声方差`1/N`。有限截取的连续RRC数据与逐频移位模型有窗边近似，不能用周期频域代数测试证明该近似无误差。

## 依据与实现

| 方法 | 可核验出处 | 本轮公式与适配 | 实现/核验 |
|---|---|---|---|
| 单源LS谱 | Chen、Hudson、Yao 2002，TSP 50(8)，§II.A；DOI 10.1109/TSP.2002.800420，本地Zotero YX23XVTW全文 | `sum_f |q_f^H x_f|²`，多峰不是完整双源ML；区别于消除自由站间增益后的最大特征值谱 | physics.Model.conditional / 直接LS等价 |
| 宽带MUSIC | Weiss、Amar 2005，§3.1式(7)—(12)，DOI 10.1155/ASP.2005.37，本地ID3FDGNH全文 | 每段FFT估计逐频协方差；用已知q代替未知b的最大化。`1/mean_f(q^H E_n E_n^H q)`，等价最大化聚合信号子空间能量（单位q） | physics.Spectrum / NumPy逐点参考；不是单频特征值定位 |
| 宽带MVDR | Capon 1969，High-resolution frequency-wavenumber spectrum analysis，DOI 10.1109/PROC.1969.7278；Tirer、Weiss 2016，DOI 10.1109/LSP.2015.2503921（本地Q7FCKSDN全文）作为DPD-HR区别依据 | 每频单位响应Capon功率`1/(q^H R_loaded^-1 q)`，跨频求平均。不同于原HR的跨频矩阵累加后最小特征值倒数；输出名称为MVDR-known，不标原HR | physics.Spectrum / 约束最小方差及逐点参考 |
| AP | Chen等2002 §II.B，多源AP；另见Ziskind、Wax 1988 DOI 10.1109/29.7543 | 消去全部逐频波形，固定其他位置，用同时投影后的观测和候选评价残差下降。常规初值、循环和全域10/5m补搜 | methods.ap_solver / 条件增量=联合残差下降、保留旧解 |
| 块NOMP适配 | Mamandipoor等2016 §III、Algorithm 1，DOI 10.1109/TSP.2016.2580523，本地UY79JRU2全文 | 原标量频率原子改为逐频位置响应块；发现新源、2D阻尼Newton、循环单源残差更新、逐频联合LS。已知K停止，不使用原CFAR | methods.nomp_solver / 梯度、Hessian及联合残差检查 |
| SBL-PDL-known | Wang等2026 §III、Algorithm 1、Remark 1/2，DOI 10.1109/JSEN.2026.3693237，本地5VI34FEP全文 | 本轮重新从复高斯模型推导，保留全域共享稀疏先验及活动位置修正；不估自由链路增益，q随位置更新。用精确证据下降的局部搜索替代原线性化增益/坐标更新。不是逐式原版PDL | methods.sbl_solver / Woodbury与直接复数后验一致，证据不增 |
| 多初值联合优化 | 同Chen等2002波形消元目标；SciPy L-BFGS-B官方接口 | 四坐标联合优化，谱初值+Sobol全域初值，最多32组；不是新的DPD评分 | methods.multistart_solver / 联合梯度及初值可追溯 |
| GA联合优化 | Deb、Agrawal 1995，Simulated Binary Crossover for Continuous Search Space，Complex Systems 9:115–148；https://content.wolfram.com/uploads/sites/13/2018/02/09-2-2.pdf | 实数编码，SBX eta15，0.9交叉，三元锦标赛，精英4，eta20多项式变异概率0.25，边界裁剪；种群64。同一四维LS，末尾同一局部精修。项目通用GA组合，不称为DPD论文原版 | methods.ga_solver / 种子、可行域、保留最优与原目标一致 |

## SBL适配推导（复数共轭不可省略）

每频`x_f=A_f s_f+n_f`，`s_f~CN(0,Γ)`，`Γ=diag(γ)`、噪声`ν=1/N`。

- `C_f=νI+A_f Γ A_f^H`。
- `μ_f=Γ A_f^H C_f^-1 x_f`。
- `Σ_f=Γ−Γ A_f^H C_f^-1 A_f Γ`。
- EM更新`γ_q=mean_f(|μ_fq|²+Σ_fqq)`，不将复数均值直接相加当功率。
- 最小化证据`mean_f(log det C_f+x_f^H C_f^-1 x_f)`。更新当前K个主要候选坐标时，其余项固定；坐标改变时幅度和时延同时重算，并以原精确证据接受。
- 全网格始终保留，噪声方差是所有算法共同允许的已知量。跨频共享γ是本适配的稀疏统计假设，不宣称真实BPSK各频功率相等。
- 利用4×4协方差求解，不建立Q×Q后验；不是删频或只看真值附近。

## 理论成本口径

令L=4，K≤2，F=N，J为段数，F_s=N/J，G为网格点数。LS扫描约O(GFL)；投影扫描约O(GFLK)，含投影构造；联合LS每次O(F(LK²+K³))。MUSIC/MVDR需O(J F_s L²+F_s L³)统计/分解及O(G F_s L²)扫描。SBL每轮约O(F(GL²+L³))加活动坐标证据线搜索。AP、NOMP、多初值和GA须再乘实际网格访问、循环、初值、种群及代数求解次数，不引用论文的简化阶数替代本实现。所有频率保留，报告实际调用计数及完整墙钟。

## 原始出处访问范围

R001、MUSIC、NOMP、R002在本地PDF全文中核验上述相关章节；R002正文有转置和公式交叉编号问题，本适配使用上面的复高斯推导与数值对照，不静默照抄错误编号。Capon原文大学公开副本：https://epsc.wustl.edu/~ggeuler/reading/cam_noise_biblio/capon_1969-ieee-high-resolution_frequency-wavenumber_spectrum_analysis.pdf 。原HR仅作为方法关系依据，不把已知响应Capon合成谱称作其复现。
