# Fourier extension / AZ 最近邻矩阵对照

2026-10-23。只读原文与现有数学记录，不运行实验、不改代码。未完成盲 pilot 及其局部成功例不用于本结论。

## 结论

**通用“规则域快速求解＋低秩纠正”已被 Fourier extension / AZ 框架覆盖；当前算法可以明确嵌入 AZ，不能再将其宣称为一种独立的新求解原理。** 这不自动否决适中计算贡献：剩余可讲的增量是多源多站算子族的特化——位置无关的边界基、跨位置摊销、固定 ridge 解的统一容差选秩，以及在这一定位问题上的端到端计算收益。当前只与直接 Cholesky、LSMR 比较，还不足以排除最直接的快速近邻。

## 1. 实际读到的原文

Matthysen–Huybrechs，*Fast Algorithms for the Computation of Fourier Extensions of Arbitrary Length*，SIAM J. Sci. Comput. 38，2016，A899–A922，DOI 10.1137/15M1030923。读到[作者全文](https://roelmatthysen.github.io/slides/paperfastfe.pdf)，核心位置为 §2.1 式 (2.3)–(2.4)、§3.1 式 (3.1)–(3.4)、§4.1/4.2 与 Algorithms 1–2、§5.1。其归一化受限 Fourier 矩阵可写成 A_FE=S_t F_Pᴴ S_fᵀ；显式周期 DPSS 与隐式低秩纠正两条路线均已给出。隐式路线使用 (A_FE A_FEᴴ−I)A_FE，其奇异值映射为 σ³−σ；多数计算可为同参数重复拟合复用。其主要数值目标是正则化/截断 SVD 意义的小残差拟合，不是本项目指定 λ 的唯一 ridge 系数解。

Coppé–Huybrechs–Matthysen–Webb，*The AZ Algorithm for Least Squares Systems with a Known Incomplete Generalized Inverse*，SIAM J. Matrix Anal. Appl. 41(3)，2020，1237–1259，DOI [10.1137/19M1306385](https://epubs.siam.org/doi/10.1137/19M1306385)。出版元数据来自出版社，公式核对用 [arXiv:1912.03648v1](https://arxiv.org/pdf/1912.03648v1)。已读 Algorithm 2.1、Lemmas 2.1–2.3、§3 开头的低秩误差定义、Algorithm 3.2 与 Theorem 3.4。它允许用问题先验构造 Z，使 A−AZ* A 数值低秩；分析包含低秩加小范数误差，重点是后向残差与稳定拟合，不仅限于标量 Fourier extension。

KU Leuven 的 TW662 链接本次超时，2016 DOI 跳转未能由网页工具打开；没有把这两个失败写成已读取。作者镜像取得完整正文。预登记查询与关键正文返回保存在 `phase2_sources/A_finite_window/round21_fe_az_*`。

## 2. 逐项映射：不是只因都用了 Woodbury

保留此前记号：完整周期多站合成 A₀(p)，观测 B(p)=SA₀(p)，缺失行 V(p)=MA₀(p)，ridge 正规阵

\[
 H(p)=B^HB+\lambda I=D(p)-V(p)^HV(p),\quad
 D(p)=\operatorname{blkdiag}_q
 \{P H_q(p)^HH_q(p)+\lambda I_K\}.
 \tag{1}
\]

每个站/源块都是 scalar FE 矩阵在右侧乘延迟相位、再乘增益；把这些块横向连接源、纵向连接站，就得到当前 B。多源相位耦合使完整周期逆从标量变成每频率 K×K 块，不能直接把各站当独立函数拟合后称为同一消元，但仍是规则域 Fourier 算子的矩阵值推广。

AZ Algorithm 2.1 的步骤为

\[
 (I-AZ^*)Ax_1=(I-AZ^*)b,\quad
 x_2=Z^*(b-Ax_1),\quad x=x_1+x_2.
 \tag{2}
\]

现在直接取

\[
 A_{AZ}=H(p),\qquad Z_{AZ}^*=D(p)^{-1},\qquad b=B^Hy.
\]

则其低秩条件恰好成为

\[
 H-HD^{-1}H=HD^{-1}V^HV.
 \tag{3}
\]

也就是说，缺失行低秩修正并不是 AZ 不包含的特殊原理。对当前缺失行的正交压缩，写成 VᴴV=CᴴC+EᴴE，取

\[
 \widetilde H=D-C^HC,
\]

则 H̃−H̃D⁻¹H̃ 的秩至多 rank(C)。在精确算术、AZ 第一阶段精确求解时，AZ 与当前 Schur 公式产生同一个唯一的 H̃⁻¹b；它们不是两个不同统计估计量。不同低秩构造、容差与浮点实现可以产生不同时间和误差，但须据此定位差异。

AZ Lemma 2.3 的先验假设也可直接映射：其易求逆部分 W=D，低秩项 L₁=−CᴴC，小误差 E₁=−EᴴE；Z*=W⁻¹，无需额外 L₂、E₂。这比“都用了矩阵求逆引理”更强：连“已知逆＋低秩＋小误差”的框架都重合。

## 3. 尚可成立的适中增量，以及不能夸大的部分

本项目的具体先验是

\[
 V_{lk}(p)=g_{lk}(p)V_0\Phi_{lk}(p),
 \quad\Phi_{lk}^H\Phi_{lk}=I,
 \quad\sum_l|g_{lk}|^2=1/N.
\]

因此一次从 V₀ 得到的 U_r 可以覆盖整族 p，而不是对每个几何重新识别低秩子空间；并有

\[
 \|E(p)\|_2^2\leq(K/N)\|(I-U_rU_r^H)V_0\|_F^2=\eta,
 \quad
 \|\widetilde c(p)-c(p)\|/\|c(p)\|\leq\eta/\lambda.
 \tag{4}
\]

可将它作为“多站相位耦合算子族的、可摊销且定容差的 AZ/边界消元实现”。原文没有直接给出本项目的 g、Φ 分解及这条几何一致选秩规则。但单位模不变性、Frobenius 界和正定矩阵扰动本身都常规；**不能仅把一般残差界换写成 η/λ 系数界就称理论突破**。有价值与否取决于这项特化是否在相关尺寸和反复位置搜索中，确实比直接套用快速通用方法更便宜、更可控。

U_r 仍是周期 Slepian 型边界集中子空间；2016 文献已经使用周期 DPSS。其“固定参数可复用”也已有来源。这里剩余差别是：p 不断变化而共享的是未调制的缺失区间基，D 与调制后的小修正仍随 p 变化。不能笼统声称首次预计算或首次边界低秩。

另外，2016 FE 的 plunge-region 小秩结论不能直接原封不动赋给当前耦合 H：各频率源响应条件数、ridge 以及保系数精度的要求都不同。只保样本拟合残差的 TSVD，可能丢弃对指定 ridge 系数重要的方向；比较时必须保持这些条件。

## 4. 必须补的最强数值参照

**同 ridge、保留多源耦合的 block-AZ：对 Hc=b 使用 Z*=D⁻¹，以原文 Algorithm 3.2 的随机低秩 SVD 求解 AZ 纠正阶段。** 可采用对称归一化等价式

\[
 K=D^{-1/2}HD^{-1/2}=I-W^HW,
 \quad W=VD^{-1/2},\quad Z^*=I,
\]

并将解换回原系数。这里 K 只是归一化正规阵记号，不是源数。需要核验同 λ 下的原目标、正规残差与实际系数误差，不能把默认 TSVD 的小拟合残差冒充同 ridge 解。

这个参照应获得相同的 FFT、小频率块 D、线程、缓存及重复求解机会；计入低秩识别、预计算和随位置更新成本，不只计最后的小系统回代。重点比较“每个几何通用识别纠正子空间”与“本候选共享 U_r”的差异。若把同一 U_r 和同一 Woodbury 实现也交给 AZ，二者已是同一算法特化，不能改名后当作独立方法相互比较。

解析 Gram–Cholesky 仍是必要的强精度/成本参照；未预条件 LSMR 不是充分的最近邻对手。将 2016 标量 FE 逐站运行、去掉耦合或改成另一种正则化，也不是公平替代。当前没有运行 block-AZ，所以此处不预言它快或慢，更不因已有盲 pilot 局部成功就判定创新完成。

## 最终边界

**保留一个适中的应用计算候选，但将其明确置于 FE/AZ 家族内。** 一般求解结构已直接重合；可争取的是利用多站算子族先验得到的复用、误差控制和实际计算优势。若这些增量相对同目标 block-AZ 也不成立，就应把它归为高质量常规实现，而不继续以新的有限记录定位方法立项。
