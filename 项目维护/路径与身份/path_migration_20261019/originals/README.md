# DPD_Strong_Weak — 强弱源物理定位

维护已知双源场景下的物理定位、AP/NOMP及参照方法和投稿论文。论文收尾已完成，当前入口为 paper/paper_draft.tex。G9为前期探索，G10为当前论文证据；各轮场景和评价合同保持区分。

G8与旧网络联动的完整历史验证已归入学习项目。本项目仅保留G9/G10必需的G8物理和场景模块，以及跨阶段重评必需的两份结果；不再加载CH3/D8、G0—G7或其他项目源码。少量固定基础副本用于保持本地独立运行，不使用共享目录或链接。

当前项目根目录为 `F:/DPD_Strong_Weak`，专门维护当前论文研究；不再另设 `DPD_WeakSource_Localization`。保留模块名和相对层级，不使用跨项目源码链接。本项目已建立独立Git仓库，保留main-e2e分支历史，不再依赖原主仓库；未配置远程。项目解释器保持原状。

- 研究与历史：[研究总结](研究总结.md)、[研究路线历史摘录](统一端到端模型研究.md)、[实验记录](实验结果记录.md)。摘录是历史原文，不是新的运行授权。
- 代码与入口：保留原来的中文源码目录和运行入口目录，不另设第二份src副本。
- 迁移身份：[文件清单](docs/迁移文件清单.jsonl)、[路径映射](docs/迁移映射.json)、[运行边界](运行基础说明.md)。
- 当前归属与独立Git状态见[项目位置](docs/当前项目位置.json)；之前的迁移及清理清单描述当时状态。
- 当前状态：[当前任务](当前任务.md)。验证详情见原项目 outputs_e2e/maintenance/project_split_20261019/交付记录.md。

日常研究与论文修改均在本目录进行。端到端项目 `F:/DPD_EndToEnd_Learning` 和复现项目 `F:/DPD_Thesis_Reproduction` 保持独立且不作变更。原根目录中的混合研究文档归档在 `outputs_e2e/maintenance/paper_home_20261019/previous_root/`；旧迁移清单和清理记录保留历史含义，最新位置以[本次交付记录](outputs_e2e/maintenance/independent_git_20261019/交付记录.md)为准。解释器继续使用 D:/Software/anaconda3/envs/PyTorch/python.exe。
