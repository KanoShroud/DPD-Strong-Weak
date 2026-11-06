# 首轮CPU工程检查失败（依据本轮终端输出补记）

- 状态：失败，不作为通过证据；未执行GPU短测或性能实验。
- 检查源位置交换时，NumPy倒序视图带有负步长，原PyTorch转换拒绝该输入：`ValueError: At least one stride in the given numpy array is negative`。
- 最小修复：仅在R3的目标函数接口将位置数组转换为连续存储，数值和位置顺序不变。R2源码未改。
- 后续检查在新的`cpu_checks_*`目录执行，本目录保留。
