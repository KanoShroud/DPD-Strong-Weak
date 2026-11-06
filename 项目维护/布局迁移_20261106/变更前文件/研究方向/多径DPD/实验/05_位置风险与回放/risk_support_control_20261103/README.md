# 固定清单风险诊断

本轮已完成，无需重跑。

- [执行协议](F:/DPD_Strong_Weak/outputs_e2e/mp_dpd/history/04_位置风险与回放/risk_support_control_20261103/执行协议.md)
- [报告](F:/DPD_Strong_Weak/outputs_e2e/mp_dpd/history/04_位置风险与回放/risk_support_control_20261103/run_20261103_165036_887173/固定清单局部响应报告.md)
- [原始结果](F:/DPD_Strong_Weak/outputs_e2e/mp_dpd/history/04_位置风险与回放/risk_support_control_20261103/run_20261103_165036_887173/result.json)
- [恢复后230项回读检查](F:/DPD_Strong_Weak/outputs_e2e/mp_dpd/history/04_位置风险与回放/risk_support_control_20261103/run_20261103_165036_887173/readback_audit.json)

执行184.176859秒：2 donor拟合、14固定清单局部重拟合、8预定小扰动；没有新全盲定位。原冻结源码与输入未修改。

结论：小扰动支持已有非零残差局部公式；完整噪声差分即使固定清单也仍有明显预测差异。不能只把风险排序失准归因于支持选择，也不能把本轮局部求解的RMSE作为新的完整算法成绩。严格重选组未执行，原因在报告中公开说明。

## 复现命令

在本目录使用项目Python：

```powershell
& D:\Software\anaconda3\envs\PyTorch\python.exe -X utf8 -m py_compile control.py test_control.py audit_report.py
& D:\Software\anaconda3\envs\PyTorch\python.exe -X utf8 -m ruff check control.py test_control.py audit_report.py --select F,E9
& D:\Software\anaconda3\envs\PyTorch\python.exe -X utf8 -m unittest test_control -v
& D:\Software\anaconda3\envs\PyTorch\python.exe -X utf8 audit_report.py
```

最后一条仅离线核验。`control.py`会执行新一批24次局部拟合并新建目录，不能把它当只读报告入口。
