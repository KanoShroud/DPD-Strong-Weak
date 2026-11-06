# SBL共同状态比较

DEVELOPMENT

本报告不改变SBL内部优化。分组γ与单点γ比较分组作用；相干聚合另与单点后验功率比较，避免混淆评分变化。

| Candidate | Reference | Target | Subset | N | Rescued | Harmed | Difference | 95% CI |
|---|---|---|---|---:|---:|---:|---:|---|
| SBL_GROUP_GAMMA | SBL | both | all | 5 | 0 | 0 | 0.0 | None |
| SBL_GROUP_GAMMA | SBL | both | duplicate_triggered | 4 | 0 | 0 | 0.0 | None |
| SBL_GROUP_GAMMA | SBL | weak | all | 5 | 0 | 0 | 0.0 | None |
| SBL_GROUP_GAMMA | SBL | weak | duplicate_triggered | 4 | 0 | 0 | 0.0 | None |
| SBL_GROUP_GAMMA | SBL | strong | all | 5 | 0 | 0 | 0.0 | None |
| SBL_GROUP_GAMMA | SBL | strong | duplicate_triggered | 4 | 0 | 0 | 0.0 | None |
| SBL_GROUP_COHERENT | SBL_GROUP_GAMMA | both | all | 5 | 0 | 0 | 0.0 | None |
| SBL_GROUP_COHERENT | SBL_GROUP_GAMMA | both | duplicate_triggered | 4 | 0 | 0 | 0.0 | None |
| SBL_GROUP_COHERENT | SBL_GROUP_GAMMA | weak | all | 5 | 0 | 0 | 0.0 | None |
| SBL_GROUP_COHERENT | SBL_GROUP_GAMMA | weak | duplicate_triggered | 4 | 0 | 0 | 0.0 | None |
| SBL_GROUP_COHERENT | SBL_GROUP_GAMMA | strong | all | 5 | 0 | 0 | 0.0 | None |
| SBL_GROUP_COHERENT | SBL_GROUP_GAMMA | strong | duplicate_triggered | 4 | 0 | 0 | 0.0 | None |
| SBL_GROUP_COHERENT | SBL_POINT_POWER_AUX | both | all | 5 | 0 | 0 | 0.0 | None |
| SBL_GROUP_COHERENT | SBL_POINT_POWER_AUX | both | duplicate_triggered | 4 | 0 | 0 | 0.0 | None |
| SBL_GROUP_COHERENT | SBL_POINT_POWER_AUX | weak | all | 5 | 0 | 0 | 0.0 | None |
| SBL_GROUP_COHERENT | SBL_POINT_POWER_AUX | weak | duplicate_triggered | 4 | 0 | 0 | 0.0 | None |
| SBL_GROUP_COHERENT | SBL_POINT_POWER_AUX | strong | all | 5 | 0 | 0 | 0.0 | None |
| SBL_GROUP_COHERENT | SBL_POINT_POWER_AUX | strong | duplicate_triggered | 4 | 0 | 0 | 0.0 | None |

首次重复仅指保存检查点观察到。详见sbl_output_metrics.csv和sbl_first_observed.csv。
