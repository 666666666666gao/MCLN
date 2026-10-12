# 几何诊断复核状态

`geometric_capacity_audit/EXPERIMENT_AUDIT.md`与JSON已完成。独立上下文审核为WARN、无本次限定诊断的阻塞项；CPU确定性重算通过。审核属于same-family/provisional，请求模型为gpt-6-astra、max，实际身份UNATTESTED。

两个探查结果及24条保存案例重算一致，范围为9508条、141个场景、2068个场景—目标组合。原生/Mask参考/迁移输出的float32 IoU逐项一致；float64在0.25与0.5上没有命中翻转。连续轴节点法的实数算术证明和GT六面投影的合法性、支配性均通过核对。

必须保留两个范围限制：

1. 端点没有包含A校正。实际完整迁移输出在43条上的IoU超过该保存端点轴向族的最大值，故它不是部署A+B的上界。严格阈值上，轴向族有448条命中位于完整迁移命中集合之外，完整迁移有3条位于轴向族之外；净计数差445不能直接称为可修复445条。
2. GT投影的9508个六面框均有效，但任意独立预测的六个系数在10条保存的端点组合中可能使上下界交叉。没有实现或验证新的六面网络，也没有把GT选择用于推理。

原始`GEOMETRIC_PARAMETERIZATION_DIAGNOSIS.md`保留审核时的字节；其最后一行“审核关闭前”状态由本文件更新。原始`MCLN_INTERFACE_AND_TRAINING_NOTES.md`是8条工程检查时的历史快照，其“完整回放仍等待”“正式预算未启动”等状态已经由完整回放结果、真实两步检查和正式控制器启动记录取代；不得作为当前状态引用。

当前正常训练的实际指标以`normal_training/normal_training_*_OBSERVATION.json`及对应收集的`training/native_metrics.jsonl`为准。几何缓存重算不认证活动训练效果，也不完成PV60.0%/52.0%、三项有效贡献或Nr3D/Sr3D目标。
