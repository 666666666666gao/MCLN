# 归一化控制执行表

| 阶段 | 状态 | 证据 |
|---|---|---|
| 代码实现 | REVIEWED | 复用runner；仅扩展最后层对比项分母改变；审查见EXPERIMENT_CODE_REVIEW |
| CPU原生公式核验 | PASS | cpu_test.json；A=0损失/梯度；A>0直接调用；其他GT保护 |
| GPU batch8预检 | PASS | preflight.json；2步、有限梯度、完整模型/优化器恢复与显存 |
| normalized训练 | COMPLETE | 3723步／29778条各一次；train.exit=0；complete/normalized/receipt.json |
| 正式9508验证 | COMPLETE_NOT_PROMOTED | bbs5575／4426；较原G−40／−69；控制器19:32:52结束；analysis/ |
| 终态审查 | WARN_ZERO_BLOCKING | 新鲜上下文；same-family/provisional；保存记录重算一致，范围限制见analysis/EXPERIMENT_AUDIT.md |
| 非最佳权重清理 | COMPLETE | 仅删除normalized/terminal.pth 342299951字节；原G保留；nonbest_retirement_receipt.json |
