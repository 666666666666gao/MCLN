# 归一化控制执行表

| 阶段 | 状态 | 证据 |
|---|---|---|
| 代码实现 | REVIEWED | 复用runner；仅扩展最后层对比项分母改变；审查见EXPERIMENT_CODE_REVIEW |
| CPU原生公式核验 | PASS | cpu_test.json；A=0损失/梯度；A>0直接调用；其他GT保护 |
| GPU batch8预检 | PASS | preflight.json；2步、有限梯度、完整模型/优化器恢复与显存 |
| normalized训练 | RUNNING | launch.json；原G/fresh AdamW；3723步；29778条一次 |
| 正式9508验证 | NOT_STARTED | 由同一控制器在训练后执行，尚无新结果 |
