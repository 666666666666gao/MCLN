# 归一化控制执行表

| 阶段 | 状态 | 证据 |
|---|---|---|
| 代码实现 | READY_FOR_REVIEW | 复用runner；仅扩展最后层对比项分母改变 |
| CPU原生公式核验 | PENDING | A=0损失/梯度；A>0直接调用；其他GT保护 |
| GPU batch8预检 | PENDING | 2步、有限梯度、恢复与显存 |
| normalized训练 | NOT_STARTED | 原G/fresh AdamW；3723步；29778条一次 |
| 正式9508验证 | NOT_STARTED | 原生last/bbs；所有候选与GT离线诊断 |
