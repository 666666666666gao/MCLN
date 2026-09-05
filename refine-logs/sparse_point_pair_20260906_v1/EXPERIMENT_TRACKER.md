# 稀疏局部记忆实验执行表

| Run ID | 阶段 | 内容 | 状态 | 决定 |
|---|---|---|---|---|
| V0 | 原生预检 | 16扫描、12次forward、0更新 | PASS | 允许固定学习对照 |
| V1 | 配对学习 | 26747 fit、6172 holdout、两臂6687步 | STARTED | 8 tests/source/data PASS;06:11初始化中 |
| V1-audit | 终态复算 | 两参考质量门、条件Mask、参数/AdamW实物 | PENDING | 不自动推广到正式模型 |
