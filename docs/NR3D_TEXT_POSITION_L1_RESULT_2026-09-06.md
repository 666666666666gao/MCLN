# L1 终态：完整性通过，固定质量筛选失败

两臂各完成 6,687 次更新，退出码 0。position-key 相对同容量 text-key 控制和
受保护起点均没有达到预先要求的 REC@0.25 净增 10 个命中；不进入正式验证，
不升级任一控制模型，不延长本次训练或重新扫描学习率。

以下均是 **6,172 条、98 个训练场景的模块留出集**；冻结主干曾见过这些场景。
不是正式 Nr3D 验证成绩，也不是整个系统的新场景泛化结果。

| 输出 | REC hits@.25 | REC hits@.50 | Mask hits@.25 | Mask hits@.50 | Mask mIoU |
|---|---:|---:|---:|---:|---:|
| 受保护原生起点 | 6005 | 5306 | 5767 | 5057 | 68.881519715% |
| 终态 text-key | 6005 | 5308 | 5766 | 5053 | 68.877019645% |
| 终态 position-key | 6005 | 5307 | 5767 | 5057 | 68.877936727% |

position-key 相对起点：REC@.25 修复 0、破坏 0；REC@.50 修复 1、破坏 0；
Mask@.25 修复 0、破坏 0；Mask@.50 修复 1、破坏 1；Mask mIoU -0.003582988pp。
仅 3/6172 行改变选中的 REC Query。多数未改变 Query 的框数值发生微小变化，
不能把这种数值变化当作语义消歧改善。

position-key 相对终态 text-key：REC@.25 修复 2、破坏 2；REC@.50 修复 12、
破坏 13；Mask@.25 修复 1、破坏 0；Mask@.50 修复 7、破坏 3；
Mask mIoU +0.000917082pp。严格 REC 净减 1，也不满足非退化条件。

## 执行与完整性

两臂均为零初始化的 288×288 矩阵，各 82,944 参数，其余参数及模型 buffer 冻结。
26,747 条 fit、413 场景，每行恰好一次，B4、无增强、fresh AdamW lr=1e-5、
weight_decay=.0005、clip=.1。起点一次、两臂终点一次完整留出评估，没有中间挑点。
总耗时约 9,929.20 秒（不含数据集初始化）；峰值 CUDA allocation 6,214,786,560 字节。

完成回执、输入清单、逐行输出 hash、训练顺序、预检引用以及两份实际序列化权重和
optimizer 均已独立核验。两份 artifact 的矩阵和 optimizer 一阶、二阶矩有限，
均为 6,687 步；parent 权重未修改。

- input manifest SHA：`354c680b44ac799a1c6d573aebfd8653b474e9a0cda8ae71818a76ac237a3746`
- training receipt SHA：`e27c990915f728780f7f4bd99dfff054780fd9a9add9f27ab8ca1abe9fb26621`
- terminal rows SHA：`e47de917d07fefd230ded395990a5ac1c4df027d7c4d81f97177cc9f6c0b6914`
- text artifact SHA：`b7d24aef0b5502def03392a1d2142d15b1dfb4a33bc351175cab1960d655ab50`
- position artifact SHA：`5f984e72f25d6b54168730d5a3fe4117f86fae3892c84b67bed0bad4b0287632`

完整 metrics、修复/破坏、scene 聚类 bootstrap 与逐行证据在
`refine-logs/text_position_l1_20260905_v2/train/`。模型 artifact 保留在对应远端隔离目录，
GitHub 不包含大权重文件。

## 对后续路线的影响

此结果不支持“仅在最后层文本 Key 加入当前点位置证据，就能在该固定短训中带来
有效收益”。它不构成对完整 EG-3DVG 或所有语义—空间交互的否定。前期零初始化
一致、梯度连通和训练 loss 变化只证明实现可运行，均不能替代本次失败的质量筛选。

下一项独立实验为高分辨率点细节记忆，不同时更换主干、任务头或损失。当前正式
Nr3D 仍保留 4475/3759，Sr3D 12139/10335，ScanRefer V99 不变；三数据集目标未完成。
