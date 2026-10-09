# Nr／Sr 接口准备进度

| 任务 | 状态 | 证据与下一步 |
|---|---|---|
| 对应作者初始化 | SOURCE_PREPARED | 两份 init.json，不载入 ScanRefer 状态；真实权重构建尚未执行 |
| 原生 G／C 标签适配 | SOURCE_PREPARED | 14 文件 AST；5 文件必要变更，等待独立复核 |
| CPU loss／梯度检查 | NOT_RUN | 不计为真实数据或准确率 |
| 真实 Nr／Sr loader 与 GPU M0 | NOT_RUN | 待 ScanRefer 正常训练结束及最终架构固定 |
| Nr3D 完整独立训练 | NOT_STARTED | 单 seed2027，使用对应作者权重 |
| Sr3D 完整独立训练 | NOT_STARTED | 与最终 Nr／ScanRefer 同架构，单卡串行 |
