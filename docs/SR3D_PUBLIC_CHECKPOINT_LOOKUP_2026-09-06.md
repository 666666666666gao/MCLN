# Sr3D公开checkpoint来源检查

2026-09-06 08:18 CST，在既有私人备份检查之后，对原论文官方公开来源进行一次有范围的补查。
本次未找到可用的Sr3D完整checkpoint链接；不能据此断言网上不存在任何副本。

| 来源 | 本次可核实的内容 | 结论边界 |
|---|---|---|
| [官方固定提交README](https://github.com/qzp2018/MCLN/blob/9744a4ed219062d448ed0dba587eeb864491f158/README.md#3-models) | Models只有ScanRefer一行；Nr/Sr有训练和评估脚本 | 脚本、PointNet++预训练权重均不能代替完整Sr模型 |
| [官方Releases](https://github.com/qzp2018/MCLN/releases) | 本次页面显示没有release | 限于该公开仓库的release |
| [Hugging Face上传讨论](https://github.com/qzp2018/MCLN/issues/5) | 可见2024评论讨论计划，未给出具体模型仓库 | 旧缓存；issue关闭不能证明上传完成 |
| [关联论文页](https://huggingface.co/papers/2407.05363) | 本次显示关联模型0个 | 不覆盖未引用论文的模型或私人仓库 |

实际git ls-remote只返回main，HEAD为9744a4ed219062d448ed0dba587eeb864491f158，
没有其他公开head/tag。另读实验细节issue12未见checkpoint链接；网页API访问失败只记录
工具限制，不推导资源不存在。可见页面观察与原始refs输出见
refine-logs/sr3d_public_checkpoint_search_20260906/lookup.json。

历史受保护Sr权重仍未恢复。本次没有重新扫描旧磁盘备份、下载权重、联系作者、运行模型
或更新优化器。受保护路径缺失的最近服务器观察仍是04:15；公开检查不是新的服务器检查。
即使以后取得官方Sr模型，也必须单独标明其来源，不能替代项目历史受保护SHA的恢复证明。

Nr3D稀疏局部记忆V3按原冻结计划运行。本地观察会话在08:15仍有效；最近实际GPU进度仍为
07:53:40两臂各192/6687步。下次服务器查询约10:53，终态初估10:58，随后240秒观察。
该时间是此前速率外推，本次未查询中间质量或改变运行设置。正式三数据集结果保持，整体目标继续。
