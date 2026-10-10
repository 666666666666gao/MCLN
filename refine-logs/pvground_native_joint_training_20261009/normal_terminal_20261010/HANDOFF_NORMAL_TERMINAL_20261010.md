## §20.376.138 — 正常联合训练三轮终态、完整权重恢复与当前保留结果（2026-10-10）

### 实际训练终态

正常PV网络联合训练于北京时间03:19:09启动、17:57:10完成，退出码0，耗时14.63小时。源代码与原定三轮配方未改。18:03:39到期观察确认控制器与训练子进程均已结束；该观察句柄已闭合，没有重启训练或在观察期限前反复查询GPU训练。

| 同一完整模型、9508条、原生last/bbs | @0.25命中／百分比 | @0.5命中／百分比 | 相对E0 |
|---|---:|---:|---:|
| E0：本轮适配权重初始化 | 5677／59.7076% | 4920／51.7459% | — |
| E1 | 5652／59.4447% | 4596／48.3382% | −25／−324 |
| E2 | 5552／58.3929% | 4552／47.8755% | −125／−368 |
| E3：固定终点 | 5576／58.6454% | 4488／47.2024% | −101／−432 |

按预定规则，best仍为E0。E0超过用户最新59.5%／51%的数值门槛，但这次三轮正常联合训练没有净增益，不能将E0既有适配能力写成新正常训练所得，也不能据此宣称三个模块均已有效。未采用单项最高列拼接、多seed、第二套排名或GT推理筛选。

原36665条训练表达、batch8、native drop_last，每轮4583次更新，三轮共13749次更新与109992条处理输入（含重复遍历，并非唯一表达数）。原核心／骨干LR为1e-6，新增模块1e-5，WD5e-4、clip0.1、seed2027。原RoBERTa参数冻结，其train模式与原生运行统计行为沿用作者代码；这些是源码事实，尚不是本次下降原因的实验证明。当前运行不是作者联合检测训练配方的完全复现。

### 保存状态确实可恢复

两份完整状态在实际native TrainTester工厂、CUDA/DDP与原load_checkpoint路径上冷恢复成功：best为epoch0、1295个模型状态张量、0个已填充Adam状态；latest为epoch3、1295个模型状态张量、820个Adam状态，步数均13749，恢复start_epoch4。模型、优化器参数组及张量、调度器和Python／NumPy／Torch／CUDA随机状态逐项一致。该恢复没有新forward、训练输入或optimizer更新，也不是重新跑了一次精度验证。

首次恢复核对因JSONL额外记录epoch、保存的retained_metrics未含epoch而失败；失败保留，第二次仅修正这项核对，不改训练或模型。SCHEMA_CORRECTION中的6ad98e…为LF文本SHA；实际部署文件原始字节SHA为9787b25f693fb7ca7f773e2c419dbfda3dae479b12b30b82dc0052ae8e3b3097，与RECOVERY_PLAN一致，另有BYTE_IDENTITY_CLARIFICATION说明，旧收据未改。

实际best：615023752字节，SHA256 ed8455ddc67e4d17018e9cf499197140e15e35db3f5daee55e0eec6846faf0c4；实际E3 latest：841676832字节，SHA256 a000a1d3ea56db3ec9bcb44b49c1db68933dfaaae7c807c351ee0e5e76c2ca2f。

### 审查、资源与后续

新终态完整性审查为WARN，0项hard integrity failure；支持上述有限的完整9508汇总、固定终点、预定best和状态恢复描述。审查没有独立重算逐条IoU／表达唯一覆盖，也未检查全部原始GT；请求模型与实际运行模型不能混同，实际运行身份UNATTESTED，属于same-family／provisional审查。没有因WARN将负训练结果改写成正结果。

C-off直接控制的源代码审查为WARN／SOURCE_ONLY，0项阻塞源码缺陷。它与主组只差use_selected_mask_supervision，但尚未通过实际控制预检、E0与初始状态等价、实时磁盘／GPU准入。其历史父权重已接受过C训练，因此将来只能回答同起点正常续训的C增量。实验计划要求先分析主组E0后的退化，不能仅凭源码通过自动追加多组长训练。

本节形成时两份完整权重正在下载归档，尚未删除远端文件；只有本地字节数与SHA256完整验证、再核对远端身份后才按用户既有授权清理较差latest。最好权重、官方PV／G／d06／f989以及V99、Nr／Sr父权重继续保护。18:03数据盘空闲1626402816字节；按现行2605151860字节准入预算，清理latest后仍须核对实际可用空间，不能将归档启动当作空间已释放。

Nr3D／Sr3D只有接口、作者权重CPU初始化与元数据准备，尚未启动当前完整模型的新训练。总体目标仍未完成：保住同一完整模型>59.5%／>51%，证明三个有效机制，再以同一最终版本分别训练Nr3D／Sr3D。

证据：NORMAL_TERMINAL_SUMMARY_20261010.json；normal_epoch3_boundary_observation/原始终态和native_metrics.jsonl；normal_terminal_recovery_20261010/attempt2/RECOVERY_RESULT.json；terminal_actual_review_20261010/EXPERIMENT_AUDIT.json；normal_controls_20261010/source_review_20261010/SOURCE_REVIEW.json。本节新增证据位于refine-logs/pvground_native_joint_training_20261009/normal_terminal_20261010/，历史pending标志保留原文，当前恢复与审查状态以新增实际收据为准。
