# Nr3D/Sr3D作者输入协议的正常训练入口

承接§20.376.142已经完成的作者核心加载和有限真实数据检查。本次只准备正常入口，尚未部署、执行或启动GPU训练。

已有隔离入口断言butd=true，拒绝butd_cls/joint_det，与保存的作者train/test脚本和两组实际数据配置不一致。新隔离目录只改train_dist_mod.py末尾的配置断言，要求butd_cls/joint_det/detect_intermediate=true，butd/butd_gt/augment_det=false。其余13份Python源码和Nr/Sr初始化JSON保持原字节；不改模型、loss、数据类、优化器或活动ScanRefer源码。

作者输入协议是场景实例框与cls_results预测类别，训练保留原生增强、联合ScanNet检测和表达目标接口。作者脚本的batch/LR/轮数与本项目当前入口并非相同，本次不宣称完整复现作者训练配方。历史Sr3D脚本中出现Nr3D/ScanRefer权重字面路径，本项目使用已经核验并实际加载过的对应Sr3D作者权重；不照抄错配的路径。

作者预训练核心按各自Nr/Sr加载，新G/A/B状态fresh，不叠加ScanRefer终点。当前复制的G/A/B组合是已准备接口，最终完整结构及C开关仍等待ScanRefer控制结果，不能提前称为论文最终模型。

## 原生评估协议的明确边界

原作者bbs评估在butd_cls=true时，先依据预测框与任一输入场景对象框IoU>0.25构造资格，然后将不合格候选分数乘0。这里不是目标root GT IoU资格，也不是新增GT Anchor选择。

当前项目评估器也使用同一任意场景对象重叠条件，但通过候选有效性掩码排除不合格候选；如果全部不合格则记miss。它与作者“分数乘0再排序”不保证逐样本等价，例如分数可能因other_entity项而为负，以及全部不合格时的行为不同。本次不修改评估器，也不宣称该差异不存在。正式baseline与新模型必须使用同一明确协议，作者日志不能代替本地对应评估。

还发现当前bbs路径应用对象过滤，而普通mask_pos路径按未过滤的token汇总分数另选Query；butd_cls=true时二者可能选择不同Query。本次仅进行了源码检查，没有测出真实Nr/Sr发生率。正式完整模型启动前，必须落实同一Query交付Box与Mask的要求，并在对应GPU/真实数据检查中验证。不能将Mask的独立最好选择拼入REC结果。

## 下一步及准入范围

1. 对两行入口断言修改和上述协议说明进行fresh源码复核。
2. 等ScanRefer正常C-off对照完成，确定最终模型；必要的同Query输出修正放在隔离源码，不改活动训练。
3. 使用对应作者权重、真实输入，检查GPU前向、原生loss、模块更新、全模型/Adam/scheduler/RNG保存恢复，再制定明确的同起点训练预算并串行启动Nr/Sr。

本轮没有数据构造、PV前向、criterion、优化器更新、新权重、REC精度或训练状态查询；单seed2027和唯一A100串行规则保持不变。总目标仍为同一完整ScanRefer模型超过59.5%/51%，三项有效贡献及同一完整方法的Nr/Sr独立训练。
