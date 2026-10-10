# Nr3D/Sr3D原生两批训练及完整恢复检查：源码准备

本目录承接§20.376.143的15份隔离源码，原字节不改。对应作者权重仍按各自Nr/Sr加载，新G/A/B参数fresh，不叠加ScanRefer状态。两份新初始化清单仅关闭C。该选择是尚待ScanRefer对照支持的候选版本，不是已确定的论文最终模型。

当前PREPARATION_PROTOCOL.json的final_method_admitted为false；脚本在导入Torch、构造模型或接触CUDA前要求该门槛成立。此布尔值本身不是准入证据。没有启动器、自动队列、远端部署或GPU执行；不能通过单独修改它绕过活动ScanRefer终态、最终结构选择、fresh源码审查和资源检查。

## 实际GPU检查将怎样做

使用原生TrainTester.get_loaders正常加载完整train/val数据，不设debug/eval/eval_train。原debug会将训练split改为val，因此不能用于真实训练预检。训练集保留原生点云、颜色、框增强和10倍ScanNet检测混合。真实全量数据行数与每轮更新数在将来的执行中记录，不用6887、9508或有限128条检查代填Nr/Sr数量。

从真实训练数据取8条表达和8条检测，交错组成两批B8；这不是原sampler的一整轮或完整预算。Sr3D的首条表达明确选择有辅助实体及anchor的真实训练行，检查非空双GT；检测行也必须有多个有效GT。GPU预检只调用原生训练循环处理这两批，保持原生前向、criterion、AdamW、裁剪和scheduler。没有合成GT、替换训练目标、复制预检权重继续正式fit或全量精度声明。

观察原生criterion七个prefix调用，记录last_的原Hungarian匹配。逐行要求有效GT均有对应，G额外集合排除所有原匹配Query和联合检测行；检查G修正对未入选logit的直接梯度为0。这不保证共享参数更新后其他预测绝不改变，也不是C的过滤部署等价检查。

分别记录原核心、骨干、G读取、Mask支撑和span头的两步梯度及实际参数变化；零输出初始化下内部第一步可为零，span内部按两步合计检查。完整1295状态、Adam、scheduler及Python/NumPy/Torch/CUDA RNG必须实际保存并重建重载，且逐项一致。保存的工程checkpoint不附正式指标，不能替代保留最好。

## 等待条件

当前唯一A100仍由ScanRefer正常C-off对照占用，原观察器21631/PID48772于10月11日01:03:05首次读取；本轮不提前轮询或改活动源码。如果最终保留C或改变A/B结构，本目录须按最终方法更新并重新审核相关差异，不能把此C-off准备当成完整最终模型准入。

仅在最终结构明确、当前GPU任务终态与资源实际核验后，使用独立目录和原单卡锁串行执行Nr/Sr工程检查，再制定/执行对应作者起点的完整训练。作者原配方的LR/batch/轮数与此检查用的项目配方不同，不能称完整复现作者训练。单seed2027、同Query Box/Mask、一套原生bbs及最高权重保护保持。

本目录仅SOURCE_PREPARATION；数据构造、前向、loss、优化器、新权重、GPU、正式成绩及当前训练查询均尚未发生。
