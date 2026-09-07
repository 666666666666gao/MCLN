# 完整预训练点—体素网络的ScanRefer原生微调对照

本对照在完整前向、作者Evaluator和完整GT反向均实际通过后启动。它属于采用现成预训练的原生网络控制，不声称PV-Ground为本项目原创，不复跑已封存的小型外观/范围/冻结读出实验。保护V99及三数据集目标不变。

## 固定起点、输入和训练

- 起点：官方ScanRefer epoch81完整权重，SHA `6f24c67cc3409f44befdef188f14383497a824d8ab309b8fd572a999ae8bdec3`；上轮两步临时更新不保存、不加载。
- 上游源码、隔离环境spec、三个CUDA扩展均与§20.160—20.161相同。不改旧运行环境；Dataset使用已经核验的原项目固定625文件来源，模型/loss/Evaluator使用固定官方PV源，实际导入文件路径和SHA记录到receipt。
- 数据：正确mesh superpoint、原50,000 XYZ/RGB点、GroupFree预测检测框。仅ScanRefer原36665条训练表达，按已有physical-scene划分29778 fit、6887模块留出；保持所有annotation参与原生干扰物计数。GT只在模型输出后用于训练/评估。
- 一次完整fit遍历，batch8、末批2，预计3723次更新，seed2027、2个DataLoader worker。恢复原数据集训练增强及检测框增强；模块留出关闭增强，和旧四行fixture逐项核对基本输入。
- 完整原生GT损失、全部可训练路径：稀疏主干、三层双向编码器、Gumbel、六层Decoder、Box/Mask头。保留作者冻结RoBERTa方式。AdamW，lr/backbone lr均1e-5，weight_decay0.0005，clip0.1，固定LR，无新Loss、Gate或教师。工程反向已确认783个requires_grad张量、27959611参数，其中24个注册参数没有有效loss路径，保留不改。
- batch8先进行一次真实fit输入的全反向容量检查；随后恢复原state、清空梯度/优化器、重设随机状态，才能记录起点评估及正式训练。这次容量检查不作方法性能门。

## 记录与比较

同一原生模型、同一固定输出规则，训练前后分别评估完整6887行模块留出。主指标固定soft-token position/bbs的REC两阈值及其同Query Mask三项；contrastive/bbf独立完整记录，不能根据结果换主模式。尺寸处理沿用作者`clamp(min=1e-6)`，Scan不增加对象框重叠过滤。

保存每行对应、实际选择、root框、IoU、Mask IoU、所有256个原框及两种原生分数，以支持回归变化与Query选择变化的后续分解。每个阶段汇总必须与作者完整Evaluator一致。记录整个fit顺序，确认所有29778行恰好出现一次，且无holdout行用于更新。

此集合的场景在作者预训练时已见，绝对准确率不构成新场景泛化证据。固定末步是唯一训练终点，不挑中间epoch、均值或最好checkpoint。先检查主REC两项相对原生起点是否不退化；该对照若明显退化则保留负结果，不扫描LR/额外遍历。可继续的终点需按固定9508正式路径独立评价，才能判断是否达到V99的5572/4797及Scan Mask58.70/50.70/44.72底线；不能用模块留出成绩晋级Nr/Sr。

## 持久化与磁盘

仅新实验目录写入。保存一个定期更新的latest恢复文件及固定terminal，内容为原预训练state的可训练部分/变化buffer、AdamW和随机状态，并绑定原始checkpoint SHA；不复制冻结RoBERTa大权重。中间checkpoint用于故障续跑，不参与选优。现有保护权重不改。

按实测batch2全训练约1.30秒（首批CUDA初始化9.16秒）、峰值5.08GB估计，batch8完整fit可能需要数小时；以实际batch8记录更新ETA。后台screen持久运行，按180—300秒或接近预计完成时观察；网络观察超时不重启原进程。

本次只做预训练原生控制，正式REC和论文创新尚未得到新结论。达到Scan正式保护线后按用户优先级立即推进同结构Nr3D/Sr3D REC，Mask不再作为Nr/Sr晋级门。
