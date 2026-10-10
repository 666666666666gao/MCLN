## §20.376.148 — 候选自身Mask参与原生训练匹配的隔离源码

依据§147的限定路径诊断，已实现训练阶段直接Mask成本的来源对照。源码放在独立query_mask_assignment_20261011/source目录，没有修改当前C-off模型源或正在运行的配方；当前仅SOURCE_ONLY，不是新训练启动或精度结果。

十四份共同源中，十二份逐字节保留，仅main_utils.py与models/losses.py变化。新增--matcher_mask_source参数，默认text、另一选项query；原生factory将它显式传给HungarianMatcher。text仍读取公共Text Mask，query读取同候选的sp_pred_masks。其后的>0硬二值判断、超点到原始点映射、点级L1、0.0002系数、分类1/框L1为0/GIoU为2均保留。

正常PV的proposal与0head至4head没有Mask输出，其匹配路径不变；末层才具备自身Query Mask。本实现保留原生全部有效GT的一对一匹配，不按GT槽0将多目标或anchor全部改为root。原生定位、语义、对比、Mask损失及G修正保持；但分配变化可能改变各任务被监督的Query，因此不能说只影响几何，也不能保证原正确候选绝不退化。

新增模型参数为0，没有修改模型前向或last/bbs评分，保留256候选，也没有新增推理GT、候选裁剪或另一套排名。当前C仍关闭；若正式采用，该比较仍是同一已适配父状态上的增量实验，父状态已有历史C训练，不能声称从未训练C的从头消融。

Mask DINO官方matcher已使用候选Mask与GT的CE/Dice成本并与分类及框成本组合；本次不是首次Mask参与匹配，也没有复制其二维采样或sigmoid分类。此处保留PV现有三维点成员与成本形式，只隔离“候选自身支撑是否参与定位责任”。参考[官方实现](https://github.com/IDEA-Research/MaskDINO/blob/main/maskdino/modeling/matcher.py)。它目前是问题导向的训练对照，不是已经验证有效的第三个论文贡献。

新上下文源码审查WARN/0阻断，核对了两份修改、十二份未改源、argparse到factory到matcher、实际warm modules与dataset绑定及早期prefix。审查为same-family/provisional，实际模型身份和推理档位UNATTESTED；不能将静态审查写成实际CPU反向、GPU预检或正式评估通过。

下一步先进行限定原生CPU工程检查：默认/text控制、query实际使用自身Mask、多GT及无Mask早期prefix、真实原生框loss对应的直接输出梯度。合成输入只用于实现验证；当前尚未执行。之后仍须等待当前三轮C-off终态，决定是否安排真实数据GPU预检与同起点同预算正常训练；没有创建额外GPU控制器或排队任务。

若采用，明确使用同一保留E0、fresh优化器、seed2027、B8、三轮及现行学习率，9508条原生同模型评估。既看直接控制，也看是否超过起点5677/4920；只优于退化控制不足以证明新增能力。当前最好权重、父链及V99继续保护。

本节0模型前向、0数据行、0criterion执行、0优化器更新、0GPU及0当前训练状态读取；没有ScanRefer/Nr/Sr新增指标。唯一原观察器52851/PID51540仍按2026年10月11日05:00:40检查正常第2轮，三有效机制与完整Nr/Sr正式训练仍未完成，目标ACTIVE_UNMET。
