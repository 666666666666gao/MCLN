# 候选自身支撑参与原生末层匹配：隔离对照

## 依据与当前边界

§20.376.147已确认：当前直接Mask匹配成本使用给256个Query广播的Text Mask。固定框和语义分数后，该项不区分候选自己的分割支撑；Query Mask仍可通过最终框间接影响匹配。九次合成CPU检查没有真实精度或掉点因果证据。

本目录先实现最小来源对照，不修改活动C-off源目录。当前只准备源码；CPU检查、GPU预检、正式训练及最终方法准入均未完成。下一次活动训练观察仍为2026-10-11 05:00:40北京时间。

## 唯一处理变量

原生 `HungarianMatcher` 新增 `mask_source`，命令行 `--matcher_mask_source` 仅允许 `text` 或 `query`，默认 `text`。

- `text`：保持原生 `pred_masks[idx].squeeze(0)`，作为同源码控制。
- `query`：末层直接成本读取 `sp_pred_masks[idx]`，即候选自己的Query Mask；其后仍用原生 >0 二值判断、点到超点映射、原始点级L1与0.0002系数。

六个早期prefix没有Mask输出，仍按原生语义／几何成本匹配。两组末层均保留分类1、框L1为0、GIoU为2。不换损失、不改Mask权重、不更换监督分母，不加入额外正例或第二套排名。模型前向、参数和原生`last/bbs`输出接口相同，全部256候选保留。

读取原生有效GT集合，为每个真实有效目标做一对一匹配；不将多GT/anchor/detection行全部映射到root。定位、语义、对比及Mask损失沿用原生索引。因为分配可能改变，实际获得各任务监督的Query可能改变；不能说它只影响几何梯度或保证不伤害原已匹配实例。

已知前作：Mask DINO的官方matcher使用候选Mask与GT的CE/Dice成本，并与分类、框及GIoU成本组合。因此“Mask用于Hungarian匹配”不是新增创新。本目录保留PV的原生三维点级硬L1与系数，仅隔离候选自身证据是否形成有用训练责任。

参考：[Mask DINO官方matcher](https://github.com/IDEA-Research/MaskDINO/blob/main/maskdino/modeling/matcher.py)、[官方criterion](https://github.com/IDEA-Research/MaskDINO/blob/main/maskdino/modeling/criterion.py)。此处只借鉴职责组织，没有复制其二维随机采样、CE/Dice实现或分类sigmoid。

## 检查及执行顺序

1. 源码审查：两份修改和十二份未修改源逐字节对照；核对argparse→native factory→matcher、两路Mask形状与原始点映射。
2. 限定CPU工程检查：加载隔离源码，覆盖默认与text分配一致、query确实读取候选Mask、多GT一对一保护、早期prefix不变、原生框loss直接输出梯度落到真实匹配Query。不执行PV前向、真实数据、完整criterion的CUDA路径或优化器；合成结果不能写成定位精度。
3. 活动三轮对照终态后决定是否采用该方向。没有当前GPU准入，也没有额外排队训练控制器。
4. 若采用，真实原生模型／真实增强训练行的GPU预检需核对正常反向、GT职责、完整保存恢复和同起点；不承接预检更新状态。
5. 正式来源对照按相同完整预算：seed2027、B8、三轮，核心及骨干1e-6、新结构1e-5，其他既定参数一致。两组从相同保留E0模型状态、fresh优化器开始；父状态已有此前C适配历史，不能声称从未训练C的独立消融。

## 验收

至少同时检查9508条同一模型的原生双阈值、相对共同起点、相对直接控制，以及已选Query的框/Mask与匹配覆盖。目标仍为Acc@0.25>59.5%、Acc@0.5>51%，对应至少5658/4850个命中，并如实报告修复／破坏；仅超过退化控制不足以宣布新增能力。

当前最好5677/4920继续保留。该小改动只有源码实现，没有精度结论或有效论文贡献结论。三项有效机制和完整模型Nr/Sr训练要求继续保留。
