# 六面边界方案的接口准备

记录时间：2026-10-04。状态：只读官方源码与 CPU 方程核对，尚未实现 PV 新模块，尚未部署或训练。当前 head-only local/whole 实验不受影响；whole 的正式结果仍待发布。

## 已核对的官方来源

D-FINE 固定提交 `956d1709314c2c6a4df6f34de232054578a7449f`。下载内容和 SHA256 见 `SOURCE_IDENTITIES.json`，原 Apache 2.0 LICENSE 随资料保存。

- [Integral / LQE / Decoder](https://github.com/Peterande/D-FINE/blob/956d1709314c2c6a4df6f34de232054578a7449f/src/zoo/dfine/dfine_decoder.py)：Integral 274–295，LQE 298–313，逐层解码 402–444。
- [偏移位置、目标编码和坐标解码](https://github.com/Peterande/D-FINE/blob/956d1709314c2c6a4df6f34de232054578a7449f/src/zoo/dfine/dfine_utils.py)：weighting_function 10–53，translate_gt 56–116，distance2bbox 119–142，bbox2distance 145–169。
- [边界监督与匹配职责](https://github.com/Peterande/D-FINE/blob/956d1709314c2c6a4df6f34de232054578a7449f/src/zoo/dfine/dfine_criterion.py)：loss_local 139–231，匹配并集和分母 245–334，分布监督 498–519。

## 可以借鉴的接口，及必须保留的区别

1. `reg_max=32` 对应 **33 个概率位置**，并非 32 个位置。偏移位置对称、非均匀，中央位置为零。零 logits 的均匀分布具有零期望，可保持正尺寸粗框；这不意味着分布已校准。
2. 非均匀位置的监督由相邻位置的实际坐标距离插值。不能把连续偏移直接取整作为 bin 标签。超出表达范围的目标在原实现中饱和到端点，PV 适配时应记录训练 GT 的覆盖情况，而不自动新增一套范围搜索。
3. 原四边偏移是向外的有符号量。候选尺度参与编码与解码，粗框被作为参考；不能把它误写成绝对世界坐标。PV 如采用六面顺序，需固定为 `x-, y-, z-, x+, y+, z+`，与 GT 编码、解码及统计保持一致。
4. 六面零偏移保留粗框，但 softmax 并不保证相对面有序。已构造出宽度为负的合法偏移组合，仅证明这种参数化的数学限制，**没有测定当前 PV 模型中的发生率**。实际实现必须在参数化阶段明确处理有效尺寸，不能在评估器中补第二套框。
5. 原 FGL 对边界交叉熵使用停止梯度的预测 IoU 权重；四边损失求和后按匹配数归一化。原模型还为定位使用跨层匹配并集。这些都不是 PV 原生监督的等价接口。首个六面实验应保留 PV 原匹配和最终框损失，新增边界项单独记录权重与分母；不同时迁入 D-FINE 匹配并集、定位蒸馏、LQE。
6. LQE 将概率统计变成一个标量并广播到类别 logits。PV 当前 token-softmax 下，相同的 token 标量平移会抵消。后续质量回写需改变实例表示或 token 相关证据；首个边界实验先不改变评分。

## 条件性的下一步

先完成当前 whole 正式评估及 local/whole 比较。若 frozen-G 下的统计摘要仍无法产生有效修正，再考虑方向支撑 token 和明确六面监督；若摘要已有净收益，则先以该收益为基础控制后续变化。质量回写和训练期教师分别后置，不在首轮同时添加。

不把本目录称为已实现六面模型，不将 CPU 检查解释为 GPU 通路或精度证据。公开源码仅用于接口参考，当前训练源码、权重和优化器均未改动。

## 已执行的方程检查

`check_parameterization.py` 只依赖 Python 标准库，结果见 `CPU_PARAMETERIZATION_CHECK.json`：33 个位置单调且对称、五个连续目标插值复原、六面坐标往返、同 token 标量平移抵消，以及软分布不保证有序面的构造例。没有执行 D-FINE 或 PV 模型。

同样本 `29778` 条只遍历一次时，batch8 是 **3723 次更新**，batch16 是 **1862 次更新**；因此调整有效 batch 必须同时重新说明更新预算。目前 GPU 实验仍维持 batch8 / 3723，不改学习率或遍历预算。
