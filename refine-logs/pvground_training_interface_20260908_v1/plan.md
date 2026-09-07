# PV-Ground原生评分与完整训练接口检查

此项沿用§20.160—20.161已经实际通过的隔离环境和官方ScanRefer完整权重。环境spec不变，直接复用；不重建、不重复环境验收。目标是进入完整预训练训练路径，保持三数据集目标和ScanRefer保护结果。

## 已定位的输出规则

固定上游提交`262e2592589baec7bb83a0d46aae6542d4ccedfb`的`main_utils.py:616`在计算损失后将所有`pred_size`截到`1e-6`，随后交给`GroundingEvaluator`。其ScanRefer配置`butd_cls=False`，不启用按对象框重叠过滤。上一轮197个非正尺寸原始Query并非197条独立表达失败，也不能自创正尺寸过滤替代作者评估。

最终原生分数有soft-token和contrastive两个模式，均使用主实体＋修饰＋代词＋关系－其他实体的文本位置图。主实体图在评估中二值化；其他图保留原值。原生Mask按对应模式选择Query，再融合该Query Mask与Text Mask。此检查记录两种模式，不在四个样本上选择新规则或宣布精度提升。

## 固定执行范围

1. 读取已有fit列表的同四行0、173、237、455。重新取得GT只放在独立监督字典；逐张量核对模型输入与上一轮fixture完全相同。GT二值Mask以bool存盘，值不变。仍无增强、seed2027、每批两行。
2. 严格加载全部1234个state张量，沿用已验证的固定position_ids保存格式适配。保留作者PVGround、DataProcessor、完整SetCriterion与compute_hungarian_loss，不删Mask或中间Decoder损失。
3. 先做两批eval：真实调用作者完整Evaluator；按作者位置图和尺寸截断规则独立记录最终Query、IoU、Mask和原始非正尺寸情况。核对汇总结果一致。四行只是执行与接口验证，不能参与方法筛选。
4. 再做两批完整train/反向/AdamW更新，原RoBERTa冻结方式不变，其余沿用模型实际requires_grad。工程检查lr与backbone lr固定`1e-5`，其余AdamW参数与clip_norm来自checkpoint配置。不增加新损失、不检查过拟合比率或用这两步判定方法有效。
5. 记录实际可训练参数、逐模块有效梯度、零/缺失梯度、参数变化、BN变化、损失分项、耗时和显存。损失及出现的梯度必须有限；稀疏主干、编码器、Gumbel、Decoder、Box与Mask路径必须有实际梯度。无学习权重或checkpoint落盘，两步临时模型随后释放。
6. 输入GT在forward完成后才进入loss/evaluator；保护权重和官方checkpoint文件SHA不变。保存失败日志与退出码。发生实际工程错误则先查具体源码和算子，不改训练目标来让检查通过。

通过后用测得的完整反向速度制定ScanRefer短周期训练预算。训练模型必须重新从原checkpoint开始；两步临时状态不成为实验起点。原生预训练控制与新方法的比较仍需独立训练和正式评价，当前没有新增SOTA或三数据集晋级结论。
