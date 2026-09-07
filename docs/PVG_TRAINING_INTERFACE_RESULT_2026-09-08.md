# 完整预训练网络的原生评分与GT反向检查结果

2026-09-08 07:32:05 CST，固定四行输入的检查完整退出0。作者Evaluator、两次完整训练反向及AdamW更新均通过；07:50:39 CST的本地独立重计也通过。此项为训练接口证据，正式评估行数为0，不产生新的模型成绩。

## 原生输出规则

固定PV-Ground提交`262e2592589baec7bb83a0d46aae6542d4ccedfb`的`main_utils.py:616–618`在损失计算后、Evaluator之前，对所有预测尺寸执行`clamp(min=1e-6)`。ScanRefer的`butd_cls=False`，不使用对象框重叠过滤。§20.161中197个包含非正尺寸的原始Query是1024个原始框中的统计，不能当作197条表达失败；也不能额外过滤这些框来替换作者协议。

两个原生模式分别为soft-token position/bbs和contrastive/bbf。它们均读取“主实体＋修饰＋代词＋关系－其他实体”的位置图；主实体图二值化，其他图保留实际值。对比响应使用Query-token相似度、温度0.07和softmax，再补齐到256维。Mask分别使用各模式选择的Query，将Text Mask与该Query Mask按原alpha融合、sigmoid后以0.5阈值映射回原始50000点。

本次实际执行完整作者GroundingEvaluator，并用显式评分、轴对齐框IoU及原始点Mask计算对照。两模式、两阈值、Top-1/5/10的12项REC计数及两项Mask IoU总和共14项，本地重新读取逐行结果重计后与Evaluator一致。四个训练样本不用于选择模式或声称准确率。

## 输入、完整训练和梯度

使用同一fit列表的行0、173、237、455，四个不同物理场景，每批两行。新增监督独立存入`labels`，模型输入的点、检测框、预测类别、superpoint及文本与上一轮无GT输入fixture逐项SHA一致。GT Mask仅将二值int64存储改为bool，标签值不变；GT框、Mask和token标签在forward完成后才用于loss和Evaluator。

从官方ScanRefer epoch81权重重新严格加载1234个state张量，沿用已验证的固定position_ids保存格式适配。完整SetCriterion和Hungarian损失保留中间层及最终层Box、soft-token、contrastive、Query/Text Mask路径。未加新Loss、Gate或读出器。

| 检查项 | 实测结果 |
|---|---:|
| `requires_grad`参数张量 | 783 |
| 可训练参数 | 27,959,611 |
| 每步实际有梯度的参数张量 | 759 |
| 每步无梯度的注册参数张量 | 24 |
| 未改变的冻结RoBERTa参数张量 | 199 |
| 改变的buffer张量 | 252 |
| 第1步完整前向、反向、更新 | 9.15572秒 |
| 第2步完整前向、反向、更新 | 1.29969秒 |
| 两步最大allocated显存 | 5,084,949,504字节 |

稀疏主干、双向编码器、Gumbel、Decoder、Box及Mask投影都有非零梯度。所有损失和实际梯度有限。AdamW的核心/主干LR均1e-5、weight_decay0.0005、clip0.1；裁剪前梯度范数为108.33858和154.37651。

24个无有效loss梯度的注册参数在两步完全相同：三个`swa_layers.*.norm`的6个参数，其forward调用LayerNorm后未使用返回值；`text_query_proj`的6个参数没有被forward使用；六层`decoder.*.norm1`的12个参数未被当前BiDecoder路径使用。这些是固定上游的实际行为，本次不顺带修复或改变其网络。

两步临时更新未保存checkpoint，随后释放模型。官方权重SHA仍为`6f24c67cc3409f44befdef188f14383497a824d8ab309b8fd572a999ae8bdec3`。后续微调必须重新加载该原始权重，不能使用此两步状态作为起点。

## 来源与后续

隔离环境spec仍为`966235b2ead7fa5a1de63e9537745b457374a4e5749ac4a7a26566b7b630c82c`，不重建、不升级保护环境。核验脚本SHA为`5fbc20943a9f9eb8c6494ca52bd742985ea61826b99f83eafd05dfb3cd2cc93d`，终态receipt SHA为`fe744df3c6f70d2b0e4539c560d55786fb0baac572763052839ed15609f0acb0`。

完整证据位于`refine-logs/pvground_training_interface_20260908_v1/`，包括固定plan/spec/source、输入回执、原始日志及退出码、逐行评分接口、逐模块梯度和本地重计。此检查不替代6887条模块留出和9508条正式评估。

下一步按`PVG_SCANREFER_FINETUNE_PLAN_2026-09-08.md`执行一次完整fit遍历的原生预训练微调控制。保护V99及Scan优先、Nr/Sr REC后续目标不变；直接采用外部完整预训练网络不作为本项目原创。
