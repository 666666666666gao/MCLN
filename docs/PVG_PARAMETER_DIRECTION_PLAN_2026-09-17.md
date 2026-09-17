# 固定D终点：原生损失通过共享参数的局部方向检查

## 固定问题

§20.215在128个训练场景中未发现最后层CE的logit局部方向使有框错误更严重。本次保持相同终点与输入，检查实际共享参数反传后是否仍成立。不增加网络、修改loss、执行optimizer或选择checkpoint。

## 预先固定

- 沿用native_score_diagnostic_20260917_v2的128行清单、顺序、seed2027、batch8、无增强。全部36665条annotation保留用于原生计数，只解析这128条；逐行核对文本和原始点SHA。不得用错误结果另选训练输入。
- D真实3723步terminal严格恢复。环境/父权重/source port/三个模块不变。沿用实际820个可训练参数张量、28883227参数，冻结参数不加入梯度空间。
- 16次带梯度的eval前向，保持部署输入及BN/Dropout评估状态；这不是原train-mode、增强、AdamW及momentum的重放。
- GT只在模型forward之后进入原完整criterion。通过matcher forward hook记录原criterion真正使用的匹配，取第二个(last_)调用；不改匹配算法。
- 对完整原生总损失Ltotal和其中0.5/7倍的最后层CE Lce分别用autograd.grad取得参数梯度。保留实际模型中未被该损失使用的None梯度，其贡献按0处理；不改变requires_grad列表，不写入参数.grad。
- 每条的root匹配与部署selected由当前前向决定并在求导时固定。margin=s(root_match)-s(selected)。逐行求参数梯度及logit梯度；root与selected同Query时margin恒0，直接记录0，不重复反传。
- 记录vtotal=-grad_theta(margin) dot grad_theta(Ltotal)，vce=-grad_theta(margin) dot grad_theta(Lce)，vremainder=vtotal-vce；以及同一margin在logit空间的CE方向。梯度内积使用float64累计，网络/梯度保持float32。
- 按真实参数名的主模块(Decoder与prediction_heads细到层)记录内积贡献和梯度范数，用于定位来源；不直接将负贡献称为有害任务冲突。也不把CE与其余项的简单差值当作独立反传验证。
- 与上一轮保存输出比较选择、匹配、分数、IoU及点SHA；若图模式/数值变化导致差异，记录当前实际结果，不能重跑挑有利结果。

## 输出与边界

rows.json含128行的当前score/IoU/固定索引、相对旧输出差异、三类参数方向及logit方向；batches.json含16批loss、梯度范数、参数分组贡献摘要和用量。input_selection、脚本、绑定spec、原始日志与失败均保留。结束核验整个state_dict未变、所有参数.grad为空、0optimizer步骤、0正式行和0新权重。

本次只回答固定评估前向和当前共享参数处的局部竞争方向，不测量有限步更新、裁剪/动量/AdamW、框回归后IoU变化、数据增强/Dropout/BN训练态或泛化。正方向不保证修复，负方向也不单独证明总体训练有害。若完整损失仍普遍改善margin，则停止把当前错误归因于该局部训练方向；若出现明确反转，再预先设计一个有对应机制的最小训练控制，不立即调loss权重。
