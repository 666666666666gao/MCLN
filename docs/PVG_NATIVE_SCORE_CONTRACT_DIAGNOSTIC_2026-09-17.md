# 原生分类监督与部署bbs排序：固定训练输入诊断

## 问题与历史边界

A/B/C/D均未通过自身起点不退化筛选；D任务矩阵干预确认了路径作用，未证明性能有效。既有V133等已尝试冻结候选上的连续IoU listwise及自由分数残差，本轮不重做该实验、不增加质量头。

本次只检查已训练D的最后层原生soft-token分类监督是否在当前输出处，与部署bbs候选竞争存在具体不一致。Hungarian一对一训练不等同于代码错误；它还承担集合去重与联合回归任务。只能从实测回答哪些候选被监督、当前局部梯度朝什么方向，不能把logit方向直接当成全网络更新或泛化收益。

## 预先固定的范围

- D真实3723步终点；官方父和全部delta严格恢复，与§20.214相同源/环境/权重。所有模型参数保持。
- 原有ScanRefer train fit划分中，按原annotation顺序选前128个不同物理场景各第一条fit表达；不按预测或失败筛选，不读holdout/9508输出。输入清单在第一次forward前落盘。
- seed2027、batch8、关闭增强、原生eval输入处理、16次无梯度模型forward。GT框/Mask/token标签在forward之后才供训练损失分析。
- 使用实际SetCriterion.matcher及loss_pos_align，保持作者cost_class=1/cost_bbox=0/cost_giou=2及原Mask匹配项。最后层soft-token logits detach后单独requires_grad；只求其梯度，不对模型反向、不执行optimizer。
- 部署分数使用原bbs文本跨度规则。几何描述使用评估尺寸clamp和root GT IoU；匹配使用原训练框与Mask规则，两者分开记录。

## 记录与解释

每条记录部署所选Query、Hungarian匹配root Query、root IoU最高Query，三者的分数/IoU/排名及全部256原始候选的IoU和分数。记录各Query是否匹配其他训练目标。

对最后层logit z，原生分类损失L，bbs分数s，计算v_i=-(grad_z L_i) dot (grad_z s_i)。这是在固定匹配和当前logit处，单独沿该分类损失下降时分数的局部方向；best与selected的v差是对应margin局部方向。另按softmax解析导数核验实际autograd，不选择步长或作模型更新。

统计matched root与selected的两阈值覆盖、存在合格框但选择失败的行、合格且未匹配root的候选在分类项下的分数方向。区分另一个目标匹配与完全未匹配。计数仅是这128个已见训练场景的描述，不是正式精度。

禁止从本检查直接决定更换部署score、匹配或loss权重。若原生梯度已纠正大多数排序错误，则本假设缺乏支持；若有可复核的反向作用，再考虑一个改变真实训练责任的最小控制，不能只换名重跑旧质量头。

本轮0训练更新、0正式行、0新checkpoint。结果和失败均归档；A/B/C/D既有结论与晋级要求不变。
