# 当前PV末层匹配的Mask来源：源码范围澄清

本记录只读取与实际R4清单SHA一致的源码，未读远端进度、运行新网络、改活动实验或统计真实匹配成本矩阵。

`models/pv_ground.py:549`从选定文本token生成一个Text Mask，550将其expand到256个候选；560/561分别保存自身Query Mask与广播Text Mask。570–574的A更正写回自身Query Mask。

`models/losses.py:854`区分多Mask路径；在当前PV末层路径，911把Text结果传为pred_masks，912把自身Query结果传为sp_pred_masks。`HungarianMatcher.forward`的328–342只读前者，按logit>0变成点级前景，再用L1/Hamming和GT比较，系数294为0.0002。自身Query Mask仍接受分割损失，不能写成它完全没有训练。

对某场景的一个GT列，广播Text Mask在256个Query行相同，所以该项成本是列常数，不能区分这些候选。当前constructor在`main_utils.py:291`为class1、bbox0、GIoU2；原生匹配并非整体常量，其他成本仍区分候选。

实际PV的proposal/decoder预测头源码也已绑定同一R4 SHA；该头不产生proposal_pred_masks。本分析不推广到其他MCLN多Mask分支或所有原作者模型版本。

这指出的是自身支撑与当前匹配接口的具体差异，不证明更换来源一定涨点、不是所有退化的唯一原因，也不能将普通Mask匹配声称为首次提出。当前selected_geometry训练保持原规则；若后续需要调整责任分配，应在真实同起点控制中检验具体来源与原生损失尺度，而不是同时改推理排名、匹配和多个监督项。
