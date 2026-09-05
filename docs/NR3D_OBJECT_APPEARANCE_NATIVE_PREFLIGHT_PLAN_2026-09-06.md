# 对象点外观的原生Decoder预检

前置证据：修正显式AABB裁剪后，原环境5项CPU测试PASS；完整511场景/16181有效槽
复核无空裁剪、无非正轴，模块Torch predicate与NumPy显式边界逐点一致。
本预检使用同16条M3 fit、16个训练场景，原受保护Nr3D parent、同50,000点输入。
不加载C1/B训练artifact，不与C1同时占用GPU，不读取模块留出或正式验证行。

每批B4，共4批；每批执行原生、零输出外观、固定输出扰动各一次，共12次forward。

1. 原生无新增模块forward，记录六层Decoder Query、seed/采样索引、最后Box、
   soft-token分数、contrastive Query投影、最终源分数、两种原始Mask和alpha。
2. 接入仅最后Decoder的对象外观残差，输出矩阵为零；所有快照须精确一致。
   分别检查原生REC主项和完整原生loss对新增输出矩阵的有限非零梯度。
   输出矩阵为零时，前面4个新增参数张量的REC梯度应为零。
3. 固定输出矩阵为.001倍288×128矩形单位阵。采样、前五层Query、Text Mask及alpha
   须不变；末层Query、soft-token分数和contrastive Query投影须有数值变化。
   记录Box、最终分数、raw Query Mask的变化；REC主项对5个新增张量的梯度须有限，
   每个张量在16行中至少一次非零。完整原生loss梯度也检查有限性。

这里的REC主项严格按当前Nr3D代码取
`(loss_ce + 5*loss_bbox + loss_giou + loss_sem_align)/(num_decoder_layers+1)`。
不改变实际训练损失，不据这16行梯度宣称解决了全数据集任务冲突。

每批移除attachment并恢复输出矩阵零。最终检查全部parent参数/buffer、附加模块状态、
原点hash、源文件、数据和保护权重。0 optimizer更新、0 checkpoint写入，0正式或
模块留出指标。这只证明新视觉证据的原生接入和梯度路径，不证明REC/Mask提升。

入口`scripts/run_nr3d_object_appearance_native_preflight.py`。
先做原Py3.7编译和清单检查；完成当前C1终态核验后，根据其固定筛选结果决定后续
GPU任务顺序。本计划不自动跳过C1通过后的正式验证，也不自动启动对象外观训练。
