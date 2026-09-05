# P2输入接入位置与最小对照边界

这是对main `35f9820e377cd31e519bc451c1d1d196fa682bf4`的只读接入审计，尚未实现或训练P2。
性能决策仍等待G0与两个P1真实前向回执。代码行号以该提交为准。

| 输入 | 当前实际位置／语义 | 首个P2比较需共同固定的内容 |
|---|---|---|
| Query表征 | `models/mcln.py:2077–2082`的`decoder_query_last`，完整`[B,256,288]`；`2136`之后的`query_last`已进入Mask投影 | 两组都取Mask投影之前的同一Query；保留原Query轴映射 |
| 文本序列 | `models/mcln.py:1962`的`end_points['text_memory']`，cross-encoder之后的`[B,L,288]`，也是Decoder接收的`text_feats` | 两组同一完整序列，旧读出对其池化，新读出让候选对读取token；池化屏蔽规则需按P1结论统一 |
| 易混淆的其他文本 | `end_points['text_feats']`是cross-encoder之前的投影输出；`proj_tokens`是64维对比投影 | 不把新组换文本阶段、换维度带来的差异一起计入关系读出贡献 |
| 最终框 | `last_center`与`last_pred_size`，`models/mcln.py:2068`生成 | 两组都使用同一最终框，仅用于几何证据，不在首版更新Box／Mask |
| 现有Decoder几何 | `BiDecoderLayer:470–487`由传入`query_pos`取中心算5维关系；`query_pos`是此前proposal／前层预测框；6维框还经过位置嵌入进入q/k | 不能声称现有网络完全不读取尺寸；两组若在最终框后比较，需共同使用该阶段的几何，而不是旧组前层框、新组最终框 |
| 几何符号 | `calc_pairwise_locs:17–18`是`center_i - center_j`，没有距离归一化 | 固定目标i／记忆j方向，保留同一5维输入；首版不混入尺寸比或新坐标系 |
| 目标预选分数 | `compute_default_source_scores`，含现有parser成分图的Default分数 | 同一合法候选内选Top-32；分数只用于同一候选预选，首版最终决策仅一条直接读出路径 |
| 合法候选 | `build_mcln_source_choice_batch:159–161`当前创建的是全True mask；它不是REC的对象重叠合法性 | 不直接把这个同名mask当作训练／部署共有的合法集合；显式复用现有`build_detector_overlap_valid`规则 |
| 过滤所需对象框 | `TrainTester._get_inputs:3446–3447`将batch的`all_detected_boxes/mask`原样交给`inputs['det_boxes']/['det_bbox_label_mask']` | 这些现有输入足以计算相同过滤；无需把root目标框、target ID或IoU标签传入网络 |
| Anchor记忆 | 完整256 Query，按同一对象重叠合法性屏蔽；不复用目标Top-32截断 | 两组保持同一记忆范围／无关系状态；当前Nr3D加载器无可靠Anchor实例GT，覆盖诊断仅称可用性代理 |

当前普通Nr3D输入还保留数据管线追加的`. not mentioned`；首版不顺手修改token序列或将该文本后缀当作已标注的“无参照关系”真值。

已有CPU探针证明旧`MultiHeadAttentionSpatial`支持32目标读取256记忆，等价于完整计算后取同样32行，
见`SPATIAL_TARGET_MEMORY_PROBE_20260905.json`。这只解决旧机制接口复用，不代表新的关系证据已有效。

`audit_rec_mask_selection.py`的v3合成反例同时调用真实SourceChoice adapter、现有对象重叠过滤和实际REC/Mask evaluator：
adapter允许的集合与REC允许的集合可以不同，即使对象框已经提供给adapter。远端CPU已通过全部断言：
adapter mask为`[True,True]`，REC对象重叠mask为`[False,True]`；无共享Mask过滤时REC选1、Mask选0。
原有候选诊断也与实际evaluator输出一致，回执为`REC_MASK_SELECTION_COUNTEREXAMPLE_V3_20260905.json`，
script SHA `6c4af3a514621f5c94eac4c6127f45157c1e26a460f8c5cef3947614d6e4bab1`。
实际执行的adapter与filter文件原始字节SHA均与当前本地源文件一致，未做源码替换。
这不量化受保护Nr3D的影响频率，也不支持直接改变正式Mask过滤规则。

## 训练框扰动与过滤的额外约束（2026-09-05）

`Joint3DDataset._get_target_boxes:1112–1113`与`_get_scene_objects:1152–1153`在train且augment开启时，
分别对中心和尺寸乘独立的`[.95,1.05)`随机量；`butd_cls`随后将scene对象框作为提议框。
两个取框函数与原MCLN固定提交的AST相同，见`BOX_JITTER_UPSTREAM_SOURCE_AUDIT_20260905.json`，
不能据此归因为新增模块引入的问题。

CPU探针实际调用这两个函数与当前过滤器，以中心`[5,0,0]`、尺寸`[.1,.1,.1]`的合成对象、
seed0、64次扰动为例，把每次root GT自身作为候选：关闭增强时64个均保留；开启增强时45个
未通过提议框IoU>.25过滤。两处框中心最大差为`.41901398`。这是合成机制反例，**45/64不是Nr3D发生率**。
回执为`TARGET_PROPOSAL_JITTER_COUNTEREXAMPLE_20260905.json`；原G0代码及增强策略未修改。

因此，首轮冻结候选读出比较应把候选产生时的模型模式和augmentation一并登记并在两组保持一致，
不能只让同名mask或过滤函数相同。若使用eval、augment=False的冻结候选，且root实例确实存在于
有效对象提议中，则它与root GT来自同一框：任意IoU>.25正确候选必然与至少一个有效提议IoU>.25。
这个条件下Full-256正确候选不会被该过滤删除；若实测违反，应先检查root提议存在性、框阶段和映射。
该条件命题不等于已核验全部真实行，也不授权在G0中同时修正框扰动。
