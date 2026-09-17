# 固定 D 终点共享参数方向诊断：独立静态与终态产物审计

日期：2026-09-17。审计范围仅为 `PVG_PARAMETER_DIRECTION_PLAN_2026-09-17.md` 及其诊断实现、绑定来源和上一轮原生 logit 结果。

- `audit_skill: experiment-audit`
- `review_independence: same-family`
- `acceptance_status: provisional`
- `reviewer: fresh Codex reviewer; gpt-6-astra / max route`
- `agent: /root/parameter_direction_audit`
- `review_type: static source review, completed runtime evidence, independent deterministic export recount`
- `overall_verdict: WARN`
- `integrity_status: warn`
- `execution_verdict: PASS for the completed bounded diagnostic; raw autograd not independently re-executed`
- `artifact_recount: PASS`

本报告由读取原始源码和本地文件的新审计上下文编写。审计者没有运行远端命令，没有执行模型诊断、训练或优化，没有改动 driver、模型、loss 或授权配置。静态初审时完整输出尚未存在；随后收到 `refine-logs/pvground_parameter_direction_20260917_v1/` 的完整终态产物，已独立核查退出码、日志、文件 SHA、JSON 数值与 NPZ 候选导出。下文包含终态补充；执行与导出复算 PASS，整体 WARN 保留的是解释及独立重算范围限制。

## 结论与直接影响

没有发现 criterion 键名错误、matcher hook 顺序错误、最后层 CE 系数错误、梯度内积符号错误或把 GT 输入模型的源码证据。实际运行于 2026-09-17 16:53:24 CST 完成、退出 0，16 批完整日志与输出证明本次完整 CUDA 图的预定 retained passes 已执行完成。独立导出复算得到：在 root 覆盖错误中，0.25 阈值的完整 loss 方向为 12 正/2 负，0.5 阈值为 17 正/4 负；不存在“所有错误在共享参数空间均改善”的结果。

该诊断测量固定 D 权重、固定 128 个 fit 场景、batch 8、seed 2027、eval 模式下，**原生 autograd 所定义的固定图方向**。它不重放原训练中增强、Dropout/BN、裁剪、AdamW、momentum、weight decay 或离散/stop-gradient 路径重算的效果。即使完整 loss 的所有方向均为正，也只能排除这一个固定条件下的负方向解释，不能排除原训练过程中的干扰机制。

仍需保留的限定有四项：

1. 旧结果只保存了 scene/text、原始点 SHA、预测摘要，缺少可逐行对照的 GT root_box/target_id、检测框及 token-map 指纹，不能称全部输入与 GT 均已逐字节复核。
2. 实际 `GumbelSampling` 在 eval 模式仍无条件调用 `F.gumbel_softmax`，结果依赖固定 RNG 实现，不能解释为随机采样的期望。
3. 模型中原生 geometry detach、离散选择及 no-grad 观测保留；梯度内积对应这些路径固定时的 autograd 图，不能直接当作完整前向重算函数的有限差分导数。
4. `vremainder = vtotal - vce` 包含其余层 CE、box/GIoU、语义对齐、真实 mask 监督及原生 mask 自一致性项。它不是“纯几何 loss”或一个已独立反传的任务。

## A. Ground Truth Provenance — PASS，跨轮身份核验有 WARN

`joint_det_dataset.py:595–597,633–648` 从 ScanRefer annotation 读取 `scene_id/object_id/token`，不是由 PV-Ground 的预测生成目标。`joint_det_dataset.py:1086–1119` 将 root target 放在索引 0，再按原规则附加 anchor；box 来自 `scan.get_object_bbox(tid)`，mask 来自对象所属点。`center_label/size_gts/gt_masks` 在 `1387–1392` 写入 batch。该临时文件的完整路径为 `D:/Program Files/UserCache/gb/codex/tmp/pvg_loss_20260917/joint_det_dataset.py`。

driver 在 `scripts/diagnose_pvground_parameter_direction.py:185–189` 向模型传递点/体素、文本、BUTD 预测检测信息与 superpoint，`butd_gt=False/butd_cls=False` 在 `85,142–144` 固定。`189` 完成模型前向之后才在 `198–204` 合入含监督标签的 batch 并调用 criterion。score 使用 root 文本图的正/修饰/代词/关系/其他实体映射（`192–196`），这些是原部署 score 的文本解析权重，不是由预测框生成的目标。

criterion 的目标来自 batch 的 GT box/mask/map（实际 `models/losses.py:856–885`）。root 匹配通过原 matcher 获得，driver `220` 同时要求 target index 0 存在且 root box label 有效。masked target 保持顺序，因此 `target_ids==0` 对应 root，而非重新寻找一个“表现较好”的任意 GT。

输入身份方面，driver `117–150` 沿用参考清单，验证 128 唯一行、128 个物理场景、fit/holdout 划分、scene/text；`180–183` 验证体素化前后原始点次序及旧记录原始点 SHA。`FitDataset._scene_graph_parse` 只解析所选行但保留全部 36665 annotations（`121–146`）；原 loader 仍在完整 annotations 上计算 distractor/unique（dataset `677,692–732`），符合方案。

**限定：**参考 `rows.json` 的字段没有 `root_box`、`target_id`、检测框/检测标签/superpoint/token-map SHA；`input_selection.json` 每行仅有 row_id/scan_id/text。当前 driver 写出 root_box（`239`），但没有旧 GT 可直接逐行比较。原始点 SHA 不能证明 GT annotation、检测框或语言 token-map 都完全相同。driver 确实调用 `verify_scanrefer_superpoints` 并验证来源 manifest（`63–72,141`），本次审计没有独立展开所有数据源字节；不得把源码绑定、点字节一致扩大成完整输入字节一致。

完整 criterion 中 `corresponding_loss_*` 的 target 是阈值化预测 `sp_src_masks_2`（loss `592,611–621`），且 BCE 权重由 detached 预测构造（`618–620`）。这是本来存在的训练自一致性项，不构成 root 评估 GT 伪造；但必须和真实 mask 标签监督（`579–581,626–627`）区分。

## B. Score Normalization 与公式 — PASS

部署分数为 token softmax 与 root 文本权重的加权和：

`s(q) = sum_t softmax(z_q)_t * [1(positive_map_t>0) + modify_t + pron_t + rel_t - other_t]`。

driver `192–197` 与训练留存 evaluator 对照式 `train.py:280–284` 一致。没有按模型输出最大值/最小值/平均值重新缩放 metric，也没有把 score 或梯度内积转成伪准确率。softmax 是原模型 score 的定义，不是用模型自身最大分数归一化结果。IoU 使用真实 GT root box，原生 box conversion 与固定正尺寸 clamp（driver `217–223`，loss `35–45,70–75`）；阈值仍为严格 `> .25` 与 `> .5`（driver `264–273`）。

`main_utils.py:268–280` 返回实际 `compute_hungarian_loss` 与包含 boxes/labels/masks/contrastive_align 的 criterion。ScanRefer 下原总损失为（每项先按原函数构造，`sum_h` 覆盖 7 个预测头）：

`Ltotal = [0.5*sum_h CE_h + 10*sum_h bbox_h + 2*sum_h GIoU_h + 0.5*sum_h align_h]/7 + 10*mask + 2*dice + 5*sp_mask + sp_dice + 10*corresponding_mask + 2*corresponding_dice + 10*adaptive_mask + 2*adaptive_dice`。

证据：loss `888–955`。`query_points_generation_loss` 在 `936–941` 可被计算并在 `961` 记录，但实际 `947–956` 返回的总损失不包含它。诊断按原返回值取完整 loss 是正确的；不得另称“所有已计算的 auxiliary loss 均参与本次总梯度”。

最后层 CE 在 loss `919` 以 `f'{prefix}_{loss_key}'` 保存，因此真实键名为 **`last__loss_ce`**。driver `208` 的 `0.5/7` 是原总 loss 中该项的系数，未重复除 target 数；原 CE 自身已经除以该批匹配 target 数（loss `511,831–835`）。这也是 batch 级梯度，而非单条 CE 梯度。

固定 `matched_root/selected` 后，`m = s(root)-s(selected)`；driver `209–243` 实现：

- `vtotal = -<grad_theta(m), grad_theta(Ltotal)>`
- `vce = -<grad_theta(m), grad_theta((0.5/7)*CE_last)>`
- `vremainder = vtotal - vce`
- `vlogit_ce = -<grad_z(m), grad_z((0.5/7)*CE_last)>`

因此正号表示在该 native autograd 图、当前 batch 梯度与固定索引下增加 root-minus-selected margin。相同 query 时 `m` 恒为 0，直接记录 0（`225–226`）正确。每个内积先把 float32 梯度转为 float64 再乘和累积（`162–167`）；结果是原始内积，不是 cosine，也没有用梯度范数掩盖大小。

**跨轮比较尺度：**旧已归档 `refine-logs/pvground_native_score_diagnostic_20260917_v2/diagnose.py:189–190,214,239` 使用未乘总 loss 系数的单层 CE。新 `logit_last_ce_velocity` 在完全相同 logits/maps/matches 下，应与旧 `matched_margin_velocity * (0.5/7)` 对应，允许 float32 求和次序造成的数值差异。不能直接把两轮原始绝对值做等值比较。

## C. Result File Existence 与数值追踪 — PASS（限已存在的终态与导出记录）

本地已完成以下确定性检查：

- 旧 `rows.json` 与 `input_selection.json` 的 SHA 分别精确等于旧 `diagnostic.json:15,9`。
- 旧 rows 有 128 个唯一 row_id，selection 有 128 个唯一物理场景，row 顺序一致，所有 point SHA 是 64 位十六进制。
- 从旧 128 行重算 `diagnostic.json:23–45` 的全部阈值摘要，所有字段一致；选择公式差异行数重算为 0。
- 当前 staged spec 中所有 10 个 `files` 条目均与本地字节 SHA 一致，包括 driver、restore helper、方案、3 个 reference 文件和3个模型模块。
- 当前 source_port 的 SHA 等于 D training spec 中固定值；实际 losses/main_utils/pv_ground/encoder_decoder_layers/modules 的本地字节 SHA 均匹配当前 port。

旧结果的已验证摘要仅归属于旧 **logit** 诊断：

| 阈值 | selected 命中 | root 匹配命中 | raw oracle 命中 | 有好候选的错误 | 其中 root margin 正向 |
|---|---:|---:|---:|---:|---:|
| >0.25 | 114/128 | 128/128 | 128/128 | 14 | 14 |
| >0.5 | 106/128 | 127/128 | 127/128 | 21 | 21 |

旧结果本身明确写明 `not shared-parameter or total-loss update`（旧 diagnostic `16`）。这组数字不是当前共享参数检查的结果，不能沿用为新诊断的 PASS。

终态补充已核查新 `diagnostic.json:8–13,23–25`、`controller.exit:1` 和 `run.log:7–23`：实际 16 批完成、32 次 loss 参数反向、72 次非恒零 margin 反向，state unchanged 与 all gradients none 的成功断言已通过，原始日志含 16 条 batch sentinel 与 1 条 complete sentinel。下文给出独立导出复算细节；没有保存原始参数梯度或全部 GT/候选 box，因此原始梯度和 GT IoU 不能由导出独立重算。

## D. Call Path、Hook 与反向图 — PASS（源码与本次完整执行）

matcher 顺序已经逐行核对：loss `852–853` 构造 `proposal_, last_, 0head_, 1head_, 2head_, 3head_, 4head_`；每个 SetCriterion.forward 只有 `817` 的一次实际 matcher 调用，`828` 的辅助匹配已注释。driver `200–207` 在完整 criterion 上挂原 matcher forward hook，要求 7 次输出并取第二次，实际对应 last_，未自行重匹配或更换算法。

批级记录请求的全部 12 个 aggregate loss 键在原 loss `957–971` 存在；不存在 `loss_ce`/`last__loss_ce` 混用或 mask 字段名字猜错的问题。参数梯度、CE logit 梯度、逐条 margin 梯度实际都在 batch 循环内调用（driver `209–213,228–233`），结果流入 row/group/batch 记录（`236–257`）和终态写出（`274–289`）。没有发现定义后从不调用的诊断指标。

一个 batch 只做一次前向，然后顺序求总 loss、最后层 CE、最多 8 个非恒零 margin 的梯度；所有请求均 `retain_graph=True, allow_unused=True`，未启用 create_graph（`209–210,228`）。这是重复的一阶反向，不需要二阶梯度。未被某个 loss 使用的 trainable parameter 保留在相同 tuple 中，None 项在内积时按 0 处理（`154–166`），没有为了让梯度可用临时解冻模型。

代码没有 `optimizer.step`、`backward()` 累积参数 `.grad` 或参数写入。每批检查 `.grad` 为空（`216`），终态 CPU 逐项比较全部 state_dict 与恢复值并再次检查 `.grad`（`260–262`）。完整运行已越过这些断言并写出 complete；“模型全状态未变”由已绑定代码的成功断言支持，而非审计者另外取回完整权重再次比较。

**运行限定：**源码中没有发现会必然破坏重复反向的操作；本次真实完整运行证明预定 retained passes 在该现场图上成功，不能将这个结论推广到任意二阶梯度或其他配置。旧 `pvg_observation_source_20260909/pointnet2_utils.py` 的 SHA 不是当前 port 中 PV-Ground `pointnet2/pointnet2_utils.py` 的 SHA，二者还是不同路径的算子；没有将它作为当前全部底层反向实现已审计的证据。

## E. Scope、参数身份与归因 — WARN

模型身份绑定较强：driver `23–47,81–105` 核查 D 训练成功/3723 steps、训练 spec、terminal、环境、port、三模块与父 checkpoint。恢复 helper `scripts/evaluate_pvground_scanrefer_task_observation.py:53–88` 严格恢复 1234 原 state、安装 37 个新增 state，要求 delta 键集合恰为可训练参数与应保存 buffer，并检查冻结参数保持原值。driver `154–160` 固定 820 个实际 `requires_grad` 张量、28883227 个参数并记录真实参数名分组。因此静态上不是任选某一层来冒充完整可训练梯度空间。

这里的 parameter-space 点积使用 8 行 batch 总 loss。某一行的负方向可能来自批内其他样本、其他监督头或任务的共享参数贡献。模块 group 细化到 decoder/prediction_heads 层（driver `157`），但没有把 decoder.5 内语义/几何 task-query slice 单独分组；不能从一个 group 的正负号直接定位某个 task slice 为原因。

实际 D 路由已按当前 port 核对：`pv_ground.py:494–515` 拆出 semantic/geometry，semantic 供语义/对比头，geometry 供 box 与后续 query mask；`modules.py:145–178` 的语义与中心/尺寸路径相符；`encoder_decoder_layers.py:524–527` 调用 task finish helper，`models/pvground_task_observation_query.py:38–53` 共享 decoder 尾部参数。

固定图解释必须同时包含以下事实：

- driver `.eval()`、无增强、seed 2027、batch 8 是固定的（`53–62,106,142–153`），但当前 native `pv_ground.py:574–588` 无条件 Gumbel 采样；这是一个固定随机实现的条件方向。
- 原生 proposal/inter-layer geometry 在 `pv_ground.py:460–461,512–513` detach；`models/pvground_observation_query.py:19–33,60–99` 的观测状态 no_grad；mask token 的 argmax 也存在（`pv_ground.py:536`）。本次保持这些路径，不能把 native autograd graph gradient 等同于所有派生量随参数重算的精确光滑导数。
- 原训练 driver `train.py:210–221,318–335` 使用 train mode、增强、梯度裁剪及 AdamW；当前没有执行这些步骤，也未应用保存 optimizer state。
- 128 个场景各一条表达来自训练 fit 的首批固定清单，不是随机代表性测试集，没有多 seed、重复 batch 分组或正式 validation 结果。

方案 `9–23` 已明确多数边界。建议结果沿用“固定 eval 图的 native autograd 参数方向”，避免单独写“当前错误不是 loss 导致”或“共享训练无冲突”。remainder 只是线性差值；其负贡献也不能称为已证明的负迁移机制。

## F. Evaluation Type — PASS，限定分类

主分类为 **`real_gt` 的训练集局部机制诊断**：root box、mask 与匹配 target 由数据集对象标注确定，非模型自行生成 root reference。它不是正式 benchmark 性能评估、训练收益实验或泛化证据。

完整原生目标包含 supervised loss 与内部 prediction-derived self-consistency loss；后者需标为 **`self_supervised_proxy` objective component**。这不会把真实 GT margin 诊断变成 fake-GT 评估，也不能因此把总损失的每个目标都标成真实 GT。

## 终态独立导出复算

新输出在 `2026-09-17T16:53:24.327641+08:00` 完成，driver 用时 228.834581 秒（`diagnostic.json:5,59`）。本审计使用 PowerShell 独立读取 JSON，并使用 .NET ZIP/NPY v1 little-endian float32 reader 解码 `candidate_values.npz`；没有再次运行模型，也没有覆盖执行方的 `recount.json`。执行方重计脚本 `scripts/analyze_pvground_parameter_direction.py:8–46` 及其 `recount.json` 也已阅读，以下核验由审计者另行复算得到。

| 检查 | 独立核验结果 |
|---|---|
| 退出与日志 | exit 0；16 个 batch sentinel、1 个 complete sentinel；没有 Traceback/RuntimeError/AssertionError 行 |
| 行与批次 | 128 行、128 个物理场景、16 批；selection、rows、batch 展开顺序一致 |
| 反向计数 | 72 条 selected≠matched，与 margin_backwards=72 一致；其余56条的4种 velocity全部为0 |
| 参数清单 | 32 groups，820 个参数名，无重复；声明 28883227 参数与运行断言一致；每批 full_grad_nonempty=770、CE=546 |
| 导出候选 | 128 个 NPY member，每个256×2，共32768个候选和65536个有限float32标量 |
| 导出数值 | selected score为本行最大值、best-IoU query为本行最大值；selected/matched score与IoU和JSON精确一致 |
| 内积导出代数 | 所有行 group和与对应velocity最大差0；remainder与full−CE最大差0 |
| loss组成 | 按已导出12项和原权重复算总loss，float64重组与运行float32总loss最大绝对差3.07849e-6；不构成原始梯度复算 |
| 有限性/内存 | rows/batches/diagnostic全部已导出数值有限；峰值allocated 18715622912 bytes，约17.4303 GiB |
| 文件绑定 | 新rows/batches/selection/NPZ四个SHA均匹配diagnostic；diagnostic SHA与recount绑定一致 |

从新 `rows.json` 重算 diagnostic 两阈值的全部摘要字段均一致：

| root覆盖错误阈值 | 错误数 | logit CE正 | 参数CE正/负 | 参数完整loss正/负/零 | 参数CE正→完整loss负 |
|---|---:|---:|---:|---:|---|
| >0.25 | 14 | 14 | 11/3 | 12/2/0 | 1行：2455 |
| >0.5 | 21 | 21 | 13/8 | 17/4/0 | 1行：3839 |

两组重叠，不能合称35个独立错误。两阈值完整loss负向记录合并后有5个唯一row；全部列出如下，数值保留6位显示，原始完整精度在rows中：

| row / scene | 属于哪个阈值的覆盖错误 | selected/root IoU | logit CE方向 | 参数CE方向 | 参数完整loss方向 | remainder方向 |
|---|---|---|---:|---:|---:|---:|
| 848 / scene0010_00 | 0.5 | 0.494085 / 0.525347 | 0.000741 | -0.144473 | -579.116189 | -578.971716 |
| 2455 / scene0044_00 | 0.25 | 0.000000 / 0.455668 | 0.000868 | 0.622636 | -5.830738 | -6.453373 |
| 3839 / scene0068_00 | 0.5 | 0.470097 / 0.797564 | 0.001036 | 0.877264 | -4.041589 | -4.918853 |
| 4424 / scene0082_00 | 0.25与0.5 | 0.000000 / 0.914371 | 0.000817 | -1.792586 | -19.831413 | -18.038828 |
| 9248 / scene0173_00 | 0.5 | 0.289670 / 0.580704 | 0.000928 | -0.488961 | -11.815154 | -11.326193 |

原始行定位为 `rows.json:787,3163,4549,5539,11974` 的row_id记录及各自前面的velocity字段。参数完整loss方向在这两组覆盖错误中的最小绝对值分别为2.780739和0.405160；报告没有把非常接近0的符号当作这里负向计数的依据。数值绝对大小仍依赖当前参数尺度、batch loss及其归一化，不能直接换算成IoU变化或有限步收益。

参考一致性另行重算得到：128行原始点SHA、selected与root match的差异数全部为0；selected IoU最大绝对差3.066659e-5，matched IoU为1.072884e-5，selected/matched score最大差均6.794930e-5。所有行新logit方向与旧方向/14的最大差9.435806e-8。故可以说**离散选择、匹配和原始点字节一致，浮点输出存在已保留的小差异**；不能说前向输出逐字节完全一致，也没有将这点差异归因为某个未经诊断的具体算子。

这些导出支持的结论是：在固定D、固定batch及native eval autograd图下，logit空间的正向CE方向确实不能保证共享参数空间同样正向；完整loss也没有对所有root覆盖错误都给出正向margin方向。它们尚不足以区分批内其他样本干扰、其他层CE、几何监督、语义对齐或自一致性项各自的因果贡献，不能据此给某个loss调权、把全部负向归因于D模块或宣称已改善训练/REC性能。方案规定的下一步只能先据具体机制设计最小控制，不能把此诊断当作训练控制已经完成。

**独立验证上限：**成功断言支持本次全state未变及.grad为空；导出重计支持存储数值、索引和分组求和的一致性。未导出原始参数梯度、当前全部predicted boxes、完整GT/token maps与全输入指纹，故本审计无法从导出独立重算参数梯度、内积或真实GT IoU，也无法完整证明跨轮GT/全输入逐字节身份。该边界与A/E/F限定继续有效，same-family/provisional状态保持。

终态产物SHA：

| 产物 | SHA256 |
|---|---|
| diagnostic.json | `1d267bb37152dc6e875e4407fca4c93ddc4e4ea5d63f90da2ec39ceabf04297f` |
| rows.json | `271f50a390afebeecd012e179c91612dbde39455c94698276b7f916e44962e27` |
| batches.json | `ee76fb8cd6ee6b902a30c2044e9d1b732843ec1c02ccaac84fc54755ac0491aa` |
| candidate_values.npz | `9ae57406d59db72a1a73a7068866db55f81adb49da7f089d625315359a401db2` |
| parameter_groups.json | `2c8f43d6a98a0dc4778a4a9b331f6dd3a1aebcd3320ee51d2dc451e104898e1a` |
| run.log | `9622444bec40eb4aa8b152bae262354932bc405e965a7b3f3b4ca6e4ef4b1acd` |
| recount.json | `4eec02cfbb3319d213ac16b389141f6443be8f9c0b8d880a8183ea8c7fb425b5` |

## 本次读取的关键身份

| 文件/来源 | SHA256 |
|---|---|
| 方案 `docs/PVG_PARAMETER_DIRECTION_PLAN_2026-09-17.md` | `25d619c641370cb3fc660f4687a9f9b55ec249914e9c9f7a5822a18da6d81ea7` |
| 参数方向 driver / staged diagnose.py | `2ac6aa173c0fad9198121e54002ef3c1d9494eb7ba5597f6b52b617f936dcad8` |
| prepare driver | `3e0401ea224496d9b92dac45349e8cd08664aa72cd5e35a3dcf0ac5d46f63e34` |
| observer driver | `d0974bb756f38af3f3fdb4573804e576369a98af3909b91b6e92919d75bb8911` |
| current diagnostic staged spec | `ff175c175deecb3f14faf4aa02a1cada7485b2c19dd28bcc4368173018ac8e7b` |
| restore helper | `a51e891f04ede88b9e83a44b63e025d30653b2c822df9717e0427a46bfdaba3a` |
| D training spec | `895430216793cfe64c365023c878c4890d9bef804325065a6a1e8900d078149b` |
| D training driver | `819e723d2dac40a191f7e057a9ba1b340519454c4f241f2458a40b431376a872` |
| source_port | `0375886f9df2b3ac18a63ce7571d672ceb433613b6defdfcd9f8d244a20bf855` |
| native losses.py | `920051cc7d1009493829eebbb6185da834a4efd037e9d73ad43ce918a81031de` |
| native main_utils.py | `14bd101eb78ec3d729968aff0a975684efa8388574583ca9e4b0f512877b4f5c` |
| actual D pv_ground.py | `20c353512939ac9087cd8f3d07b82a648bcefe42510563e26afc37235facdb7c` |
| actual D encoder_decoder_layers.py | `e2cb80a29651f8540edd0a84c6d07072c9933430513c55f3bbe254006788d736` |
| actual D modules.py | `1609b3608e017669ae0ff63e7618f2078e5f72f20955da02e7ff230481b846fc` |
| supplied dataset source | `3afebfc69232f9547e6810563ba6dd4f34a0f770f929df2f4203b21493de0e3d` |
| archived old logit driver | `1d51a20431793062b041745a39965515597e3c75f3ea6fec1241792b613ad31c` |
| old logit diagnostic.json | `28be9e74ae279984f53c026ee2dee5c153e3cb9e4d551f69de62578349095b77` |
| old logit rows.json | `42e9bff1058c09d01e831c71a354163de6f677831f71e8ee3dc17d28f8fa1db1` |
| old input_selection.json | `d2fbed7e9aa4546af703f90cff47928e7950c783be03f49c9ed916b66192ca22` |

实际 D forward/decoder/modules/port 的本地来源目录为 `D:/Program Files/UserCache/gb/codex/tmp/pvg_parameter_20260917/`。最初提供的旧 `pvg_task_read_interface_20260917/` 快照 hash 与当前 D port 不同，最终 forward 结论已全部以本表的实际 D 文件为准。dataset 源属于独立 dataset_source namespace，不以 PV-Ground 自带的同名 dataset 文件 hash 代替其身份；本次尚未独立验证该 dataset_source manifest 的全部字节。

历史 `docs/PVG_RUNTIME_AND_FORWARD_RESULT_2026-09-08.md:3,38,46` 仅证明当时完整预训练前向/工程能力，且明确没有训练反向或正式 REC 结果；本次未将它当作多次共享参数反向或性能完成证据。

## 发布文字的有限核对

已对照 `docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md:18930–18975` 的§20.216关键表、负向行和解释边界；数值与本次导出复算一致。该节保留 batch-loss、Gumbel实现、native stop-gradient图、内部自一致性目标、GT/全输入身份及原始梯度未导出的限制，不把正方向计数写成修复样本数。停止本分支、不据少数负行降低Mask权重或新增优化器机制，属于没有超出本次证据的执行决定。本审计未再次核验该节引用的远端健康、历史保护线和其他实验完成状态。

tracker 的结论应表述为固定诊断中完整loss拥有更多正margin方向（严格阈值17对13），不把该符号计数表述成实际margin已改善或总体幅度收益。本报告以本节所列终态产物为最终审计输入，审计已完成；没有等待中的模型实验或新增复跑要求。
