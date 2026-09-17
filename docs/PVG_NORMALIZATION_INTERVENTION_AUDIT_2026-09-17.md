# 固定 D 终点 BatchNorm 运行统计干预审计

日期：2026-09-17。审计技能：`experiment-audit`。审计者：新的原生 Codex reviewer 上下文 `/root/normalization_intervention_audit`。`review_independence: same-family`；`acceptance_status: provisional`。

**总体结论：WARN / provisional；确定性产物重计通过，未发现静态执行阻塞。** 本次证据支持“固定 D 参数、固定这 128 个训练场景输入时，整体替换 BN 运行统计会改变输出”。它不支持训练收益、正式 ScanRefer 性能通过、Mask 通过或模型晋级。正常重复臂与正常臂逐位一致；父 BN 臂在 IoU > 0.25 的命中数持平，在 IoU > 0.50 净增 4，未达到计划所述的“双阈值改善”。

本审计只读取本地代码及已取回产物，没有远端操作、实验启动、模型前向、参数更新或凭据读取。唯一写入文件为本报告。独立重计先从原始 NPZ/JSON 推导完成，未调用执行者的分析脚本，也未以其 `recount.json` 代替独立判断。

## 1. 固定对象与静态可执行性

| 对象 | 固定身份或范围 |
|---|---|
| D 终点 | 第 3,723 步；`ce03188965491a82bcb1c5a6d26f590d3a243a01985457f220d5503c75b2fcf5` |
| 官方 Scan 父 checkpoint | `6f24c67cc3409f44befdef188f14383497a824d8ab309b8fd572a999ae8bdec3`；计划标为 Scan epoch 81 |
| 训练 spec | `895430216793cfe64c365023c878c4890d9bef804325065a6a1e8900d078149b` |
| 模型 source port | `0375886f9df2b3ac18a63ce7571d672ceb433613b6defdfcd9f8d244a20bf855` |
| 本次 spec | `80a6319dbf0e43e6bf3c7f9ff3f442f81b9150b3b6160bba2d90d50ab591c384` |
| 本次输入 | 预先固定的 128 个 fit 物理场景，每场景一表达；1 个 seed 2027，batch size 8，无增强 |
| 三臂 | D 正常；D 正常重复；D 参数保持不变、84 组 BN 运行统计全部恢复父值 |
| 实际工作量 | 16 批 × 3 臂 = 48 次 no-grad forward；0 backward、0 optimizer step、0 新 checkpoint、0 正式评估行 |

四份任务脚本以及归档 `diagnose.py`、`evaluate.py`、`controller.py` 均通过纯内存 Python AST 解析；没有导入或执行它们。spec 绑定的 11 个文件均存在且 SHA256 匹配，工作区诊断和 evaluator helper 源码与归档版本一致。启动器在独立目录、独立 controller 中使用原 GPU `flock`，检查旧任务退出和 GPU 空闲；本审计只核对源码与本地回执，不重复远端检查。证据：`scripts/prepare_pvground_normalization_intervention.py:27-45,46-87`；归档 `controller.py:4-9`。

本地默认 `python` 指向损坏的 `E:/Scripts/python.exe`，报 `No pyvenv.cfg file`；这是本地审计工具入口问题，不是诊断执行故障。重计使用已缓存的 `uv run --offline --no-project --with numpy python -B -c ...`，没有新建项目环境或运行实验代码。

归档 `controller.exit:1` 为 `0`；`run.log` 包含恰好 16 个 batch 记录及唯一完成记录，与 `batches.json`、`diagnostic.json` 完全一致。完成时间为 2026-09-17 17:15:22 +08:00，诊断计时 103.338846 秒。没有只凭 launch 回执宣称完成。

## 2. A：GT 来源 — PASS（源码来源与产物一致性）

GT 来自 ScanRefer 训练标注的 `object_id` 及 ScanNet 对应实例框，没有由模型输出生成 GT。实际 dataset 源码缓存的 `load_scanrefer_annos` 读取 `ScanRefer_filtered_train.txt/json` 并保留 `object_id`、文本；`_get_target_boxes` 用 `scan.get_object_bbox(tid)` 构造 center/size，root 为 target 列表的第 0 项；`__getitem__` 将其写入 `center_label`、`size_gts`。诊断直接使用这两个字段的 root 框，并检查 root label 有效。

证据：`D:/Program Files/UserCache/gb/codex/tmp/pvg_loss_20260917/joint_det_dataset.py:584-646,1086-1119,1387-1392`；`scripts/diagnose_pvground_normalization_intervention.py:199-207`。数据集缓存 SHA256 `3afebfc69232f9547e6810563ba6dd4f34a0f770f929df2f4203b21493de0e3d` 与训练归档 `source_manifest.json` 一致；该 manifest 的 SHA256 `75cd5f87a8e715c15b458ed1964c2e6f1d35046bb6beffca1b3388d5903bbe36` 与训练 `input_manifest.json` 一致。

模型输入只包含 points/voxels、文本、外部检测框/类别/有效性及 superpoint；没有传入 GT center/size、GT masks 或 GT instance boxes。`butd=True`、`butd_gt=False`，模型输入的 `det_boxes` 来自 `all_detected_boxes`，GT 在 forward 之后单独用于评分。证据：诊断脚本 `:85,139-142,187-200`。语义 token maps 来自固定数据解析流程，用于与原生 evaluator 一致的评分；它们不是用预测 IoU 选择出来的评分权重。

独立重计检查了 128 个保存 GT 均为有限的 6 维框、尺寸为正，NPZ 的 `row_id_gt` 与 `rows.json.root_box` 一致。审计未重新读取远端 128 个原始 ScanNet 实例文件；因此 GT 的物理来源由绑定的数据加载源码和原执行检查支撑，本地直接验证的是保存 GT 的完整性及使用方式。

## 3. B：评分、尺寸与 IoU — PASS；raw oracle 保留限定

原生 `bbs` 评分为：

```text
p = softmax(last_sem_cls_scores, token_dim)
score(q) = sum_t p(q,t) * [positive_map(t) > 0]
         + sum_t p(q,t) * modify_positive_map(t)
         + sum_t p(q,t) * pron_positive_map(t)
         + sum_t p(q,t) * rel_positive_map(t)
         - sum_t p(q,t) * other_entity_map(t)
selected = descending_argsort(score)[0]
```

诊断脚本 `:194-203` 与绑定的原生 `src/grounding_evaluator.py:209-286,535-553` 一致；原生模型以 256 个类别/token score 输出，代码中不触发 evaluator 的不同宽度填充分支。这里的 softmax 是原生模型评分步骤；没有把命中率、IoU 或结果除以模型自己的最大值/均值来美化指标。

box 的原始 center/size 被保存在 NPZ。计算 IoU 时将 size clamp 到 `1e-6`，转为 xyzxyz，然后用交集体积除以并集体积；命中规则为严格 `IoU > 0.25` 和 `IoU > 0.5`。证据：诊断脚本 `:206-210,225-228`；`D:/Program Files/UserCache/gb/codex/tmp/pvg_loss_20260917/models/losses.py:35-75`；原生 evaluator `:289-303`。独立重计没有调用这些函数，而是从原始框重新实现几何计算。

`raw_oracle_iou` 是全部 256 个原始候选经过同一 clamp 后的最大 IoU，仅用于诊断候选池；它不参与选框，也不是合法候选召回。存在非正尺寸的原始候选，详见第 7 节，故不得把该 oracle 写成合法检测召回。

## 4. BN 干预、RNG 与状态还原 — PASS（固定诊断完整性）

CPU census 归档 `controller.exit=0`，脚本 SHA 与记录一致，252 项唯一 buffer 对应 84 组，每组恰好包含 `running_mean`、`running_var`、`num_batches_tracked`；252 项记录均发生变化。所有 84 个计数器从 479199 到 482922，正好增加 3723。该事实只说明训练更新过运行统计，不说明其更新错误或有害。证据：`scripts/census_pvground_normalization_buffers.py:10-43`；`refine-logs/pvground_normalization_census_20260917_v1/census.json`。

诊断按实际 `torch.nn.modules.batchnorm._BatchNorm` 类型枚举 key，检查 `track_running_stats=True`、`training=False`，并要求与 census 252 项集合完全相等。`put_bn` 的唯一写入位置是 `dict(model.named_buffers())[name].copy_(...)`；枚举 suffix 不含 affine `weight`/`bias`，也没有任何参数优化或 parameter copy。所有其他参数从固定 D terminal 加载，且模型始终 eval。证据：诊断脚本 `:92-106,151-167,174-192`；`scripts/evaluate_pvground_scanrefer_task_observation.py:53-88`。

每批先构造一次原始点云与 voxels，再在每臂前恢复 Python、NumPy、CPU Torch、全部 CUDA RNG 状态。每臂重新 clone batch tensor、deepcopy 其他 batch 内容，并从复制的 voxels 重建输入，避免前向对输入字典或 tensor 的修改污染后续臂。下一批恢复正常臂 forward 之后的 RNG 状态。实际 PVGround 的 query 生成会调用 `F.gumbel_softmax`，即使 eval 也存在随机采样，所以配对 RNG 在此不是多余措施。证据：诊断脚本 `:168-193,213`；绑定 `pv_ground.py:310-320,574-588`。

每批三臂结束后恢复 D 的 BN buffers；全部结束后将模型转回 CPU，对完整 state_dict 与 D terminal 逐 tensor 做 `torch.equal`，并断言所有 parameter `.grad is None`。完成产物只能在这些断言之后写出。证据：诊断脚本 `:213,220-241`。`diagnostic.json:2,29` 和退出码 0 支持这些执行断言确实通过。审计者未在本地加载远端 checkpoint 再做一次状态恢复；因此这是绑定源码与成功完成记录支持的执行证据，不是第二次独立模型运行。

## 5. C、D：产物与实际调用路径 — PASS

实际路径为：固定训练 dataset → `DataProcessor` → 绑定的 `PVGround` → `last_sem_cls_scores/last_center/last_pred_size` → 原生 bbs 等价公式及原生 IoU 函数 → rows/NPZ → summary。D 最后一层的 source/task read 均保持启用；实际模型源文件路径被断言为固定 task-observation source 目录。证据：诊断脚本 `:63-80,106-108,177-210`；绑定 `pv_ground.py:280-307,465-511`；绑定 `modules.py:135-178` 与 `encoder_decoder_layers.py:524-527`。

审核的 `pv_ground.py`、`modules.py`、`encoder_decoder_layers.py`、`models/losses.py`、`src/grounding_evaluator.py` 五份本地缓存均与固定 source-port manifest 的各项 SHA256 相符。数据集来自单独绑定的 detection-aligned dataset source，未误把其路径当成 PVGround 模型 source。

本次没有实例化完整 `GroundingEvaluator`，而是执行经上述核对的 bbs/IoU 公式。`evaluate.py` 在本次仅提供 `expanded_parent_state`、`terminal_state`、SHA/JSON/time helpers；其 formal `main`、Mask、`promotion_check` 不会因 import 执行。不得把“复用 formal helper”表述成“本轮执行了完整官方正式评估”。

独立产物核对覆盖：

- 128 个 row ID 唯一、顺序与锁定 selection、参考行和 16 个 batch 完全相同；128 个物理场景唯一。
- 每行 scan ID 和 point SHA 与上轮固定参考一致；输入 selection 文件与锁定副本字节哈希相同。
- 512 个 NPZ 数组恰好为每行 3 臂候选数组加 1 个 GT；候选数组均为有限 `float32[256,8]`，列为 6 维原始 box、bbs score、IoU。
- 共重计 98,304 个候选的 IoU；float32 重计与保存值最大绝对误差为 **0**。
- 额外用 float64 重计，最大绝对差为 `2.0812356899e-6`；没有阈值命中翻转。最近的所选 IoU 到任一阈值距离为 `5.6266784668e-5`。
- 所有 384 个 row-arm 的最高 score 均唯一；NumPy 从 score 独立选出的 top-1 与保存的 selected 全部相同。
- 所选 score/IoU、raw oracle、两阈值 hits/fixes/breaks，以及 repeat/parent-BN 最大差值与产物逐项一致。
- `run.log` 的 batch/complete JSON 与对应文件完全相同，最终退出码为 0。

## 6. E、F：范围 — WARN；类型为 real_gt 训练输入诊断

固定 scope 是训练集 fit 的 128 个物理场景、一场景一表达、单 seed、固定 D 终点、单一“全部 BN 恢复父值”干预。程序重算物理场景哈希分组，要求 chosen 不在 holdout，并检查全部 36665 条训练 annotation 的行序、已选场景/文本及输入点云 SHA。证据：诊断脚本 `:114-149,179-183`；计划 `:5-11`。

`evaluation_type: real_gt` 说明 GT 来源真实，不授予泛化或完整性能结论。这 128 个训练场景是预训练见过的数据；没有独立训练对照、完整正式 9508 行评估、Mask 指标或跨 seed 重复。正常重复臂用于测量本次配对执行的重复波动，不是新的独立训练 seed。

## 7. 独立重计结果

以下是固定训练输入诊断的命中数，不是正式验证集 REC 报告。

| 臂 | IoU > 0.25 | IoU > 0.50 | 所选框平均 IoU |
|---|---:|---:|---:|
| D 正常 | 114/128（89.0625%） | 106/128（82.8125%） | 0.665025979 |
| D 正常重复 | 114/128（89.0625%） | 106/128（82.8125%） | 0.665025979 |
| D 参数 + 全部父 BN buffers | 114/128（89.0625%） | 110/128（85.9375%） | 0.678163026 |

| 父 BN 相对正常 D | 修复 | 破坏 | 净变化 |
|---|---:|---:|---:|
| IoU > 0.25 | 1 | 1 | 0 |
| IoU > 0.50 | 6 | 2 | +4（+3.125 个百分点） |

| 输出差异 | 正常重复相对正常 D | 父 BN 相对正常 D |
|---|---:|---:|
| 改变的 top-1 选择 | 0/128 | 80/128 |
| 改变的候选 box | 0/32768 | 32768/32768 |
| 改变的候选 bbs score | 0/32768 | 32768/32768 |
| box 分量最大绝对差 | 0 | 6.302164078 |
| box 分量平均绝对差 | 0 | 0.194392518 |
| box 分量 RMS 差 | 0 | 0.600700254 |
| bbs score 最大绝对差 | 0 | 0.960355639 |
| bbs score 平均绝对差 | 0 | 0.007271408 |
| bbs score RMS 差 | 0 | 0.039397348 |
| 候选 IoU 最大绝对差 | 0 | 0.921688080 |
| 所选 IoU 最大绝对差 | 0 | 0.817014456 |

正常臂和重复臂的原始 box、score、IoU 全部逐位一致。父 BN 引起的输出变化超出本次测得的重复波动。该结论作用于整个被替换的 84 组运行统计；不能定位到某一来源、某一 BN 层，或把影响归于单独的 `running_mean`/`running_var`。`num_batches_tracked` 在 eval 下不是单独证明的性能机制。

raw-256 oracle 在三臂均为 `128/128`（>0.25）和 `127/128`（>0.50）。raw oracle 平均 IoU 从 `0.829494556` 到 `0.834468971`。原始 box 至少一个 size 分量 `<=0` 的候选数由 `7339/32768` 增至 `9620/32768`（repeat 与 normal 相同）。这些原始候选按原生规则 clamp 后才参与 IoU；没有删除负面候选。净增 4 个高阈值命中不能外推为所有候选整体质量改善。

阈值跨越的全部样本如下，保留破坏记录：

| 阈值 | 类型 | row ID | 场景 | 正常 IoU | 父 BN IoU |
|---|---|---:|---|---:|---:|
| 0.25 | 修复 | 8728 | scene0163_00 | 0.000000000 | 0.817014456 |
| 0.25 | 破坏 | 9248 | scene0173_00 | 0.289669961 | 0.000000000 |
| 0.50 | 修复 | 848 | scene0010_00 | 0.494084597 | 0.516792357 |
| 0.50 | 修复 | 3636 | scene0062_00 | 0.478593439 | 0.639612675 |
| 0.50 | 修复 | 4087 | scene0073_00 | 0.423249245 | 0.643016875 |
| 0.50 | 修复 | 4384 | scene0080_00 | 0.494390577 | 0.509594679 |
| 0.50 | 修复 | 6459 | scene0119_00 | 0.447929025 | 0.510024488 |
| 0.50 | 修复 | 8728 | scene0163_00 | 0.000000000 | 0.817014456 |
| 0.50 | 破坏 | 8281 | scene0155_00 | 0.554430962 | 0.488861859 |
| 0.50 | 破坏 | 8498 | scene0158_00 | 0.500056267 | 0.297166586 |

## 8. 结论边界与处置

| 表述 | 审计判断 |
|---|---|
| 训练使 84 组 BN buffers 发生变化 | 支持；CPU census 与固定终点一致。变化本身没有好坏含义。 |
| 本次固定 D 输出对整体 BN 运行统计敏感 | 支持；配对输入/RNG、重复臂一致、干预臂输出改变。 |
| 父 BN 在本次 128 行上提高 IoU > 0.50 命中数 | 支持，必须同时报告 IoU > 0.25 持平以及所有修复/破坏。 |
| 父 BN 实现双阈值改善 | 不支持；0.25 净变化为 0。 |
| 冻结 BN 训练会带来收益 | 不支持；本次没有这样的训练对照。 |
| D 的正式泛化性能、Mask 或完整 Scan 门槛通过 | 不支持；本轮没有这些评估。 |
| 可保存/部署混合状态或晋级到正式评估 | 不支持；计划禁止以本次训练集干预指标直接晋级。 |

计划第 11 行的“两个阈值均无净正变化”停止条件和“双阈值改善”条件均不精确覆盖此次 `0 / +4` 结果。保留它为“低阈值持平、高阈值局部改善”的混合结果，不事后把双阈值要求放宽成单阈值成功。本审计不要求追加实验，不建议部署混合状态，也不授权后续训练；任何另行训练方案需要作为新的方案单独设计和判断。

可复算性有两个明确边界：NPZ 保存的是合成后的 bbs score，没有原始语义 logits 或逐 token maps，因此可以独立复算 top-1/IoU/计数，但不能仅凭 NPZ 重新数值构建 softmax 与 token 求和；评分公式已通过绑定源码比对。远端 checkpoint、原始 ScanNet 文件和完整模型末态没有在本地重新载入，状态还原结论依据固定源码执行断言与完成记录。它们没有被冒充成第二次独立 GPU 复现。

发布文字已作有限核对：`docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md:18977-19013`（§20.217）及 `refine-logs/EXPERIMENT_TRACKER.md:337-339` 的本轮数字、负面候选记录和 scope 与原始产物一致；明确没有双阈值同时改善、没有训练收益证明、没有追加冻结 BN 训练或正式晋级。此核对不重审该主文档中的其他历史阶段。执行者随后补充的 `recount.json` 与审计者先前独立得到的阈值计数、均值及非正尺寸数量一致。

## 9. 审计输入哈希与机器可读摘要

相对目录：`refine-logs/pvground_normalization_intervention_20260917_v1/`。

| 文件 | SHA256 |
|---|---|
| `spec.json` | `80a6319dbf0e43e6bf3c7f9ff3f442f81b9150b3b6160bba2d90d50ab591c384` |
| `diagnose.py` | `bb0d8f2878e45ba78ca064add0b358434f6d263456a0b6261e68116a10234110` |
| `evaluate.py` | `a51e891f04ede88b9e83a44b63e025d30653b2c822df9717e0427a46bfdaba3a` |
| `plan.md` | `4d945c5b0fb187884d56a46ecca7751c3780d8fddbe14a745b50a33803977529` |
| `census.json` | `d663feb1afff8e78fb16db5fe413b930c8688f26fa4d725c89d99a8ef938cdd6` |
| `input_selection.json` | `d2fbed7e9aa4546af703f90cff47928e7950c783be03f49c9ed916b66192ca22` |
| `diagnostic.json` | `17bb48a5c4904c7f5d116ea7404189fee5e4a713258f38df214d63c17b7ca7a4` |
| `rows.json` | `f2a99d0b39e158e18c4fc884c3d98ee782872507dc1e7cb69ad2aeb462fce33b` |
| `batches.json` | `66e6f489d8d37a0fbfd448c62d4fa48d223d7feeb191c3f7f4f6bf369f27c8bd` |
| `candidate_values.npz` | `04ca3fc81e8a239fdfc37501ec89beb62deb07e276ac2cb28192cedb99d3004e` |
| `run.log` | `b0979049dcba2a7d74c71b4f4475986358a5588d9ea7ff21ed11c1e52489c3ab` |
| `controller.exit` | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |

```json
{
  "audit_skill": "experiment-audit",
  "date": "2026-09-17",
  "reviewer_context": "/root/normalization_intervention_audit",
  "reviewer_backend": "codex-native",
  "reviewer_family": "openai",
  "review_independence": "same-family",
  "acceptance_status": "provisional",
  "overall_verdict": "warn",
  "integrity_status": "pass_with_stated_evidence_limits",
  "performance_gate": "not_assessed_no_pass_claim",
  "static_execution_blocker": false,
  "evaluation_type": "real_gt",
  "evaluation_scope": "fixed_endpoint_128_fit_training_scenes_single_seed",
  "independent_recount": "pass",
  "checked_candidate_rows": 98304,
  "numpy_float32_iou_max_abs_error": 0.0,
  "numpy_float64_iou_max_abs_error": 0.0000020812356898991524,
  "repeat_bitwise_equal": true,
  "normal_hits": [114, 106],
  "parent_bn_hits": [114, 110],
  "parent_bn_fixes": [1, 6],
  "parent_bn_breaks": [1, 2],
  "parent_bn_selection_changes": 80,
  "dual_threshold_improvement": false,
  "fixed_endpoint_bn_sensitivity_supported": true,
  "training_benefit_supported": false,
  "mask_claim_supported": false,
  "formal_performance_claim_supported": false,
  "extra_experiment_required_by_this_audit": false
}
```
