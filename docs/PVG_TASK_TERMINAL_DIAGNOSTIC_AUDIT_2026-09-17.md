# D 固定终态机制诊断：独立完整性审计

静态审阅时间：2026-09-17 15:52 CST；完成证据追加核验：16:00 CST。审阅者：原独立 native Codex 子代理 `/root/terminal_diagnostic_audit`。审阅技能：`experiment-audit`。本报告由审阅者直接读取源文件、spec、receipt、归档脚本、日志和 NPZ 后编写；不以执行代理的结果解释代替证据。

```yaml
review_independence: same-family
acceptance_status: provisional
overall_verdict: WARN
integrity_status: warn
static_execution_blocker: none_identified_after_v2_correction
v1_execution_status: failed_output_snapshot_serialization
v2_completion_evidence: verified_complete
deterministic_artifact_checks: pass_with_raw_size_recompute_scope_limit
evaluation_type: synthetic_proxy
evaluation_subtype: fixed_trained_state_intervention_on_real_training_inputs
```

模型和 reasoning effort 的运行身份以父代理保存的原始 reviewer spawn 记录为准。本报告不声称跨模型家族验收。审阅者没有启动或重启作业；本轮只新增本文件。

所有下列路径均相对于 `C:/Users/gb/.codex_mcln_g0_20260905/`。行号是本次直接读取的文件版本。

## 结论与可接受范围

v2 对已观察到的输出类型错误作了最小修正，未改变模型、输入、干预、权重、RNG 或选择规则。现已收到并独立核对 v2 完成产物：两批全条件共十次 forward、完成标记、exit 0、1271 项严格恢复及结束状态复核均有代码与执行证据支持。未发现隐藏训练、正式评估或 checkpoint 选择。整体保留 WARN 是对四个固定训练样本、mask 数值变化和因果归因范围的限制，不表示本次运行未完成。

本诊断不能识别“D 相对 C 的变化主要来自共享核心更新”。每个条件都保留 D 的其余训练终态权重，仅恢复或置零两个任务矩阵。因此 `zero_both` 是 D 内部的条件干预，不是重新训练的 C，也不是保持训练轨迹不变的因果消融。它可以显示这四个输入上任务矩阵是否改变对应张量；不能证明有效任务分工、性能提高、泛化、消除梯度冲突或总体训练贡献比例。证据：`refine-logs/pvground_task_terminal_diagnostic_20260917_v2/diagnose.py:136-142`；`models/pvground_task_observation_query.py:25-30`；原后续问题位于 `docs/MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md:18837`。

## v1 失败与 v2 修正、重试的区别

v1 的 `run.log:4-11` 保存了真实 traceback：在 `diagnose.py:143` 执行 `output[key].detach()` 时遇到 `AttributeError: 'list' object has no attribute 'detach'`。`controller.exit:1` 为 `1`。错误发生在第一批 `normal` 的模型 forward 返回之后、输出快照保存时；代码尚未到 `DIAGNOSTIC_FORWARD` 打印、后续重复/置零干预或最终完整状态复核。这一次是失败尝试，不是完整的十次 forward 诊断。

该轨迹说明执行已越过前置 checkpoint 哈希、3723 步、真实 delta 严格加载和 1271 项 state 相等断言；它不证明所有干预完成或最终恢复成功。代码在首个条件没有置零任务矩阵，且此路径没有优化器更新和 checkpoint 写入。

直接比较 v1 与 v2 归档 `diagnose.py`，唯一代码差异为：

```diff
-                   'last_proj_queries','proj_tokens','adaptive_weights']
-    list_keys = ['last_pred_masks','sp_last_pred_masks']
+                   'last_proj_queries','proj_tokens']
+    list_keys = ['last_pred_masks','sp_last_pred_masks','adaptive_weights']
```

修正与真实接口一致：`refine-logs/pvground_pretrained_resources_20260908_v1/source/models/pv_ground.py:539` 将 `adaptive_weight_lists` 赋给 `end_points['adaptive_weights']`；v2 `diagnose.py:144-146` 对列表每项分别执行 detach/CPU/clone，随后仍进入同一差异统计。没有增加 fallback、修改模型或选择其他输出。

v1 与 v2 的 training root/spec SHA、terminal SHA、fixture 路径/receipt SHA、seed、四个行号、五个条件、batch size 和 forward 预算完全相同。v2 `spec.json:2-25` 与 v1 对应字段一致；差异仅为本次磁盘观测和 `diagnose.py` 文件哈希。其余五项 staged 文件哈希完全一致。v2 `launch.json:2-8` 记录 15:48:47 CST 的独立重试目录与零训练/零正式行预算；launch 不是完成证明。

15:52 静态审阅时 v2 本地产物尚未取回，曾标为 pending；16:00 已完成对 `diagnostic.json`、`boxes_scores.npz`、`controller.exit` 和完整日志的独立核对，当前状态为 verified_complete。v1 原始脚本、spec、run.log 与失败 exit 均仍保留，未将失败尝试覆盖成成功。

## A-F 检查

| 项目 | 状态 | 证据与解释 |
| --- | --- | --- |
| A. Ground truth provenance | PASS | 本诊断不计算准确率、不合成 GT；参考是同一模型的 normal 输出并明确限为机制诊断。输入来自已有真实训练 fixture，GT 与模型输入分开保存。见 v2 `diagnose.py:43-47,115-133,149-158`；fixture 导出器 `refine-logs/pvground_training_interface_20260908_v1/export_fixtures.py:46-76,85-111`。 |
| B. Score normalization | PASS | 差异为原始 max_abs、mean_abs、RMS，无模型自身统计量作分母。softmax 只用于既有 bbs 排序，未被包装为性能分数。见 v2 `diagnose.py:149-161,165-177`；正式 wrapper `scripts/evaluate_pvground_scanrefer_task_observation.py:33-40,297-308`。 |
| C. Result existence | PASS within inspected artifacts | 训练 receipt、终态 audit、正式跳过 decision、A/B/C 配对 JSON 存在，抽核第 20.213 节主指标与变化数一致。v1 明确失败；v2 完整产物存在且日志/JSON/NPZ 相互一致，详见追加核验。历史原始 6887 行和 mask 未在本审计中重新计算。 |
| D. Dead code / hidden training | PASS for v2 executed path | 主入口调用真实模型，hook 捕获实际 reader 返回值，差异进入 record。完整日志核对十个预定条件。导入 evaluate 的函数不会触发其受 `__main__` 保护的主评估。无 optimizer、backward、训练循环或新 checkpoint。v1 的后续代码因异常未执行，保留失败状态。见 v2 `diagnose.py:13,90-106,136-177,179-199`；wrapper `:354-355`；v2 `controller.py:7`、`run.log:4-14`。 |
| E. Scope | WARN / explicit ceiling | 4 行、4 个训练物理场景、seed 2027、batch 2、5 条件、10 次 forward；不代表验证集准确率、泛化或多 seed 结论。normal_repeat 的真实数值差异必须作为解释干预差异的参照。见 v2 `spec.json:7-25`、`diagnose.py:107-115,165-192`。 |
| F. Evaluation type | PASS with required classification | `synthetic_proxy`：真实数据输入上的模型内部参考与干预一致性诊断，无准确性 GT。单独的正式 wrapper 使用数据集框/mask GT，属于 real_gt 路径，但 D 正式运行被跳过，不计入本诊断。见 wrapper `:284-308`；`refine-logs/pvground_scanrefer_formal_20260917_task_observation_v1/decision.json:1`。 |

## 恢复、输入、选择与状态检查

**真实终态恢复。** v2 `diagnose.py:23-47` 核验 staged 文件、训练退出、训练 spec、实际 terminal、源文件与三个模块的 SHA；`:69-98` 核对官方父、实际 3723 步 delta、完整 fit 行计数、模块标志和哈希，再严格加载 1271 项状态并逐项相等比较。导入 helper 的 `evaluate.py:53-72` 先严格恢复 1234 项父状态，再安装 37 项新增状态；`:75-88` 对 delta 键集合、shape、dtype、finite 和冻结参数逐项核验。v2 已通过这条路径并完成，见 `run.log:14`、`diagnostic.json:1321-1338` 与 `controller.exit:1`。审阅者没有另行运行模型或重新加载远程 checkpoint。

**输入固定与 GT 隔离。** 导出器按原 fit 顺序选择前四个不同物理场景，关闭增强且排除 holdout；fixture receipt 固定行号 0、173、237、455。v2 对每个文件核 SHA，构建一次同批输入，并为每次 forward 复制张量/深复制其他值。实际 `_get_inputs` 只返回体素、点、文本、检测框、检测标签掩码、检测类别与 superpoint，没有 GT 框或 mask：`refine-logs/pvground_pretrained_resources_20260908_v1/source/train_dist_mod.py:129-169`。该文件哈希 `e604249b116309737e98e86dda1e21631d4538778dca739d7e6484855e7db750` 与 D 的 `source_port.json:24` 一致。

**固定 RNG。** v2 `diagnose.py:53-62` 重设 Python/NumPy/PyTorch/CUDA seed 2027，关闭 benchmark 和 TF32；`:94` 使用 eval；`:137-142` 在 no_grad 下逐条件恢复任务矩阵并重置 RNG。重复 normal forward 为实际数值误差参照。静态配置不代替 real normal_repeat 输出验证，不能预先声称 GPU 自定义算子完全确定。

**输出选择依据。** v2 `diagnose.py:149-156` 的 bbs 分数使用 fixture 的文本 positive/modifier/pronoun/relation/other-entity span maps，并降序选首 Query；与正式 wrapper `:297-303` 对应规则一致。这里使用文本跨度映射并不等于 GT 框/mask 输入模型。全部候选框、分数和选中索引写入 NPZ，未据干预结果更换主输出、搜索阈值或筛选样本。

**结束时恢复。** v2 `diagnose.py:179-183` 恢复任务矩阵、移除 hook、回 CPU，并将全部参数和持久 buffer 与终态逐项 torch.equal；成功后才落盘 JSON/NPZ。v2 的完成日志与 exit 0 支持这条断言实际通过；v1 未执行至此。该检查没有声称恢复 Python RNG 或模块的临时观测属性。

**无正式评估接续。** v2 controller 的唯一子进程目标为 `diagnose.py`；导入的 `evaluate.py` 主函数不自动运行。D 训练 receipt `:32` 的 `primary_rec_nonregression=false` 与正式 `decision.json:1` 的 `skipped_primary_rec_regression, formal_rows=0` 一致。本次诊断不得绕过原晋级门槛。

## 核验过的文件标识

v2 spec 的六个文件全部重新计算 SHA256 并匹配。三个模块同时与训练 spec/receipt 一致。下表的 expected terminal 是由训练 spec/receipt 与诊断 spec 固定的权重标识；审阅者未在本地重新读取远程 checkpoint 二进制。

| 文件/证据 | SHA256 |
| --- | --- |
| v1 diagnose.py | `93cec238ac3c3e60e9ba54d06bca389f7232484473aeeeefd5a3ca557346c67c` |
| v2 diagnose.py / 当前 scripts/diagnose_pvground_task_terminal.py | `22909e25751cca86b733420dbf1322b53ab1907dd3fb590ef8887573e8e900e8` |
| v2 evaluate.py | `a51e891f04ede88b9e83a44b63e025d30653b2c822df9717e0427a46bfdaba3a` |
| v2 fixture_receipt.json | `2a2a856108d34303895aad4abc40b23b266d414de5a64031d7e566343be4d420` |
| pvground_source_query.py | `e0fcc6a30402ebeba7a9ffa1be907515350b64e408602a8b47b702e83a5d05ab` |
| pvground_observation_query.py | `cc92906f9b6f388c8da702f555a099faf07d619fefa2772bfb229693ebae7742` |
| pvground_task_observation_query.py | `39349bf1fe8f3f51149f0872911020ecd0ede98fc9a8278cecb5ae466ed8643d` |
| v2 spec.json | `30da708e0f16655d047cdce80093ee4950d1072c3949f9466107a8a5fb3788ed` |
| v2 launch.json | `64a7fc35dba05ba2bb10ab213e3221a41906e8fd5326d9363d619503e827834b` |
| v2 controller.py | `f5939799dd6b2a3501eeb8d89a1dd4d6b06f19710cff5f5b4d3369e204f8015c` |
| D training spec.json | `895430216793cfe64c365023c878c4890d9bef804325065a6a1e8900d078149b` |
| D training receipt.json | `92dcd07dd8cfcabaddb1c8a6f5d1fe9e7482ea5e99bb730b3f0731fa8244d78f` |
| expected D terminal.pth | `ce03188965491a82bcb1c5a6d26f590d3a243a01985457f220d5503c75b2fcf5` |

## 完成证据追加核验（2026-09-17 16:00 CST）

**运行证据通过。** `v2/run.log:4-13` 恰有十条 forward，顺序为 batch 0、2 各依次执行 normal、normal_repeat、zero_semantic、zero_geometry、zero_both；两批对应训练行 `[0,173]`、`[237,455]`。`:14` 的唯一完成记录与 JSON 除 batches 外全部字段一致，完成时间为 15:49:15.173566 CST、用时 28.5193679333 秒；`controller.exit:1=0`。脚本、terminal、training spec、fixture SHA 与 spec 一致；恢复 1271 项和结束状态不变不是仅引用静态配置，而是由已审阅的断言路径、完成日志和退出共同支持。代码与入口仍为零 optimizer step、零 formal row、零新 checkpoint。

**NPZ 独立复核通过，范围明确。** 审阅者使用离线 uv/NumPy 2.5.3 直接读取 NPZ，无模型 forward。30 个数组恰好覆盖两批、五条件和 boxes/scores/selected 三类；shape、dtype、finite 检查通过。20 个保存选择同时匹配 JSON、日志和分数唯一 argmax，最小 top-1 间隔为 `0.0003252848982810974`。对 scores 与 centers 的 16 个条件比较重新计算 48 项 max/mean/RMS，和 JSON 差值全为 0。两批均验证：normal_repeat 的保存框/分数与 normal 完全一致；zero_semantic 框等于 normal，zero_geometry 分数等于 normal；zero_both 框等于 zero_geometry，分数等于 zero_semantic。各条件选中 Query 均为 `[50,88,237,219]`，但这不等于原始输出没变。

**原始尺寸与其他张量的复核限制。** `diagnose.py:143` 统计 clamp 前 `last_pred_size`，`:158` 保存框时执行 `clamp(min=1e-6)`。NPZ 中每条件第一批有 211 个、第二批有 174 个尺寸分量恰在下限，十次条件合计 1925 个。因此不能用 NPZ 完整重算原始 size 差异；尺寸差异只作为原执行 JSON 的记录。reader、logits、mask 和 alpha 原始张量也未保存，未声称独立重算其汇总统计。首次离线验证因审阅者假设“无尺寸触及 clamp”而断言失败，随后依据已存在的导出代码限定重算范围；未更改实验或重新推理。

**重复数值差异与实际效应。** JSON 两批 normal_repeat 的 reader、Query 点、中心/尺寸、logits、投影和 bbs 分数完全一致；mask 和 alpha 存在数值变化，最大分别为 `last_pred_masks 1.6212463379e-5`、`sp_last_pred_masks 4.1961669922e-5`、`adaptive_weights 5.2154064178e-8`，见 `diagnostic.json:6-160,660-814`，不能称所有张量 bit-exact。

| 条件与 max_abs | 行 0、173 | 行 237、455 | 证据级别 |
| --- | --- | --- | --- |
| zero_semantic：bbs 分数 | 0.0006306767464 | 0.0004911646247 | NPZ 独立重算 |
| zero_geometry：中心 | 0.0006314516068 | 0.0008619427681 | NPZ 独立重算 |
| zero_geometry：原始尺寸 | 0.005386590958 | 0.003426194191 | 执行 JSON，受上述导出限制 |
| zero_geometry：superpoint mask | 0.1028671265 | 0.1131439209 | 执行 JSON，远大于本次 repeat 数值变化 |

zero_semantic 保持中心/尺寸和 geometry reader 不变，zero_geometry 保持语义 logits/投影/bbs 分数和 semantic reader 不变；小量 mask/alpha 变化与重复运行变化相近时，不作明确机制归因。支持的结论限于这四个输入上的选择性输出响应；没有准确率或泛化证据，也没有识别 D/C 共享核心贡献，不能将 zero_both 解释为 C。现行晋级要求不变。

| 完成产物 | 直接重算 SHA256 |
| --- | --- |
| diagnostic.json | `6283855d534c31061f3adbbb70909599f5133fb51d9fef2efbaa787690d7f6fe` |
| boxes_scores.npz | `7275497dbf0144e2c1c24fcda1daf3b51e3b657fea8382d23341f73b079877c1` |
| run.log | `7d64a4f8e0068f49f55bf92458b5d56dff186fa275aa940b060a6d6ce0720b19` |
| controller.exit | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |

NPZ SHA 与 `diagnostic.json:1313` 和 `run.log:14` 一致。确定性的文件/数值检查通过；语义审阅归属仍为 same-family，验收状态仍为 provisional。
