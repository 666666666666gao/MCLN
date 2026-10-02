**PASS — 修订后的 GT 范围处理正确，无代码阻塞项；可以启动当前版本的 real8。** 本结论仅通过 CODE_REVIEW，不代表新版 GPU sanity 或 full9508 已通过。

Reviewer：原生独立审查 agent /root/pv_candidate_gt_scope_review；路由模型 gpt-6-astra，reasoning_effort max，fork_turns none。review_independence：same-family；acceptance_status：provisional。审查独立读取了源码、真实失败日志及保存数组，没有采纳旧报告的正确性判断作为证据。未使用 SSH、网络、GPU、model forward、训练、删除或认证环境读取；仅写本报告及同名 JSON。

所有下述相对路径均相对于 C:\Users\gb\.codex\tmp\pvground_fused_support_20261002。

1. **真实失败与原生数据范围。** candidate_audit/full/run.log:11–13 显示断言 root_scene_slot.numel()==1 失败，candidate_audit/full/run.exit:1 为 1，对应归档代码 candidate_audit/full/run.py:348–355。complete_tail_fused_retry/source/imported/src.joint_det_dataset.py:1086–1119 根据表达式 target_id 独立生成 root 框、有效 GT 槽和实例 Mask；root 位于表达式槽 0。该文件 :1121–1157 的场景列表另按 DC.nyu40id2class 保留对象，且只覆盖最多 MAX_NUM_OBJ=132 个原生实例槽（:59–62）；有效槽保留原场景实例下标，并非全部表达式目标集合。两类框都取同一 scan 的 get_object_bbox，val 不做 train jitter。:1327–1334、:1388–1392、:1422–1429 确认各返回字段的来源。旧代码假设所有 root 都在场景检测列表中，真实失败证明此假设错误；旧审查报告 CANDIDATE_AUDIT_CODE_REVIEW.md:16 未识别这一范围差异。原报告保留不改。

2. **修正的数学含义正确。** run_full_candidate_audit.py:335–336 从独立表达式 GT 计算所有候选的 root IoU；:344–351 只对原生有效场景 GT 求最大 IoU；:354 使用 root_iou >= best_scene_iou。这等价于 root 在“原生场景列表 ∪ {表达式 root}”中取得并列最大重叠。root 已存在时，它等价于旧的 root 列等于场景最大值；root 缺失时仍直接使用真实表达式 GT，不新增伪框、不重标检测类，也不丢弃该表达式。:355、:390、:426 记录成员身份及缺失行数。best_scene_GT_id/class 仍指原生检测词表列表中的对象，不冒充完整场景身份。零重叠也能并列；只有与 iou > .25/.5 同时成立的候选进入 qualified proxy 统计（full_candidate_audit_body.py:92–101）。

3. **生成、候选、原生语义和无更新路径。** 实际执行构建器 prepare_full_candidate_audit.py:6–21 的只读部分，在内存重建的源码与当前生成程序逐字一致。当前生成程序与归档失败程序仅有 3 个 diff hunk，均为上述代理修正、成员/数量元数据和 evidence_limits；模型、输入、排序、Mask、matching 及训练参数未改。run_full_candidate_audit.py:228–243 保留 9508 条 val，:265–267 不丢尾 batch，:332–343 连续记录每行并映射全部 matcher 结果，:329–331、:401–412 保存全部 256 个 query；没有因 root 缺失、错误、未匹配或低排名而过滤候选。native matcher 仅在 models.losses.py:817 调用一次/层，:852–853 的顺序为 proposal、last、0head…4head，:872–885 的 target 使用表达式 box_label_mask，因此抓取 matching[1] 后通过 valid_slots[targets] 映回表达式 GT 槽是正确的。bbs 与 evaluator.py:543、:254–281 一致；Mask 使用原生 scalar adaptive weight 融合 text/query logits 后 sigmoid > .5（models.pv_ground.py:540–553，evaluator.py:596–605），superpoint 成员计数等价于逐点展开。程序 :291–294 关闭梯度、设 eval 并保存 state，:419–422 验证 state 不变及 native evaluator 累计量。AST 中不存在 backward、step、get_optimizer 或 model.train 调用。没有构造优化器或权重更新路径。此段三个 native 文件均位于 complete_tail_fused_retry/source/imported/。

4. **本轮 CPU 证据。** 使用已有本地 Python/NumPy 2.5.3，未安装依赖或导入 Torch/native model。6 个完整脚本及加函数外壳后的 body 均通过内存 compile。直接提取并执行原生 dataset 的 _get_target_boxes / _get_scene_objects 方法，分别构造明确标记为合成的 present-root 与 absent-root 场景，每种保留 256 个候选。两种场景的表达式 root 框和 Mask 完全相同，场景有效实例 ID 分别为 [0,1,2] 和 [0,2]。query 0、127、255 的 root IoU 为 [1/3,0.6,1]，代理均为 [false,true,true]；覆盖竞争对象胜出、正重叠并列、root 独占胜出及零重叠并列。代理与上述 GT 并集最大值判定逐项相等，旧断言在 absent-root fixture 必然失败。实际执行 recount_full_candidate_audit.py:45–46、:85–86 的 AST 语句，present/absent 代理及混合缺失总数 1 正确，故意错误的代理和缺失总数均被拒绝。

   对**旧真实** sanity 保存的 8×256=2048 个框独立用 float64 重算，.25/.5 阈值布尔判断完全一致，命中数为 4/3，最大 IoU 数值误差为 1.105977841975303e-6；matched-root / matched-other / unmatched 合格候选计数分别为 .25：[8,0,431]，.5：[7,0,197]。保存的旧新代理逐项相同。对失败 full 已保存的 44 个完整 chunk（连续 352 行、90,112 个候选），旧新代理也逐项相同。这些结果只验证历史可用数据和本次修正的 CPU 语义，**不替代新版 real8**，也不是失败 full 的完整结果。

5. **重跑与归档门槛。** run_candidate_audit_authorized.py:15–17、:23 使用新的 sanity_gt_scope_v2/full_gt_scope_v2 本地和远端目录；:24–31 先检查父 fused 实验完整状态及正式 9508-row receipt，full 再要求新版 8-row pass、state 未改变且保留 256。旧 candidate_audit/sanity 不能满足新路径门槛。审查时两个 v2 目录均不存在。**当前允许 real8；full9508 仍须等同一份已审代码完成 real8、CPU recount 通过，并核对保存的 sanity/run.py 与待启动源码一致。** launcher 本身只检查所列 receipt 字段，没有自动比较源码或读取 CPU_RECOUNT；这些是执行流程中的检查，不能把旧通过记录当新通过记录。本次未建议新增绑定框架。旧 sanity/full 的 57 个文件在 CPU 检查前后 SHA-256 全部不变；旧失败 run.py、log、exit、partial chunks 和旧报告均未改写。

6. **证据边界。** 场景列表受检测词表及原生槽上限约束；“root 优于该列表”不证明完整场景唯一实例、语义正确或物理同一身份。native eval Hungarian 的 unmatched 也不等于物理背景，低 IoU 不能自动标成语义负样本。这里使用 GT 的代理只做诊断，不是可部署选择规则，也不构成重训 ablation。CPU recount 对所有保存框验证的是阈值判断精确一致；连续 IoU 数值允许已有浮点误差，Mask/scene maximum 仍依赖保存的 native 结果，当前 NPZ 不足以重新恢复全部场景 GT。recount_full_candidate_audit.py:78–93 对 archived formal 只统计 selected-query 与 hit 变化，不是连续 box/Mask 数值一致证明。完整新版 9508 行是否通过须由后续实际运行判定。

Blocking issues：[]。所需代码修改：无。full9508 尚未满足的运行前置条件不构成阻止 real8 的代码缺陷。

审查对象 SHA-256（仅记录现有文件身份，未引入任何哈希绑定机制）：

| 文件 | SHA-256 |
|---|---|
| candidate_audit/full/run.py | 2132287cd9bce49ca0774fe1e9246d2b777e99b227dd86e7df8e3ba7fcb7541e |
| candidate_audit/full/run.log | 24777bd0a638148bba41109a692b621546b9912ab4caac22fb6f5547ade6932e |
| candidate_audit/full/run.exit | f1b2f662800122bed0ff255693df89c4487fbdcf453d3524a42d4ec20c3d9c04 |
| candidate_audit/sanity/run.py | 2132287cd9bce49ca0774fe1e9246d2b777e99b227dd86e7df8e3ba7fcb7541e |
| candidate_audit/sanity/CPU_RECOUNT.json | 7cadfa5e928224a02a9ca40aa583832aec4e66a510f54fafd24eb3ede68e8390 |
| complete_tail_fused_retry/source/imported/src.joint_det_dataset.py | 3afebfc69232f9547e6810563ba6dd4f34a0f770f929df2f4203b21493de0e3d |
| full_candidate_audit_body.py | 7caae21a84dc3c18db706c1df24e6fcb9c29eef99be2f353753e60b412f29642 |
| prepare_full_candidate_audit.py | 49614085e34f04d8616b5a9808b6eb31cf3d51acabb635a7bfb01659db33dfed |
| run_full_candidate_audit.py | 9dc2498f844a0c6a92337fec0bb5600615d0924322aa601f3939bcee462d5f99 |
| run_candidate_audit_authorized.py | a2e417b7801c4aa07cec6c67fa964be865db10a7526cb41126f4264b527b3a16 |
| recount_full_candidate_audit.py | 8a262ce36316565156d1d725ec45ab0f036f28c5fdcb87edbfc0e73dae909b72 |
| CANDIDATE_AUDIT_CODE_REVIEW.md | 50fe91429b1cd7215e79afbcf36a4aebd0aeaae1c8cb5994288ec14367da4772 |
| run_pvground_tail_support_retry.py | 54f9df009cc5b67f27ab52745143e85d1fcd5301432b69f664f5e944999c22e6 |
| complete_tail_fused_retry/source/imported/models.losses.py | 920051cc7d1009493829eebbb6185da834a4efd037e9d73ad43ce918a81031de |
| complete_tail_fused_retry/source/imported/evaluator.py | 39c8de921aff9fea6e2a9a39f68e5634bdd765b1bfd065f0ee3c6c9dac931677 |
| complete_tail_fused_retry/source/imported/models.pv_ground.py | cc17e6c235075c4be24c752bb76dcafe0273c7a036f3dcf6bedb3696379aaa58 |

本报告对应的 CANDIDATE_GT_SCOPE_REVIEW.json 保存详细 CPU 结果、精确 diff、全部 57 个历史产物哈希及审查归属。
