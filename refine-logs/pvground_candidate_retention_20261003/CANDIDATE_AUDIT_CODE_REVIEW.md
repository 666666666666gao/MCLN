# Full-candidate audit independent code review

Verdict: **WARN — 无阻塞项；核心实现正确，静态检查及本地 CPU 模拟通过。**

Reviewer: `gpt-6-astra` / `max`; context: `fresh-context`; review_independence: `same-family`; acceptance_status: `provisional`.
Reviewed at: 2026-10-03T01:45:37.031206+08:00

Scope: 源码审查与本地 CPU 检查。未连接 SSH、未部署、未执行 GPU forward、未读取认证 wrapper 或环境密钥，未修改审查对象及活动训练。此结论不是实际 GPU sanity/full 通过记录。

## 已确认正确的实现

- 构建与只读执行：`prepare_full_candidate_audit.py` 在原训练 `step` 定义前截断，生成内容与当前 `run_full_candidate_audit.py` 逐字一致。生成脚本不包含 `backward`、`step`、`get_optimizer` 调用；第 291 行显式关闭梯度，第 293 行设置 eval，第 418 行逐张量核验模型 state 不变。没有构造优化器或更新权重的路径。
- 严格恢复与数据协议：先严格加载作者模型，再加载原 G delta，再严格恢复 fused terminal（生成脚本 134–221 行）；terminal step 为 3723。沿用 val 9508、batch 8、seed 2027、原生 loader/input preparation，包含最后不足 8 行的 batch。
- 候选集合：生成脚本 329–411 行为每行保存全部 256 个 query 的框、coarse 框、bbs score、root Box/Mask IoU、最后层 native matching、scene-overlap 代理及 no-object 概率。未按错误、未匹配或排名过滤；第 256 名仍可进入替代候选统计。
- native matching：`models.losses.py` 852–853、891–917 行的顺序是 proposal、last、0head…4head，因此 `matching[1]` 正确。matcher 的 target 下标经 `valid_slots` 映射回原生 GT 槽位；root 是 GT slot 0，而 scene instance ID 使用 `target_id`，二者没有混用。
- 场景代理：dataset 1086–1119、1121–1157 行表明 root/scene 框使用相同未增强 val 坐标；`all_bbox_label_mask` 保留原实例下标。生成脚本还断言 root 对应 scene 框逐元素一致。最大重叠仅针对原生有效场景 GT 框，`root_joint_best_scene_overlap` 正确保留并列最大值；它不证明物理实例或完整语义正确性。
- native bbs/Mask：bbs 使用二值化 main map 加 modify/pron/rel 再减 other，与 evaluator 218–286、535–553 行相同。Mask 使用同一 scalar adaptive weight 对 text/query logits 融合，再 sigmoid > .5；实际 text 为 `[1,256,S]`、query 为 `[256,S]`。按 superpoint 成员数计算 intersection/union 等价于原生 50000 点展开，另对 selected query 验证逐点精确相等。native evaluator 的 bbs hit 和 Mask 累计量均有最终核验。
- 运行与归档：`run_candidate_audit_authorized.py` 24–31 行先要求当前 fused 完整状态及 9508-row formal receipt，通过 8-row sanity 后才允许 full；40–48 行使用同一 GPU flock，合并 stderr 后流式读取，避免第二条未读输出管道。大数组只在内存中编码并经 stdout 保存到本地 NPZ，不写远端大缓存。chunk 顺序、退出码、receipt/chunk 数都核验；正式成绩文件没有写入路径。
- CPU 复算：保存的 256 个框全部参与 float64 root IoU 重算、两阈值判定核验、matched-root/other/unmatched 计数及错误守恒；first-qualified 原生排名用 score-tie 上下界检查，不假定 NumPy 与 Torch 的并列顺序相同。跨 formal 的身份、root 框与 input fingerprint 要求一致，query/hit 变化独立计数，未强制数值 bitwise 一致。
- G 的含义：`pvground_semantic_assignment.py` 9–23、34–50 行确实仅替换训练期 root IoU > .5 的未匹配 query 的原生 eos token loss。诊断保留此类 query，且没有把“未匹配”解释为物理背景身份。

## 非阻塞项 W1：未汇总连续输出差值

位置：`recount_full_candidate_audit.py:77–89`。

当前 `versus_archived_formal` 只记录 selected-query 变化与 IoU .25/.5 hit 变化。若 query 和阈值判定相同，selected box、IoU 或 Mask IoU 的连续数值仍可能不同，但现有报告不会汇总这些差值。本地合成检查将同一 query 的 formal IoU 从 1 改成 .875、Mask IoU 从 .5 改成 .375，并改变框坐标，当前比较仍返回三项变化数均为 0。

这不改变候选覆盖、matching 或错误分区结果，也不把 fresh forward 写回正式成绩；现有 evidence_limits 已声明 fresh forward 可能有数值差异，因此不是启动阻塞。使用当前报告时只能称“query/hit 是否变化”，不能声称“连续输出数值一致”。若后续需要完整数值对照，直接用已有 NPZ/row/formal 字段增加 selected box、IoU、Mask IoU 的差异行数和最大绝对差即可，无需 GPU 重跑。本轮保持该遗漏明示，不改部署代码。

## 实际完成的验证

1. 五个完整脚本通过 Python 编译检查；构建器在内存中重建的源码与产物一致；无优化器/反向更新调用，梯度关闭存在。
2. 执行实际 stream 接收分支：两块 base64 NPZ 往返、顺序校验及 complete receipt 解析通过。
3. 执行实际 CPU recount（仅把 formal 路径指向明确标记的合成文件）：8 行 × 256 框，覆盖低至第 256 名的正确候选、matched-other、unmatched、全无合格框、精确 .25/.5 边界及相同 score。两个阈值的期望 hit、错误分区与候选计数全部正确；两个 4-row chunk 同时覆盖小尾 batch 的保存/复算形状。
4. 独立读取并重算 `SAVED_ERROR_CANDIDATES.json` 的两个系统 × bbs/bbf × 两阈值，排名区间、错误数、替代数与 loose-only 分区全部一致。same_tail_raw / bbs 的严格错误数为 5051 = 966 + 508 + 980 + 886 + 1711，其中 3340 有替代框；loose-only 1137 = 675 + 462。这里的 rank 是已有 GPU top-K coverage 所确定的区间，不是精确 first rank；也不是物理实例身份结论。

CPU 证据保存在 `candidate_audit_review_fixture_20261003_014237/REVIEW_CHECKS.json`；该目录全部是明确标记的合成审查数据，不是正式实验结果。默认系统 `python` 的启动器损坏，本检查使用已缓存的 `uv run --offline --with numpy python`，没有安装或下载新包。未在本地执行 Torch/native model，因此 native GPU 张量与完整 9508 行的实际通过结论仍由训练结束后已有的 sanity/full 运行给出。

Blocking issues: **[]**。本报告以 exclusive create 写入，不覆盖已有报告。
