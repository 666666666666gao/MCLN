# Experiment Audit Report

日期：2026-10-03（北京时间）  
审计对象：已完成的 ScanRefer tail_fused full256 诊断。  
审计者：gpt-6-astra，reasoning_effort=max，fresh native agent `/root/pv_full256_result_integrity`。  
review_independence：same-family；acceptance_status：provisional；executor_model：unavailable。

**Overall verdict：WARN。Integrity status：warn。Blocking findings：[]。**

已独立检查完整原始结果，支持按下述限定如实封存该诊断；没有发现需要修改模型、重跑 GPU 或阻止该事实性封存的材料错误。本审计不重做此前 fused 训练审计，不把旧审查结论当成本次结果证明，也不将代码审查 PASS 当作性能 PASS。

所有相对路径以 `C:/Users/gb/.codex/tmp/pvground_fused_support_20261002` 为根；下文 D 指 `candidate_audit/full_gt_scope_v2`，S 指 `complete_tail_fused_retry/source/imported`。

## A. Ground Truth Provenance：PASS

表达式来自数据集的 ScanRefer annotation JSON，root 的实例编号来自 `object_id`；root 框和 Mask 分别取该实例的原生 bbox 与点成员，不由模型输出构造。证据：`S/src.joint_det_dataset.py:585–646,1086–1119,1324–1334,1387–1392`。输入准备仅向模型提供点、文本、检测结果和 superpoint；GT 在前向后进入原生 loss/evaluator 与离线诊断，见 `D/run.py:269–286,308–336`。butd_gt/butd_cls 均关闭，`D/run.py:138,238–241`。

场景 GT 是另一个受检测词表和最多 132 个原生槽限制的列表，不能假定包含表达式 root（`S/src.joint_det_dataset.py:1121–1157`）。修订版独立计算 root IoU，再与该列表的最大 IoU 比较，且单独记录成员身份（`D/run.py:335–355`）。完整 9508 行中 87 行 root 不在该检测 GT 列表；这些行及其所有候选仍在结果中。零重叠也可能并列，因此只有另加正 IoU 条件的统计才构成“合格 root 重叠代理”。

本地归档的六个原生导入文件 SHA-256 均与本次 imports.json 及归档 formal 的导入记录一致；七个附加模块与 spec 的摘要一致，split_protocol 摘要一致，141 个实际场景均有 manifest 中的 val superpoint 条目。全部行的 row_id、scan_id、target_id、root_box、point_sha256 与归档 formal 精确相同。本轮没有重新读取远端原始 annotation/点云/Mask 文件或父 checkpoint 当前字节；这是源码与归档输入身份的核验，不是对原始数据重新下载后的独立重建。

## B. Score Normalization：PASS

框 IoU 使用普通交并比，体积交集除以两个框体积之和减交集；阈值为严格 `>.25`、`>.5`。命中率分母为全部 9508 条表达；错误中存在备选的百分比分母分别是 3942、5102 条实际错误。没有用模型自己的最大值、均值或预测数量重标性能。证据：`D/run.py:299–305,379–388,420–422`；`S/evaluator.py:289–303`；`analyze_completed_candidate_audit.py:22–43`。

bbs 的 softmax 是原生候选评分的一部分，不是报告 IoU/命中率的自归一化。分数公式与原生 evaluator 的 main、modify、pron、relation 加和及 other 减项一致（`D/run.py:324–328`；`S/evaluator.py:254–281,535–550`）。

## C. Result File Existence and Arithmetic：WARN

审计者用独立的本地 NumPy/float64 代码逐块读取全部 **1189 个 NPZ、9508 行、2,434,048 个候选**，未调用作者的会写报告脚本。文件序号为 0000–1188，前 1188 块各 8 行，末块 4 行；行号连续 0–9507，无重复、缺行或候选筛除。数组共 129,459,547 字节，与 INTAKE 每块文件名和字节数一致。全部数组形状、有限值、框正尺寸、IoU/概率范围、选择字段、匹配槽、代理布尔量、逐行资格计数及摘要算术一致。实际退出码为 0，完成日志与 receipt 精确一致（`D/run.exit:1`；`D/run.log:8–26`）。

独立得到以下原生 GPU 口径结果；相互重叠的备选类别不可相加：

| 数量 | @.25 | @.50 |
|---|---:|---:|
| 实际选择命中 | 5566 | 4406 |
| 实际选择错误 | 3942 | 5102 |
| 错误中存在合格备选框 | 3357 | 3408 |
| 错误中存在未匹配合格框 | 3254 | 3209 |
| 错误中存在同时满足 root 重叠代理的未匹配合格框 | 3229 | 3201 |
| 错误中全 256 无合格框 | 585 | 1694 |
| 全部候选中的 matched-root / matched-other / unmatched 合格数 | 8883 / 0 / 370849 | 7738 / 0 / 249347 |

错误行首个合格候选的排名分箱 2–16、17–32、33–64、65–256、none 分别为：@.25 **628/544/1138/1047/585**，@.50 **959/533/1072/844/1694**；32 名之后分别 2185、1916。所有已选 query 都是保存 bbs 分数的最大值；全部 first-qualified rank 的独立分数上下界相等，没有影响这些排名的同分歧义；top-k 标志与实际 first rank 一致。八个示例的行号、query、IoU、Mask IoU、匹配状态与分数排名均逐项核对。

未匹配且满足 root 重叠代理的候选在 `(0,.25]`、`(.25,.5]`、`(.5,1]` 三个区间分别为 **65974/114046/248601** 个，覆盖 **7778/6945/7603** 条表达。严格错误中，额外要求该代理的 unmatched Box>.5 且 Mask>.5 备选存在于 **2911** 行；不要求该代理时是 **2919** 行。摘要字段的实际条件在 `analyze_completed_candidate_audit.py:55–66`，本次已核对更新后的 `candidate_completion_appendix.md:32` 明确写出该限定。

**必须保留的数值 WARN：** float64 CPU 与原生 GPU 的最大框 IoU 差为 **5.394267800662433e-06**。唯一阈值不一致位于 row3499/query189，CPU=**0.2500001376978458**、GPU=**0.24999968707561493**；它未被选择，matched_slot=-1。@.25 合格候选总数 GPU 为 **379732**、CPU 为 **379733**，差异全在 unmatched（370849 对 370850）；@.50 两者均 **257085**。全部已选 .25/.5 判定、所有候选 .5 判定和错误中存在备选的表达数一致。审计复现了原始差异，没有容差抹平、阈值放宽或改写原生决定。证据：`D/CPU_THRESHOLD_DIFFERENCES.json:5–31`；`D/CPU_RECOUNT.json:9–24,46–73`；`recount_full_candidate_audit.py:28–39,53–113`。保存的严格旧 checker 哈希与失败记录一致，其实际 AssertionError/exit1 被保留。

归档 formal 对照的 query 变化 **1** 行（row2236：63→196，IoU 0.15109270811080933→0.1647103875875473），.25/.5 命中变化均 **0**。选中框的连续值在 **9503** 行不完全相同，选中 Mask IoU 在 **2** 行不同；不得声称连续框或 Mask 逐位等价。formal rows 的摘要及 receipt 中的 REC/Mask 命中也由原始行重新核对。此诊断不能将前向数值变化记为方法增益。

## D. Live Evaluation and No Update Path：PASS

本次真实执行路径先 model forward，再调用原生 criterion，捕获七次 matcher 返回；原生层序为 proposal、last、0head…4head，因此 `matching[1]` 对应最后层。将压缩 target 下标映射回 `valid_slots[targets]` 正确（`D/run.py:308–343`；`S/models.losses.py:817,849–885,891–917`）。所有实际行有效 GT 槽均为 **[0]**，每行恰有一个 root 匹配和 255 个未匹配候选。因此 matched_other=0 是此数据与目标构造的结构性结果，不能证明多实例训练保护或完整场景身份判别已通过。

原生 evaluator 实际被调用，累计 REC 与本次 GPU 计数的断言通过；Mask 使用原生 adaptive weight 融合 logits 后 sigmoid>.5，再按 superpoint 点成员计算交并比（`D/run.py:321,357–375,420–425`；`S/evaluator.py:194–206,594–605,646–653,897–902`）。所有保存 Mask/场景代理字段和派生计数均已检查，但 NPZ 没有原始 GT 点成员及完整 scene boxes，因此本轮不能在 CPU 从零复算每个候选的 Mask IoU、scene maximum 或重新解完整 Hungarian cost。

`D/run.py:291–294` 关闭梯度、设 eval、保存 state，`:419` 逐项检查 state 未变，成功 receipt 位于这些断言之后。AST 中没有 backward、step、get_optimizer、Adam/AdamW/SGD 或 model.train 调用；`S/main_utils.py:268–280` 只构建 criterion，优化器构造是另一个未调用的方法。本诊断没有构建优化器、反向传播、更新权重、新建验证训练标签或裁剪候选。已有 G 的训练代码已替换部分 IoU>.5 未匹配 query 的 CE（`complete_tail_fused_retry/source/pvground_semantic_assignment.py:9–50`）；本次诊断没有新增该实现。

## E. Scope and Failure Preservation：WARN

实际范围为 **一个 seed2027、一个 tail_fused 终点、一个 ScanRefer 长期开发验证集、141 个场景、9508 条表达**。它支持全候选离线几何诊断，不能证明重训增益、多 seed 稳定性、独立测试泛化、Nr3D/Sr3D 达标或一个可部署评分器能兑现 oracle 差距。现有 5566/4406 对应 58.54017669%/46.33992427%，不能标为达到 50% 的严格 REC 目标。

v1 full 的真实 exit1 与 root-membership 断言保留（`candidate_audit/full/run.log:8–13`）；实读 44 个旧完整块，连续 352 行/90112 个候选，没有将其算作完整成功。新 sanity_gt_scope_v2 的 8 行、2048 个框、exit0 及 .25/.5 CPU 判断独立核对通过；其结果不代替 full 的 9508 行核验。sanity/full/当前待执行源码字节一致，SHA-256 为 `9dc2498f844a0c6a92337fec0bb5600615d0924322aa601f3939bcee462d5f99`；该源码身份记录明确发生在 full 启动之后（`candidate_gt_scope_v2_source_identity.json:2–7`），不能回填为启动前见证。旧 source-only PASS 不承担本次性能验收。

更新后的 appendix 对 full256、浮点差异、root 范围、2911 的代理条件、root-only 匹配及 formal 连续输出限定与实际产物一致。publisher 仅作为待执行发布代码检查：它要求实际 full exit0、完整数量、sanity/source 一致及无阻塞审查，并纳入摘要、失败记录和本审计（`publish_completed_candidate_audit.py:17–33,49–84`）。本审计未执行发布，也不证明远端或公开仓库已同步；此前 fused 训练结果的进一步结论仍属于原审计范围。

## F. Evaluation Type：real_gt（附有限范围的几何身份代理）

root Box/Mask 使用数据集真实 GT，分类为 **real_gt**。root IoU 不劣于“检测词表/槽范围筛选后的 scene GT 最大值”只是一种明确限定的 **GT 几何代理**，不证明唯一物理实例或完整表达语义；unmatched 不等于真实背景，低 IoU 也不能直接推出负语义标签。它不是模型生成参考的 synthetic_proxy，但其身份解释必须保留 proxy 限定（`D/run.py:429–433`；`candidate_completion_appendix.md:32,38–40`）。

## Action Items and Claim Impact

- 无阻塞修改项；保留 GPU/CPU 两套原始阈值计数、原严格 CPU 失败以及 v1 GPU 失败，保持 WARN。
- 支持“完整保留 all256、部分选择错误内已有几何合格备选”；支持本表具体计数。
- “2911”仅支持带 root 重叠代理的联合 Box/Mask 条件；matched_other=0 不支持多实例职责保护已经验证。
- 不支持“未匹配均为背景”“备选均为语义正确答案”“可直接转正验证标签”“连续输出完全等价”“该诊断证明新训练策略有效”等超出证据的结论。

审计只做本地读取、CPU 数值/哈希/AST 检查，并写本 Markdown 与同名 JSON；未使用 SSH、网络、GPU、模型前向或凭据。全部 NPZ 的有序文件名/字节数/SHA-256 清单摘要为 `b5a03f9a28012504841e916a62615b4837e4a0aaab7b53ebb88d047ff7cf431d`，定义为按文件名排序的 `name\tbytes\tsha256\n` UTF-8 串的 SHA-256；详细审查输入哈希和独立统计保存在 JSON。完整响应原文交由父执行器按 review-tracing 归档，本子审查不另写 trace 文件。
