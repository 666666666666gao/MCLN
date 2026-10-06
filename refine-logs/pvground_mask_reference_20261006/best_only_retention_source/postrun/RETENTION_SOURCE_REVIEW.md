# 闭合后 best-only 权重清理工具源码审查

结论：**SOURCE_ONLY PASS，0 个未解决阻断项；未执行清理**。本报告单独审查 `retain_metric_best.py` 与 `retain_metric_best_authorized.py`，不修改或追认已有分析、重载、发布审查。当前 reviewer 为同一上下文后续审查，`fresh_context=false`、`same-family/provisional`；请求的 Astra/max 不是后端 attestation。

## 已解决的问题

初版远端清理器只检查 terminal checkpoint SHA，遗漏了本轮明确要求的训练日志 SHA。M1 生产源码 `run_geometry_fit.py:586` 实际写入 `train_log_sha256`，因此这不是假设性风险。root 已在 terminal 分支加入 `sha(root/arm/'train.jsonl') == fit['train_log_sha256']`，位于所有删除之前。删除该新增单行可精确复现初版 SHA `f34f6794582ef8a7cf0d8f76e4d4fc8a633430a47f66291a028c5c2c236c9382`；没有其他代码变化。

最终受审 SHA：

- `retain_metric_best.py`：`d7c38bda11a13fc974460e0ac0664dd288b69b12bef9566a17533b94243d5ec0`
- `retain_metric_best_authorized.py`：`1bc3be71301ab9799993448411696cba204c32054f1ea8e78140d55dd8bb592e`

## 正确性和删除范围

- `paths` 严格由五个候选组成：两臂各自的 `initial.pth`、`terminal.pth`，加固定旧保护4511 delta。根目录及旧父实际解析路径必须等于指定绝对路径；四个 M1 路径必须 resolve 后不变且位于本轮根目录。PV、G、V99 不进入删除集合；G 在清理前后另行核对 SHA。
- 唯一删除语句是固定集合上的 `path.unlink()`，位于所有五个候选的身份、来源、结构、formal 指标和 winner 校验完成之后。没有递归、glob、fallback、try/except、checkpoint 复制或负权重归档。日志、NPZ、原始证据不在删除集合中。
- wrapper 要求真实 `fit_wait` 显示 observer 闭合、controller 不存活且 exit0；远端再要求 controller exit0、complete/protected-parent 状态，以及精确 controller 命令的 pgrep 不存在。`fit_wait` 字段与现有 observer 生产源码一致。
- wrapper 要求四份9508实际复算的 summary、相同训练顺序，并在删除前要求所有候选按框类型的 CPU/native threshold flips 和 all256 oracle mismatches 为零。没有重新推理、重排行为或 GT 质量门控。
- 远端按固定 parent → native initial → native terminal → fused initial → fused terminal 顺序建立候选；比较键是 native Acc@.50 命中数、保护旧父优先、Acc@.25 命中数。其顺序和 tie 行为与当前分析器一致，包括新候选完全相同指标时的稳定顺序。旧父胜出和新 initial/terminal 胜出均有合法路径。
- 四个候选均检查真实 step、spec SHA、reference mode、共同 output reset、history 和来源字段；完整 geometry 10 张量、456102 参数、名称/shape/dtype，以及实际 CPU head 严格 state load 均在未来执行时校验。initial hidden8 与保护父逐张量一致，output2 全零，optimizer state 为空。terminal 为 step3723、hidden累计14892、output3723、29778唯一训练行和10个 optimizer state/step3723，并核对 terminal 与 train log SHA。
- 远端清理器的 optimizer 检查是 state 数量和 step 等元数据；新 winner 的完整 AdamW moments/groups 恢复必须由此前真实 M2 重载 receipt 提供。不能将该 CPU head 检查写成新一次完整模型/AdamW 重载。
- 新 winner 还必须有终态审计绑定的 `selected_candidate_CPU_restore.json`，其 summary SHA、arm/stage、真实 step 与 winner 对应；远端要求重载 checkpoint SHA 与待保留文件一致、完整模型逐张量一致、optimizer exact，以及 direct factory 不再依赖旧 geometry。只有这些门槛完成，旧4511才能作为非 winner 删除。旧父胜出时不要求制造新候选恢复记录。

## 未来终态审计契约

本轮明确要求未来真实终态审计提供 `execution_scope='TERMINAL_SOURCE_AND_ACTUAL_ARTIFACTS'`、`fresh_context=true`、PASS/WARN、空 `blocking_findings`，以及 `actual_file_digests` 中相对路径 `analysis/SUMMARY.json` 的 SHA；新 winner 还必须绑定 `selected_candidate_CPU_restore.json`。另需真实 `TERMINAL_REVIEW_CALL.json` 的 `result_received`。源码与这个明确契约相符。

历史审计文件的 scope 和 digest schema 并不完全相同，不能复用它们作本轮通过证据；本次不添加兼容或 fallback。当前没有本轮 fit_wait、summary、终态 audit/call、selected restore 或 retention receipt。此报告没有生成任何这些未来结果。

## 静态检查与边界

两份最终源码均通过 Python3.7 grammar AST 与内存编译，使用现有本地 Python3.12.6。`shlex.join` 仅由本地 wrapper 使用；远端代码无该3.8+ API。wrapper 只上传已审单个脚本并回读，环境口令仅用于未来 SSH 认证；没有输出凭据或新增通知。

本次没有执行 SSH、网络、GPU、模型构造、`torch.load`、optimizer 恢复、训练、删除或归档；没有修改活动 M1 源码或唯一 observer，也没有改已有审查字节。只写本审查及追加本地审查 manifest。SOURCE_ONLY PASS 仅表示工具源码通过，不能称为真实终态通过、已清理、已恢复或已达到新精度。

最终输入路径、实际 SHA、已解决问题和执行范围见 `RETENTION_SOURCE_REVIEW.json`。
