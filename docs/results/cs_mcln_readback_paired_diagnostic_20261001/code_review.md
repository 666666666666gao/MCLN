# R fixed-prediction diagnostic review

Reviewer: `/root/readback_paired_diagnostic_review`; fresh context, `gpt-6-astra`, reasoning `max`.
Review independence: same-family. Acceptance status: provisional.

The first review identified one blocking score-path mismatch: the draft combined token maps before reduction, whereas the native evaluator reduces each map separately. Aggregate hit equality would not guarantee per-row Query/rank equality. This was corrected by capturing the actual score tensor passed by the native evaluator to its own ranking method, reusing that method for top1 and Full256, and asserting each row's ranking prefix equality. No evaluator or training source was edited.

Final reviewer response:

> 当前两文件复核通过：PASS，same-family / provisional。
>
> BLOCKING：无。已直接捕获原生 evaluator 的实际分数，复用原生排序并逐条验证 top1／Full256 前缀一致；每批缓存正确重置。此前问题已解决。
>
> R hook、语义头重放、共享框与 Mask、GT／阈值、oracle、排名和汇总逻辑正确。正式指标不匹配时会先保存诊断 JSON。
>
> 启动脚本的训练完成检查、best 规则、空闲 GPU 检查、冻结源码导入及 first12→full9508 顺序符合计划。
>
> NON-BLOCKING：无须追加代码修改。
>
> 本次确认的是当前 11359／4902 字节版本，两文件 AST 均通过。未修改文件、运行 GPU 或验证精度；真实 sanity、全量重放及正式指标一致性仍待执行。

Reviewed SHA256:

- `diagnose_cs_mcln_readback.py`: `7ab14d5fbf2efe582662214b65981bbfdeec6ac2e08afb7bb695baab3d4cc933`
- `run_cs_mcln_readback_posttrain_diagnostic.py`: `980af8d38775afa97346c8bad7403b3b3d08e2f1e4d8de7f1e1c30115893030e`

This review is source-level evidence, not a GPU sanity run or accuracy result.
