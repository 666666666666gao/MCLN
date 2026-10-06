# Doc85 清理工具发布器源码审查

结论：**SOURCE_ONLY PASS，0 个未解决阻断项；未发布、未清理**。本次为同一 reviewer 上下文的后续审查，`fresh_context=false`、`same-family/provisional`，无实际后端 attestation。旧分析、重载、清理及 Doc84 发布审查保持原字节。

最终发布器 SHA：`458f58903a36dc1bfbf0d93eda66802053d5f1e3f84406c3d2cd17baa3f2e220`。

唯一文案修正：原句容易将五份候选描述成做相同的 head/formal/train-log 检查；最终分别说明五个固定候选核对身份、四个新候选核对头状态和正式结果、两份 terminal 核对训练日志 SHA。逆转这一段即可精确复现初稿 SHA `f6b32e365f30e2e5b373207370a82ab17b643c755375f0e4d64bb68888ddeaed`，发布执行逻辑未变。

## 本地核对与代码结论

- 三个当前 HEAD 为 `3d64dde06cc3e739262f538d0ddb2564793cddf2`、`e86e783142e7cd2afa368c35f13db8cdb9c59e4c`、`8cfecad5e6e24881c76e8639a4b7f2b470145849`，均匹配真实 Doc84 发布记录；工作树干净。
- 三仓库及 desktop 的 Doc84 字节一致，SHA 为 `beeb8dbeb5a018f88268898b1fa6ef4e527f859678c7169fc76ff4f34cd5c5a0`，尚无 Doc85。三个 Git tracked 文档也彼此一致，SHA 为 `e6d1a542a09bbdf56edfe5166bf8875b7b49e81d16e383dd37b050fb2b951840`；与工作副本的差异仅为已有 CRLF→LF 规范化。发布器分别保护工作副本和 Git 历史前缀。
- `RETENTION_SOURCE_REVIEW.json` 为 SOURCE_ONLY PASS、空阻断项，其27份受审输入全部匹配。准备记录指向该真实审查 SHA，明确未执行清理、无新精度。当前 continuation 指针与 observer27853 条件一致，本地无 complete_fit、analysis、清理 receipt 或 Doc85 发布 receipt。
- payload 固定为8个源码/审查/准备文件，加一份 `** -text\n` 的 `.gitattributes`，总9项。源码发布到前两个仓库；第三个 V99 仓库只更新交接文档。没有 checkpoint、NPZ、env 值、凭据或 `.aris` 归档，没有目录 glob。
- 远端要求旧 Doc SHA 相同，并将 evidence resolve 后严格限制为 `/root/autodl-tmp/mcln_published_evidence_20261005/pvground_mask_reference_20261006/best_only_retention_source`；目标目录必须不存在，文件必须在其中，以 `xb` 新建并回读核对。该约束适用于既有 refine-logs symlink；本次没有 SSH 检查远端状态。
- 本地 staging 限固定清单，payload 在 Git index 中逐项比较原始字节，文档 staged 内容必须保留旧 tracked 前缀；三个提交后要求工作树干净。向第一仓库 origin/main 的 push 不带 force，并回查远端 ref。四本地与远端文档最终比较同一新字节。
- guard 中旧 Doc84 SHA 当前恰好出现一次，位于 `expected_previous_sha`。代码要求唯一匹配后仅替换该 SHA；成功记录和 continuation 更新在提交、main push及文档一致性验证后写入。
- Doc85 只报告准备与 SOURCE_ONLY 审查。旧5616/4511对应的59.0660%/47.4443%计算正确，没有新增精度或实际恢复/清理声明。当前进度检查时刻没有被执行；publisher 不调用清理器、模型、训练脚本或 observer。
- 本次发布属于已有明确授权的源码/文档同步范围。SSH 环境口令只用于认证，不写入 payload 或日志；无新通知 API。发布不执行授权中的未来删除动作。
- 最终发布器及内嵌远端脚本均通过 Python3.7 grammar AST 与内存编译；检查使用本地 Python3.12.6，`shlex.join` 仅在本地使用。

本次仅做本地读取、SHA、AST/内存编译和只读 Git 查询。没有网络、SSH、Git 修改、publisher 执行、训练进度查询、模型加载或删除；仅写本审查及追加本地审查 manifest。真实 Doc85 发布、远端落盘、新 main 和 guard SHA 均未发生或未由本审查验证。

JSON 绑定最终发布器及实际 payload/input 字节。本审查 JSON 自身无法包含固定自身 SHA，其完成后的实际 SHA 在 reviewer 完成消息中单独交付；其余 payload（含本 Markdown）均有实际 SHA，固定 `.gitattributes` 字节亦单列。
