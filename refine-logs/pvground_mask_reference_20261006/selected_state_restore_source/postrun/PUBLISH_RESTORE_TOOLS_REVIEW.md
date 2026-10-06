# Doc84 重载工具发布器独立源码审查

结论：**SOURCE_ONLY PASS，0 个阻断项，尚未发布**。本记录单独审查 `publish_restore_tools_authorized.py`，不将其追认到旧 `RESTORE_SOURCE_REVIEW`。这是同一 reviewer 上下文的后续审查，`fresh_context=false`、`same-context followup`、`same-family/provisional`。请求的 `gpt-6-astra / max` 不等于实际后端 attestation。

## 已核对

- 当前三个本地 HEAD 为 `0d0d22bbfaaabad6d758a8205795ceec0c7d9669`、`3fad5cd1e6c20c29da8818f1ede29c9940594f6a`、`ebfd5352b59f1808c0a281683c504beb018ac62e`，均匹配 `fit_launch_publication.json`；工作树干净。三个仓库及 desktop 的 Doc83 均为 SHA `17b56cbc76b42f6ceae6b871627a9817dcc7637619b2c6746909be9fde97d1ba`，不存在 Doc84。
- Doc84 以原工作副本字节为前缀追加，要求章节只出现一次；所有四份本地文档与远端文档最终比较相同字节。Git 内的文档保留原 tracked 前缀。当前三个 tracked 文档一致，SHA `71989262fe4fa421824e2310599fc8e30df4f8ff9759993bff88d1b3bc570354`；与工作副本的差异已核实仅为现有 `core.autocrlf=true` 的 CRLF→LF 规范化，不是文档内容变化。
- 旧发布记录、活动 continuation 指针、observer27853、本地无 `complete_fit`/`analysis` 等前置条件均由发布器检查。Doc84 将 11:33 的本地 native handle 见证与实际远端进度检查分开；保持首次17:14:14、后续240秒。没有写成当前已完成 GPU probe 或 M1 终态。
- 新文案仅说明四份预声明结果的未来复算和 step0/3723 的未来严格恢复；旧 trained best5616/4511、目标 ACTIVE_UNMET 均未变。没有新增精度、实际 CPU 模型构造、实际 optimizer 恢复或 raw Mask 重算声明。`POSTRUN_PREPARATION.json` 的旧 analyzer SHA 是11:32历史快照；当前 gate 使用新的实际分析审查和重载审查。
- 两个 source gates 在任何 SSH/Git 写操作之前检查 `SOURCE_ONLY`、PASS/WARN、空阻断项及所有受审文件字节。现有重载审查的31项 SHA全部匹配，其中包含当前分析器及其已通过的独立 follow-up。
- 固定 payload 共14个文件，加一份内容为 `** -text\n` 的 `.gitattributes`，总15项。范围是源码、审查、准备清单、历史准备快照与已有 observer handle 见证；没有目录遍历或采集目录的 glob，没有 NPZ、checkpoint、凭据值、env 文件或 `.aris` 上传。SSH 口令仅从环境取得用于认证，不打印、写入 payload 或写入日志。
- 远端先核对旧 Doc SHA，再将 evidence 路径 resolve 后严格限制为 `/root/autodl-tmp/mcln_published_evidence_20261005/pvground_mask_reference_20261006/selected_state_restore_source`；该目录必须尚不存在。每个新文件要求位于其中，以 `xb` 创建并回读核对；不覆盖活动 M1 源码或 observer。远端 symlink 和路径的真实状态仍待执行时验证，本次没有 SSH 查询。
- 源码 payload 同步到前两个仓库；第三个 V99 仓库只更新 handoff 文档。staging 集合受固定清单约束，payload 的 staged 字节逐项回读比较，提交后三处工作树必须干净。仅向第一个仓库的 origin/main 做普通 push，不使用 force，并回查远端 ref。
- guard 当前旧 Doc SHA 恰好出现一次，位于 `expected_previous_sha`；代码要求唯一匹配后只替换这64个十六进制字符。成功记录和 continuation 更新在远端文档、三个提交、main push、四本地文档一致性检查后写入。
- 发布器及其内嵌远端脚本均通过 Python3.7 grammar AST 和内存编译；检查使用现有本地 Python3.12.6。`shlex.join` 只在本地使用，远端脚本只依赖3.7标准库。

## 执行边界和文件绑定

本次只执行本地文件读取、hash、AST/内存编译，以及 `git rev-parse`、`git status`、`git show`、`git check-attr`、`git config --get`。没有运行发布器、SSH、网络、GPU、git add/commit/push、模型加载、训练、通知或权重操作；仅新建本审查及追加本地审查 manifest。

旧重载审查保持原字节：JSON SHA `0de4c4dd66d3471aa928e4fd787d592c0dbce7fa5235ecf2f29a72a063d62193`，Markdown SHA `697ba48beb072416fe1355bd6fc36059bec0e9d5ecef6896625ad4f6b5a74b19`。本次发布器源 SHA 为 `03745acb5588375b202834023e05afa57b22aaf660195c27fed47b3503bc5bc6`。

JSON 绑定所有既有实际 payload 文件和本 Markdown 的 SHA；发布审查 JSON 不能包含自身固定 SHA，因此其完成后的真实 SHA 在 reviewer 完成消息中交付。`.gitattributes` 为固定字节，单独给出 SHA。没有制造发布 receipt 或预写 Doc84；真实发布、远端 evidence 落盘、GitHub main 和新 guard SHA 均未验证。
