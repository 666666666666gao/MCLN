# Publication resume source review

**SOURCE_ONLY PASS — blocking_findings: []**

本报告是同一 reviewer 上下文的最小源码 follow-up，`fresh_context=false`；不重新进行终态实证审查。原 `EXPERIMENT_AUDIT` 的 WARN、无阻断项结论及两个报告字节均保持不变。模型请求为 gpt-6-astra / max，后端身份未证实，same-family / provisional。

当前复核源码：`postrun/publish_reference_terminal_resume.py`，SHA256 `05fbef72348e3870311052fd88b345e71c3c6bcd83a2b08d96139c5c13391a98`。生成器 SHA256 为 `c5661df1897df01d640e98a5032672443fdb73217934fb6fdaf680de4a35cb2f`。

## 实际状态与恢复范围

三个仓库 HEAD 仍分别为 `109269e82892e006b36ef014a89c5379d033b80a`、`b7ab48e64e7dbd62d3ca42a409742d948b907d53`、`e4a44da3d43c20f9b3d1e8de38ef5d4b0b118636`，与 Doc78 凭据一致。主仓库已暂存本轮文件，MANIFEST 中有且仅有一条真实闭合发布条目；PV 尚无该条目，R 只有文档修改。实际 `terminal_publication.json` 尚不存在。恢复源码要求原 HEAD 未改变，沿用四份已写 Doc79，不重新追加文档。证据：[resume:13](C:/Users/gb/.codex/tmp/pvground_support_reference_20261005/postrun/publish_reference_terminal_resume.py:13)、[resume:37](C:/Users/gb/.codex/tmp/pvground_support_reference_20261005/postrun/publish_reference_terminal_resume.py:37)、[resume:46](C:/Users/gb/.codex/tmp/pvground_support_reference_20261005/postrun/publish_reference_terminal_resume.py:46)。

四份本地 Doc79 都是 2,318,744 字节，SHA256 为 `001e8e40bcd91f8d2aff3a5be320ef9d5a4868d0e599e2e65c88ae717fdbdbc1`，Doc79 标题均只出现一次。主仓库当前 Git index 文档保留旧 Git 文档的完整前缀。恢复源码将通过远端 CPU 重新核对远端 Doc SHA，并要求退出 0；之后仅上传四份新增小文件并逐份读回，不重传 complete 原始大文件。证据：[resume:70](C:/Users/gb/.codex/tmp/pvground_support_reference_20261005/postrun/publish_reference_terminal_resume.py:70)、[resume:78](C:/Users/gb/.codex/tmp/pvground_support_reference_20261005/postrun/publish_reference_terminal_resume.py:78)、[resume:109](C:/Users/gb/.codex/tmp/pvground_support_reference_20261005/postrun/publish_reference_terminal_resume.py:109)。

## 已修复的问题

首个恢复版本（SHA256 `3c6082aba23d3eee0ffbbbe229b89b745f53aa0549438a0ae602d67afc695e52`）枚举全部 analysis 文件，却只复制及上传中断凭据和恢复源码。本次要求写出的两份新审查报告会进入 payload，但两个仓库缺少副本，导致提交前读取失败，也不会上传。

当前版本在复制及上传列表中明确加入 `analysis/PUBLICATION_RESUME_REVIEW.json` 和 `analysis/PUBLICATION_RESUME_REVIEW.md`，生成器同步修正。这两份报告将在所有 payload 字节检查及 stage 之前写入两个仓库并上传、读回，因此上述实际阻断已消除。证据：[resume:60](C:/Users/gb/.codex/tmp/pvground_support_reference_20261005/postrun/publish_reference_terminal_resume.py:60)、[resume:78](C:/Users/gb/.codex/tmp/pvground_support_reference_20261005/postrun/publish_reference_terminal_resume.py:78)、[resume:90](C:/Users/gb/.codex/tmp/pvground_support_reference_20261005/postrun/publish_reference_terminal_resume.py:90)、[generator:56](C:/Users/gb/.codex/tmp/pvground_support_reference_20261005/postrun/prepare_terminal_publication_resume.py:56)。本报告末尾仅保留一个换行。

## 原始字节、Git 与状态更新

- 两份 complete/SOURCE_REVIEW 历史报告与源文件、两个仓库工作树字节相同，主仓库 index 也相同；没有修剪或重写。第一层严格空白检查新增排除这两份原始报告；第二层仍检查全部 stage，仅关闭 blank-at-eof。对当前主仓库 index 实际执行这两个只读检查，退出码均为 0。证据：[resume:103](C:/Users/gb/.codex/tmp/pvground_support_reference_20261005/postrun/publish_reference_terminal_resume.py:103)、[resume:107](C:/Users/gb/.codex/tmp/pvground_support_reference_20261005/postrun/publish_reference_terminal_resume.py:107)。
- 三个仓库实际修改路径均落在计算得到的 stage 集合内。此前已传输的全部 payload 工作树字节与本地原文件一致，主 index 已存在 payload 的 Git blob 哈希与按原字节独立计算结果一致。主仓库和 PV 只 stage 文档、MANIFEST 及本轮指定 payload；R 只 stage 文档。`.gitattributes` 保留证据原始字节，源码在 commit 前逐份检查 index 字节。证据：[resume:51](C:/Users/gb/.codex/tmp/pvground_support_reference_20261005/postrun/publish_reference_terminal_resume.py:51)、[resume:64](C:/Users/gb/.codex/tmp/pvground_support_reference_20261005/postrun/publish_reference_terminal_resume.py:64)、[resume:94](C:/Users/gb/.codex/tmp/pvground_support_reference_20261005/postrun/publish_reference_terminal_resume.py:94)、[resume:100](C:/Users/gb/.codex/tmp/pvground_support_reference_20261005/postrun/publish_reference_terminal_resume.py:100)。
- 主 MANIFEST 只读取已有真实条目；PV 检查尚无同一条目后补入该条目，R 的 MANIFEST 不在 stage 范围。证据：[resume:46](C:/Users/gb/.codex/tmp/pvground_support_reference_20261005/postrun/publish_reference_terminal_resume.py:46)、[resume:95](C:/Users/gb/.codex/tmp/pvground_support_reference_20261005/postrun/publish_reference_terminal_resume.py:95)。
- 读取的当前 summary、review_call、audit、wait、retention、resources 实际 schema 能满足入口门槛：终态 WARN 无阻断、review 已返回、控制器闭合退出 0、两份非最佳权重已删除的记录、无本地权重归档、保留父模型 [5616,4511] 与 summary 一致。这只是本次恢复源码的 schema／状态复核，未重新连接远端或核算模型结果。证据：[resume:17](C:/Users/gb/.codex/tmp/pvground_support_reference_20261005/postrun/publish_reference_terminal_resume.py:17)、[resume:23](C:/Users/gb/.codex/tmp/pvground_support_reference_20261005/postrun/publish_reference_terminal_resume.py:23)、[resume:28](C:/Users/gb/.codex/tmp/pvground_support_reference_20261005/postrun/publish_reference_terminal_resume.py:28)。
- commit、工作树干净、主仓库 push 及远端 main head 核对之后，才替换同步 guard 中唯一旧摘要、复核四份文档，写终态发布凭据并更新 continuation state。当前 guard 中旧摘要出现一次。源码保留原 heads 顺序、Doc79、父模型指标及严格目标缺口 243，不运行实验或清理入口。证据：[resume:112](C:/Users/gb/.codex/tmp/pvground_support_reference_20261005/postrun/publish_reference_terminal_resume.py:112)、[resume:115](C:/Users/gb/.codex/tmp/pvground_support_reference_20261005/postrun/publish_reference_terminal_resume.py:115)、[resume:118](C:/Users/gb/.codex/tmp/pvground_support_reference_20261005/postrun/publish_reference_terminal_resume.py:118)、[resume:134](C:/Users/gb/.codex/tmp/pvground_support_reference_20261005/postrun/publish_reference_terminal_resume.py:134)。

历史 SOURCE_REVIEW.json 原始 SHA256 为 `1f31ef4ce946e598979757f527d60073bb8dedba9a23fa2437af42d0baed1939`，SOURCE_REVIEW.md 为 `889d2eaf1f3046e8c0daeb68b00ceeab33b03b247e46c4f0d0a11b1c77d31ff1`。

## 证据边界

PUBLICATION_INTERRUPTION.json 是作者生成的中断记录，记录 native session 64090、exit 1 和两份原始报告的 EOF 空白诊断；本 reviewer 没有重放或独立观察该 native session。实际本地 Git 状态与所提供中断位置一致。

该记录的 `original_publisher_sha256` 为 `60f7ff6b8961ef7b2ca4b5be45dc9145c98b646d0c25d9e8ca87051949f1baed`，独立核验它等于生成器第 22 行所计算的 `read_text().encode()` 换行规范化源码摘要。原 fast publisher 的原始字节 SHA256 则是 `062ddb93fc7ab544dc244728021e657b00705d74c930525bf584944030f847e9`；两个值不混用。

恢复凭据中的 complete 文本 SHA 校验计数承接第一次发布器已完成的检查；恢复源码本次重新检查的是远端 Doc 和四份新增文件，未重跑 complete 探针。这不作为本 reviewer 的远端实测。

本次只做本地标准库读文件、AST 解析、字节核对及只读 Git 检查；未执行发布器或生成器，未访问 SSH、网络、凭据、个人记忆或权重，未 stage、commit、push 或清理。恢复成功仍须由实际执行结果证明。本报告仅适用于已观察到的首次失败状态，不是泛化重启框架。全部 23 个实际读取文件的 SHA256 与具体读取范围见配套 JSON 的 `reviewed_files`。
