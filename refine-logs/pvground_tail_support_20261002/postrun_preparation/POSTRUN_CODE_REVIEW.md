# Same-tail support 终态辅助脚本与初态发布代码审查

**Verdict: PASS。阻塞项：无。具体最小修复：无。**

审查归属：fresh Codex reviewer，gpt-6-astra / max；`review_independence: same-family`，`acceptance_status: provisional`。本结论只覆盖代码与既有接口，不是新终态成绩审计，也不证明脚本已执行或发布成功。

按请求直接读取两个终态辅助脚本及列出的协议、训练源、历史实现和历史审计；另按追加委托只读审查 `publish_initial_boundary_and_helpers.py`。所有训练源码和实现保持不变。没有 SSH、凭据读取、远端访问、GPU、checkpoint 反序列化或收集/训练/发布脚本执行。

## 正确性与范围检查

**completion_and_budget — PASS**

收集必须完成指定 arm、3723 更新、通过9508 formal及两阶段exit=0；CPU独立检查3723个按序batch、29778个fit ID、完整holdout次序和6887/6887/9508行数。既有runner实际构造唯一fit分区并验证完全一致，因此这里的Counter检查满足当前协议，无需假设性重复ID分支。

证据：`collect_complete_arm.py:34-46`；`recount_complete_arm.py:25-39`；`recount_complete_arm.py:53-59`。

**raw_evidence_and_archive — PASS**

原始凭据/行/日志/导入来源均按字节保存并记录SHA；行文件、train日志、run/spec/source_port及imported源与既有凭据核对。terminal.pth完整复制，大小、前后stat和SHA核验；runner保存state_delta、optimizer、RNG、fit顺序及父身份。没有删除或父文件改写，元数据明确依赖保留的author/G父权重。

证据：`collect_complete_arm.py:47-119`；`run_pvground_tail_support.py:527-540`。

**selected_and_coarse_geometry — PASS**

所选框与同一query粗框分别对保存root_box计算轴对齐IoU；严格>.25/>.5逐行等价检查与连续最大绝对差异同时保留。没有重新引入已被历史数据否定的2e-6固定误差门槛，也没有用其他模型输出充当GT。

证据：`recount_complete_arm.py:42-101`；`run_pvground_tail_support.py:441-482`。

**mask_and_coverage_limits — PASS**

Mask仅对已保存mask_iou重计；Full256/top16仅对GPU已保存标记求和并检查单调/0或1/所选命中一致性。文本明确没有原始Mask或所有256框的新重放。

证据：`recount_complete_arm.py:63-69`；`recount_complete_arm.py:86-100`；`recount_complete_arm.py:143`。

**paired_identity_and_scope — PASS**

完整fit batch逐步比较，initial、terminal及formal的row/scan/target/root_box/point SHA逐行核对后才统计修复/破坏。tail_raw对历史G仅背景；tail_fused对应完成的同尾部raw。E0精确比较只针对已保存selected REC与身份，Mask差异原样留在行/比较凭据中。

证据：`recount_complete_arm.py:14-19`；`recount_complete_arm.py:29-39`；`recount_complete_arm.py:102-143`。

**primary_metric — PASS**

bbs与bbf分开输出；原G5615/4495差值及4754严格命中目标均只读取formal bbs，没有用bbf或6887留出代替。

证据：`recount_complete_arm.py:62-101`；`recount_complete_arm.py:136-148`。

**publication_content — PASS**

续写只描述19:35时142步与6887初态bbs6176/5602、bbf6206/5647；保留row18482双模式Mask差异及ETA暂估，未声称终态/正式融合成绩。7个payload正是2份边界证据、请求、2份原样审查报告和2个辅助脚本。

证据：`publish_initial_boundary_and_helpers.py:19-57`；`tail_raw_training_boundary_193500.json`；`training_boundary_record.json`。

**publication_write_boundary — PASS**

三个被审脚本SHA受本报告约束；旧交接SHA、三个HEAD和干净工作树均有运行前门禁。发布限定文档、7 payload、manifest及该新目录的gitattributes；第三repo仅交接文档。另在既有本机sync guard内替换旧交接SHA（当前恰好1次）。远端只写文档/归档payload目录，不触及活动训练源码；push为授权main的非强制push并读回HEAD。当前本地门禁已只读核实，远端门禁须执行时验证。

证据：`publish_initial_boundary_and_helpers.py:13-31`；`publish_initial_boundary_and_helpers.py:47-122`。

## 实际静态验证

- 三个被审脚本及两个权威 runner/controller 均通过 `ast.parse`；仅解析源码，没有导入或执行。
- 发布脚本文字中的142步、6887行、bbs6176/5602、bbf6206/5647及双模式row18482差异，与既有边界JSON逐项一致；边界摘要的原始凭据SHA也一致。
- 三个本地工作树当前均干净，HEAD依次为 `56fd154f21766be0012e6972df54cf184d18c7de`、`3e82ea8de6f93dd9535d7854a35db43af6de4f7b`、`38613bf37041d19eee335bbd816b036d10931d5b`。
- 三个仓库与Desktop的旧交接文档均为2,100,373字节，SHA256 `c424232c18eb4f03deec92b9dbda1940f00116a02c33efe9d71ca501d8404c36`；新payload目标当前不存在。
- 既有sync guard含旧交接SHA恰好一次。其更新属于发布脚本明确的额外本机写入，不修改活动实验源码。
- 历史P3 JSON保留了约2.93e-6的连续误差及逐行阈值检查，本轮只读取这些旧记录，未重新计算或生成历史成绩。

## 非阻塞证据限制

- 这是代码审查。没有新的same-tail终态成绩，没有执行这三个脚本、SSH、远端读写、GPU、checkpoint反序列化或科学实验。PASS不等于已完成运行/发布/结果完整性审计。
- E0 Mask差异仍未解释。边界凭据只有row18482在bbs/bbf中不同；保存selected REC与身份相等不代表全部输入、logit、排名或全模型逐位相等。
- CPU框计算仅复核保存的GT/所选及粗框；Mask和all256限定为保存值/标记。terminal归档是需要保留author/G父权重的delta，不是独立完整模型或已执行的新restore。
- 现有原始GT来源和历史成绩以本次指定runner/既有审计为背景；未重新从ScanNet原始标注生成全部GT，也未重跑历史控制。

现有方案在请求范围内正确，无需加入fallback、额外try/except、兼容层或假设性防御分支。实际终态到达后仍需执行原定收集、CPU复核及新的结果完整性审查；本报告不提前替代这些步骤。

## 实际审查文件 SHA256

以下是本次实际读取的任务文件；大交接文档仅核对字节身份，sync guard仅做字面量/语法核对，范围在JSON逐项注明。

| Path | Bytes | SHA256 |
|---|---:|---|
| `C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\POSTRUN_REVIEW_REQUEST.md` | 3236 | `0f538853ab81ba439360595303a6b849034c2f8b3d8b30af21e9540750be3fc4` |
| `C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\collect_complete_arm.py` | 6134 | `d9ec4305afef6c066be16bc9ce9655356d240935bb33ba092ba04e36088bbe6c` |
| `C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\recount_complete_arm.py` | 8537 | `23724c51a50a46fadcad75fdff662c22572223e2131b731924721329d6410e94` |
| `C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\publish_initial_boundary_and_helpers.py` | 8443 | `84074fec1cf2fc080f8efca365248df17b459304723cadf5552f9ad2b3eed8d8` |
| `C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\PLAN.md` | 4954 | `5e8999f10ec2a3b6e69e71496f5edce863e3131912c33baefd002143f42973ff` |
| `C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\run_pvground_tail_support.py` | 38567 | `483a7db4e0d11d8b79af63ea6e02f6d9455bed218190efc17d42e48929648346` |
| `C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\run_pvground_tail_support_control.py` | 4120 | `21774197eb1ee5a9c4e7742953c1d496e75d6192c80c82de24eb6084398bdcb6` |
| `C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\tail_raw_spec.json` | 3710 | `30f8a11ae02f0af3cae1ae3db6d5f225820b13b426c44466615cca01cf5caf13` |
| `C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\tail_fused_spec.json` | 3731 | `03c6d64f322210998e2cc35b71f42dd42628465d5d4303588a8e269c90311c76` |
| `C:\Users\gb\.codex\tmp\pvground_p3_next_20261002\collect_complete.py` | 5629 | `7b1ccfb947a4d5e6a107247326b747a92db737d3e02d6f0347e86321bc574ccb` |
| `C:\Users\gb\.codex\tmp\pvground_p3_next_20261002\recount_complete.py` | 7764 | `2645e872a2d5a1e0cc46ae3dba50d5e5fa5918b5d18aa021fadaabf6cc4c6524` |
| `C:\Users\gb\.codex\tmp\pvground_p3_next_20261002\complete\CPU_RECOUNT.json` | 9891 | `fc68682965073245d5371a1a0de71991bc78d1dc7b6d5da747f0b7647e9e2cd5` |
| `C:\Users\gb\.codex\tmp\pvground_p3_next_20261002\complete\EXPERIMENT_AUDIT.md` | 15050 | `5423833a305cb4e6dc32f0c39d4febcfff1f5bbc682d39415693f31a48ada61e` |
| `C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\tail_raw_training_boundary_193500.json` | 6108 | `4466c33689f26949419ebda4a85f8209b25d050947329d7b5d3dd9976c1d163f` |
| `C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\runtime_publication.json` | 880 | `57427cacde71c7a5776d543c2723f1357777d5b687670cbaa61f3edec3c6fec4` |
| `C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\training_boundary_record.json` | 765 | `aa0e3527d8378ae63485d389fab93f611ab246db7bcbe16e6aaa9f492c57ecad` |
| `C:\Users\gb\.codex\tmp\sync_cs_handoff_remote_20260923.py` | 1435 | `27fa322999f13a8f11b4b8d773f90eeac6d245d5774d9e0c9b52cb63eba26384` |
| `C:\Users\gb\.codex_mcln_g0_20260905\docs\MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md` | 2100373 | `c424232c18eb4f03deec92b9dbda1940f00116a02c33efe9d71ca501d8404c36` |
| `C:\Users\gb\.codex_pvground_cs_20261002\docs\MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md` | 2100373 | `c424232c18eb4f03deec92b9dbda1940f00116a02c33efe9d71ca501d8404c36` |
| `C:\Users\gb\.codex_mcln_v99_internal_20260928\docs\MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md` | 2100373 | `c424232c18eb4f03deec92b9dbda1940f00116a02c33efe9d71ca501d8404c36` |
| `C:\Users\gb\Desktop\document\MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md` | 2100373 | `c424232c18eb4f03deec92b9dbda1940f00116a02c33efe9d71ca501d8404c36` |
| `C:\Users\gb\.codex_mcln_g0_20260905\.gitattributes` | 25426 | `17bbf1837249dd89b33cfe2290258732f11d815d214032588f81d3a264c34339` |

审查记录时间：2026-10-02T19:54:55.400892+08:00。
