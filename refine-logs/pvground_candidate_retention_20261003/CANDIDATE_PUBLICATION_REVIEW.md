# Candidate retention publication independent review

Verdict: **PASS — 本次发布审查无实际阻塞项；无需修改发布脚本。**

Reviewer: `gpt-6-astra` / `max`; context: `fresh-context`; review_independence: `same-family`; acceptance_status: `provisional`.
Reviewed at: 2026-10-03T02:03:50.948073+08:00

本报告审查 `publish_candidate_retention.py` 的 §20.376.20 追加事实、发布文件和修改范围。实际完成源码检查、本地 Git/文档只读检查及已归档行的 CPU 数字复核。没有执行 publisher，没有 SSH、网络发布、GPU forward 或优化更新；没有读取认证 wrapper、环境密钥或 MEMORY.md。本报告自身是将要发布的审查材料，不是 GPU 结果。父任务确认本 reviewer 的实际 native spawn 参数为 model=gpt-6-astra、reasoning_effort=max、fork_turns=none。

## 审查输入与方法

完整读取：`publish_candidate_retention.py`、`SAVED_ERROR_CANDIDATES.json`、`analyze_saved_error_candidates.py`、`RAW_GEOMETRY_PROFILE.json`、`profile_completed_raw_geometry.py`、`CANDIDATE_AUDIT_CODE_REVIEW.json/.md`、`fused_retry_publication.json`。

读取并按结构提取两个大边界文件的状态、initial receipt/comparison、实际更新和 checkpoint 元数据：`tail_fused_retry_boundary_002900.json` 与 `tail_fused_retry_boundary_005006.json`。为解释 `rec_inputs_exact` 的历史字段名，另读 `initial_zero_output_comparison.py` 和训练 runner 的调用位置；为核对现有观察安排，读 `record_candidate_audit_ready.py` 及 `mcln_active_wait_20261001.json` 的相关字段。

本地校验三个仓库的 HEAD/branch/porcelain 和四份 handoff 字节；只用 AST 提取 publisher 的 appendix/payload 定义，在内存中构造带审查占位时间的追加内容，未 import 或运行 publisher。独立读取两个系统的 formal `rows.jsonl`/`receipt.json`，复算两种模式、两个阈值的全部发布计数；另按保存的 bbs coarse/final 框独立计算本次引用的几何量。没有重复先前候选诊断的完整 synthetic/mock 检查，也没有运行 native model。本轮 Python 检查使用本地可用的 uv Python 3.12；系统默认 python 的启动器报 No pyvenv.cfg，不影响后续完成的只读检查。

## 追加事实正确

- same_tail_raw 是 §20.376.18 已完成的 9508 条正式开发验证，不是当前 fused 的结果。bbs 严格错误 5051 = 966 + 508 + 980 + 886 + 1711；其中 3340 有另一个 IoU > .5 的框，即 66.1255196991%，文中 66.13% 正确；第 32 名后首次覆盖为 980 + 886 = 1866。宽松错误 3914 中另有合格框 3340，即 85.3346959632%，文中 85.33% 正确。loose-only 1137 = 675 + 462。
- 两系统（same_tail_raw 与 G_continued）、bbs/bbf、.25/.5 的错误数、覆盖区间、替代数、Full256 标志总数和 loose-only 分区均与归档行/receipt 一致。这里复核的是已保存 GPU coverage 标志；appendix 正确说明它只给 rank 区间，不是恢复全部候选坐标、精确名次或物理身份。
- “未匹配、低排名、当前未过阈值不能统一等同无用背景”符合保留全部 256 的用户要求。文中同时保留 root IoU 不等于完整语义正例的限制。G 只替换部分未匹配高 IoU query 的 CE，来源为已完成独立源码审查的明确结论；本次 publisher 没有改标签、损失或候选选择。
- raw 的本次几何数字对应主模式 bbs：同一已选 Query 严格净 +9；中心移动中位数 2.9464145125913124 mm，四舍五入为 2.946415；最大单面位移中位数 3.7278011441230774 mm，四舍五入为 3.727801；超过 10 mm 的 6/9508 条。GT 体积分组表达数为 2377/2380/2374/2377，严格净变化 6/1/1/1。原文正确区分表达计数、体积与点密度，并未将内部前后作用写成独立训练消融。
- E0 receipt 为 2026-10-03T00:24:25.336634+08:00，6887 条、bbs6176/5602；输入身份/root GT 以及两种模式双阈值判断一致。字段 `rec_inputs_exact=false` 实际由 query/box/iou 输出差异计算，不是否认 `input_identities_exact=true`；appendix 明确保留跨进程 Query/框/Mask 不逐位等价，未声称全部原始输入张量或完整输出一致。
- 00:50:01.249981 的快照显示 stage=tail_fused/train、618/3723、latest.pth 346642129 B。追加文字明确把 618 限定为该旧快照，既不推断当前步数，也不声称独立反序列化验证了 latest。03:20 是现有本地 observer12278 元数据的安排，本审查未重新连接远端确认实时状态。
- 原 G 5615/4495、raw5594/4457、V99 历史保护和 50%/Nr/Sr 未完成的表述保持原历史文档边界。没有 fused 终态数字、候选 GPU sanity/full 结果或新的优化收益被制造。

## 代码审查与尚未执行状态区分正确

已有 `CANDIDATE_AUDIT_CODE_REVIEW` 的实际结论为 WARN、blocking_issues=[]、STATIC_AND_CPU_MOCK_ONLY、GPU_executed=false、SSH_connections=0。它审查并见证了全部 256 query 保留、native matching、root/scene IoU、原生 Mask 成员计数、stdout 压缩流和终点恢复流程；本次发布只引用该结论，没有把其静态/CPU 证据提升为实际 GPU 通过。

该上游报告 W1 指出：与旧 formal 的比较只汇总 Query 和阈值命中变化，不汇总连续 selected box/IoU/Mask 差值。appendix 明确保留这一限制，故本次发布没有新增错误的输出一致性声称。先完成 fused formal/归档，再 8 条真实 sanity/CPU recount，随后 9508 条/CPU recount，仍被写为未来流程。没有以本发布 PASS 替代上述真实执行凭据。

## 发布修改范围正确

实际本地前置状态：

| 仓库 | HEAD | porcelain |
|---|---|---|
| 主仓库 .codex_mcln_g0_20260905 | b99f888293f155b4af77c8b41c25af181fc98c1c | 空 |
| 隔离 PV .codex_pvground_cs_20261002 | 9739d287bdd13ca5bfc8a3dac2832269b6711bb5 | 空 |
| 历史 R .codex_mcln_v99_internal_20260928 | bc1e13fb1de19328d06e596e3dce07660ff8f3c3 | 空 |

四份本地 `MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md` 各 2111034 B，逐字相等且符合 §19 凭据的既有 SHA。内存构造的文档保留全部旧前缀，且 §20.376.20 标题只出现一次。publisher 还会在任何文档写入之前读取远端 handoff 并要求等于旧内容；本审查没有假称已检查当前远端字节。

13 项 payload 为 7 份脚本、2 份 CPU 分析 JSON、2 份候选代码审查文件和本报告的 2 份发布审查文件。所有 13 个主/PV 目标路径当前不存在；11 个既有源文件存在，本报告两源文件在本次 exclusive create 后补齐。publisher 没有将 auth wrapper、checkpoint、数据集或外部 private 文件加入 payload。

逐项修改路径明确：
- 主仓库与隔离 PV：追加 handoff，新增 13 项 payload，并追加 `MANIFEST.md` 和这些 payload 专用 `.gitattributes` 条目。
- 历史 R：只提交追加后的 handoff；桌面：只写相同 handoff。
- 远端：写 handoff、7 个独立诊断/分析脚本和新建证据目录；远端命令只有引用固定路径的 `cat --`，SFTP 不调用脚本。当前活动 run.py、训练配置、优化器和任何权重均不是写入目标。
- 发布后更新既有同步 guard 中唯一的旧 handoff 摘要，并创建本地 `candidate_retention_publication.json`。guard 内容及凭据没有被本 reviewer 读取；脚本在修改 guard 前断言旧摘要出现一次。没有任何删除调用。

Git 操作先以允许路径集合检查 status，再显式 add/commit；`git add -f` 仅用于已列出的文件，不是强推。前两仓库的每个暂存 payload 都与源字节比较，提交后要求工作区干净。只有主仓库执行 `git push origin HEAD:main`，无 force/删除 ref/镜像 push，随后检查远端 main 等于新 HEAD，并核对主/PV 的文档和 payload blob 一致；不推送隔离 PV 或历史 R 分支。

SSH 凭据仅从 `os.environ['MCLN_SSH_PASSWORD']` 使用，没有输出或写入发布材料。SSH 加载已有 host keys，没有自动接纳新 host key；读取命令用 shlex.quote 引用路径。没有将活动训练或完整历史结果纳入提交/覆盖范围。

## 结论与实际边界

Blocking issues: **[]**。Required changes: **[]**。本次发布源码及其新事实正确，PASS 限于本次静态/本地证据发布审查。先前候选代码 WARN 及其 W1 保持原样。

本审查没有证明尚未执行的远端写入、Git push 或 GPU 诊断已经成功。新建远端目标、远端文档相等、push 成功以及同步 guard 的运行时断言，须由实际 publisher 产生后续凭据；目前没有发现要求增加 fallback、hash 绑定框架或无关防御逻辑的实际缺陷。

本报告和 JSON 以 exclusive create 写入，已有文件不覆盖。
