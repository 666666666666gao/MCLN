部署入口源码续审结论：**PASS，阻断项 0，非阻断项 0。** 在协作操作、既定原生环境和本次工程预检范围内，实际 launcher／controller 的终态门槛、模块放置、全程锁和失败停止逻辑正确，未发现确定性代码缺陷。此结论不是已部署或已运行的证明。

本次为同一审查者 `/root/pvg_whole_model_preflight_review` 的续审，保持 `SOURCE_ONLY / same-family / provisional`。原启动配置由父代理确认为 `gpt-6-astra / max / fork_turns=none`；没有独立后端 SKU 查询，也不声明第二次新鲜独立审查。

**核对结果。**

- [launcher 第 19 行](C:/Users/gb/.codex/tmp/pvground_whole_mask_integration_20261003/launch_preflight_authorized.py:19)先读取 normalized 的完整收集、分析和审查文件，并检查 complete、退出码 0、控制器不存活、分析时间对应本次终态以及审查 verdict。最新[第 24 行](C:/Users/gb/.codex/tmp/pvground_whole_mask_integration_20261003/launch_preflight_authorized.py:24)要求审查为 PASS／WARN 且 `blocking_issues` 为空。这些都在 SSH 客户端创建之前；文件缺失或断言失败就停止，不会代造未来凭证。
- 字段与实际生产者一致：[collector](C:/Users/gb/.codex/tmp/pvground_candidate_normalization_20261003/collect_complete.py:59)输出嵌套 `status`、`controller_exit`、`controller_alive`；[analyzer](C:/Users/gb/.codex/tmp/pvground_candidate_normalization_20261003/analyze_complete.py:173)将 intake 终态时间写入 `SUMMARY.json.finished_cst`。experiment-audit 的[机器报告约定](C:/Users/gb/.codex/skills/experiment-audit/SKILL.md:189)确实包含大写的顶层 `verdict`；小写 `overall_verdict/integrity_status` 是另两个字段。新准备的[终态请求生成器第 30 行](C:/Users/gb/.codex/tmp/pvground_candidate_normalization_20261003/prepare_terminal_audit_request.py:30)明确要求未来 `analysis/EXPERIMENT_AUDIT.json` 包含 `verdict` 和列表型 `blocking_issues`，与最新 launcher 一致。生成器要求实际 intake、summary 及列出的证据文件存在后才排他写入请求。本次只读其源码合约，没有执行，也没有假定未来请求／分析／审查文件已存在。该未来请求提及的清理实现留给新鲜终态 auditor，不在本次部署审查内。
- [远端 probe](C:/Users/gb/.codex/tmp/pvground_whole_mask_integration_20261003/launch_preflight_authorized.py:37)使用原 launch.json 中的根目录及控制器 PID 391849，要求该 PID 已不存在、实际 status=complete、controller.exit=0、无 compute 进程、预检路径尚不存在及至少 64MiB 日志余量。检查通过后才建立预检目录，不会抢占或终止旧作业。
- [上传路径](C:/Users/gb/.codex/tmp/pvground_whole_mask_integration_20261003/launch_preflight_authorized.py:58)正确：公共 runner／controller 位于预检根目录，十个必需模块和各自 spec 位于 local_range／whole_range。spec.root 与该目录逐项一致；[每个子进程的 PYTHONPATH](C:/Users/gb/.codex/tmp/pvground_whole_mask_integration_20261003/preflight_controller.py:23)以本组目录开头，符合 runner 的检查和导入位置。实际上传内容有读回比对；旧 warm 环境、模型、loss、数据及权重路径保持不变。
- [一个 flock](C:/Users/gb/.codex/tmp/pvground_whole_mask_integration_20261003/launch_preflight_authorized.py:86)包住整个 controller；controller 按 local_range、whole_range 串行等待。根目录须全新，上传采用排他文件模式，已有产物被保留。没有自动重试、清理、删除、GPU 抢占或正式拟合命令。
- [子进程失败处理](C:/Users/gb/.codex/tmp/pvground_whole_mask_integration_20261003/preflight_controller.py:29)记录本组日志和退出码；非零即写 failed 并停止后续组。只有退出 0 且原生 receipt 为 pass／batch8／两步，才把本组加入 completed；两组满足后才写 complete。缺失或不合格 receipt 不会转成 PASS。
- launcher 保存 controller 日志和实际 shell 退出码；实际唯一 PID 匹配成功后才记录 `LAUNCHED_NOT_COMPLETED`。该状态明确只代表启动，不代表模型检查已通过。
- SSH 密码仅从已授权环境变量传给 Paramiko，不进入 shell 字符串、日志或 JSON；代码加载系统 host keys，未放宽未知主机策略。probe／flock 参数和嵌套 shell、日志路径、pgrep 模式均正确引用。子进程以 argv 列表调用 `env`，不经过 shell 解释环境值。未读取任何凭据值或授权包装器。
- 子进程固定调用已审阅的 `--mode preflight`。runner 只接受 cpu／preflight，执行 batch8 两步；权重／优化器序列化仍是 `BytesIO`，没有磁盘 checkpoint 或正式 fit 路径。远端 controller／probe 的语法和 API 保持 Python 3.7 可用；`shlex.join` 运行在本地 launcher，与已有本地预检入口一致。

**适用范围。**

未执行 launcher、controller、probe、模型、导入、AST／编译或测试；未连接 SSH、使用 GPU、查询作业、读取二进制权重／fixture 或凭据。此源码审查不声称已经观察到当前远端结束、空闲资源或未来终态审查通过。实际开始仍须由入口检查取得这些证据。

启动凭证只证明控制器启动。成功须结合 controller.exit=0、两组 complete、各组退出 0 和原生 receipt 判断，不能由经过时间或观察失败推断。64MiB 仅为日志余量；无 compute 进程不等于新模型已证明能装入显存。

600 秒总时长、420 秒首次检查和 240 秒后续轮询均是明确标注的估计／安排；本入口不启动监控循环。预检仍只覆盖每组一个 batch8、两次更新，不证明精度、容量等价或完整数据有效性，也不替后续正式实验选择共同父权重／loss。

原实现审查的限定结论全部保留：共享上游仍可能改变语义／Mask，缓存零头重放不证明跨进程完整模型逐位一致，内存重载不证明磁盘／RNG 恢复或继续训练等价。`verify_native_replacement` 仍不会被两次 `update=True` 调用执行。

**审阅身份。**

- 最新 launcher：6088 字节，SHA256 `77c7aad9a6ece761dea2cf0d202b6c83ce10a25076a841ac2df698e5361c9178`。封存前发现更新后已完整重读，包含新增的终态审查阻断列表门槛。
- 终态请求生成器：3832 字节，SHA256 `9f7ece897d7e39ed1da64be9fab73b37e2b80ad5c47caee8dac9b717eba07909`。
- controller：2147 字节，SHA256 `db943667ea60f3bfe1892cc8137eaafa52f30f73eb0c480d57403f36e90055a9`。
- runner：28513 字节，SHA256 `485a8e04fecbcb83a10beee73af52e46d5fee206884eaae1b148820003699bff`，与已审阅版本相同。
- 已重新核对先前报告的 43 项文本／源码身份，全部一致。先前两份 EXPERIMENT_CODE_REVIEW 报告没有改动。

本次全部 14 项阅读文件的完整路径、字节数、SHA256、逐项源码依据和未执行事项保存在 [DEPLOYMENT_CODE_REVIEW.json](C:/Users/gb/.codex/tmp/pvground_whole_mask_integration_20261003/DEPLOYMENT_CODE_REVIEW.json)。本次只新增这两份部署审查报告。
