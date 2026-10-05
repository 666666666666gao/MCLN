# 启动、收集和观察脚本源码复核

**PASS，blocking_findings=[]。** 范围为 **SOURCE_ONLY**。复用原委托审查者的后续源码审查；请求路由为 Astra/max，实际 backend 未获独立证明，仍为 same-family / provisional。原主体 SOURCE_REVIEW.json 的 39 个文件身份全部保持一致，主体源码未修改。

启动条件与实际产物接口相符：collector 和 launcher 都读取预检 observer 的扁平字段，要求 observer_closed、controller 已退出、exitcode=0、status=complete。launcher 随后检查两组真实 preflight.json 的两次更新、零磁盘权重、456102 参数/10 tensors、冻结父状态，以及缓存 bbs/Mask、额外梯度、空行与裁剪端点见证。

特别核对了共享 helper：optimizer_restore_exact 的实际返回字段为 `moment_and_step_states`、`param_groups`、`all_keys_moments_steps_and_groups_exact`。launcher 读取 `optimizer_exact_check['all_keys_moments_steps_and_groups_exact']`，字段正确。

launcher 会核对原主体和本次 review 的本地文件身份、两组远端预检 receipt 的字节，以及远端 runner/helper/controller/两份 spec 与已审查本地文件的字节。资源检查要求预检远端闭合、尚无 fit 状态、GPU 无活动计算进程，并按实际序列化体积预留 `3 * max(serialization_bytes) + 256 MiB`。full fit 在现有 flock GPU 锁下执行；controller 和 runner 保持原 official/G/geometry 起点检查。

两组 train 各自新建进程，从同一受保护起点重建模型和全新 AdamW。两步预检状态只留在 BytesIO，train 没有恢复它的路径；formal 恢复各自 terminal。每组 29778 行仅一次，B8、尾 B2、3723 更新；父几何阶段 3723，terminal 累计 7446，与已审查 runner 一致。

observer 的生成结果与来源的指定替换完全一致：首次读取锚定 launch+6200 秒，随后每 240 秒；最多 106 次，最后一次计划观察为 31400 秒。终态记录保留在 `fit_wait.json.terminal`，随后关闭 SFTP/SSH 句柄并退出。训练结论读取 `terminal.exitcode` 和 `terminal.status`。

collector 使用最终 21 文件白名单，包含两组 imports.json、load/preflight receipts、日志、exit 以及指定代码/spec；拒绝 arm 目录中的 .pth/.pt/.tmp，不收集权重，不运行模型或重放预检。

四个新脚本与内嵌远端 probe 均通过 Python 3.7 AST 检查；生成 observer 一致性、Windows 上远端 root 推导、批次与累计更新算术检查通过。本次只做本地读源和标准库检查，未执行 SSH、GPU、模型、训练或优化器。实际预检完成和 full fit 启动仍由脚本运行时 gate 验证。精确路径与 SHA256 见 companion JSON。
