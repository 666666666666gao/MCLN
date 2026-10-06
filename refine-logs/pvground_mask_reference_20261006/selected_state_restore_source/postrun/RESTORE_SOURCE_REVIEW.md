# MaskReference 候选 CPU 重载工具源码审查

结论：**SOURCE_ONLY PASS，0 个阻断项**。这是全新上下文的同系列审查，`review_independence=same-family`、`acceptance_status=provisional`。请求的 `gpt-6-astra / max` 不是后端身份凭证；本审查没有可验证的实际后端 attestation。

本次仅审查 `selected_mask_reference_factory.py`、`restore_candidate_state.py`、`restore_selected_candidate_authorized.py` 及有关生产源码。未执行 SSH、GPU 查询、模型构造、checkpoint 加载、optimizer 恢复、训练、分析器、部署、通知或权重保留/删除。没有读取凭据值或其他授权包装器；仅按任务明确要求读取第三份受审 SSH 包装器。`publish_restore_tools_authorized.py` 未读、未审。

## 正确性结论

- 新工厂保持原路径的 PVGround → strict official PV → TaskObservation/G → BoundaryBoxRefiner → R → MaskReference 构造顺序。旧 geometry 的加载不消耗 RNG；省略其加载后，以 selected 的完整 10 张量覆盖 geometry，保留了 R 构造前的相同随机层创建序列。CPU checker 对两个工厂分别设 seed2027，并比较完整 `state_dict`。这是未来执行的见证，当前没有实际逐张量比较结果。
- 完整持久 state 数由源码推出为 `1234 + 37 + 10 + 23 = 1304`。geometry 的 10 个参数为 member 两层、condition、aggregate、output 的 weight/bias，共 456102 参数。替换模块保持参数注册顺序；M1 的 `tuple(trainable.values())` 与 checker 的按 `model.parameters()` 顺序筛选所得顺序一致。
- direct 模型最终只有 geometry 可训练，全模型为 eval，PV/G/R 冻结；R output 由原构造器严格置零。没有新 rank、第二个 box 源或语义 head。
- `initial_formal` 读取真实 `initial.pth`：step0、空 AdamW state、output2 全零，并逐张量核对 hidden8 与保护 4511 checkpoint 一致。其 header 不要求原工厂仅 terminal 才有的字段，也没有伪造 step3723。
- `formal` 读取真实 `terminal.pth`：step3723、hidden 累计 14892、output 更新 3723、29778 唯一训练行、10 个 optimizer state，所有 step3723。实际 optimizer 是 AdamW；严格恢复 helper 检查 param groups、state keys、step 和 moments。
- 原工厂仅接收保护 4511 的真实 terminal 作为本次比较见证，然后加载 selected 的完整 delta；没有将 selected step0 传入原工厂、改写 header 或用 terminal 代验 initial。
- 包装器只接受四个预声明候选中真实复算后胜过保护 4511 的新候选。arm/stage、step0/3723 和 native `bbs` 的 @.25/@.50 计数与重载 receipt 对应。初始架构收益没有被称为训练收益。
- 执行前要求实际 `fit_controller.exit=0`、complete/protected-parent 状态、原 controller 命令不存在，并核对远端 spec；checker 重复闭合检查并核对 env、official PV、G、旧 geometry、MaskReference、helpers 和 source-port 所列文件 SHA。旧 geometry 在 checker 中仍必须存在，未来 direct factory 不依赖它的结论不构成删除授权或已删除声明。
- 三份文件和内嵌远端 probe 均通过 Python 3.7 grammar AST 与内存编译。检查在本地 Python 3.12.6 完成，不等于实际 Python 3.7/Torch 1.10.2 运行成功。`shlex.join` 仅用于本地 SSH 包装器；远端 3.7 代码不调用它。使用的 Torch API 与当前 1.10.2 路径相符。
- SSH 包装器从环境获取口令，只用于认证；源码没有输出口令或 env 字典。无新增 API 通知、无 GPU forward、无新 9508、无 `torch.save` 或权重删除调用；仅未来上传隔离重载源码并写检查记录。

## 独立门槛与边界

审查期间，分析器已将保护 parent 的 `optimizer_updates` 从混用累计历史的 11169 修正为真实 fresh optimizer 3723，并另存 `total_geometry_fit_updates=11169`；候选排序和新候选的 0/3723 语义不变。旧 `ANALYSIS_SOURCE_REVIEW.json` 一度不再匹配当前分析器。其独立 follow-up 必须通过后，包装器的分析源码 SHA 门槛才会放行。Final binding: the independent follow-up is now SOURCE_ONLY PASS with zero blockers. Its report SHA is `39e08732d6e009987b752d88405ef6f480bd3ea1d02378d67a3579b7dbd63c61` and its analyzer SHA is `f9c3f6d4b4fda3a71819b2b89f2bdd416fbd93ebaf98ebe310f14a15f7a7f42d`. Both stable artifacts are included in the final JSON source binding; this remains source review only.

历史 `pvground_runtime_bundle_20260908_v1/env_spec.json` 本地快照与当前 spec 的 canonical SHA 不同，因此没有用它证明当前远端环境。当前运行时仍须由闭合后的包装器和 checker 验证。

当前本地没有 `complete_fit`、`analysis` 或 `selected_candidate_CPU_restore.json`。尚未证明实际任务终态、四份新 9508 指标、真实 selected checkpoint 恢复、训练 continuation RNG 恢复、raw Mask 重新计算或权重可删除。工厂的 RNG 结论是相同入口 seed 下相同构造序列；未来 checker 的完整持久 state 比较不能被提前报告为实际 M1 全状态见证。

本次没有修改三份工具、活动 M1 源码或 observer。受审路径与最终字节 SHA 见 `RESTORE_SOURCE_REVIEW.json`。
