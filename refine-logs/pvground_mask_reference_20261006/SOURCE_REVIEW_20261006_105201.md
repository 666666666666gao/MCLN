# PV-Ground Mask reference 源码审查

**PASS — SOURCE_ONLY，零阻断。** Fresh context；same-family / provisional；backend identity 未认证。当前修正版本可用于 M0 的源码门槛，M1 仍需两组实际预检闭合通过。原只读部署 probe 的 FAIL 记录保留。

模型实现符合计划：保留 4511 的 hidden8，两组共同清零 output2；native / fused-Mask reference 影响局部采样、DFL target 和 decode，原粗框仍作为六维 prior features。全部 256 候选保留，只有原生 final semantic head / bbs；Box 与 Mask 使用同一 Query。456102 个几何参数、10 个 trainable tensors，其余 PV/G/Mask/语言/语义及 zero-output R 冻结。

reference 从实际 SP 成员范围和预测 fused logits 构造，不读取 GT 或质量阈值。39 个 fixture 逐项对应已闭合的 9508 行诊断、原始 NPZ 和历史粗框：234 个复制数组、row/query/point identity、已有摘要全部一致；39 例 fused logits 全部为负，最大值 -0.1764220893。保存的 fixture 没有 GT 字段。该核对不是本轮 torch/GPU reference 执行，历史离线 invalid-IoU=0 规则未被替换。

native+G+matched DFL+原 extra geometry loss 的配方正确；extra 源码与旧版本逐字节一致。matches[1] 对应原生 final Hungarian，排除全部 matched candidates；真实空集合公式保留。数据、seed2027、B8、LR1e-5、WD5e-4、clip0.1、29778 行一次/3723 步、尾批 2 与计划一致。每组有 initial/terminal 6887 和 initial/terminal formal9508；formal 保存全部候选 prior/reference/final/scores/valid 与真实 root GT。

M0 代码覆盖 neutral decode、真实 all256 raw member extrema、39 例规则、step2 hidden 梯度、冻结状态、单原生评分/Mask 和 CPU 载入后的严格模型/优化器值恢复。M1 gate 校验两组实际 M0 结果。唯一观察器按预定首查与 240 秒后续查询；closed collector 在本地及远端检查 exit0/complete，直接流式 tar 收集非权重文件，并核验路径范围、大小和现有 SHA。

本地检查：17 个 Python 文件和 3 段远端 Python AST 通过；17 个 helper 摘要、原生 evaluator 摘要通过。3 个生成脚本的源码/规格输出可在拦截写盘的内存重放中精确复现；计划文档仅存在 LF/CRLF 差异。没有修改生产源码、联网、部署或执行 GPU。

后续阶段必须保留以下边界：

- 两组实际 M0 尚未执行；本审查不代替梯度、吞吐、显存或严格恢复实测。
- 新增 initial.pth 保存 step0、全部 10 个 geometry state 和空 AdamW 正确。现有 formal 恢复只接受 3723 步 terminal；若 initial 入选，M2 必须单独严格重建它并恢复空优化器，不能借用 terminal witness。
- 单 seed、复用 development val、共同 output reset 已披露。离线 4848 或 step0 改善不能称为训练学出的提升；仍需完整固定预算比较及 fresh terminal audit。

主代理已在部署前修正 SUMMARY 字段为 protected_trained_model_hits，当前生成器和 deployer 一致，实读值为 [5616,4511]。没有遗留代码修正要求。

精确版本、逐文件 SHA256、证据路径和审查范围见 [SOURCE_REVIEW.json](SOURCE_REVIEW.json)。M2 最佳选择、恢复与清理工具尚未实现/审查，本报告不提前验收 M2。

## 实际失败后的源码补审

第一次真实部署尝试在前置只读 probe 以 exit1 失败，错误为 `KeyError: status`，发生在 mkdir、上传和 GPU 启动之前。旧 E/controller.py 实际写入 `completed` / `exit_code`，归档 formal_status.json 没有 `status`。这是初审漏检的 schema 错误；旧初审 timestamped 报告和 DEPLOY_PROBE_FAILURE.json 均保留。

最小修正只把 guard 改为 `closed_status['completed'] and closed_status['exit_code']==0`，继续要求 formal.exit 为 0；没有新增 fallback 或 try/except。本次从当前 probe AST 抽取该断言，在实际归档 JSON 上验证通过，并确认归档 formal.exit 为 0。未执行网络或 GPU。

M1 首查同步从 19500 改为 21600 秒，预计 22000 秒结束，即预计终点前 400 秒（约 7 分钟）；后续仍 240 秒。M0 首查 720 / 预计 900 秒不变。fit observer 直接读 launch metadata，源码未变。

版本比较确认：deployer 仅 guard 改动；generator 仅同 guard 与首查/说明改动；fit launcher 仅首查/说明改动；plan 仅第 26 行 LF 规范化为 CRLF。反向这些精确替换均能恢复初审记录的旧 SHA，其余 93 个绑定输入摘要不变，包括模型、runner、规格及其他生命周期源码。当前生成器的 8 个输出在纯内存重放中均逐字节一致。

**补审结论：PASS，零未解决阻断；SOURCE_ONLY。** 本次是同一 reviewer 的 followup，不声称新 fresh-agent 调用或 backend 认证；SOURCE_REVIEW_CALL.json 未被审查代理改写。实际重试、M0 两步以及后续训练仍未由本审查执行。step0 若入选时的 M2 独立严格恢复要求保持不变。
