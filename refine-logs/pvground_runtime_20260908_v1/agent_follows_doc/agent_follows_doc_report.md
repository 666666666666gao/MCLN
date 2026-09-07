# PV-Ground agent-follows-doc 核验报告

结论：**已有环境的文档执行为 PASS；总体为 WARN，仅因输出文件扩展名与实际序列化格式不一致。** 逐字命令实际退出 **0**，`PVG_RUNTIME_WITNESS.status=pass`，且出现 `DOCUMENTED_WITNESS_PASS`。没有发现已暴露运行核验项的实际失败。

## 核验身份与边界

- agent：`/root/pvg_runtime_doc_witness`
- model：`gpt-6-astra`
- reasoning_effort：`max`
- review_independence：`same-family`
- acceptance_status：`provisional`
- 输入为指定技能、compute-env-contract、local-codex-policy、执行计划中的环境 ledger 与完整命令；使用新任务上下文完成核验。
- 只核验既有环境。未读取凭据、未修复、未安装、未训练，也未修改源码或执行文档。没有检查模型精度。
- 完整模型前向、严格权重加载、真实 ScanRefer 输入和 REC 指标没有在本次运行中验证。

## 文档与实际命令

执行计划：`C:/Users/gb/.codex_mcln_g0_20260905/docs/PVG_RUNTIME_EXECUTION_PLAN_2026-09-08.md`

技能：`C:/Users/gb/.agents/skills/run-experiment/SKILL.md`；引用的 contract 与 policy 均来自相邻 `shared-references/`。执行计划内三个 `env:` 块承担 ledger 功能。专用 `.aris/compute` 目录不存在，但 contract 明确允许 ledger 放在既有项目文档中，因此这不是失败。

文档命令逐字执行一次；`shell=powershell`、`login=false`、`cwd=C:/Users/gb`：

```powershell
uv run --no-project --with paramiko python -u C:/Users/gb/.codex/tmp/run_mcln_authorized_20260908.py C:/Users/gb/.codex/tmp/mcln_remote_script_20260907.py C:/Users/gb/.codex/tmp/run_pvg_runtime_documented_witness_20260908.py --out C:/Users/gb/.codex/tmp/pvg_runtime_agent_witness_20260908.json
```

- 开始记录（UTC）：2026-09-07 22:51:56 UTC
- 完成观测（UTC）：2026-09-07 22:52:17 UTC
- witness 时间（CST）：2026-09-08T06:52:14.172250+08:00
- witness 实测时长：15.204147100448608 秒
- 进程实际退出码：0
- 两个 sentinel：`PVG_RUNTIME_WITNESS`、`DOCUMENTED_WITNESS_PASS`

## 逐项对照

| 检查 | 文档预期 | 实际证据 | 判断 |
|---|---|---|---|
| 逐字调用 | 按给定 PowerShell 命令执行，不修复、不安装、不训练 | 原命令未更改，执行一次，login=false | PASS |
| 退出与 sentinel | 退出 0，输出 PVG_RUNTIME_WITNESS，包装脚本核对 receipt | 退出 0；两种 sentinel 都存在；status=pass | PASS |
| GPU 与主要运行包 | A100 40GB，Torch 1.10.2+cu111，spconv 2.3.6 | NVIDIA A100-PCIE-40GB；版本完全相同 | PASS |
| 扩展解析路径 | 专用 runtime 下的 PV-Ground 与 OpenPCDet 扩展 | point_extension 与 stack_extension 都位于该 runtime 根目录 | PASS |
| FPS 两批合成点 | 对照 CPU 最远点采样 | indices=[[0,3,2],[0,3,2]]，固定 witness 通过 | PASS |
| 分组与空邻域 | 逐点参考、空邻域核验 | 最大绝对误差 0.0；empty_groups=[false,true,false,true]；固定 witness 通过 | PASS |
| 特征梯度 | 精确匹配参考 | grouping_gradient_matches_reference=true | PASS |
| RoI 点内判断 | 对照轴对齐几何 | roi_membership_matches_reference=true | PASS |
| 体素 | ZYX、float32 参考、保持原点顺序 | zyx；float32；raw_point_order_preserved=true；固定 witness 通过 | PASS |
| 完整类导入 | 导入 PVGround，不执行模型前向 | full_model_imported=PVGround；model_forwards=0 | PASS |
| 零训练与零数据集行 | optimizer step=0；formal evaluation row=0 | optimizer_steps=0；dataset_rows=0 | PASS |
| 本地 --out 文件格式 | 命令给出 .json 扩展名；正文没有明确承诺 JSON 序列化 | 文件实际是两行 sentinel 日志，不能直接作为 JSON 解析 | WARN |

## 唯一观测差异

`C:/Users/gb/.codex/tmp/pvg_runtime_agent_witness_20260908.json` 使用 `.json` 扩展名，但实际内容是 `PVG_RUNTIME_WITNESS {…}` 加 `DOCUMENTED_WITNESS_PASS` 两行文本，**不是独立合法 JSON**。这属于文件命名与机器读取预期之间的不一致；文档正文并没有明确保证该文件是 JSON，因此不将其扩大为算子或运行环境失败。

只读解析复核使用 `ConvertFrom-Json -ErrorAction Stop`，退出 **1**：

```text
Conversion from JSON failed with error: Unexpected character encountered while parsing value: P. Path '', line 0, position 0.
```

未修改该文件或执行脚本；报告目录另外提供有效的 `agent_follows_doc_report.json`，并保留原输出为 `documented_stdout.txt`。

## 证据范围与未独立复核事项

命令已按文档运行，包装脚本打印了 `DOCUMENTED_WITNESS_PASS`。本审核没有另外打开远端 `build_receipt.json` 或 `agent_kernel_receipt.json`；receipt 核验结论来自包装脚本。未重新计算 env spec 的 canonical SHA、源码 Git blob SHA、依赖 wheel SHA、预训练权重 SHA，也未重建过去的构建历史、编译参数或逐包对比原 bdetr 清单。这些文档历史/构建声明不能仅凭当前简短 stdout 判为再次独立证实。

witness 报告的源码 SHA256：`22955d73b72e5c49b3d6c811f9480392676e3d230309a86ec89e1a95b6132a75`。

三个扩展相关实际行为中，FPS、分组/梯度及 RoI 检查均通过；stdout 显式列出 point 和 stack 扩展路径，没有独立列出 roiaware 二进制路径。本报告不自行补出该路径。

## 留存文件

- `agent_follows_doc_report.md`：本完整报告。
- `agent_follows_doc_report.json`：有效结构化报告，包含命令、工具退出码、完整 witness、逐项判断、未验证事项、原始输出与文件 SHA256。
- `documented_stdout.txt`：按文档生成的原本地 --out 文件的原字节副本。

原本地输出 SHA256：`975c592efa10cc7db84e206834f22a17ffb3663856f07f9a6e08726033f0125b`。
执行计划 SHA256：`23d1a7dadc9ab81857a0e6503b14b4ebfae0afcfc7beebba7b622b6dc7b417fb`。
