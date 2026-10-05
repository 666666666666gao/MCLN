# PREPARED_SOURCE_REVIEW

结论：**FAIL**。范围：**PREPARED_SOURCE_ONLY**。发现两处确定的生产者/消费者契约错误；核心固定候选、缓存几何头比较与 CPU 统计逻辑未发现其他阻断问题。

请求的审查配置为 `gpt-6-astra / max`；实际后端及推理配置未获工具证明，归属为 same-family / provisional。此报告不满足最终 `SOURCE_ONLY` 启动门槛。

## 阻断问题

1. **B1 — 成功探针会被控制器的旧回执字段阻断。** `controller.py:39` 读取 `model_state_unchanged`；新 `probe_body.py:158` / `run_cohort_probe.py:368` 在恢复父头并核对完整状态后只输出 `model_state_restored=True`。成功子进程退出后必然触发 `KeyError`，磁盘状态仍为先前的 running，外层 `controller.exit` 为 1。最小修正：控制器改读 `model_state_restored`，并同步 `prepare_probe.py:20–35` 对旧控制器的生成变换，避免再生成旧字段；不增加别名或 fallback。

2. **B2 — CPU 分析检查了不存在的退出文件。** `analyze_probe.py:20` 读取 `complete/child.exit`；`controller.py:32` 生成 `probe.exit`，收集器保留原文件名。即使 B1 修复，正常收集后的分析仍会在加载 NPZ 前触发 `FileNotFoundError`。最小修正：该检查改为 `complete/probe.exit`，保留退出码必须为 0 的要求。

## 已核对的正确部分

- `prepare_closed_spec.py` 要求真实闭组观察、控制器结束/零退出、两组各自 train/formal 完成、3723 更新、29778 训练行、9508 正式行、终点/恢复 SHA 一致和文本 intake；不会把现有单组 control 回执当成双组终态。
- 每批只做一次完整父前向，并在同一 Query、点、粗框、Mask 和原生分数上替换全部 10 项几何状态。父缓存重放必须等于父最终框；三个头只改变几何输出，没有优化器、反向或权重写入。最后恢复父头并核对全部模型状态。
- 资格固定于父输出和父原生最终层匹配：自身 Query Mask 与融合 Mask 均 >0.5，父最终框 ≤0.5，且排除所有已匹配 query。训练后跨阈值不会改变 cohort。GT 来自数据集标签。
- 保存的整数 Mask 交并支持 CPU 精确资格重算；CPU float64 框 IoU、面误差和末次移动支持固定组的 paired repair/damage。DFL 的非均匀结点、裁剪及插值与原训练函数一致。

## 只读验证与限制

12 个 Python 文件通过 Python 3.7 语法解析（`probe_body.py` 按嵌入函数体解析）；PREPARATION 哈希、生成 runner 与 source/body 拼接结果一致，7 个已审运行依赖符合继承 spec 的哈希。旧闭合 NPZ 只读重算：64 行、16384 候选、1090 资格候选；最大 CPU/GPU IoU 差 `2.3695575378512856e-06`，0.25/0.5 均无阈值判定变化。严格阈值的修复/破坏小型数值检查通过。两份既有 64 行记录的 row/scan/point/GT 身份一致，全部仅有 GT slot 0。

当前 `spec.json` 与 `launch.json` 均不存在。没有执行新 cohort、读取终点权重、访问远端、启动 GPU、修改训练、安装包或删除权重。语法检查与旧数据核算不是目标 Python 3.7/Torch 1.10.2 运行证明。该固定增强训练面板不能证明未见场景有效性、非空多 GT 保护或物理实例身份。GPU DFL 没有独立 CPU 重算；CPU 与 GPU 的近阈值差异按代码明确报告，缓存头一致性也不代表跨 CUDA 的完整前向逐位一致。

修正 B1/B2 后应复审改动；真实双组闭合及文本收集后，才能绑定实际终点生成 spec，再做最终 `SOURCE_ONLY` 刷新。实际 GPU 诊断、收集和 CPU 核验必须在删除任何非最佳终点头之前完成。

## 已审文件与 SHA256

- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\analyze_probe.py` — `c894078e6e16b78b3fd022e6b943bac4089c91a554e80a3dc12e4d441c6f181e`

- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\check_prior_metric.py` — `4057628a6993d74f0d431aad708ea4483353d5842c262649b16bb0886941a13c`

- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\cohort_metrics.py` — `608a717c2257844172705ba774f9e61bf17d1faa9748b8a773b5cd6ba750a90a`

- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\collect_probe_authorized.py` — `de5a39203bd0e8ff8058a6ea5ea15a4639c1cfd4834c52ec8778d5f596c9cf21`

- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\controller.py` — `26b867cb760d16b47d59241316caa807111ea937270e5761f7118c5432c5aa83`

- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\launch_probe_authorized.py` — `436d981dfdd1b409b959d8cdbb56f846456829785146bec8971a61b87c3981d7`

- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\observe_probe_authorized.py` — `a6570ad337c693048d0424f08166b967e3b2f2bf2ec2589864d710f6cde89bbe`

- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\prepare_closed_spec.py` — `5a04450354826479f0edb1f37f7af40067d7c273ac260aeb2e1d81399f33b2fd`

- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\prepare_probe.py` — `f4fabce3fb6bb7b07dd22b2709367f5a560148de8a033cdcea5cf8ec217aa9bc`

- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\probe_body.py` — `0502ade839808218b943c87bb3dbaf49c02352413bf8785170afe7bbe6346ee2`

- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\publish_prepared_source.py` — `945f917fb56bcb3ce2960d94aaf43bf3fb68275bf9061b86e9634044445f04f1`

- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\run_cohort_probe.py` — `ff006960b9a1c0754ba6dc3ef163fc997c3594e2232e98c9bc25f0639766dd18`

- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\EXPERIMENT_PLAN.md` — `893f9dcdf38e2221b69fd11eaab594c0c5dbcf03b62d81ec1c1cbffe9654e23d`

- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\GENERATION.json` — `1a06080d772e7ea3e63f3e4fb53b20ec779c953e53b0f52051ccd37d3443e3c4`

- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\PREPARATION.json` — `fe4d02ff16cfc658b36bfe69dee4c093ca3c476a0e0668239561342efa3332b1`

- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\METRIC_PRIMITIVE_CHECK.json` — `f01b25d0328243ac61bdefe4f6f737737f85b4821b31c7ae8221791b4b289e95`

- `C:\Users\gb\.codex\tmp\pvground_mask_branch_responsibility_20261005\run_mask_branch_probe.py` — `84c0858166781214cf241a04d5ee42243b92246fe7c4705579568f58fc6602c7`

- `C:\Users\gb\.codex\tmp\pvground_mask_branch_responsibility_20261005\spec.json` — `cd02a2687fbee148256e101f7709382dcedc0834f4d46f00c0c4e7cebd2a9a1a`

- `C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\runtime_bundle\pvground_boundary_box_refiner.py` — `88debfef64f4295f87d697b7b9d4cd1eab17b39f4041f6d9ae773fc65dda9461`

- `C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\runtime_bundle\pvground_whole_mask_box_refiner.py` — `4a701338bcbf042a4eb7793209999434d68c269e563ac09008e758228fc72118`

- `C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\runtime_bundle\pvground_tail_support_box_refiner.py` — `665e94c150492a9fc3d52ddca7da2b7dca7d7ddbd981e759640f15846bb3f757`

- `C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\runtime_bundle\pvground_candidate_box_refiner.py` — `e971346230d0e0547139823c106e48f970c58a0b0ef26ef46b75beaa672f6212`

- `C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\runtime_bundle\readback_model_factory.py` — `c235461179969fdd0f50d503a831512f27c083d9866479ab1777857a507c3763`

- `C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\runtime_bundle\whole_mask_range.py` — `22c4adb0cb8dfb929825dcbfcb5f9696cea879da212cebeea1a1a5acb6347a4c`

- `C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\runtime_bundle\readback_preflight_checks.py` — `1a0be39f8dc0b1b10c0ace1a0e5d560280d4ac9e2dc2c5694a4b5cff571094d3`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\controller.py` — `8f43a56efa9a6d1097899fd7474054e2663bfb46547fce157705344f8baec895`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\fit_body.py` — `a67200e7d37bbc87d8303cee56eb63812d0725e9a3b0e364306fec5c64737233`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\observe_fit_authorized.py` — `698a5203af439fc2fe90299c606a95eaa6014bdfa1e8443bf6fe47b8614e96e4`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\collect_formal_authorized.py` — `a4711fc24e4e55f4d1e1515092d07ecdc7b991ded68af088eb60021bdd00e890`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\run_geometry_fit.py` — `ecfaeb9e73aba2d317abd80f5dba340f508b224ebf02f2b262549402ef926889`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\closed_control_receipts\control\receipt.json` — `3779254e981d3ba1c3ad5da7e5c7499b1d06efc29cc78fe1604bfc58dbdfe2e6`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\closed_control_receipts\control\formal_restore.json` — `1f7cc61d14972c4d3d56cae0333c26d35047486e2afc94d5cec492e07ac75d4d`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\closed_control_receipts\control\formal\receipt.json` — `63a24be5db5dd873eae6cea6e2f2776fd8cb8b24b12f65e5753dab36f7016574`

- `C:\Users\gb\.codex\tmp\pvground_mask_branch_responsibility_20261005\complete\receipt.json` — `5441778fd4e4021c68215a700bc950c65a2acc714caf28bb10c464d7267c7064`

- `C:\Users\gb\.codex\tmp\pvground_mask_branch_responsibility_20261005\complete\batch_00.npz` — `71e5148220341d0f76480bd1a7e002ce6064d6a992d86f3d5cc98311d023bd32`

- `C:\Users\gb\.codex\tmp\pvground_mask_branch_responsibility_20261005\complete\batch_01.npz` — `4fd47625792533a532433894d0f94175cf576c888583c5b80ee7fec8e6c75fa2`

- `C:\Users\gb\.codex\tmp\pvground_mask_branch_responsibility_20261005\complete\batch_02.npz` — `fb737abb8b934f4f2ed3e6d49dd7b94d739862b08e54105949b1e8b71a4bbaaf`

- `C:\Users\gb\.codex\tmp\pvground_mask_branch_responsibility_20261005\complete\batch_03.npz` — `3a86c5ec2c3337bb2d4a9a0421ebb58824d032965090d3067cf93cf58c60002e`

- `C:\Users\gb\.codex\tmp\pvground_mask_branch_responsibility_20261005\complete\batch_04.npz` — `bd1aeb4f9a35e0f56d2c419a909c229387411e819ac943395db6f58f09382c84`

- `C:\Users\gb\.codex\tmp\pvground_mask_branch_responsibility_20261005\complete\batch_05.npz` — `c134c31445ce9c1d07ff6106b5de46f82d14c39cc0fda831303060d29424a626`

- `C:\Users\gb\.codex\tmp\pvground_mask_branch_responsibility_20261005\complete\batch_06.npz` — `f641f87c0920b8b6f81b661e1b3d655144fb0b0f971203ea664e7ee402e8277b`

- `C:\Users\gb\.codex\tmp\pvground_mask_branch_responsibility_20261005\complete\batch_07.npz` — `946f286c107fa5e066f4cd84defb04eec112f44b17214a76386480a1d44edbfc`

- `C:\Users\gb\.codex\tmp\pvground_mask_branch_responsibility_20261005\complete\rows.jsonl` — `5f93346437c0c154759af270815e52ccc840a7e80f451f67a25904fd1b0e7c0f`

- `C:\Users\gb\.codex\tmp\pvground_mask_geometry_responsibility_20261005\complete\rows.jsonl` — `959bf18c908b44e84e5822b7c19f8ef939414207b833e6c2dbe3ebf9e63dd6b7`

- `C:\Users\gb\.codex\skills\experiment-bridge\SKILL.md` — `ba0a305c9fc250966092ec2789d3ddc67007486cf9d1106691edd425276d7e35`

- `C:\Users\gb\.codex\skills\shared-references\local-codex-policy.md` — `b0249bfccd48ff79a5976afa7a54c64a3cfb65a5496a2105c5bb21069d197783`
