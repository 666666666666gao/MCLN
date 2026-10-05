# PREPARED_SOURCE_REVIEW

结论：**PASS（修后复审）**。范围：**PREPARED_SOURCE_ONLY**。阻断问题为 0；初审的 B1、B2 均已按最小修正解决。此结论不满足最终 `SOURCE_ONLY` 启动门槛。

请求配置为 `gpt-6-astra / max`；实际后端及推理配置未获工具证明，归属仍为 same-family / provisional。

## 修正核对

- **B1 已解决：** `controller.py:39` 改读生产者实际输出的 `model_state_restored`；`prepare_probe.py:26–27` 同步该生成变换。控制器继续要求所有模型状态恢复及梯度缺失，没有增加别名或 fallback。
- **B2 已解决：** `analyze_probe.py:20` 改读 `complete/probe.exit`，与控制器生成、收集器保留的文件名一致，继续要求退出码为 0。

逐字节逆向移除这些修改后，三份文件都恢复为初审哈希；其余 9 份 Python 源码、`probe_body.py`/`run_cohort_probe.py` 所定义的方法和训练规则未变。四份生成脚本与旧模板的实际变换结果一致。12 份源码通过 Python 3.7 语法核对（body 按嵌入片段解析），PREPARATION 的 12 项哈希与实文件一致。

## 保留的审查结论

初审已核对的父模型缓存前向、完整 10 项几何状态替换、固定父资格/匹配排除、数据集 GT、整数 Mask 交并和 CPU float64 paired 统计逻辑保持不变。初审只读旧数据核算为 64 行、16384 候选、1090 资格候选；最大 CPU/GPU IoU 差 `2.3695575378512856e-06`，0.25/0.5 无阈值判定变化。相关源码及证据哈希未变，本次未重复运行旧指标脚本或覆盖 `METRIC_PRIMITIVE_CHECK.json`。

初审 FAIL 已原样保留为 `PREPARED_SOURCE_REVIEW_initial.json` 与 `.md`，两份文件均与初审交付哈希一致。修正证据 `PREPARED_SOURCE_FIXES.json` 的前后哈希已核对。

## 尚未满足的执行条件与证据范围

`spec.json`、`launch.json` 与最终 `SOURCE_REVIEW.json` 仍不存在。必须等待真实两组 train/formal 闭合和文本收集，绑定实际终点生成 spec，再做最终 `SOURCE_ONLY` 审查；实际诊断、收集和 CPU 核验应在任何非最佳终点头清理之前完成。没有执行 GPU、访问网络、读取终点权重、修改训练、安装依赖或删除权重。

该固定增强训练面板全部仅有 native GT slot 0，不能证明未见场景有效性、非空多 GT 保护或物理实例身份。资格仍使用父 GPU 框门槛与整数 Mask 计数，CPU 阈值差异会被报告；GPU DFL 未独立 CPU 重算，缓存头一致性不代表跨 CUDA 完整前向逐位一致。本地语法及旧数据检查不等于目标 Python 3.7/Torch 1.10.2 的实际运行证明。

## 已审文件与 SHA256

- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\analyze_probe.py` — `ac8b3988dd16703b9250b6869563eeeaa734dc1b499d96a6fd622ee879195b8d`

- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\check_prior_metric.py` — `4057628a6993d74f0d431aad708ea4483353d5842c262649b16bb0886941a13c`

- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\cohort_metrics.py` — `608a717c2257844172705ba774f9e61bf17d1faa9748b8a773b5cd6ba750a90a`

- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\collect_probe_authorized.py` — `de5a39203bd0e8ff8058a6ea5ea15a4639c1cfd4834c52ec8778d5f596c9cf21`

- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\controller.py` — `a5658bf487fcd701f76478af1b78419c90d1dea17f92c815a011912c9b6cb35c`

- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\launch_probe_authorized.py` — `436d981dfdd1b409b959d8cdbb56f846456829785146bec8971a61b87c3981d7`

- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\observe_probe_authorized.py` — `a6570ad337c693048d0424f08166b967e3b2f2bf2ec2589864d710f6cde89bbe`

- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\prepare_closed_spec.py` — `5a04450354826479f0edb1f37f7af40067d7c273ac260aeb2e1d81399f33b2fd`

- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\prepare_probe.py` — `79504839e8492b578b0574cecf8cb41d09cf42683080bac8a4f93891f405e81f`

- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\probe_body.py` — `0502ade839808218b943c87bb3dbaf49c02352413bf8785170afe7bbe6346ee2`

- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\publish_prepared_source.py` — `945f917fb56bcb3ce2960d94aaf43bf3fb68275bf9061b86e9634044445f04f1`

- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\run_cohort_probe.py` — `ff006960b9a1c0754ba6dc3ef163fc997c3594e2232e98c9bc25f0639766dd18`

- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\EXPERIMENT_PLAN.md` — `893f9dcdf38e2221b69fd11eaab594c0c5dbcf03b62d81ec1c1cbffe9654e23d`

- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\GENERATION.json` — `1a06080d772e7ea3e63f3e4fb53b20ec779c953e53b0f52051ccd37d3443e3c4`

- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\PREPARATION.json` — `a01cba52c462a5a81a676441e64190c55bd7562d93c11fafdb1ad06773c3e6de`

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

- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\PREPARED_SOURCE_FIXES.json` — `01b6cfffec5c90bd05b1ac2f102e064c9f989a2a47e93a29730e22f247ee838d`

- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\PREPARED_SOURCE_REVIEW_initial.json` — `94865062f07601195718b562817838885d6e68a56cec4236a9d610451f577cda`

- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\PREPARED_SOURCE_REVIEW_initial.md` — `e2b80c6226c9c23378de81359a281b3cd0ed2902ca53fdf9347a1331d9310f6b`

- `C:\Users\gb\.codex\tmp\pvground_mask_branch_responsibility_20261005\controller.py` — `6ba190275bfd121de585d7a70f0fab16a824063f16608a916526e3bf2bc4eceb`

- `C:\Users\gb\.codex\tmp\pvground_mask_branch_responsibility_20261005\launch_probe_authorized.py` — `3eb44f6adb5762949dfc36f8f700dae544ec627237beb9e3b74ab938024f9fba`

- `C:\Users\gb\.codex\tmp\pvground_mask_branch_responsibility_20261005\observe_probe_authorized.py` — `2b0e4cad962cddf69873af3fdb95202a907f455fe8f5c6e895aa6913096a7880`

- `C:\Users\gb\.codex\tmp\pvground_mask_branch_responsibility_20261005\collect_probe_authorized.py` — `de5a39203bd0e8ff8058a6ea5ea15a4639c1cfd4834c52ec8778d5f596c9cf21`
