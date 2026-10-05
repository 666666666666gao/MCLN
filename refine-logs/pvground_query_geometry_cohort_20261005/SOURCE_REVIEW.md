# SOURCE_REVIEW

结论：**PASS**。范围：**SOURCE_ONLY**。阻断问题为 0。实际双组闭合证据、新 spec 终点绑定与最终源码门槛已核对；未发现新的源码或绑定差异。该结论不是 GPU 执行结果、结果完整性结论或正式指标接受结论。

请求配置为 `gpt-6-astra / max`；实际后端及推理配置未获工具证明，归属为 same-family / provisional。

## 实际闭合与输入绑定

- 双组总控制器于 `2026-10-05T13:53:50.086084+08:00` 结束，观察器于 `2026-10-05T13:54:07.953051+08:00` 记录控制器不存活、退出码 0。control/query_supported 的 train/formal 四项全部完成且退出码为 0。
- 每组训练回执记录 3723 次更新、29778 条 fit 输入恰好一遍、10 项几何状态/456102 参数；每组正式回执均为 pass、9508 行。两组正式恢复回执均记录严格恢复成功，终点 SHA 与训练回执及新 spec 一致。
- 实际 intake 于 `2026-10-05T13:57:58.757545+08:00` 收集完成，73 个文件共 45176490 字节与清单一致。两个 arm spec 仅 root 和 extra_geometry_weight（0/1）不同。原始 checkpoint、G、4506 几何父模型以及源码/环境/输入身份与新诊断 spec 一致。

实际 spec SHA256：`748809b0e1c740b3253d992de719b8e8e1bac55518e7e1e0a5a6651932f1de71`。其内容已在内存中用未变的父 spec、真实两组回执及 fit_wait/INTAKE 哈希重建并比对一致；未执行 spec 写入器。

| 几何头 | 终点 SHA256 | 对应训练 spec SHA256 |
|---|---|---|

| control | `fcce28c679193e0ede4b1127080bc53376cf2f7b33fbbda94fcaf5f8d66a1ea8` | `282b7f117c3c242970416608c05e8f9b7684f1bf3b0ccfc0159990771505a9a1` |

| query_supported | `0b37986d4448b125405d573272282ef37396a781ba970789dfc9b080445cec51` | `30fb451ec8ab08b8825e0ab15d4c47504d5f3e9d1027f0ad52363a931fb79c33` |


## 源码契约与门槛

先前审查的 53 个文件均保持原哈希；12 份诊断源码、生成体、指标、修正记录及父级依赖未变。重读要求的源码后，确认每批一次完整父前向、三次同上游缓存几何头重放、完整 10 项状态替换，原生语义头每批只调用一次，原生 logits/分数和 Mask 保持固定。所有父状态及 zero-R 冻结，结束后恢复父头并逐项核对完整模型状态。没有优化器、反向、更新或新权重写入。

资格固定为父模型自身 Query Mask 与融合 Mask 均 >0.5、父最终框 ≤0.5，并排除父原生最终层所有匹配 query；不随任一训练后头的表现重新筛选。GT 来自数据集。`model_state_restored`、`probe.exit` 与 `controller.exit` 生产者/消费者契约保持修复，生成器也保留修复。

启动器要求 `SOURCE_ONLY`、PASS/WARN 且无阻断项，并核对所审文件哈希；本报告列入实际 spec。它仍会在启动时检查远端闭组状态/零退出、GPU 空闲、可用空间与资源锁，探针会在加载前校验两份终点字节。本次没有执行这些远端检查或启动动作。

## 指标与证据范围

CPU 使用整数 Query/fused Mask 交并及固定父 GPU 框门槛重建资格，对保存的框以 float64 核算 IoU、最大单面误差、末次移动及固定组的严格阈值 repair/damage，明确列出 CPU/GPU 阈值差异。GPU DFL 与训练插值定义一致，但不独立 CPU 重算。

该面板是已见过的增强训练 64 行，已核实的参考数据仅有 GT slot 0，候选彼此相关；不能据此声称未见场景有效性、非空多 GT 保护或物理实例身份。缓存头一致性也不代表跨 CUDA 完整前向逐位一致。初审旧数据指标检查仍只作为旧数据工具验证，未覆盖或重跑其记录。

本次仅写 SOURCE_REVIEW.json/.md，保留所有 PREPARED 历史；未运行 GPU/SSH、读取终点权重、修改实现/训练、安装依赖或删除权重。实际诊断、收集、CPU 核验及后续结果审查仍待执行，并必须先于任何非最佳终点头清理。

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

- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\PREPARED_SOURCE_REVIEW.json` — `c8d390cd80a34d4a5e244524a857dc9ef3cd25395b447f865404ad350452e037`

- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\PREPARED_SOURCE_REVIEW.md` — `27aec7ab469fb04aa213f777ba395ba87494c544f52ff03031afa5f5be6eed31`

- `C:\Users\gb\.codex\tmp\pvground_query_geometry_cohort_20261005\spec.json` — `748809b0e1c740b3253d992de719b8e8e1bac55518e7e1e0a5a6651932f1de71`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\fit_wait.json` — `701b071d8a92f8bc82de50e37d15ea511960f3cb2a71aef65eb6a77f6ab02d87`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\INTAKE.json` — `4c6b2457880e28aeda378e70b388f379fc8819f74405c2c6dfaf9b5fe16bd938`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\EXPERIMENT_PLAN.md` — `b62e838197e85eacd19076696190fdad0590e93d910876bcf98f3fc795a9a947`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\SOURCE_REVIEW.json` — `a71f65de0423e7ee722b96e92641eb0a7ede8ab6588a43477879e7e3b8d33fd7`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\SOURCE_REVIEW.md` — `c77a2ca3fbbdee21b2b7d09dc0e6bfd0aebe81d27ab3857c871c2f6fde1940f9`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\control\formal\receipt.json` — `63a24be5db5dd873eae6cea6e2f2776fd8cb8b24b12f65e5753dab36f7016574`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\control\formal\rows.jsonl` — `f21000ce1dfa75309627d71d0e56f1580b7d1d3f6d758a5f8acdad8a2179b00f`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\control\formal.exit` — `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\control\formal.log` — `14529032e8d76f3fcb135460936bae6bc126f35b83f68035ee389a061c54ecc3`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\control\formal_restore.json` — `1f7cc61d14972c4d3d56cae0333c26d35047486e2afc94d5cec492e07ac75d4d`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\control\imports.json` — `51fd47332fe71a1ce94c22eedd9256c095346f9808304630395e0f2c24c98a23`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\control\initial\receipt.json` — `875edb6d167676b42c42e1b85dbb9b6c807f9b42073c372486d9da607736ed4c`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\control\initial\rows.jsonl` — `944313ba5ceb0aca7f7d37ff2d6e419efba6eb10bfd6d6bfcbfe92937c7d0563`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\control\load.json` — `0de7fcbcbe21c94b035209b3329a8a54c1c4194fe89bc5448425cae9bd17b491`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\control\preflight.exit` — `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\control\preflight.json` — `a2b904c5145ba3ac0f13b3a80edf8c743150c3a6782e35bf1fed8a3f9d64e37c`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\control\preflight.log` — `bc944f386991cc7a5e81fdad7c077a841c3466b43d1bacbb9081a66b71e35300`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\control\receipt.json` — `3779254e981d3ba1c3ad5da7e5c7499b1d06efc29cc78fe1604bfc58dbdfe2e6`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\control\terminal\receipt.json` — `84c53976a2f73cc70adf33ff934121b9eadc7bb124cf5e101af32073bb8201af`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\control\terminal\rows.jsonl` — `e2979299fba80a8850d19c9c7271ad1c2ff0367e59ba0f07117447a1884fe338`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\control\train.exit` — `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\control\train.jsonl` — `b4c9f13a8aa388753fc6c3d80a64f4ecadc4c0c2251f0818e0665d91f28dadff`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\control\train.log` — `16a8dcf75284e7979a38aacfff8e12850696aaec425ff188ca119e805e7134f7`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\control_spec.json` — `282b7f117c3c242970416608c05e8f9b7684f1bf3b0ccfc0159990771505a9a1`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\controller.py` — `8f43a56efa9a6d1097899fd7474054e2663bfb46547fce157705344f8baec895`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\fit_controller.exit` — `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\fit_controller.log` — `e3c98c416efbdfce06fb89fe65e4dc341e4fcfd621d29f1cbf012c0c565a1e66`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\fit_launch.json` — `db9e2911484a538b8640fd6d8db0a965467b3a3125dd41e005eeba88ed41c488`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\fit_status.json` — `1a9c0de09e70af2d7676277dad0571701222f82646fdcb3a59efc8a0423f2a75`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\preflight_complete\EXPERIMENT_PLAN.md` — `b62e838197e85eacd19076696190fdad0590e93d910876bcf98f3fc795a9a947`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\preflight_complete\INTAKE.json` — `965099206cedbd8b37a2cdd46610680b752652bb4eb7e04d2a06a50549860617`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\preflight_complete\SOURCE_REVIEW.json` — `a71f65de0423e7ee722b96e92641eb0a7ede8ab6588a43477879e7e3b8d33fd7`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\preflight_complete\SOURCE_REVIEW.md` — `c77a2ca3fbbdee21b2b7d09dc0e6bfd0aebe81d27ab3857c871c2f6fde1940f9`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\preflight_complete\control\imports.json` — `51fd47332fe71a1ce94c22eedd9256c095346f9808304630395e0f2c24c98a23`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\preflight_complete\control\load.json` — `17b08e65feedfd128571d0a6a8c3c13ef8e953963bf1f4b99c010a4b213e85c9`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\preflight_complete\control\preflight.exit` — `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\preflight_complete\control\preflight.json` — `a2b904c5145ba3ac0f13b3a80edf8c743150c3a6782e35bf1fed8a3f9d64e37c`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\preflight_complete\control\preflight.log` — `bc944f386991cc7a5e81fdad7c077a841c3466b43d1bacbb9081a66b71e35300`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\preflight_complete\control_spec.json` — `282b7f117c3c242970416608c05e8f9b7684f1bf3b0ccfc0159990771505a9a1`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\preflight_complete\controller.py` — `8f43a56efa9a6d1097899fd7474054e2663bfb46547fce157705344f8baec895`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\preflight_complete\preflight_controller.exit` — `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\preflight_complete\preflight_controller.log` — `de394bed8819341fe0ff7b9163cad4de402daf4c6e0ac3946cae24e5d8bfc5b5`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\preflight_complete\preflight_status.json` — `897407cab630d81048761c29e8ba83ac11c37a66fcdea3ffae8faea382a78d8e`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\preflight_complete\query_supported\imports.json` — `51fd47332fe71a1ce94c22eedd9256c095346f9808304630395e0f2c24c98a23`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\preflight_complete\query_supported\load.json` — `278d3c5a9d220f4232308c6a72c47002dd5ad2af23ced52fbc9ee77e3035fd78`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\preflight_complete\query_supported\preflight.exit` — `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\preflight_complete\query_supported\preflight.json` — `4655ccbbf97c16cf7b9fe283f22084577a4a33cdc448df72f44d2679a0c472fe`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\preflight_complete\query_supported\preflight.log` — `044b8cbee813540eda266ed18b4dbd8b26397a4ba11285b8c03855b49f082a8f`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\preflight_complete\query_supported_geometry.py` — `aebeb6ec667a3a2e61098268193664bf2ef02a1ce8dc72303badca8b19dc964c`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\preflight_complete\query_supported_spec.json` — `30fb451ec8ab08b8825e0ab15d4c47504d5f3e9d1027f0ad52363a931fb79c33`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\preflight_complete\run_geometry_fit.py` — `ecfaeb9e73aba2d317abd80f5dba340f508b224ebf02f2b262549402ef926889`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\preflight_controller.exit` — `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\preflight_controller.log` — `de394bed8819341fe0ff7b9163cad4de402daf4c6e0ac3946cae24e5d8bfc5b5`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\preflight_status.json` — `897407cab630d81048761c29e8ba83ac11c37a66fcdea3ffae8faea382a78d8e`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\query_supported\formal\receipt.json` — `fdab1dd33b81fa70551b76d6eed6de5c8fbce95b43199297bbe35e19353a33f1`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\query_supported\formal\rows.jsonl` — `5ef50c722abb31f10e02a06615d0f2cd437b0bcb354b32f5135ebf1304db9e16`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\query_supported\formal.exit` — `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\query_supported\formal.log` — `60df0b70eb586bf2ed5e12eb629653829b9c65a80074fce8e79fb6e14291f067`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\query_supported\formal_restore.json` — `2fdbc5e800c8239906a0ca072179af13a3b228f668fd5ae41395eaa8c483bb82`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\query_supported\imports.json` — `51fd47332fe71a1ce94c22eedd9256c095346f9808304630395e0f2c24c98a23`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\query_supported\initial\receipt.json` — `fb821d87e39e9624009059bc4720769ffa7e8ea343104b2cb3d33e58780985d2`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\query_supported\initial\rows.jsonl` — `05f0585ea8aafbfe1e2f394359ff9071402c0d380d3cd732d5f7688c5622c2d8`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\query_supported\load.json` — `333602b496fa48c5d47d6ee795a7ee7e16b4d37a4648f0b3f4f153c4300f8448`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\query_supported\preflight.exit` — `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\query_supported\preflight.json` — `4655ccbbf97c16cf7b9fe283f22084577a4a33cdc448df72f44d2679a0c472fe`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\query_supported\preflight.log` — `044b8cbee813540eda266ed18b4dbd8b26397a4ba11285b8c03855b49f082a8f`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\query_supported\receipt.json` — `086c948430ae5d8c08d971155ef01834d5d21b581204d7a6d02634d4b93729f6`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\query_supported\terminal\receipt.json` — `19f925e625c1e918653c04dbc86e9220b48473cf391e4233f7bc7a13bbcee458`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\query_supported\terminal\rows.jsonl` — `7bff0ca8678f3037c8a890ee9f55bdd4b82073a400fdb3a56afbabc6a199659c`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\query_supported\train.exit` — `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\query_supported\train.jsonl` — `05b98a4da41867b0200246e49040fd744e969e746daf0744321d7c577b286a68`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\query_supported\train.log` — `ab786955be7dc48fc10e26e3290f4e6d39e4158080101433b1bc26f3a025e2a9`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\query_supported_geometry.py` — `aebeb6ec667a3a2e61098268193664bf2ef02a1ce8dc72303badca8b19dc964c`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\query_supported_spec.json` — `30fb451ec8ab08b8825e0ab15d4c47504d5f3e9d1027f0ad52363a931fb79c33`

- `C:\Users\gb\.codex\tmp\pvground_query_supported_geometry_20261005\complete\run_geometry_fit.py` — `ecfaeb9e73aba2d317abd80f5dba340f508b224ebf02f2b262549402ef926889`
