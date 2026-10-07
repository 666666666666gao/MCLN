**PV-Ground face-support：真实终止产物审查**

时间：2026-10-07T13:00:37.494347+00:00。结论 **WARN；blocking findings=0；确定性实件核验PASS**。fresh context；same-family / provisional；请求路由Astra / max，实际backend与effort身份未认证。范围：TERMINAL_SOURCE_AND_ACTUAL_ARTIFACTS。

真实产物一致，但本轮不能证明face-region模块增益。两臂正式阈值逐表达标签相同，均低于同次前向reference及保护父模型。保护父模型仍是最佳实际候选。

| 实际候选／阶段 | 新增更新 | @0.25命中／准确率 | @0.5命中／准确率 |
|---|---:|---:|---:|
| 保护父模型 | 0 | 5598 / 58.8767% | 4848 / 50.9886% |
| 初始face-center | 0 | 5598 / 58.8767% | 4848 / 50.9886% |
| 初始face-region | 0 | 5598 / 58.8767% | 4848 / 50.9886% |
| 训练后face-center | 3723 | 5593 / 58.8241% | 4832 / 50.8204% |
| 训练后face-region | 3723 | 5593 / 58.8241% | 4832 / 50.8204% |

初始中性评估没有生成新的可选checkpoint。候选仅是保护父模型及两个实际terminal；按joint5620/4764、strict、wide、父模型平局优先的既定顺序，父模型胜出。联合目标仍缺22/0个命中。

**gt_provenance: PASS。** ScanRefer annotation object_id selects scan.get_object_bbox; target point membership creates gt_masks. FormalDataset uses val, predicted detections with butd_gt=False, and disabled augmentation. root_gt is dataset center_label[:,0,:3] plus size_gts[:,0], not a model reference. Historical/current row, scene, target, root_box and point SHA align exactly. No full raw-dataset reload was performed locally.

证据：run_face_support_pair.py:204-220；run_face_support_pair.py:261-274；paired_geometry_loop.py:291-318；pvground_g_p2_20261002/complete/source/joint_det_dataset.py:585-647；pvground_g_p2_20261002/complete/source/joint_det_dataset.py:1086-1119；pvground_g_p2_20261002/complete/source/joint_det_dataset.py:1324-1329；pvground_g_p2_20261002/complete/source/joint_det_dataset.py:1387-1392。

**score_normalization: PASS。** Accuracy is strict IoU>0.25/0.5 count divided by9508. Box IoU is geometric intersection/union, Mask IoU is point intersection/union. Native logits softmax and the original bbs textual-component score are retained; no reported metric is divided by a prediction max/min/mean. Every stored selected score is the unique maximum among256 candidates; zero score ties or argmax mismatches. Raw logits are not available for independent native-score regeneration.

证据：run_face_support_pair.py:243-250；run_face_support_pair.py:284-290；paired_geometry_loop.py:307-368；pvground_final_quality_20261005/runtime_bundle/native_root_bbs.py:4-9；pvground_runtime_bundle_20260908_v1/PV-Ground/src/grounding_evaluator.py:209-303；analysis/TERMINAL_AUDITOR_CHECKS.json。

**result_existence: PASS。** Actual controller and all3 modes closed exit0. All2421 files/382890067 bytes match intake hashes and sizes. Both stages have9508 rows/1189 NPZ, B8/final B4, all256 candidates and the exact8-field paired schema. The independent reviewer program recomputed19472384 float64 IoUs: zero selected threshold flips and zero full256 oracle label mismatches; SUMMARY and transitions agree. Root receipt formal_rows=0 refers to6887 internal holdout; separate formal receipts carry9508.

证据：fit_wait.json:3-34；complete_fit/fit_status.json:1-23；complete_fit/INTAKE.json；complete_fit/initial_formal/receipt.json；complete_fit/formal/receipt.json；analysis/TERMINAL_AUDITOR_CHECKS.json。

**dead_code_and_executed_call_chain: PASS。** Controller executes initial_formal -> train -> formal. Runner constructs PairedGeometryRun and calls run. Formal mode restores both3723-step terminals before evaluate. Evaluation calls native bbs and native evaluators, persists arrays/rows and asserts selected counts equal native bbs top1 counts. Existing analyzer output is independently reconciled; the production analyzer was not imported or rerun by this reviewer. Native bbf/top-k routines may also run within the native evaluator but are not claimed as the primary metric.

证据：controller.py:40-59；run_face_support_pair.py:292-299；paired_geometry_loop.py:239-258；paired_geometry_loop.py:289-386；postrun/analyze_face_support_formal.py:155-238；complete_fit/formal.log。

**paired_comparison: PASS。** One actual frozen parent forward and one native semantic-head call feed two disjoint heads with separate AdamW/gradient graphs. All3723 training records reflect the asserted route. Formal arms share actual query, masks, score and reference. Initial final arrays equal reference exactly. Trained arms differ by12088503 coordinate elements across9508 expressions, max0.030805587768554688, while selected and full256 coverage threshold labels agree per expression. Equal totals are not cancellation and do not imply a no-op.

证据：paired_geometry_loop.py:48-96；paired_geometry_loop.py:164-217；face_region_box_refiner.py:23-59；complete_fit/preflight.json；analysis/TERMINAL_AUDITOR_CHECKS.json。

**cross_stage_drift: WARN。** Initial/formal scores and original_prior are exact. Reference differs at444 elements/123 candidate slots/36 expressions, max2.5314231738448143. reference_valid differs at12 slots/8 expressions. All selected common fields and selected changed slots are0. Historical protected-parent query differs for3 expressions. The within-stage same-forward comparison is valid; not every all256 cross-stage change is a pure-training effect under bitwise fixed inputs.

证据：analysis/TERMINAL_AUDITOR_CHECKS.json；analysis/SUMMARY.json；EXPERIMENT_PLAN.md:30-46；postrun/analyze_face_support_formal.py:191-211。

**training_budget_and_checkpoint_identity: PASS。** 29778 unique IDs exactly match fit partition and are disjoint from6887 holdout;3723 sequential updates per arm,3722 B8 plus B2. All logged gradients are finite positive. Each head456102 parameters/10 states; frozen parent/R; seed2027,LR1e-5,WD0.0005,clip0.1. Eight hidden tensors inherit11169 updates and reach14892; two reset output tensors receive3723. Actual closed CPU inspection binds3 checkpoint identities, strictly restores each declared terminal sampler and10 Adam states, and reuses the unchanged protected1304-state fullCPU witness. Reviewer checked local source/receipt/hash relationships, without loading PTH or running NN.

证据：run_face_support_pair.py:127-170；paired_geometry_loop.py:219-258；paired_geometry_loop.py:416-462；complete_fit/train.jsonl；complete_fit/receipt.json；closed_weight_inspection.json；postrun/inspect_closed_face_weights.py:27-89；pvground_mask_reference_20261006/selected_candidate_CPU_restore.json。

**scope: WARN。** One ScanRefer development split:9508 expressions/141 distinct physical scenes, one seed2027, one paired fit. Internal train-split holdout6887/106 scenes is not another benchmark. No new Sr3D/Nr3D result. Mask IoUs are stored-row recounts, not raw-mask CPU regeneration. Full256 GT coverage is an oracle diagnostic, not deployed accuracy. No broad robustness, statistical equivalence, speed gain or3-effective-module conclusion.

证据：pair_spec.json:54-83；EXPERIMENT_PLAN.md:59-76；POSTRUN_TASKS.md:11-17；analysis/TERMINAL_AUDITOR_CHECKS.json；pvground_referit_mask_reference_20261006/current_research_goals.json。

**evaluation_type: PASS。** Formal boxes and producer mask metrics use annotated dataset targets. Extra-candidate GT qualification exists only in supervised training losses. Predicted reference and face sampling do not consume GT. Full256 coverage remains explicitly oracle diagnostic.

证据：mask_reference.py:10-28；mask_reference.py:42-66；query_supported_geometry.py:15-65；run_face_support_pair.py:261-274；paired_geometry_loop.py:307-350。

**独立重算的细节。** 全部2378个NPZ严格包含row_ids、root_gt、original_prior、reference、reference_valid、scores、final_face_center、final_face_region；scores/validity为[B,256]，四种boxes为[B,256,6]。所有selected slices与行记录一致。浮点值不逐位相同：最大选中CPU/GPU IoU误差1.1198971423764803e-5，但两阈值翻转均0。

| Full256 GT coverage（表达数） | @0.25 | @0.5 |
|---|---:|---:|
| 原始prior（两阶段） | 8952 | 7884 |
| reference（两阶段） | 8925 | 8163 |
| 初始两臂final | 8925 | 8163 |
| 训练后两臂final | 8915 | 8130 |

训练后每臂3322/3298个表达有合格候选但native未选中，另有593/1378个表达无合格候选。均是GT oracle诊断，不能算已恢复命中或新推理分数。选中invalid reference为39；全256 invalid初始/终止为1162195/1162191。

相对终止同次前向reference，两臂均为@0.25 repair4 / damage9 / net−5；@0.5 repair13 / damage29 / net−16。两臂之间repair/damage均0，Full256 coverage逐表达标签也相同。不同训练盒坐标不能当作精度贡献。

内部holdout6887/106 scenes：初始两臂6149/5667，终止控制6147/5665、方法6146/5666。方法相对控制−1/+1，不是正式增益或一致收益。保存Mask命中5812/5133，mIoU47.10762848393134%；本次仅stored IoU重计，**未独立重算当前raw masks**。

**Checkpoint实件绑定。** {
  "protected_geometry_parent": {
    "path": "/root/autodl-tmp/pvground_mask_reference_20261006/fused_mask_reference/initial.pth",
    "bytes": 1828167,
    "sha256": "2301e90a9391af4e4cd56edae09ce49db1cc8f67463e1c2568352ee2dafaff61",
    "actual_step": 0
  },
  "face_center/formal": {
    "path": "/root/autodl-tmp/pvground_face_support_20261007/face_center/terminal.pth",
    "bytes": 5586053,
    "sha256": "18efc943c3598dc801620cf5c505e166c7237cfc69639376adb1b5025385569d",
    "actual_step": 3723,
    "declared_sampler_restored": "face_center",
    "strict_CPU_geometry_restore": true,
    "optimizer": {
      "moment_and_step_states": 10,
      "param_groups": 1,
      "all_keys_moments_steps_and_groups_exact": true
    }
  },
  "face_region/formal": {
    "path": "/root/autodl-tmp/pvground_face_support_20261007/face_region/terminal.pth",
    "bytes": 5586053,
    "sha256": "4c18824ee7bb4e086328f25b7a19ede0faefcf521f2db3d025d169be3aa6f5bc",
    "actual_step": 3723,
    "declared_sampler_restored": "face_region",
    "strict_CPU_geometry_restore": true,
    "optimizer": {
      "moment_and_step_states": 10,
      "param_groups": 1,
      "all_keys_moments_steps_and_groups_exact": true
    }
  }
}

两个terminal各5586053 bytes，step3723，声明sampler分别face_center/face_region；10 geometry states与Adam的10份moments/steps/groups严格CPU restore。与旧formal_restore及fit receipt的SHA一致。保护initial为1828167 bytes、step0，checkpoint SHA为2301e90a9391af4e4cd56edae09ce49db1cc8f67463e1c2568352ee2dafaff61。

保留父模型沿用原1304-state完整CPU witness，当前factory字节与旧factory一致。按已读取的原writer序列化方式从本地envelope.receipt复原得到SHA 02d88b5770b395815e77df115f0a9d9e04f64032d65a0ad30b0362f08a6d7a22，与本次及上次actual inspection相同。新inspection绑定SUMMARY SHA 1d24e87ded2d99a5d83a05ab53d3fad72b0a6c7e8da54b064c2687226f34e2cd；其实际文件SHA c9d70101c02d71f76b61852d53e71ca8ac0ff0a9089e1f8ccccf9ee7b0236455。审查者未重载PTH、构造模型或执行GPU。

**论断限制与后续。** Same-family/provisional semantic judgment; actual backend and reasoning effort identity are UNATTESTED. Reviewer used local stdlib/NumPy only:0 SSH,0 NN forwards,0 optimizer steps,0 PTH loads,0 cleanup,0 downloads. GT provenance uses exact imported-source bindings, dataset lineage and aligned stored roots; full original raw dataset scans were not reopened locally. Native score call path and actual stored argmax were checked, but raw logits were not archived for independent score regeneration. Formal Mask metrics were only recounted from stored IoUs; no current raw-mask CPU recomputation. Actual new inspection restores each terminal geometry head and Adam only. The selected full1304-state model witness is the unchanged historical protected witness, not a new full-model inference. SUMMARY is the immutable pre-audit snapshot; its pending flags are superseded by these reports and the actual inspection for audit/selection evidence only. Later cleanup/publication still requires actual receipts.

独立程序terminal_auditor_checks.py已实际运行一次，23.817秒，退出0；结果保存在TERMINAL_AUDITOR_CHECKS.json。原生产analyzer未二次执行。完整文件摘要见EXPERIMENT_AUDIT.json.actual_file_digests。该WARN限定科学解释，不把负结果当成完整性失败。清理与发布状态必须由后续真实receipt单独证明。
