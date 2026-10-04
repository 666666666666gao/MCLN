# Source review: native final-quality arm

Verdict: PASS — SOURCE_ONLY. No blocking or nonblocking code defect was found; no source correction is requested. This is a same-family, provisional review. Astra/max was requested, but actual model and reasoning effort are not attested.

No GPU/runtime/accuracy PASS is granted. I performed source reads, standard-library AST, file-identity, diff and historical-control JSON checks only. No Torch import, weight load, model construction, training, evaluation, SSH/SFTP operation or credential-wrapper read was performed.

**score**. Native root bbs is reproduced exactly at source level: softmax token probabilities times binarized main-positive map, plus raw modify/pronoun/relation maps, minus raw other_entity map. No sigmoid class-probability interpretation or extra inference score. Evidence: runtime_bundle/native_root_bbs.py:4; native evaluator:218-281,535-553,565-646.

**pool_and_target**. Pool is the exact unchanged original-G qualified_unmatched mask plus the unique original final-matcher root query. All queries matched to nonroot GTs remain excluded. Final centers/sizes and root GT are detached; provider sizes are floored at 1e-6. Eligibility remains a training geometry proxy. Evidence: runtime_bundle/native_final_quality.py:12; runtime_bundle/pvground_semantic_assignment.py:10; runtime_bundle/pvground_boundary_box_refiner.py:27.

**matcher**. matching[1] is correct: the native criterion calls matcher in proposal_, last_, 0head_ through 4head_ order, once per prefix. The captured root is target zero after valid-GT filtering; qualified_unmatched asserts slot-zero validity. Evidence: native models/losses.py:817,852-853,872-885,891-917; runtime_bundle/run_final_quality_fit.py:373-381,454-462.

**loss_reduction**. The implemented sample mean of within-pool centered squared errors equals half the ordered-pair mean squared score-minus-IoU difference. Gradient to score k is 2*(e_k-mean_C(e))/(batch_size*pool_size) inside C and zero outside. The fixed coefficient is 1.0; native CE/G/contrastive denominators and clipping are unchanged. Evidence: runtime_bundle/native_final_quality.py:centered_quality_error; runtime_bundle/run_final_quality_fit.py:460-470; quality_fit_spec.json; EXPERIMENT_PLAN.md.

**preflight_witnesses**. Source contains a separate pairwise tensor construction and loss/logit-gradient comparisons, exact nonpool zero direct gradient, explicit other-matched-GT exclusion, equal-score high-IoU promotion/low-IoU reduction, and isolated quality gradients to R before/after the first update. It also checks native bbs independently and same-frame Box/Mask preservation. These real-batch assertions have not been executed in this review. Evidence: runtime_bundle/native_final_quality.py:verify_quality_loss; runtime_bundle/readback_preflight_checks.py; runtime_bundle/run_final_quality_fit.py:360-394.

**preflight_restore**. Preflight starts a fresh zero-output R, runs exactly two AdamW steps on one real eight-row batch, serializes R delta and AdamW into BytesIO, perturbs one R parameter, reloads strict complete state and optimizer, checks all moments/steps/groups and step=2, and checks frozen parent states. It writes no weight file. Formal evaluation uses a separately started process with a fresh model/optimizer and strict terminal restore. Evidence: runtime_bundle/run_final_quality_fit.py:216-235,360-421; runtime_bundle/whole_model_preflight_checks.py:optimizer_restore_exact.

**control_fairness**. The 15 inherited helper modules are byte-identical to the completed evidence-visible control. The runner diff adds only the requested quality term, its two-update preflight and associated metadata. R remains 96672 parameters/23 tensors with the same seed-2027 zero-output initialization, frozen4506 parent chain, AdamW, lr1e-5, decay0.0005, clip0.1, batch8, one 29778-row pass and 3723 updates. Controller compares every new batch row list to the actual old trace. Evidence: runtime_bundle/readback_model_factory.py; quality_fit_spec.json; formal_draft/complete/evidence_visible/train.jsonl; controller.py:100-105.

**native_route_eval**. Existing delayed semantic source has one native semantic-head call after native Mask, final Box refinement and R. R receives only prediction/input/text evidence. New quality code is invoked only in preflight/train. The runner explicitly imports the installed native evaluator, runs all 9508 val rows/all 256 candidates, uses the selected bbs query for both final Box and native fused Mask, and checks aggregate Box hits and Mask IoU against that evaluator. Cached-head and candidate-availability outputs are separate diagnostics. Evidence: revision2/source_preview/PV-Ground/models/pv_ground.py:507-575; runtime_bundle/run_final_quality_fit.py:206-212,272-358; runtime_bundle/readback_preflight_checks.py:observed_readback_forward.

**controller_and_launcher**. Controller requires the new local preflight receipt before train/formal; it does not consume the inherited unused preflight_root field. It verifies exact control order, strict formal restore and CPU selected-Box threshold counts before deleting only the new owned terminal when Acc@.50 is not above 4506; the three reconstruction parents are checked and retained. Launcher verifies this source review, the known remote environment, prior controller closure, GPU idleness and free space; uploads only a new absent root; launches only the bounded preflight under the existing GPU lock. No remote code or credential wrapper was executed/read by this review. Evidence: controller.py:59-135; launch_quality_preflight_authorized.py:12-84.

**fit_launcher_and_observer**. Full-fit launcher requires the previous observer to have closed on controller exit zero, the actual two-update preflight receipt and exact remote/local receipt bytes, unchanged spec/runner payloads, GPU idleness and storage reserve. It starts only the planned train/formal controller under the existing GPU lock. Observer snapshots its explicitly supplied stage once, waits until the estimated first check, then uses 240-second polls; it records process/status/exit evidence without replaying inference or optimization and stops after a window of roughly twice the estimate. Preflight receipt collection into preflight_complete is an explicit prerequisite of the fit launcher, not an action claimed complete by this review. Evidence: launch_quality_fit_authorized.py:12-81; observe_quality_authorized.py:13-59.

The centered loss is mathematically consistent. For e_i = s_i - u_i and n = |C|, mean_C[(e - mean_C e)^2] = (1 / (2 n^2)) sum_(i,j in C)(e_i - e_j)^2. Its score derivative is 2(e_i - mean_C e)/(B n) within C and zero outside. At equal scores this becomes -2(u_i - mean_C u)/(B n), so gradient descent promotes above-mean IoU candidates. A singleton pool contributes zero by the stated formula. The isolated direct-gradient guarantee concerns the new term's query logits; it does not claim that shared R parameters cannot change other queries after an update.

The exact historical visible control has 3723 sequential records, 29778 distinct rows, 3722 eight-row batches and one two-row tail. Its formal receipt records 9508 rows and native bbs hits 5615/4477. Its runner, spec and train-log identities match the executed receipt. The actual artifacts are under formal_draft/complete/evidence_visible; the shorter complete/evidence_visible path in the request was resolved without modifying anything.

All 23 new Python files parse with Python 3.7 grammar. This is syntax evidence, not a dependency/runtime compatibility test. Fifteen inherited helpers match the completed control byte-for-byte. Fit/preflight specs differ only by root; both copies of native_final_quality.py are identical. The supplied model previews, native loss and evaluator match the executed source-port/import identities.

Outstanding evidence is the actual new real-batch preflight and then completed training/native evaluation. Weight 1.0 is an explicitly fixed experimental budget, not a demonstrated optimum. No accuracy improvement, semantic identity, novelty, Nr3D or Sr3D conclusion follows from this review.

SOURCE_REVIEW.json contains the complete list of inspected files and their SHA256 identities. The administrative SOURCE_REVIEW_CALL.json and active_continuation_state.json, plus the future OBSERVER_STAGE.json, are excluded from the launch identity list because they change with task status.

## Inspected file identities

| File | SHA256 |
|---|---|
| C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\EXPERIMENT_PLAN.md | d6fd25599560119221493225633062d51bc0dd1d14bb53628dc5e2daf6d217be |
| C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\native_final_quality.py | b101f5b1cb4f9ca6eacb8a6f6edfa8630e3dd5361e5af47b0f88f4f9387a3bb4 |
| C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\prepare_quality_draft.py | d00f043afddabcada265dbb1e69af6b58a7409055783de325fcecd48c9f26003 |
| C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\quality_fit_spec.json | 7d84b843e16ccb140ac4f3a4f1b9839606110bdf297f7a8a7148b11a61273ec0 |
| C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\quality_preflight_spec.json | 24e62ee89f8fed345175d7cea29620bb759cd3e18a19567e596933228575027b |
| C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\IMPLEMENTATION_STATUS.json | 24eff7ed6a523b593b80eb9a626418d94aa119f51babbc0d9e859e16bb8d09b6 |
| C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\controller.py | 1d92f2b92514316a3b7cb5fecbab047833786e8c93e59e6fddad677841d1aada |
| C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\launch_quality_preflight_authorized.py | 7897057b970284712faa1920f5cb6a33c344cdbf35da72653310991a2e0536df |
| C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\runtime_bundle\install_boundary_evidence_readback.py | 531e7dbaa365d14deb870f9f205b2fb8b4654c8ea8356c06d844950aeca10fad |
| C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\runtime_bundle\native_final_quality.py | b101f5b1cb4f9ca6eacb8a6f6edfa8630e3dd5361e5af47b0f88f4f9387a3bb4 |
| C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\runtime_bundle\native_root_bbs.py | b0357bd54e8753672ee31d3056e50c1d53a854d7f0c71cd01a7bc8cecf422c8b |
| C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\runtime_bundle\pvground_boundary_box_refiner.py | 88debfef64f4295f87d697b7b9d4cd1eab17b39f4041f6d9ae773fc65dda9461 |
| C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\runtime_bundle\pvground_boundary_evidence_readback.py | ef3460373bc02533bbcff5e8e9e4b62e45836f601cc3bfd5e7d2304ec310aa4a |
| C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\runtime_bundle\pvground_candidate_box_refiner.py | e971346230d0e0547139823c106e48f970c58a0b0ef26ef46b75beaa672f6212 |
| C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\runtime_bundle\pvground_observation_query.py | cc92906f9b6f388c8da702f555a099faf07d619fefa2772bfb229693ebae7742 |
| C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\runtime_bundle\pvground_semantic_assignment.py | 3bf9c5f50095e7e447af91e7f61a34c930bf04ec8590f7e4da54134b859cf773 |
| C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\runtime_bundle\pvground_source_query.py | e0fcc6a30402ebeba7a9ffa1be907515350b64e408602a8b47b702e83a5d05ab |
| C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\runtime_bundle\pvground_tail_support_box_refiner.py | 665e94c150492a9fc3d52ddca7da2b7dca7d7ddbd981e759640f15846bb3f757 |
| C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\runtime_bundle\pvground_task_observation_query.py | 39349bf1fe8f3f51149f0872911020ecd0ede98fc9a8278cecb5ae466ed8643d |
| C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\runtime_bundle\pvground_whole_mask_box_refiner.py | 4a701338bcbf042a4eb7793209999434d68c269e563ac09008e758228fc72118 |
| C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\runtime_bundle\readback_model_factory.py | c235461179969fdd0f50d503a831512f27c083d9866479ab1777857a507c3763 |
| C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\runtime_bundle\readback_preflight_checks.py | 1a0be39f8dc0b1b10c0ace1a0e5d560280d4ac9e2dc2c5694a4b5cff571094d3 |
| C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\runtime_bundle\run_final_quality_fit.py | 47457bc0a95161287e633aa423274de4f701a7cc482bfc95f9a890d9207a5379 |
| C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\runtime_bundle\whole_mask_range.py | 22c4adb0cb8dfb929825dcbfcb5f9696cea879da212cebeea1a1a5acb6347a4c |
| C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\runtime_bundle\whole_model_preflight_checks.py | 115aa821de0fbc27d73aff9689354ad3b1f18796fdc23aab09b77d52b7666094 |
| C:\Users\gb\.codex\tmp\pvground_geometry_readback_20261004\formal_draft\evidence_visible_fit_spec.json | 65d373eccce4d503514620303c06a4eee3b9d08b3b2e5a62b096c6ec17e1b57f |
| C:\Users\gb\.codex\tmp\pvground_geometry_readback_20261004\formal_draft\runtime_bundle\run_readback_fit.py | 773213e3035688df304d4dd88d63b8b27bbf636fec8af2f49be47f3339564a08 |
| C:\Users\gb\.codex\tmp\pvground_geometry_readback_20261004\formal_draft\complete\evidence_visible\train.jsonl | 42fbfea82eb4708f98cbf29c8f3811464124c4402a9038ab1dafa14bffb698ad |
| C:\Users\gb\.codex\tmp\pvground_geometry_readback_20261004\formal_draft\complete\evidence_visible\receipt.json | 38362d09ddbb817c029ff627a0513837488d08b22dbeb5f1c4406f6219a82127 |
| C:\Users\gb\.codex\tmp\pvground_geometry_readback_20261004\formal_draft\complete\evidence_visible\formal\receipt.json | eaa763a5ecede315d7117176005122c81b20651ab814af4ca1c8f031a7ac47a3 |
| C:\Users\gb\.codex\tmp\pvground_geometry_readback_20261004\formal_draft\complete\evidence_visible\imports.json | 9cc629e89199b5cc22c128b402b0b197c4fa7bae81069e65550e8246ca2be595 |
| C:\Users\gb\.codex\tmp\pvground_geometry_readback_20261004\formal_draft\complete\evidence_visible\spec.json | 65d373eccce4d503514620303c06a4eee3b9d08b3b2e5a62b096c6ec17e1b57f |
| C:\Users\gb\.codex\tmp\pvground_geometry_readback_20261004\formal_draft\runtime_bundle\install_boundary_evidence_readback.py | 531e7dbaa365d14deb870f9f205b2fb8b4654c8ea8356c06d844950aeca10fad |
| C:\Users\gb\.codex\tmp\pvground_geometry_readback_20261004\formal_draft\runtime_bundle\native_root_bbs.py | b0357bd54e8753672ee31d3056e50c1d53a854d7f0c71cd01a7bc8cecf422c8b |
| C:\Users\gb\.codex\tmp\pvground_geometry_readback_20261004\formal_draft\runtime_bundle\pvground_boundary_box_refiner.py | 88debfef64f4295f87d697b7b9d4cd1eab17b39f4041f6d9ae773fc65dda9461 |
| C:\Users\gb\.codex\tmp\pvground_geometry_readback_20261004\formal_draft\runtime_bundle\pvground_boundary_evidence_readback.py | ef3460373bc02533bbcff5e8e9e4b62e45836f601cc3bfd5e7d2304ec310aa4a |
| C:\Users\gb\.codex\tmp\pvground_geometry_readback_20261004\formal_draft\runtime_bundle\pvground_candidate_box_refiner.py | e971346230d0e0547139823c106e48f970c58a0b0ef26ef46b75beaa672f6212 |
| C:\Users\gb\.codex\tmp\pvground_geometry_readback_20261004\formal_draft\runtime_bundle\pvground_observation_query.py | cc92906f9b6f388c8da702f555a099faf07d619fefa2772bfb229693ebae7742 |
| C:\Users\gb\.codex\tmp\pvground_geometry_readback_20261004\formal_draft\runtime_bundle\pvground_semantic_assignment.py | 3bf9c5f50095e7e447af91e7f61a34c930bf04ec8590f7e4da54134b859cf773 |
| C:\Users\gb\.codex\tmp\pvground_geometry_readback_20261004\formal_draft\runtime_bundle\pvground_source_query.py | e0fcc6a30402ebeba7a9ffa1be907515350b64e408602a8b47b702e83a5d05ab |
| C:\Users\gb\.codex\tmp\pvground_geometry_readback_20261004\formal_draft\runtime_bundle\pvground_tail_support_box_refiner.py | 665e94c150492a9fc3d52ddca7da2b7dca7d7ddbd981e759640f15846bb3f757 |
| C:\Users\gb\.codex\tmp\pvground_geometry_readback_20261004\formal_draft\runtime_bundle\pvground_task_observation_query.py | 39349bf1fe8f3f51149f0872911020ecd0ede98fc9a8278cecb5ae466ed8643d |
| C:\Users\gb\.codex\tmp\pvground_geometry_readback_20261004\formal_draft\runtime_bundle\pvground_whole_mask_box_refiner.py | 4a701338bcbf042a4eb7793209999434d68c269e563ac09008e758228fc72118 |
| C:\Users\gb\.codex\tmp\pvground_geometry_readback_20261004\formal_draft\runtime_bundle\readback_model_factory.py | c235461179969fdd0f50d503a831512f27c083d9866479ab1777857a507c3763 |
| C:\Users\gb\.codex\tmp\pvground_geometry_readback_20261004\formal_draft\runtime_bundle\readback_preflight_checks.py | 1a0be39f8dc0b1b10c0ace1a0e5d560280d4ac9e2dc2c5694a4b5cff571094d3 |
| C:\Users\gb\.codex\tmp\pvground_geometry_readback_20261004\formal_draft\runtime_bundle\whole_mask_range.py | 22c4adb0cb8dfb929825dcbfcb5f9696cea879da212cebeea1a1a5acb6347a4c |
| C:\Users\gb\.codex\tmp\pvground_geometry_readback_20261004\formal_draft\runtime_bundle\whole_model_preflight_checks.py | 115aa821de0fbc27d73aff9689354ad3b1f18796fdc23aab09b77d52b7666094 |
| C:\Users\gb\.codex\tmp\pvground_geometry_readback_20261004\revision2\source_preview\PV-Ground\models\pv_ground.py | 8d2aad665ce7d5e19f7423309ef0ba05ba406f302159c6460a36323cbae1d1c9 |
| C:\Users\gb\.codex\tmp\pvground_geometry_readback_20261004\revision2\source_preview\PV-Ground\models\modules.py | 1f262505e4c9d6fd186609c6da1497bf5cf6f62f982fa7746c7d4f5400d8f2df |
| C:\Users\gb\.codex\tmp\pvground_geometry_readback_20261004\revision2\complete_preflight\source_port.json | 99eb0c92822eec9c28543a7324d761c1322573ecc88f01e175902a12c8f16304 |
| C:\Users\gb\.codex\tmp\pvground_geometry_readback_20261004\revision2\readback_remote_source_receipt.json | 4363dc05b81626767ead5dd9347aa99a57d58ca7d882757304df595f78d43ddb |
| C:\Users\gb\.codex\tmp\pvground_runtime_bundle_20260908_v1\PV-Ground\src\grounding_evaluator.py | 39c8de921aff9fea6e2a9a39f68e5634bdd765b1bfd065f0ee3c6c9dac931677 |
| C:\Users\gb\.codex\tmp\pvground_runtime_bundle_20260908_v1\PV-Ground\models\losses.py | 920051cc7d1009493829eebbb6185da834a4efd037e9d73ad43ce918a81031de |
| C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\launch_quality_fit_authorized.py | 946737a13ffef02cc2df2f209e1941e4c7ac544b333f8d9dfaee85fd8f013cb5 |
| C:\Users\gb\.codex\tmp\pvground_final_quality_20261005\observe_quality_authorized.py | 67d6cdadbd21d999950c55f8025b6bf05242ad17d5313d16eb9494b36214f146 |
