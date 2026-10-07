# Fresh result-to-claim review

- verdict: **NO**; integrity_status: **warn**
- execution_scope: `TERMINAL_SOURCE_AND_ACTUAL_ARTIFACTS`
- fresh_context: `true`; executor narrative and historical SOURCE verdicts used as proof: `false`
- reviewer: `/root/pvg_reference_keep_terminal_audit`; same-family / provisional; `backend_identity_attested=false`
- model attribution: requested gpt-6-astra / max routing, without independent backend attestation
- read-only inputs; network/SSH, model execution, optimizer updates, package installs, deletion and input edits: **0**
- all relative evidence paths below resolve against `C:/Users/gb/.codex/tmp/pvground_reference_keep_20261006`

**claim_supported: no.** Confidence is high for the observed negative contrast and unmet numerical gate, not for a universal claim that the mechanism can never work.

**结论：本轮没有证明参考退化惩罚能减少正式评估中的修坏或增加原生定位命中；没有证明学习式六面细化增益。** 同模型 5620/4764 与三项有效贡献两项准入条件均未建立，因此所声明的后续 Nr/Sr 训练前提尚未满足。

## Claim under review

单一待验证主张：固定预测融合Mask空间参考和六面分布结构，参考相对几何退化惩罚在同起点/同预算/seed2027下减少修坏、增加原生实际定位。与本轮控制及5598/4848起点均比较；只超过差控制或历史4832不足以证明学习式增益。保持完整256候选、sameQuery Box/Mask、唯一bbs，无教师/新排序/GT推理。最新最终准入5620/4764，并需三个经直接消融验证的有效模块后才从对应作者权重独立训练Nr/Sr。本项新增推理参数0，是待验证训练机制。

Source: `research_contract.md:1`.

## Decisive comparison

| Comparison | Hits @0.25 / @0.5 | Interpretation |
|---|---:|---|
| Own initial state, each arm | 5598 / 4848 | Zero current updates; historical hidden/G training remains |
| Control λ=0, after 3723 updates | 5593 / 4832 | −5 / −16 against own initial |
| Keep λ=1, after 3723 updates | 5593 / 4832 | −5 / −16 against own initial; no gain over control |
| Same-forward original prior | 5615 / 4495 | Reference changes the two thresholds by −17 / +353 |
| Required same-model gate | 5620 / 4764 | Best reference misses wide threshold by 22 |

Both terminals use the same selected Query and have exactly the same hit labels on all 9,508 expressions. Each loses more cases than it repairs: 4 repairs / 9 damages at 0.25, and 13 repairs / 29 damages at 0.5. Equality of aggregate accuracy is not hiding opposite per-expression effects. The 6,887-expression module holdout also declines, from 6149/5667 to 6147/5665 in each arm.

Evidence: `analysis/fresh_terminal_recount.json:331`, `analysis/fresh_terminal_recount.json:809`, and all four actual formal receipts at `complete_fit/control/initial_formal/receipt.json:9`, `complete_fit/control/formal/receipt.json:9`, `complete_fit/keep/initial_formal/receipt.json:9`, `complete_fit/keep/formal/receipt.json:9`.

## What results support

- In fixed seed2027, each arm completes one29778-expression pass/3723 updates from the same recorded step0 geometry state. Both native formal endpoints yield5593/4832 on9508 development expressions, versus own initial5598/4848.
- Keep versus control changes neither selected Query nor either threshold hit label on any of9508 expressions. Both lose5 wide and16 strict hits; each has4 repairs/9 damages at.25 and13 repairs/29 damages at.5.
- The training keep term is active, with positive isolated M0 gradient after the first update; aggregate logged keep loss is0.21893579928413387 versus control0.22696565735385565, and degradation occurrences938001 versus940960. These are trajectory/support-dependent training observations, not localization gains.
- The zero-update reference has5598/4848 versus its same-forward original prior5615/4495: -17 wide, +353 strict hits. It is a threshold tradeoff from prediction-derived coordinates with retained historical hidden/G states; it is not new learned refinement.
- The protected parent remains the metric-selected candidate under the declared joint-gate/strict/wide/tie rule; the new trained endpoints are inferior.

## What results do not support

- The contract claim that reference-relative keep reduces formal repairs-to-damages or improves actual native localization.
- A learned six-face refinement gain over the zero-update Mask-reference state; both learned endpoints are worse.
- Same-checkpoint5620/4764 admission: best5598/4848 misses the wide requirement by22; terminal5593/4832 misses it by27. Strict metric alone passes, joint admission does not.
- Three independently effective contributions: this pair isolates one training weight and obtains no performance benefit; it is not a three-module ablation. A frozen zero-output R branch is not an effective contribution.
- Cross-benchmark or cross-seed stability, unseen-scene generalization of the official-pretrained pipeline, complete scratch-training budget14892, or bit-identical full Mask/reference inference.

## Missing evidence and admission

- A same-model endpoint attaining both5620 and4764; metrics from different checkpoints cannot be combined.
- Direct controlled ablations establishing three effective contributions in the final fixed architecture.
- If a future learning improvement is claimed, an improvement over both the same-budget control and its own true zero-update initialization, with paired repair/damage reporting.
- Full raw Mask/logit replay and new local checkpoint reconstruction are outside this audit evidence scope; do not claim they were performed.

The 5620/4764 condition must hold at one checkpoint; two checkpoints cannot contribute separate passing columns. The retained reference is 5598/4848 (gap 22/0), while both trained endpoints are 5593/4832 (gap 27/0). Passing the strict threshold alone is insufficient. A positive zero-update reference tradeoff is not evidence that the current trainable refinement works. This pair isolates one loss weight and finds no localization benefit; it cannot establish three independently effective contributions. A frozen zero-output R branch likewise supplies no effectiveness evidence.

No current integrity blocker prevents recording the negative experiment. The positive claims have three blockers: C1 no gain over control or own initial; C2 joint same-model gate unmet; C3 three effective contributions not established.

## Suggested claim revision

在固定seed2027、同起点和3723步预算下，参考退化惩罚产生真实梯度并降低记录的训练期退化损失，但未改变ScanRefer开发集任一阈值命中集合。λ=0与λ=1终点均为5593/4832，低于零更新Mask参考5598/4848；本轮未证明学习式定位增益或第三项有效贡献。

## Next action

- No rerun is needed to record this negative result. Keep the protected reference as the measured starting comparator and do not promote keep as effective.
- For a later authorized mechanism experiment, predefine one fixed-seed2027 same-budget intervention and compare its terminal with both its own initial state and the paired control; retain complete native bbs/same-Query/256-candidate artifacts.
- Only after a single fixed model clears5620/4764 and direct ablations establish three effective contributions does the user-authorized independent author-initialized Sr3D/Nr3D stage meet its stated prerequisite. No multiseed experiment is requested or required.

## Scope

- One fixed seed2027; two lambda configurations; no multiseed requirement.
- 9508 shared development expressions across141 scenes; 6887 module-holdout expressions across106 official-training scenes.
- Eight hidden geometry tensors have recorded11169 prior updates, then3723; reset output tensors receive3723. Author and G pretraining history is additional.
- CPU reanalysis covers saved arrays/receipts and source-bound restoration records, not model execution, raw Mask reconstruction or fresh full-model CPU restoration.

Frozen state equality and same-forward geometry checks are not full-forward output identity: the saved native scores/prior/selected references are exact across the current runs, but a few Mask IoUs and non-selected references drift; three Queries differ against the older protected-parent output. Raw Mask/logit reconstruction, original dataset rebuilding and new local checkpoint deserialization were not performed. The closed inspection reuses the unchanged-parent prior CPU witness. These limitations narrow the audit scope; they do not provide evidence for a keep gain.

Evidence: `analysis/fresh_terminal_details.json:21`, `analysis/fresh_terminal_details.json:114`, `closed_weight_inspection.json:61`, `postrun/inspect_closed_weights.py:92`.

## Supporting file:line evidence

- `research_contract.md:1`
- `complete_fit/control/formal/receipt.json:9`
- `complete_fit/keep/formal/receipt.json:9`
- `complete_fit/control/initial_formal/receipt.json:9`
- `complete_fit/keep/initial_formal/receipt.json:9`
- `analysis/fresh_terminal_recount.json:331`
- `analysis/fresh_terminal_recount.json:809`
- `analysis/fresh_terminal_details.json:143`

## Actual file bindings

| Actual file | Bytes | SHA256 |
|---|---:|---|
| `analysis/SUMMARY.json` | 12524 | `69568ce12e94aec5971102c1e2583503cb0aae54782661220d43d6e99441ac82` |
| `closed_weight_inspection.json` | 3314 | `0a4a27bf86dd771099eb62e7d335d4588a59a3dc99490c84c728646974c79f0c` |

The JSON report binds 83 actual files by absolute resolved path, bytes and SHA256, including reviewed sources, raw receipts/rows and the intake manifest. `analysis/fresh_verified_manifest.json` records the independently verified full 4,823-file intake. All 83 report-bound files and 100 distinct file:line references were checked again when assembling these reports.

