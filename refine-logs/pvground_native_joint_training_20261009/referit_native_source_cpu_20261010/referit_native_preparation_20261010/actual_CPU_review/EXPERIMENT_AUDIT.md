# Experiment Audit Report — actual synthetic CPU check

Date: 2026-10-10 (Asia/Shanghai)  
Project: Nr3D/Sr3D native target preparation  
Overall verdict: **WARN**. Bounded CPU assertion/receipt verdict: **PASS**. **0 blocking issues; 1 nonblocking reporting issue.**  
Reviewer: fresh Codex agent `/root/pvg_referit_cpu_targets_actual_20261010`; actual model and reasoning effort **UNATTESTED**. Requested route: `gpt-6-astra`, effort `max`. `review_independence: same-family`; `acceptance_status: provisional`.

The closed artifacts support one successful synthetic CPU target and local-gradient check. They do not support native REC accuracy, a full criterion/model run, author-weight initialization, epochs, or scientific effectiveness. This review grants no GPU or native-training admission.

Paths below are relative to `C:/Users/gb/.codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/referit_native_preparation_20261010`; `../source/models/losses.py` is the protected parent criterion. The root resolves to `D:/Program Files/UserCache/gb/codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/referit_native_preparation_20261010`. The four D-drive intake paths resolve to the same observed files.

## A. Target/reference provenance — PASS within synthetic scope

The targets are hand-constructed, not dataset annotations. `check_native_targets_cpu.py:36–78` creates random logits/mask leaves, fixed centers/sizes, two valid target boxes, token maps of mass 0.5, eight-point masks, and manual match indices. The name `language_dataset` selects a formula; it does not show that ScanRefer, Nr3D, or Sr3D data was loaded. The detection row is also a hand-labeled fixture.

G reads root box/token labels from those synthetic batch tensors (`source/pvground_semantic_assignment.py:9–42`). Its qualifier uses detached candidate/root geometry; prediction dependence selects a slot and does not manufacture a ground-truth target. C takes the first synthetic target mask and pools it by the supplied four superpoint groups (`source/selected_query_mask_objective.py:30–37`). C's choice of query uses detached synthetic logits and the native score helper (`source/native_root_bbs.py:4–9`).

The numerical “expected” CE and gradient are a recomputation using the same synthetic logits, target maps, and hand-written indices (`source/pvground_semantic_assignment.py:66–110`), compared with a direct native `loss_pos_align` call (`check_native_targets_cpu.py:75–77`). This is a legitimate algebra/implementation consistency oracle. It is not an independent scientific reference or an official benchmark evaluation. Explicit synthetic labels in `check_native_targets_cpu.py:1,116–123`, the receipt, and `research_contract.md:7–11` prevent a fake-ground-truth accuracy claim here.

## B. Normalization and metric meaning — PASS

G uses the actual fixture match count: 4 in the initial two-row comparison and 6 in each three-row panel. The native CE call receives `2 * count`; G's denominator is the sum of match lengths (`check_native_targets_cpu.py:65–76`; `source/pvground_semantic_assignment.py:52–60,82–95`; `../source/models/losses.py:458–515`). Its external coefficients are 0.5/7 for ScanRefer and 1/7 for Nr3D/Sr3D, consistent with the parent full-loss formula at `:945–958`. Supplying 6 as a scalar does not execute six decoder layers.

The target mass is deliberately unnormalized: 0.55 for the fixture's ScanRefer/Nr3D root and 0.5 for Sr3D. The checked vectors at `check_native_targets_cpu.py:110–111` match the parent weighting at `../source/models/losses.py:483–487`; this is not rescaling a performance metric.

C combines focal/Dice losses with coefficients 5, 1, 10, 2; each per-query loss uses `num_boxes=1`, then the sum is divided by the actual batch size (`source/selected_query_mask_objective.py:36–40`). Focal averages across mask positions; Dice uses its usual soft prediction-plus-target overlap denominator (`../source/models/losses.py:393–437`). Softmax, IoU's union, and Dice's denominator contain predictions as part of their definitions. None is a post-hoc normalization of a reported accuracy by the model's own maximum, minimum, or mean.

Reported CE corrections are signed loss deltas. The near-zero CE/gradient errors measure numerical agreement, not high accuracy. All printed values are raw outputs of the stated loss/error calculations.

## C. Result existence, exits, provenance, and chronology — PASS

All 26 primary requested files (including the request and parent criterion) exist and were read. Local stdlib verification established:

- Transport, remote-child, and copied execution exit codes are all 0 (`cpu_execution/CPU_TRANSPORT_EXIT.json:1`, `CPU_REMOTE_RAW_STDOUT.json:1`, `CPU_EXECUTION.json:4`).
- Decoded raw stdout equals the 2,082-byte `CPU_STDOUT.json` exactly. Decoded stderr equals the empty child stderr; raw transport stderr is also empty.
- `CPU_PARSED_RESULT.json` is structurally equal to the JSON parsed from that same stdout. It is a reencoding, not another run.
- Every non-base64 field in the raw receipt equals the corresponding execution-copy field. The recorded R2 source-audit hash equals the observed R2 report hash.
- 75 of the R2 report's 76 recorded inputs match directly. The sole changed file is the live tracker. `source_review/R2_CLOSED_TRACKER_INPUT.md` has the exact R2-expected hash `03bca0f2ce19ac70a59c5a38449dcf322b24ee0d46895f51181b00dfccb7343a`; `R2_SEALED_TRACKER_SNAPSHOT.json:2–5` identifies that alias. The current tracker has hash `f662c38d6eb125544c38ae54e7b2ff855f52a1d1a854d5b619f0ae183fcf28fb`.
- All 14 prepared-source hashes and all 14 protected parent-source hashes match their manifests. Both fixed R1 report hashes match R2's recorded values. The fixed preparation JSON is byte-identical to the R2 preparation JSON.

The R2 review was created at 06:40:41 +08:00; the child ran 06:41:25–06:41:35; intake was recorded at 06:43:49. R2's `NOT_RUN` tracker and pre-execution plan are historical source-review inputs. The current tracker says `EXECUTED_OUTCOME_AUDIT_PENDING` for CPU and keeps real loader/GPU/epochs unrun (`refine-logs/EXPERIMENT_TRACKER.md:5–10`). This chronology does not constitute a phantom result or unexplained source change. Do not modify R2's archived expected tracker hash to match later status.

The launcher binds the reviewed runner/checker through R2 input hashes, guards the original remote source hashes, checks payload bytes before writing, disables CUDA visibility, and preserves raw receipts before result assertions (`run_native_targets_cpu_authorized.py:13–17,21–56,68–81`). The checker asserts the imported criterion path (`check_native_targets_cpu.py:12–15`); the receipt names `/root/autodl-tmp/pvground_native_joint_training_20261010/PV-Ground/models/losses.py`. The current local parent criterion hash is `00b8627364358f35de936d7efa74ff10cb53de4664c01fc6f7de7a93413f70d6`.

These are consistent local source/receipt records, not an independently observed remote-host attestation. The receipt's zero GPU/training-query counters are literals written by the launcher, not runtime instrumentation. The reviewed executed script contains no full-model construction, checkpoint load, optimizer step, training-state query, or explicit CUDA tensor operation. This reviewer made no remote call.

## D. Executed versus defined/transported code — WARN on coverage

The successful terminal print follows all assertions in `check_native_targets_cpu.py:81–123`. The executed path constructs `SetCriterion(matcher=None, losses={}, eos_coef=.1)`, then calls only `loss_pos_align` directly. It exercises old/new G and C in the initial ScanRefer panel, new G verification and new C in each of the three language panels, native focal/Dice functions, the native root-score helper, and local `autograd.grad` operations.

It does **not** call `SetCriterion.forward`, `HungarianMatcher.forward`, `compute_hungarian_loss`, `SetCriterion.loss_masks`, box/contrastive losses, the PV factory, a model forward, an optimizer, or a loader. In the parent source those full paths are separate at `:267–391,562–778,808–846,849–988`. The original G verification function is defined but not invoked. That is a coverage limit, not evidence of a fabricated executed metric.

The launcher transports 16 Python files from `source/` and `original/`, plus the checker. Transport does not execute their contents. In particular, `models.losses` is imported from the native source before the prepared source directory is inserted (`check_native_targets_cpu.py:12–16`), so the prepared `source/models/losses.py` integration is not the imported criterion. Its only diff against the parent is the extra `num_decoder_layers` argument at line 962; this audit checked that diff and hashes statically, not at runtime.

## E. Scope and protection assertions — WARN; narrowly supported

One recorded CPU process uses one seed, 2027. It makes one initial two-row ScanRefer compatibility panel, then one three-row panel for each of the three language labels: 11 artificial rows in total, with only the latter nine represented by the three result entries. Each three-row panel has two expression rows, one detection row, two targets per row, five candidate queries, eight tokens, eight points, and four superpoint groups. There are zero real dataset rows, zero real scenes, and zero training runs.

For each language panel, query 2 on the two expression rows is G-qualified; detection rows and all manually matched logits have zero correction gradients. The G witness checks CE reconstruction, corrected CE/gradient agreement, zero changes outside selected slots, positive qualified null-token gradient, and no gradient to box center/size. Evidence: `check_native_targets_cpu.py:91–100`; `source/pvground_semantic_assignment.py:87–110`.

C's forced highest-score query is 4. It is unmatched on row 0, matched to the other target on row 1, and ignored on the detection row. The script checks these three roles, one added row, nonzero own-mask gradient only on row 0/query 4, no own-mask graph for rows 1/2, and nonzero row-0 text-mask and alpha gradients (`check_native_targets_cpu.py:101–109`). The selected-`matched_root` branch is not exercised, nor are real annotation root/anchor mapping, variable padding, real tokenization, model-parameter gradients, optimizer updates, or full native loss integration. No new speculative tests or implementation changes are required to accept this bounded receipt.

**W1 — nonblocking reporting ambiguity.** `CPU_PARSED_RESULT.json:12` reports `ScanRefer_original_loss_and_gradient_values_identical=true`. The initial panel compares G loss and G logit gradients exactly (`check_native_targets_cpu.py:82–87`), but compares only C scalar loss and `extra_rows` (`:88–90`). It does not compare old/new C gradients. Interpret the flag as **G loss/gradient equality and C loss equality for the two-row fixture**. The minimum correction is this wording qualification in downstream reporting; retain the closed raw result. No C gradient-equality claim is supported by that flag.

The exact recorded results are:

| Language label | CE correction | Extra mask loss | Native CE reconstruction error | Corrected CE error | Corrected gradient max error |
|---|---:|---:|---:|---:|---:|
| scanrefer | -0.004633262287825346 | 1.0204280614852905 | 0.0 | 0.0 | 5.820766091346741e-11 |
| nr3d | -0.006834529805928469 | 1.3277350664138794 | 0.0 | 0.0 | 1.0186340659856796e-10 |
| sr3d | -0.013621876947581768 | 1.9363161325454712 | 0.0 | 5.960464477539063e-08 | 1.709850039333105e-10 |

Source: `cpu_execution/CPU_PARSED_RESULT.json:14–61`, verified against decoded raw stdout. Each entry reports one extra mask row. None of these numbers is REC, accuracy, a generalization estimate, or a before/after training result.

## F. Evaluation classification — PASS with precise label

**`synthetic_target_and_gradient_engineering_not_accuracy`**: hand-constructed target/selection fixtures and local numerical/gradient consistency, including a native CE method witness. The closest broad checklist category is `synthetic_proxy`, but its usual “model-generated reference” wording would be inaccurate here: no neural model generated these target labels. It is not `real_gt`, human evaluation, or benchmark simulation accuracy.

## Claim impact and admission

- Supported: the recorded child completed the specified synthetic assertions; native direct CE consistency and the stated fixture protection/gradient paths passed.
- Needs qualifier: ScanRefer old/new compatibility is G loss plus G gradient and C scalar loss on two synthetic rows. “All matches protected” refers to the fixture and G correction; it is not comprehensive C branch or model behavior coverage.
- Unsupported by this package: C old/new gradient equality, real-data root/mask correctness, full criterion/Hungarian/factory integration, author checkpoint initialization, native epochs, REC accuracy, or superiority over baselines.
- Research-contract C1/C2 effectiveness remains unestablished (`idea-stage/docs/research_contract.md:5–9`). Future real-data/model/restore/evaluation work already listed in the plan remains necessary before scientific claims. This review does not authorize, run, or recommend bypassing that staged work.

Scientific-result acceptance: **not granted**. GPU/native-training admission: **not granted**. Full goal complete: **false**. The zero blocking count applies only to acceptance of the closed synthetic CPU outcome.

## Review operations, continuity, and sealing

This reviewer read the listed closed sources/receipts directly, read the required skill/policy references, parsed nine Python sources plus the embedded launcher block with stdlib AST, compared source text, and checked hashes/base64/JSON equality with stdlib only. No implementation file or prior report was edited. No neural/GPU/SSH/model/checkpoint operation, current-training query, or repeated experiment execution occurred.

A first attempt to invoke the PATH `python` for read-only stdlib checks failed with `No pyvenv.cfg file`. `uv python find --managed-python --offline` located the existing managed Python; the stdlib check then succeeded. This local reviewer-tool failure is preserved in the trace and is unrelated to the closed CPU child's successful exit.

No SOUL, USER, daily memory, MEMORY, AUTH, credential, askpass, or known-host file contents were read or copied in this reviewer session. Prior R1's bootstrap limitation appears in the supplied R2 report; no such contents were accessed or used as evidence here. The skill and its shared policy references were additional instruction reads, not experiment evidence. Requested Astra/max routing is recorded separately from the unattested actual model/effort.

`EXPERIMENT_AUDIT.json` contains observed input hashes, checks, scope, claims, and attribution. Full task/request, substantive reviewer response, raw local verification output, source-diff/AST output, and route metadata are stored under `.aris/traces/experiment-audit/referit_native_CPU_20261010/`. `OUTPUT_SHA256.json` seals the new review/trace artifacts and existing executor-request metadata by SHA-256; it is a byte-integrity manifest, not reviewer-identity or remote-execution attestation.

