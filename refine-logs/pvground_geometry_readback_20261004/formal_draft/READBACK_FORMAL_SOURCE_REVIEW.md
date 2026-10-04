# Readback formal source review

**PASS — SOURCE_ONLY.** No blocking or non-blocking defects found in the exact reviewed draft.

Review independence: **same-family**. Acceptance: **provisional**. Requested reviewer routing: **gpt-6-astra / max**; the actual backend/model and effort are **unattested**. This delegated reviewer read the supplied source artifacts directly.

Reviewed at: 2026-10-04T23:10:54.847692+08:00. The companion JSON contains the exact absolute paths and SHA256 identities of all 42 reviewed source/reference files. It binds neither this review nor private tracing artifacts.

## Findings

Blocking findings: **none**. Non-blocking findings: **none**.

## Verified source behavior

- **sample and update arithmetic: PASS.** 29778 = 3722*8 + 2; one shuffled DataLoader pass, drop_last=False, 3723 optimizer.step calls, tail-size assertion and exact Counter(fit row IDs). AdamW lr=1e-5, weight_decay=.0005, clipping=.1; both seeds 2027.
- **same capacity control and no leakage: PASS.** Both 96672-parameter / 23-state arms use the same frozen officialPV+G+4506 parents, 256 queries, full native text, original semantic Query and six face embeddings. Only the 44-dimensional evidence tensor is zeroed before the evidence encoder. GT is added after the model forward for the unchanged native+G training objective; no GT inference gate/teacher/quality loss/contrastive-positive expansion.
- **single deployed head order: PASS.** Actual pv_ground.py/modules.py defer only the final semantic subhead. Native Mask generation precedes geometry refiner, then R, then one native final semantic subhead call. observed_readback_forward verifies actual hooks and clones protected outputs within the same complete forward. The separate cached-preR head replay is evaluation-only and labelled fixed-frame diagnostic; no second backbone forward or deployed dual rank.
- **original native plus G objective: PASS.** Pinned native losses.py calls matchers in proposal_,last_,0head_...4head_ order, so matching[1] is final Hungarian. Unchanged G excludes all matched queries, selects detached root-IoU>.5 unmatched slots, and replaces their eos-weighted native soft-token CE with ScanRefer .6/.2/.2/.1 target mass using the original .5/7 scaling. Geometry and native contrastive/Mask tensors remain frozen.
- **split and row identity: PASS.** The pinned scene-based protocol is reconstructed before use: 29778 fit and 6887 holdout, disjoint physical scenes; only fit IDs enter the update loop. Initial/terminal 6887 is correctly described as pretrained-seen. Formal 9508 is a separate val dataset/process with ordered row IDs and no updates. Stored row/point identities support actual later pairing without claiming bitwise complete-forward equality.
- **native evaluator and masks: PASS.** The runner explicitly imports runtime/PV-Ground/src/grounding_evaluator.py. Its native reference has SHA39c8de921aff9fea6e2a9a39f68e5634bdd765b1bfd065f0ee3c6c9dac931677. BBS uses softmax then binary positive-map sum + raw modify/pron/relation-map sums - other-entity sum.256 candidates are ranked; Box uses selected final geometry and Mask uses the same selected Query, scalar-alpha logit fusion, sigmoid>.5 and native superpoint mapping. Selected Box hits and Mask IoU sum are checked against actual native counters.
- **freezing and strict terminal restore: PASS.** Strict official load, strict 1072-state G merge, strict 10-state 4506 geometry merge and only 23 R parameter tensors enabled. Parent model stays eval; all persistent parent states are compared to initial. Terminal contains R state delta, optimizer and parent/spec identities plus exact fit IDs. A fresh process reconstructs parents and strictly restores complete model plus exact AdamW moments/steps/groups at 3723.
- **weight retention: PASS.** Controller waits for successful train+fresh formal receipts, verifies terminal identity, recounts all 9508 stored selected boxes on CPU and verifies thresholds before retirement. Strictly higher BBS@.5 improves best. Only verified terminal.pth files inside the two owned arm directories can be deleted; official/G/4506 parent chain is always preserved, including when R wins. All text survives; no local weight archive is collected.
- **launch and observation gates: PASS.** Launch requires this source review and unchanged reviewed bytes, closed successful corrected V2 controller, both actual B8/two-update receipts with exact frozen/optimizer checks and identical reused helper files. V2 controller runs CPU construction then GPU preflight for each arm. Launch uses existing GPU lock and parent identities; observer waits near the estimated milestone then uses240-second intervals. Collector copies text/results only and does not replay inference or optimization.

## Local verification and limits

- Python 3.7 AST parsing passed for 26 formal/model Python files and 3 embedded remote probe strings. All16 runtime payload identities match the source check/specs; the 15 reused helper bodies are byte-identical to corrected V2. Both actual runner copies and both parent/revision2 model copies are byte-identical. The two arm specs differ only in output/preflight paths and evidence visibility.
- Only the reviewed pure `controller.box_iou` function was executed with Python stdlib against the existing geometry-parent 9508 rows: 5616/4506, zero threshold changes. This verifies the recount calculation on actual stored parent boxes; it is not an R result. The full row artifact identity is recorded in JSON.
- No Torch/model construction, GPU run, SSH/network action, formal launch, optimizer update or weight mutation was performed by this review. No source file was edited.
- Actual corrected V2 success was not observed by this reviewer. Formal launch remains conditional on both real CPU/two-update GPU preflights and the closed successful V2 controller. The launcher source enforces that separate gate.
- The5615/4754 target remains unmet; protected geometry remains 5616/4506, strict gap 248. This PASS makes no accuracy, zero-shot, causal geometry-quality, three-module novelty or Nr/Sr effectiveness claim.
