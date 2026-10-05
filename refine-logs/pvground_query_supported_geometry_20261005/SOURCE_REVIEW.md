# Query-supported geometry source review

**PASS.** No remaining blocking source defects were found in the revised delta. Scope: **SOURCE_ONLY**. Fresh delegated Codex review; requested `gpt-6-astra`, `max`, `fork_turns=none`; actual backend not independently attested. Same-family review, provisional acceptance.

The target indices, qualification and loss budget are consistent with the plan. Native matcher call 1 is the final layer; all matched queries are excluded. Native targets and DFL both filter valid GT slots, and the checked first valid slot is root0. Qualification requires candidate Query and native fused Mask IoU >0.5 together with final Box IoU <=0.5. It uses point memberships and detached training GT, and is absent from evaluation/model inputs.

For each expression, the added term is `(10 * native L1 + 2 * native GIoU + face-mean DFL) / 7`, with candidate means followed by an actual-batch mean. Empty rows contribute zero. Existing native/G supervision, matched DFL/7, target range, knots and clipping are unchanged. Only the existing 456102-parameter geometry head is trained; all parents and fresh zero R are frozen/eval and included in frozen-state checks.

The revised preflight closes the review findings:

- It replays the updated geometry head and zero R on the same cached upstream tensors and checks exact native bbs/Mask preservation. The one diagnostic semantic replay is recorded separately; ordinary complete forwards still execute the native head once.
- The selected real batch1 has per-row qualification counts `[28, 0, 0, 1, 0, 14, 100, 0]` and 126 outside faces. Both real steps require empty and nonempty rows plus clipped targets. The existing DFL endpoints are used unchanged.
- Isolated extra gradients must be zero at every nonqualified/matched output and nonzero at the head. Parent/R states, in-memory head/optimizer restore, and formal strict full-model/optimizer restore are checked.
- Full fit now checks completed two-arm preflight and both passing receipts before launching either arm. Load metadata now reflects head-only training and frozen parents/R.

Local source checks passed: Python 3.7 AST for all seven new Python sources (the intended fit-body fragment wrapped in its function); exact generated-runner identity and embedded fit body; specs differ only in root and extra-loss weight; all 17 reused helper identities match; the readback source port and relevant native files match. Batch arithmetic is 3722 full B8 batches plus B2, hence 29778 rows and 3723 updates.

The review performed source reads and standard-library checks only: no model import/construction, PyTorch import, GPU work, optimizer update, weight output, remote call, experiment execution, or independent historical NPZ recount. The probe summary/panel are supplied evidence. Actual branch audit and both GPU preflights remain execution gates; this source PASS establishes no accuracy result.

Exact reviewed file identities and their review extent are in the companion JSON. No speculative fallback, extra compatibility layer, hashing scheme, or unrelated refactor is requested.

Serializer refresh 2026-10-05T09:01:21.011013+08:00: the head checkpoint now includes `boundary_mode=distribution`, `support_arm=whole_range`, `use_whole_range=True`, and `boundary_loss_weight=1/7`, matching the existing geometry factory. All ten head tensors are complete replacement values, so a promoted terminal loads through that factory directly. Parent/cumulative geometry budgets are recorded as 3723 and `3723 + step_number` (7446 at terminal). Removing exactly the two added lines reproduces the previously reviewed fit-body and runner hashes. Both parse as Python 3.7; only fit_body.py, run_geometry_fit.py and GENERATION.json identities changed. The source-only provisional PASS remains unchanged.
