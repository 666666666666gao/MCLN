# Range-head-only source review

**PASS — no blocking or nonblocking source defects found.** The implementation is correct for the proposed single `whole_range`, real batch-8, two-update preflight and for the subsequently supplied conditional formal-pair source. This is a **SOURCE_ONLY** verdict; the new runtime preflight has not passed yet.

Reviewer: fresh `/root/pvg_range_head_only_source_review`; requested/configured `gpt-6-astra`, reasoning effort `max`, `fork_turns: none`, as recorded in the native request trace. Backend SKU is not independently observable. `review_independence: same-family`; `acceptance_status: provisional`.

## Findings

- Blocking findings: none.
- Nonblocking findings: none.
- Required source fixes: none.

## What was checked directly

Read the plan, spec, preparation record, all 15 original Python implementation files, the review-request recorder, and the subsequently added observer/collector and formal-stage operational files (21 Python files total). The expansion includes both formal specs and `formal_preparation.json`. Read the prior runner, completed local/whole specs, imports, loads, training receipts, formal receipts and analysis report. Inspected the actual supplied native model, criterion/optimizer, dataset, processor and evaluator sources. Prior audit fields were read only to check the launcher's field names, not adopted as this review's verdict.

Executed deterministic, stdlib-only checks:

- All 21 current Python files and the four embedded remote probe strings parse with Python 3.7 syntax grammar. This is a syntax check, not execution in the remote environment.
- Both runner hashes match `source_preparation.json`; the complete runner diff is confined to the freeze-mode protocol and associated witnesses/metadata.
- All 11 declared unchanged modules are byte-identical to the prior source. All six archived native files match both completed runs' import hashes. Both archived specs equal their prior arm specs, and the three whole-range module hashes match the new spec.
- AST-extracted the actual `get_optimizer` function and executed its filtering with sentinel parameters and a capturing constructor: the groups contain exactly **10, 0, 0** parameters, with LR `1e-5` and WD `5e-4`. No Torch import or AdamW instance was created. The native optimizer already explicitly represents empty groups, including in its existing frozen branch.
- Layer-dimension arithmetic gives **400,614 parameters in 10 tensors**, with a 1,302-input aggregate layer.
- Every preflight receipt key read by the controller is produced by the runner. Both executable `step` call sites pass `update=True`; the retained `update=False` semantic-gradient witness is not reached.
- The archived split protocol matches its existing manifest digest: 29,778 unique fit rows, 6,887 disjoint holdout rows; B8 gives 3,723 updates and a final B2.

## Correct implementation details

**Frozen original G.** `run_range_head_only.py:135-196` strictly restores official PV and the trained original G delta before adding the unchanged head. The only trainable names are `candidate_box_refiner.*`. Both optimization paths call `model.eval()` and then train only the independent refiner (`:421`, `:569`). The original persistent parameters and buffers are compared to their initial values, and their gradients must remain `None` (`:343`, `:436`, `:603`). There is no call that puts the original modules back into training mode.

**Loss and deployed output.** Native `models.pv_ground.py:518-562` finishes Mask generation, invokes the refiner once, and writes its output to `last_center` and `last_pred_size`. The actual native criterion reads those fields (`models.losses.py:891-917`); bbox and GIoU retain the native `10/7` and `2/7` weighting. Thus the geometry losses train the final deployed refined box. Semantic/Mask outputs and whole-range observations are intentionally frozen, and the current preflight checks that fact rather than requiring geometry-to-Mask gradients.

**Unchanged head and inference.** The 109 whole-range statistics are either supplied or zeroed while local support, parameter count, native alpha logit fusion and all 256 candidates remain present. There is no new loss/module, teacher, P2, quality reranking or GT inference gate. The two comparison forwards belong to the disposable sanity check, not deployed dual inference.

**Sanity closure.** The preflight repeats one real B8 for two updates. Cached source-off/on replay checks exact zero-head preservation; native head-off/on checks reset RNG separately. The code requires a geometry gradient at the output layer and positive member, condition and aggregate gradients after the second update. It then verifies original G state equality and serializes/reloads the head delta and optimizer in `BytesIO`, comparing model values, group metadata, state keys, moments and steps. The branch returns before disk checkpoint code. Its controller consumes the correct `weight_files_created` field and launches no formal training.

**Later train/formal source.** The supplied conditional runner preserves seed 2027, LR `1e-5`, WD `5e-4`, clip `0.1`, one exact fit pass and matched step-by-step row order for the whole arm. Formal restoration requires exactly the 10 head delta keys and strictly reconstructs original G plus that head. Initial/terminal 6,887-row evaluation remains separate from 9,508-row development validation. Native bbs/bbf selection and Mask fusion agree with the evaluator; saved totals are checked against it. Coarse/final diagnostics use the same selected Query, and full-candidate GT oracle fields do not affect selection.

**Launch boundary.** The launcher checks the actual predecessor's terminal schema, quotes dynamic SSH command arguments with `shlex`, loads known host keys, preserves the existing GPU lock and deploys only `--mode preflight`. The screen name is constant. There is no job termination, weight deletion or formal-campaign launch in these paths. These are source conclusions; no remote condition was queried by this reviewer.

## Added operational source

The scope was extended before finalization to `observe_preflight_authorized.py`, `collect_preflight_terminal.py`, `prepare_formal.py`, `controller.py`, `launch_formal_authorized.py`, both formal specs and the formal preparation record. These files were read directly; the final verdict includes them.

The observer only reads the one launch's PID/status/exits/log. The collector requires controller termination and consistent terminal status/exit, accepts only text/source file suffixes, checks environment and original-G hashes, and verifies every copied file. It reads the protected weight remotely only to hash it, without deserialization or weight transfer. Neither helper changes a job or deletes a file. Their SSH arguments are quoted and host-key behavior matches the launcher.

The two formal specs differ only in output root, arm name and the whole-range flag. Both point to the **new** frozen-mode whole-range probe, which also performs cached local/whole zero-head replay. The formal launcher requires the actual collected successful intake and preflight receipt, rechecks remote completion, and consumes the existing `reviewed_files` contract now present in the final JSON report. The formal controller checks B8, two updates, frozen original state, cached pair equality and exact optimizer restoration before either fit. The old joint-training preflight is not used as proof of the new freeze protocol.

The formal controller diff is limited to the runner name and new-protocol proof/fit checks; its retention logic is unchanged. That logic waits for complete training and all 9,508 formal rows, verifies row hash/order and terminal/parent hashes, and recomputes selected-box thresholds. It retains strict bbs @.50 improvements, keeps the incumbent on ties, and only deletes an owned `terminal.pth` under the two new arm directories. The retained best and protected original G are excluded. The recovery checkpoint contains only the head delta and optimizer, not a newly copied full original model.

A stdlib AST check exercised five retention-selection cases without executing any file operation: both actual prior nonbest scores select their own endpoints; a first improvement preserves protected original G; ties retain the incumbent; a later improvement selects only the previous owned endpoint. Both controller hashes match `formal_preparation.json`. Actual local preflight launch, preflight intake and formal launch artifacts were still absent when checked. No operational result is claimed.

## Limits of this acceptance

Native evaluation still calls `F.gumbel_softmax` (`models.pv_ground.py:585-599`). Freezing parameters and running statistics does not imply equal candidates across different random draws or processes. The new same-RNG zero-head check and the existing recorded cross-process comparison are appropriate; no stronger reproducibility claim is justified.

The completed prior pair is motivation only: formal bbs @.50 was 4,428 local versus 4,461 whole, still 34 below original G's 4,495. Both prior training receipts report 806 changed tensors, so those runs do not certify this new 10-tensor protocol.

The real two-step freeze/gradient/restore result remains pending. No GPU, model import, weights or SSH was used for this review. The subsequently added formal launcher, controller and specs have now been inspected; their execution remains conditional on the actual new-protocol sanity result. Source inspection does not substitute for that runtime evidence.

Reviewed at 2026-10-04T05:54:23.632468+08:00.
