# Fresh source review: stable original G, existing range head only

Please read primary files directly. Source-only: no SSH, GPU, model imports,
weights, deletion, job mutation, external messaging or further delegation.
Do not evaluate by paraphrases or use previous reviews as your own verdict.

Plan and implementation:
`C:/Users/gb/.codex/tmp/pvground_range_head_only_20261004/EXPERIMENT_PLAN.md`
and all15 Python files in that directory, plus preflight_spec.json and
source_preparation.json. The launcher deploys ONE whole_range two-step probe;
formal training is not launched by it. Check controller/receipt field names.

Primary prior source and outcome:
`C:/Users/gb/.codex/tmp/pvground_whole_mask_fit_20261003/run_whole_mask_fit.py`,
`analysis/REPORT.md`, actual complete local_range/whole_range spec, imports,
load, train receipt and formal receipt. Use completed prior experiment only as
motivation/provenance, not as current frozen-protocol evidence.

Actual native imported sources, byte identity audited against both prior runs:
`C:/Users/gb/.codex/tmp/pvground_fused_support_20261002/complete_tail_fused_retry/source/imported/`
contains models.pv_ground.py, models.losses.py, main_utils.py, evaluator.py,
src.joint_det_dataset.py and prepare_data.py. Inspect actual forward, modes,
optimizer/criterion, and scores. Do not assume eval removes native Gumbel RNG.

Assess that only ten400614-parameter head tensors train; all original G
parameters AND running buffers remain fixed/eval. Native bbox/GIoU must train
the actual deployed refined box. Semantic and Mask loss/outputs are intentionally
frozen observations; do not require geometry-to-Mask gradients in this control.
Preserve all256 candidates, native bbs scoring, officialPV+trainedG init and
freshAdamW. No new modules/loss, teacher, P2, dual inference or GT gate.

Real planned sanity: B8 repeated twice, cached zero-head source on/off and
native on/off same-RNG checks, output-layer task gradient and after2 steps
member/condition/aggregate gradients, all original state unchanged, exact
BytesIO head/AdamW restore, zero disk weights. Check original three optimizer
groups can have two empty groups and include exactly head parameters.

Later conditional source pair keeps effectiveB8/LR1e-5/WD5e-4/clip0.1,
29778 fit rows once/3723 updates with lastB2,6887 pretrained-seen holdout and
9508 development validation separate. Verify train/formal restore checkpoint
contains only head delta, strict native restoration, budget and row order.
No claim of bitwise cross-process candidates or deployable GT oracle.

Please write EXPERIMENT_CODE_REVIEW.md and JSON here, with verdict PASS/WARN/FAIL,
blocking_findings list, nonblocking findings, execution_scope SOURCE_ONLY,
requested model/effort and same-family/provisional attribution. Read-only code
review may use deterministic stdlib AST/diff checks; record what was actually
checked, not prospective checks as passed. Do not edit source or manufacture
GPU results. Only these review-owned reports may be written.

Scope: report concrete actual defects, including rare cases actually produced
by this repo. Propose minimal fixes only. No speculative fallback, compatibility
frameworks, new pins/digest schemes, try/except or unrelated refactoring. Existing
hash/provenance machinery may be checked for real bugs. Trust boundaries of SSH
commands remain in scope. Say plainly when code is correct.
