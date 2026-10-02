# Same-tail support post-run helper review

Review only the proposed local post-run collection and CPU recount implementation. No result exists for completed same-tail arms yet. Do not access SSH, credentials, remote surfaces or GPU; do not run the collector, training, publishers or any code with remote effects. Read actual files, and report concrete blockers rather than hypothetical hardening.

Implementation files:

- C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\collect_complete_arm.py
- C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\recount_complete_arm.py

Authoritative interfaces and plan:

- C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\PLAN.md (original protocol; opening preparation-only status is historical)
- C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\run_pvground_tail_support.py
- C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\run_pvground_tail_support_control.py
- C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\tail_raw_spec.json
- C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\tail_fused_spec.json
- C:\Users\gb\.codex\tmp\pvground_p3_next_20261002\collect_complete.py
- C:\Users\gb\.codex\tmp\pvground_p3_next_20261002\recount_complete.py
- C:\Users\gb\.codex\tmp\pvground_p3_next_20261002\complete\CPU_RECOUNT.json
- C:\Users\gb\.codex\tmp\pvground_p3_next_20261002\complete\EXPERIMENT_AUDIT.md
- C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\tail_raw_training_boundary_193500.json

Requirements: collect one actually completed arm (3723 updates,29778 fit rows exactly once,6887 initial/terminal,9508 formal), preserve verbatim receipts/rows/import sources and exact SHA256, archive terminal state_delta+optimizer/RNG without deleting remote or parent files, independently recompute saved selected/coarse boxes against their saved GT using strict >.25/>.5, retain continuous float discrepancies and per-row threshold checks. Recount Mask only from saved IoUs and all256 coverage only from saved GPU flags; do not claim fresh raw-Mask/all-box replay. Compare tail_raw against historical G continuation only as context; tail_fused versus same-tail raw is the actual controlled support comparison. Verify row/input identities and full ordered fit batches before reporting paired repairs/damages. Keep primary bbs and original G5615/4495/target4754 distinct from alternative bbf. Preserve known E0 Mask differences; do not describe selected REC parity as whole-model bitwise parity.

Scope: simplest minimal scheme; no fallback, hypothetical edge-case branches, unnecessary compatibility or unrelated refactor. Report actual trust-boundary findings if any. Helpers are not deployed/executed; existing reviewed training sources and protocol must remain unchanged.

Output exact inspected file hashes, PASS/WARN/FAIL, blocking findings and concrete minimal fixes; same-family/provisional gpt-6-astra max. Save full Markdown report and JSON under:

- C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\POSTRUN_CODE_REVIEW.md
- C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\POSTRUN_CODE_REVIEW.json

Do not edit implementation files or execute scientific experiments. This is a code review, not a completed-result integrity audit.
