# Fresh review request: archived P3 selected-query analysis and publication

Use gpt-6-astra, reasoning max, fork_turns none. Attribution is same-family/provisional. Perform read-only analysis; only write the requested review reports in this private directory. Do not run SSH, GPU forwards, any training, Git mutation, or the publication script. Do not read MEMORY.md.

Files to read:
- C:\Users\gb\.codex\tmp\pvground_p3_next_20261002\analyze_selected_refinement.py
- C:\Users\gb\.codex\tmp\pvground_p3_next_20261002\publish_selected_refinement.py
- C:\Users\gb\.codex\tmp\pvground_p3_next_20261002\SELECTED_REFINEMENT_FINDINGS.md
- C:\Users\gb\.codex\tmp\pvground_p3_next_20261002\complete\SELECTED_REFINEMENT_MAGNITUDE.json
- C:\Users\gb\.codex\tmp\pvground_p3_next_20261002\complete\INTAKE.json
- C:\Users\gb\.codex\tmp\pvground_p3_next_20261002\complete\CPU_RECOUNT.json
- C:\Users\gb\.codex\tmp\pvground_p3_next_20261002\complete\EXPERIMENT_AUDIT.md
- C:\Users\gb\.codex\tmp\pvground_p3_next_20261002\complete\EXPERIMENT_AUDIT.json
- C:\Users\gb\.codex\tmp\pvground_p3_next_20261002\complete\p3\terminal\rows.jsonl
- C:\Users\gb\.codex\tmp\pvground_p3_next_20261002\complete\p3\terminal\receipt.json
- C:\Users\gb\.codex\tmp\pvground_p3_next_20261002\complete\p3\formal\rows.jsonl
- C:\Users\gb\.codex\tmp\pvground_p3_next_20261002\complete\p3\formal\receipt.json
- C:\Users\gb\.codex\tmp\pvground_p3_next_20261002\complete\run.py
- C:\Users\gb\.codex\tmp\pvground_p3_next_20261002\complete\source\split_protocol.json
- C:\Users\gb\.codex\tmp\pvground_p3_next_20261002\complete\source\imported\models.losses.py
- C:\Users\gb\.codex\tmp\pvground_p3_next_20261002\complete\source\imported\src.joint_det_dataset.py
- C:\Users\gb\.codex_pvground_cs_20261002\models\pvground_semantic_assignment.py
- C:\Users\gb\.codex_pvground_cs_20261002\models\nr3d_semantic_assignment.py
- C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\REFERIT_G_INTERFACE_NOTES.md
- C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\referit_g_interface_receipt.json
- C:\Users\gb\.codex_pvground_cs_20261002\refine-logs\pvground_referit_formal_preparation_20260917_v1\nr3d_formal_input_contract.json
- C:\Users\gb\.codex_pvground_cs_20261002\refine-logs\pvground_referit_formal_preparation_20260917_v1\sr3d_formal_input_contract.json
- C:\Users\gb\.codex\tmp\pvground_fused_support_20261002\initial_boundary_publication.json
- C:\Users\gb\.codex_mcln_g0_20260905\docs\MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md
- C:\Users\gb\.codex_pvground_cs_20261002\docs\MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md
- C:\Users\gb\.codex_mcln_v99_internal_20260928\docs\MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md
- C:\Users\gb\Desktop\document\MCLN_CURRENT_COMPLETE_HANDOFF_2026-08-15.md
- C:\Users\gb\.codex\tmp\sync_cs_handoff_remote_20260923.py

Audit checklist:
1. Ground-truth provenance; distinguish saved dataset GT from geometric proxy/identity labels. Assess what can actually be verified locally.
2. Recompute the new descriptive statistics independently from both archived row files, including quantile definitions, coordinate units, selected-query scope, threshold repairs/damages, valid sizes, and hash binding. Check all claims against actual keys/files.
3. Check normalization denominators, continuous versus threshold metrics, subgroup boundaries and absence of invented significance/generalization claims.
4. Confirm the script was executed and the result hash binds its source; distinguish new CPU analysis from historical GPU score/evidence. Detect dormant code and assess reporting scope.
5. Verify the future Nr/Sr G interface note against the actual frozen PV criterion and dataset fields and saved contracts, without interpreting historical preparation as new execution.
6. Inspect publication code for exact reviewed-artifact binding, old-prefix preservation, four local copies plus remote, bounded writes, staging, nonforce push, source/active experiment protection, no deletes, and no reviewer trace disclosure. Static review must not claim publication has executed.
7. Report PASS/WARN/FAIL with file:line evidence; separate blockers from honest qualifications. Classification is archived real-GT descriptive analysis, not independent training ablation.

Write full report to C:\Users\gb\.codex\tmp\pvground_p3_next_20261002\SELECTED_REFINEMENT_AUDIT.md, and JSON to the same directory with filename SELECTED_REFINEMENT_AUDIT.json. JSON must contain verdict, blocking_findings array, reviewed_files array of actual path/bytes/sha256, checks, reviewer_model=gpt-6-astra, reviewer_reasoning=max, review_independence=same-family, acceptance_status=provisional, plus limitations. Preserve your full response verbatim in your final answer. Do not alter executor files or invent PASS if evidence is missing.
