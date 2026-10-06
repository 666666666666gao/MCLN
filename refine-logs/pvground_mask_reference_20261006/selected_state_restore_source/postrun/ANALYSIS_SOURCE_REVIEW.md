# Postrun analysis source review

PASS — zero blocking findings. Followup reviewed 2026-10-06T11:44:16.567049+08:00. Original review 2026-10-06T11:31:48.090186+08:00 is preserved unchanged.

SOURCE_ONLY; same-context followup, same-family/provisional; backend identity not_attested. The analyzer main was not executed. No new terminal data was read or synthesized; complete_fit and analysis were absent at review time. Training source, SOURCE_REVIEW, publishers and running observers were unchanged.

- Four9508 paths and seven NPZ fields match the frozen producer. The analyzer covers all256 prior/reference/final boxes, scores, validity and root GT; it verifies batch order, row coverage and exact selected-box/root correspondence.
- Intake entries are checked by bytes/SHA; formal row files are checked against their receipts. The collector and controller provide the existing closed-job and complete-transfer gates.
- Historical parent rows were actually read:9508 sequential rows, receipt SHA `ec73b6acb9691d6c591136c5e5a34e80a8d7d716bedef714ca960bbbd324716c`, native hits5616/4511. New results must match parent row/scan/target/root/point identities.
- Saved selected queries must have maximal native scores. Tied maxima retain the producer query. Mask metrics recount stored IoU and do not reconstruct raw Masks.
- Initial results are labelled zero-update architecture states and must decode exactly to their stored references; native references must equal original priors. Terminal results are separate3723-update entries. Hidden11169 prior updates and output reset are disclosed.
- Fit receipts/logs require3723 sequential updates,29778 unique inputs,tail2 and identical order across arms. Extra-loss totals count recorded candidate/outside-face occurrences; they are not unique candidate counts or causal conclusions.
- Selection compares protected4511 with the four declared initial/terminal results, using native Acc@.50. Primary ties retain the parent; otherwise@.25 and stable table order resolve ties. The analyzer performs no weight deletion or restore. Fresh terminal audit and selected-state restoration remain pending.

The isolated IoU formula passed analytic values1,1/3,0 and broadcasting. Float64 CPU threshold differences are reported separately from the native metric. The initial review's source adjustment splits threshold disagreements into original_prior/reference/final, each at.25/.5; native metrics and selection are unchanged.

Current analyzer SHA: `f9c3f6d4b4fda3a71819b2b89f2bdd416fbd93ebaf98ebe310f14a15f7a7f42d`. All101 original source bindings still match. The JSON report binds 22 reviewed files. This report does not certify any new terminal score, current process state, checkpoint restore or reviewer backend identity.

Parent-step metadata correction: the initial review missed that `optimizer_updates=11169` used cumulative history. Historical protected4511 has a fresh optimizer and checkpoint step3723; its save schema adds prior7446 for cumulative11169, and the current parent factory requires step3723. The corrected table now reports `optimizer_updates=3723` and `total_geometry_fit_updates=11169` separately. Reversing only those two lines reproduces the previous analyzer SHA; metrics, comparisons and selection are unchanged. No checkpoint was opened or restored in this followup.

The original review archive `ANALYSIS_SOURCE_REVIEW_20261006_113148.json/md` remains unchanged. Root `POSTRUN_PREPARATION.json` and `record_postrun_preparation.py` remain the11:32 snapshot; their old analyzer SHA is historical, not the current analysis gate. No analyzer main or terminal recount was executed.
