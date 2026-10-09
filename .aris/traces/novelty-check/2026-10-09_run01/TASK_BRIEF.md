You are a fresh same-family provisional novelty reviewer, requested gpt-6-astra / max, actual backend UNATTESTED. User prioritizes PV-Ground achieving ScanRefer >59.5/@.25 and >51/@.5 with 3 actually effective contributions, seed2027 only, followed by independent author-initialized Nr3D/Sr3D. They explicitly stopped further pretrained reproduction investigation and ask whether our contributions overlap PV-Ground and what current problems should guide improvements.
Do not launch SSH/GPU/training, install dependencies, read AUTH/credentials or bulk MEMORY, or modify experiment sources. Use primary papers and actual local code; write reports in C:\Users\gb\.codex\tmp\pvground_novelty_20261009 (create it if needed). Preserve exact briefing, full raw response, source URLs/hashes. No unsupported claims of 3 effective modules.
Current/proposed core claims to verify (method claims, not established novelty):
C1 Current prediction-fused Mask hard foreground and actual superpoint member extrema produce candidate spatial reference for six-face decoding. In strongest reference-only state, refinement output is zero and final Box equals reference extent. Contrast to baseline's original direct regressed Box.
C2 Current learned query-specific Mask support corrector reads language-conditioned PV Query 288, SP visual features 288, existing Text/Query/fused responses and observed member count; small residual MLP corrects Query logits before original fusion and member-extents reference. Native semantic ranking remains one last/bbs. It has only a small single-seed formal gain, not a broad novel paradigm.
C3 Proposed, not yet implemented/effective: supplement native matched Mask training with root GT supervision on the actual native-score-selected but unmatched Query, preserving every other matched-instance responsibility, frozen score path and all256 deployment candidates. This aims to repair the support delivered by the query actually selected rather than adding another ranking head. Inspect closest inference-aware learning and one-to-many mechanisms. Also judge whether this proposal is a sensible evidence-targeted next pilot, without claiming likely accuracy gain.
Actual local code/evidence to read:
C:\Users\gb\.codex\tmp\pvground_compressed_geometry_support_20261008\mask_reference.py
...\mask_support_corrector.py
...\mask_support_model_factory.py
...\matched_mask_objective.py
...\analysis\EXPERIMENT_AUDIT.json
C:\Users\gb\.codex\tmp\pvground_current_error_partition_20261009\SUMMARY.json and FINDINGS.md (locate exact files via rg if name differs)
C:\Users\gb\.codex\tmp\pvground_support_identity_20261009\support_identity_readout.py and run_identity_campaign.py
C:\Users\gb\.codex\tmp\pvground_support_identity_20261009\USER_REQUESTED_STATUS_20261009_113621.json (raw user-requested completed receipt; not yet independent terminal audited)
Official local PV model mirror C:\Users\gb\.codex\tmp\pvground_mask_support_correction_20261008_v2\complete_fit\PV-Ground\models\pv_ground.py
Primary sources already located (read method, not only titles; broaden recent literature independently if needed):
https://github.com/AaNnWwTt/PV-Ground
https://openaccess.thecvf.com/content/CVPR2026/papers/Shang_PV-Ground_Text-Guided_Point-Voxel_Interaction_for_3D_Visual_Grounding_CVPR_2026_paper.pdf
https://arxiv.org/html/2206.02777v3 (Mask DINO)
https://arxiv.org/html/2410.13842v1 (D-FINE)
https://arxiv.org/html/2308.11887v2 (3DRefTR)
https://arxiv.org/html/2401.03989v1 (MS-DETR)
https://arxiv.org/html/2608.03216v1 and https://neesky163.github.io/iFAN/ (Aug2026 iFAN: adjusted quality ranking and cross-layer distillation)
https://aclanthology.org/2023.emnlp-main.656.pdf (3DRP-Net)
https://openaccess.thecvf.com/content/CVPR2025/html/Lu_Relation3D__Enhancing_Relation_Modeling_for_Point_Cloud_Instance_Segmentation_CVPR_2025_paper.html
Executor's searches covered mask-to-box initialization, query Mask correction, support/geometry responsibility, highest-score selected query supervision, and recent 2025-2026 including last6months. Search stage does not establish novelty.
Required output: NOVELTY_REVIEW.json/.md with exact overlapping original PV/MCLN features vs our deltas, per-claim nearest named prior work and verifiable difference, which parts cannot be claimed new, what a reviewer would require as direct controls, current valid vs failed vs proposed mechanisms, and concise prioritized optimization direction. Global recommendation PROCEED/PROCEED WITH CAUTION/ABANDON under these limits. Avoid excessive per-source quotation or >200-word single-paper paraphrase.
=== NOVELTY VERDICT LIMITS (these bound how you judge, never how widely you search) ===
Search exhaustively; judge calibrated. Two failures waste months equally:
passing an idea a published paper already contains, and killing a viable idea
because the territory has neighbors.
1. Proximity is information, not a verdict. Someone working nearby goes in the
   report; it is not by itself a reason to reject.
2. ABANDON has exactly one qualification: a specific published paper already
   contains this result — name that paper. No named paper, no ABANDON.
3. Crowded-but-deltaed is PROCEED: state the delta in one sentence a reviewer
   could verify. Thin or contested delta is PROCEED WITH CAUTION — say what
   would make it carry, not why it should die. CAUTION is not a safe middle:
   if you cannot name the specific thing that makes the delta thin, the
   verdict is PROCEED.
4. Concurrent or competing work is not a veto. That is a race — report it and
   let the user decide whether to run it.
5. A direct attack on a central problem is legitimate novelty when nobody has
   executed it well. "This area is hot" does not mean "this area is taken."
6. This check is an early gate, never the last one — more triage, pilots, or
   external review still stand between any idea and a paper, whatever order
   this run uses. A wrongly passed idea dies cheaply at one of them; a wrongly
   killed idea is never seen again. When torn between two verdicts, choose the
   more permissive one.
Say plainly when an idea clears the check. Do not manufacture overlap.
