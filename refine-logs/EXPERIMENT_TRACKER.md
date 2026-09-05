# MCLN Nr3D experiment tracker

Updated: 2026-09-05 18:44 CST. Detailed evidence: master §§20.37–20.44.

| Experiment | Current state | Decision |
|---|---|---|
| G0 augmentation pair | Complete; integrity PASS, scientific FAIL | Keep data fix; original G0→G1 performance route sealed |
| P1 four-row padding identity and candidate audit | Complete | Selected seeds/Masks stable in four padding interventions; full object-memory availability remains incomplete. PR #7 stays draft |
| Independent P2 v1 | Complete; both heads 6,687 updates; terminal verification PASS | Pair-global REC -1/-84, pair-protected -3/+118 and lower Mask mIoU. Both fixed screens FAIL; do not advance this variant to P3 |
| P1 official-path diagnostic | Complete; native row/CSV/source/weight checks PASS | Current-source REC4478/3763 differs slightly from history4475/3759; no promotion. Full256 misses426; legal candidates exist for2995 current errors. Mask/CSV/SP analyses complete |

P2's 6,172-row, 98-scene holdout is from train scenes already seen by the frozen
backbone. Protected/global/pair REC hits are 6005/5312, 6003/5514, 6002/5430;
Mask mIoU 68.8766%, 68.6682%, 68.7242%. Do not label these official or unseen-system
results. Full256 lacks a qualifying box on only 25 rows, so this audit does not
represent the official validation failure distribution.

Raw P2 results and independent checks are in PR #8:
`refine-logs/p2_readout_v1_20260905/`. On the server, both final addon heads are
archived at `/root/autodl-tmp/mcln_g0_view_pair_20260905/pair_readout_train_v1/results/`.
P2, P1 and R1 have ended; GPU now runs the read-only M2 Mask-branch diagnostic. No protected model
was replaced, no P2 formal validation ran, and the three-benchmark objective
is still incomplete. Nr3D protected REC remains 4475/3759 on 7899 rows.

Historical G0 boundaries: raw augmentation-permission changes 2155;
normalized text 325 (fit253/holdout72). These are permission counts, not all
numerical inputs changed by the continuous worker RNG. G0's auxiliary logging
defect did not affect its independently checked per-row decision.

Completed P1 report: `docs/NR3D_OFFICIAL_CANDIDATE_AUDIT_RESULT_2026-09-05.md`.
Existing Decoder object-memory input audit completed on four fit rows: 190 object slots, 38 covered by Queries, 155 correct predicted classes; not GT text-anchor recall. R1 completed all four arms6687 updates and6172 holdout rows. Independent verification PASS, all three object_pair screens FAIL. Its REC versus protected is+6/+24, Mask mIoU-0.1458pp; versus query_pair strict REC-94, versus object_global-329. PR#9 90861cd seals full results, bootstrap and original verifier cwd error/corrected v2. No control promotion or Decoder integration. Report `docs/NR3D_REFERENCE_MEMORY_R1_RESULT_2026-09-05.md`. P2 remains sealed; actual CSV lengths are used throughout R1.

CPU Mask-target audit complete (zero model forwards/updates): majority GT-label point mIoU81.9564 versus optimal SP83.3750 on sealed val inputs. The1105 good-box/bad-fused-Mask cases split154 representation-limited +123 majority-label-limited +828 whose majority labels already pass. Do not call these model metrics or training-example counts. Next Mask diagnosis separates raw text/query/fused branches after the fixed R1 result; no new Mask architecture or threshold sweep. Report `docs/NR3D_MASK_SUPERVISION_AUDIT_2026-09-05.md`.

M2 native Mask-branch diagnostic started18:43:03, originalscreen15553/controller15555/Python15557. Fixed7899 val rows, protected model and thresholds, zero updates. Records raw text/query/fused Mask quality, alpha and separate native REC/Mask/best-Box identities; fixed prior1105 and154/123/828 cohorts. No M2 result yet. First progress18:50; ETA18:58–19:05 and240s near-completion polls. Plan `docs/NR3D_MASK_BRANCH_DIAGNOSTIC_PLAN_2026-09-05.md`; input manifestfd44d2abe72824eed0bad7426daa946836b42551446fc1d316a31366438bf113. Do not duplicate the running observation.
