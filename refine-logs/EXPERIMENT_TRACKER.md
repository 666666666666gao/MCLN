# MCLN Nr3D experiment tracker

Updated: 2026-09-06 03:24 CST. Detailed evidence: master §§20.37–20.63.

| Experiment | Current state | Decision |
|---|---|---|
| G0 augmentation pair | Complete; integrity PASS, scientific FAIL | Keep data fix; original G0→G1 performance route sealed |
| P1 four-row padding identity and candidate audit | Complete | Selected seeds/Masks stable in four padding interventions; full object-memory availability remains incomplete. PR #7 stays draft |
| Independent P2 v1 | Complete; both heads 6,687 updates; terminal verification PASS | Pair-global REC -1/-84, pair-protected -3/+118 and lower Mask mIoU. Both fixed screens FAIL; do not advance this variant to P3 |
| P1 official-path diagnostic | Complete; native row/CSV/source/weight checks PASS | Current-source REC4478/3763 differs slightly from history4475/3759; no promotion. Full256 misses426; legal candidates exist for2995 current errors. Mask/CSV/SP analyses complete |
| R1 reference-memory four-arm screen | Complete;6687 updates per arm and6172 holdout rows; integrity PASS | All three object_pair screens FAIL; PR#9 sealed, no Decoder or control promotion |
| M2 native Mask-branch diagnosis | Complete;7899 rows,zero updates; native/source/data/state checks PASS |767/828 majority-label-pass failures also lack passing raw Masks; inspect Mask learning/alignment before fusion changes |

| M3 native Mask supervision/gradient probe | Complete;16 fit rows,4 forwards,8 gradient probes,zero updates | Native matched-Query indices and gradients correct;66/302 majority SP neighborhoods lack target seed centers; locality still needs distances |

| M4 Mask neighborhood intervention | Complete;16 fit rows,8 forwards,zero updates |31 majority SPs read distant seed0 on empty balls; nearest-two mean Mask +2.2877pp but6 rows worsen. Proceed only to matched learning screen |
| M5 existing Mask projection training | Complete;both1024-step arms and6172 terminal rows; integrity PASS, fixed quality FAIL |2048 fit/262 scenes,6172 holdout/98 backbone-seen scenes;2 arms1024 steps each, no REC changes or formal promotion |
| L1 last text-attention key evidence | Complete; 6687 updates per arm; integrity PASS, quality FAIL | REC25 net0 versus both; no formal validation, control promotion or continuation |
| Point-detail to superpoint Mask | Complete;1024 updates per arm,6172 endpoint; integrity PASS, quality FAIL | detail-native mIoU+.01507pp, detail-start-.09230pp; no continuation or formal promotion |
| C1 candidate Mask memory reading | Both1024 updates complete, terminal evaluation | 03:20:52 PID23307,800/6172 terminal rows; quality pending |
| Raw object point appearance | Explicit bounds fix;5 CPU tests and full511-scene input contract PASS | 0 empty/invalid-axis crops; native16-row preflight staged/Py3.7 compiled, not executed |

P2's 6,172-row, 98-scene holdout is from train scenes already seen by the frozen
backbone. Protected/global/pair REC hits are 6005/5312, 6003/5514, 6002/5430;
Mask mIoU 68.8766%, 68.6682%, 68.7242%. Do not label these official or unseen-system
results. Full256 lacks a qualifying box on only 25 rows, so this audit does not
represent the official validation failure distribution.

Raw P2 results and independent checks are in PR #8:
`refine-logs/p2_readout_v1_20260905/`. On the server, both final addon heads are
archived at `/root/autodl-tmp/mcln_g0_view_pair_20260905/pair_readout_train_v1/results/`.
P2, P1, R1 and M2 have ended; at19:01:06 no R1/M2 process remains and GPU use is1MiB/0%. No protected model
was replaced, no P2 formal validation ran, and the three-benchmark objective
is still incomplete. Nr3D protected REC remains 4475/3759 on 7899 rows.

Historical G0 boundaries: raw augmentation-permission changes 2155;
normalized text 325 (fit253/holdout72). These are permission counts, not all
numerical inputs changed by the continuous worker RNG. G0's auxiliary logging
defect did not affect its independently checked per-row decision.

Completed P1 report: `docs/NR3D_OFFICIAL_CANDIDATE_AUDIT_RESULT_2026-09-05.md`.
Existing Decoder object-memory input audit completed on four fit rows: 190 object slots, 38 covered by Queries, 155 correct predicted classes; not GT text-anchor recall. R1 completed all four arms6687 updates and6172 holdout rows. Independent verification PASS, all three object_pair screens FAIL. Its REC versus protected is+6/+24, Mask mIoU-0.1458pp; versus query_pair strict REC-94, versus object_global-329. PR#9 90861cd seals full results, bootstrap and original verifier cwd error/corrected v2. No control promotion or Decoder integration. Report `docs/NR3D_REFERENCE_MEMORY_R1_RESULT_2026-09-05.md`. P2 remains sealed; actual CSV lengths are used throughout R1.

CPU Mask-target audit complete (zero model forwards/updates): majority GT-label point mIoU81.9564 versus optimal SP83.3750 on sealed val inputs. The1105 good-box/bad-fused-Mask cases split154 representation-limited +123 majority-label-limited +828 whose majority labels already pass. Do not call these model metrics or training-example counts. Next Mask diagnosis separates raw text/query/fused branches after the fixed R1 result; no new Mask architecture or threshold sweep. Report `docs/NR3D_MASK_SUPERVISION_AUDIT_2026-09-05.md`.

M2 native Mask-branch diagnostic COMPLETE, controller/analysis exit0,7899 rows,717.419s. Fixed828 majority-label-pass cases split767 bothrawbranchesfail +61 eitherrawbranchpasses; requiring a legal Box>.5 reduces passes to40. Direct raw-Query replacement at native Mask selection gives+4/-4 hits and only+0.0044pp mIoU, not adopted. All source/data/state/native checks pass; aggregates equalP1. Last11 rows have explicitly retained Box numeric/raw-size differences, unchanged Query identities/Mask IoUs, cause not isolated. Report `docs/NR3D_MASK_BRANCH_DIAGNOSTIC_RESULT_2026-09-05.md` and full terminal evidence. No new model or formal best.

M3 COMPLETE with original Hungarian indices/native autograd, no indexing or disconnected Mask-projection bug. Good legal Box Queries104, direct Mask gradient absent on88; normal one-to-one objective, not automatic evidence for extra labels. All16 matched Boxes>.5,10 matched raw Masks>.5. M3 receipt b6b7b1d7903f766c0f54f576c33641d89d60e455d0e5b5159db0680d1bba163f. Report docs/NR3D_MASK_SUPERVISION_PROBE_RESULT_2026-09-05.md.

M4 COMPLETE: actual majority foreground empty balls31/302, original seed0 distances3.09–6.16m versus nearest.200–.285m. Target-seed-free neighborhoods66→19. Frozen16-row fusion mIoU+.0228774 and .50+1;6 improve/6 worsen/4 unchanged. Grounding and M3 identity PASS; no promotion. Original upstream shares the grouping behavior. Full report/receipt published0a3a93c.

M5 launched20:00:27, screen17183.mcln_m5_mask_locality, isolated directory /root/autodl-tmp/mcln_mask_neighborhood_m5_20260905_v1. CPU2 tests PASS, data724 files pinned; real two-arm gradient preflight PASS (1348960 parameters,16 tensors), initial holdout1600/6172 at20:08. PID17187 active, GPU10553MiB/30%; no optimizer updates yet. Baseline ETA20:23, whole-run estimate21:00–21:30 pending training throughput; next check near20:23–20:26. It first evaluates both start arms, trains existing16 Mask parameter tensors on2048 fit rows for1024 updates per arm, then evaluates full6172 module-holdout rows. Fixed nearest endpoint must beat both terminal native and protected start mIoU by.002, with neither Mask threshold declining; grounding/input/Query identity required. This cannot improve frozen REC or complete the three-benchmark goal. No new formal metrics. R1/P2 remain sealed.


20:43 authoritative update: M5 has finished both1024-step trajectories; native PID17187 remains active in terminal evaluation, estimated completion21:03. Earlier20:08 baseline progress above is historical. Object point input CPU audit COMPLETE/exit0:683 slots,0 empty raw crops,8 empty seed RoIs,123 crops with instance fraction<.5. Input/source/data identity PASS, no new performance result. L1 position-key versus text-key mechanism registered in docs/NR3D_TEXT_POSITION_L1_PLAN_2026-09-05.md; code/test preparation, no training or quality result yet. Original REC/three-benchmark objective remains open.


M5 terminal authoritative update: fixed quality FAIL, integrity PASS, controllerexit0; no formal promotion. Full report docs/NR3D_MASK_NEIGHBORHOOD_M5_RESULT_2026-09-05.md. L1 CPU7 tests and terminal summary4 tests PASS; actual M3 nested receipt reader corrected beforeGPU and archived. L1 real-fit preflight next, no L1 training result yet.


L1 v2 native preflight PASS/exit0,16 fit rows,20 forwards,zero updates; all tensor identity/gradient/state/data checks passed. v1's ordered-key verifier error is archived and corrected without changing the mechanism or experiment contract. Paired training launched21:13:24, screen mcln_l1_train, /root/autodl-tmp/mcln_text_position_l1_20260905_v2/train/. Start evaluation and actual training throughput still need collection; no terminal quality result. M5 remains failed at Mask25 despite positive mean-IoU evidence; full REC/three-benchmark goal stays open.


21:24 L1 v2 authoritative live state: PID18450, GPU9703MiB, baseline1600/6172, zero optimizer updates so far; latest50 B16 batches112.975s, baseline ETA21:35. Next inspect near21:33–21:35 and use actual training throughput for terminal ETA. ScanRefer four protected artifact hashes and0444 modes PASS. Sr3D CSV/train/val inputs present; documented protected/candidate/E26/E29 files absent on this instance and backup location requested. Read-only report docs/THREE_BENCHMARK_ARTIFACT_READINESS_2026-09-05.md. No new formal benchmark result; full goal remains active.


21:41 authoritative L1 state: PID18450 live, both arms logged384/6687 updates,1536 fit rows; recent128-step intervals147.045/149.627s. Training ETA23:43–23:46 and terminal completion estimate2026-09-06 00:05–00:20 CST, pending actual throughput. Full6172 L1 zero-start records exactly match M5 native start for point hashes and selected REC/Mask identities/IoUs; no trained endpoint read. New terminal position-state reader passed5 CPU synthetic tests and does not authorize promotion. Full goal active; Sr3D backup location pending.


22:15 progress: the conditional native L1 formal-pair entry is implemented and passed 11 CPU tests in the frozen server environment. It reuses the unchanged terminal gate, native evaluation branch and separate native evaluators on identical batches, preserving REC/Mask selection paths and recording pre/post-filter coverage. Full GPU evaluation is untested and no trained artifact or eligible formal manifest was created. Last live check at 22:06:41 recorded both arms 1664 / 6687 updates (6656 rows); PID 18450 continues. See master §20.54 and docs/NR3D_L1_NATIVE_FORMAL_PAIR_2026-09-05.md. Overall goal active; Sr3D backup location remains pending.


2026-09-05 23:10 CST progress: point-based local-detail residual and isolated native preflight are prepared; 54,144 parameters, 4 synthetic CPU tests pass, no voxel installation or GPU preflight. M2 cached-output identity diagnosis finds533 differing REC/Mask selections among7898 legal-REC rows; forcing existing REC-query Masks gives+47/+27 hits and+0.2884171pp mIoU on that paired subset, with17/9 breaks. Diagnostic only; no evaluator change or formal best. Identified Sr3D backup candidates were exhausted at22:27; new backup location remains needed. L1 fixed run and protected results unchanged. See master§20.55 and docs/NR3D_POINT_DETAIL_MEMORY_PLAN_2026-09-05.md. Overall goal active.


2026-09-05 23:40 CST progress + verified wait: native task audit confirms shared Decoder identity, separate existing projections and shared Hungarian indices. All533 differing REC/Mask Query selections coincide with REC pre/post-filter top changes; do not claim these prove missing shared representations. Paired historical-output deltas unchanged, evaluator unchanged. Local-detail input readiness PASS (612 source/36 data files and matching scatter); no GPU preflight. L1 actual PID18450 has6272/6687 updates at23:37, no endpoint. Collector full-command matching corrected an observed false match to its observer. See master§20.56 and docs/NR3D_TASK_QUERY_ARCHITECTURE_AUDIT_2026-09-05.md. Goal active.


2026-09-06 00:50 CST authoritative update: L1 complete and sealed FAIL; position REC6005/5307, Mask5767/5057, mIoU68.877936727% on6172 backbone-seen module-holdout rows. Integrity and actual optimizer/artifact checks PASS. Point-detail native16-row GPU preflight PASS with exact zero identity and connected Mask gradients. Independent native/detail Mask learning pair launched00:47:13, manifest9c690081...fae81;2048 fit/262 scenes,6172 holdout/98 scenes,1024 updates per arm. Both train original16 Mask tensors;detail adds54144 parameters, no equal-capacity claim. Fixed primary gate and conditional Query Mask diagnostics registered before results. Full goal active, formal protection unchanged. See master§20.57 and three new2026-09-06 reports/plans.


2026-09-06 01:24 CST authoritative update: B full6172 start identity PASS versus both zero-detail and archived M5 native, including point/grounding hashes and selected Query/IoUs. Existing REC Query substituted for Mask on these rows gives133 switches, -13/-9 hits and-0.153471988pp mIoU; cached counterfactual only, evaluator unchanged. Shared-SP B affects raw Query/Text/alpha paths, and whole Box head input changes token scores; C insertion boundaries documented. Last live 01:23:10: PID21638,384/1024 steps per arm, recent64 steps88.07s. Current run continues with240s observer; final ETA02:00–02:10, no terminal quality or formal promotion. Master§20.58; goal active.


2026-09-06 01:50 CST authoritative update: B both1024-step updates complete; 01:47:29 PID21638 continues in terminal evaluation 2400/6172. No terminal screen result. C1 candidate-specific SP memory readout implemented separately,74880 parameters,4 CPU synthetic tests PASS on original server Python/Torch;0 dataset/native/GPU forwards and0 optimizer steps. Box/identity path retained, no final selection change; native behavior untested. Master§20.59 and prototype doc. Goal active, protected formal results unchanged.


2026-09-06 02:24 CST authoritative update: B complete/controller0, integrity PASS and fixed quality FAIL against both controls. Artifact/optimizer checks PASS; cross-run M5 native endpoint differs slightly, training point tensors not stored for cross-run comparison. C1 native16-row/12-forward preflight PASS with unchanged REC/Decoder Query/Text Mask/alpha, finite connected gradients, restored state and0 optimizer updates. Full reports and master§20.60. No new formal result, no training continuation, full goal active.


2026-09-06 02:37 CST authoritative update: C1 native-vs-memory Mask Query learning launched02:32:29; real first-fit loss/shared gradients equal, source/data/native-state checks PASS.02:35:59 PID23307 in initial6172-row evaluation. Native trains x_query only664992 parameters, memory adds74880; all other branches frozen. Fixed1024-step budget and two-reference quality screen. Manifest30beb42f; master§20.61. Observer starts02:52:29 then240 seconds; no terminal/formal result.


2026-09-06 03:05 CST authoritative update: C1 complete6172 start reproduces protected B native; 03:04:39 both448/1024 updates. Raw object appearance prepared independently,4 CPU tests/32 real appearance-only forwards on683 slots PASS. Full511-scene/16181-slot input audit complete with1 empty crop; numeric diagnosis supports explicit bounds comparison, no fallback or expansion adopted. No native object-appearance GPU run or training. Master§20.62; protected formal results unchanged.


2026-09-06 03:24 CST authoritative update: explicit AABB crop predicate passes all511 scenes/16181 slots with0 empties,0 nonpositive axes. Boundary membership changes11697 slots (+16858/-4271 memberships), all inputs/old counts unchanged.5 CPU tests PASS; native object appearance preflight staged and compiled only. C1 both1024 updates complete, 03:20:52 terminal800/6172; no quality result. Master§20.63. Goal active.
