# MCLN experiment tracker

Updated: 2026-09-06 23:12 CST. Acceptance: §20.79; Scan mesh actual64 updates and post-training queue: §20.98.

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
| C1 candidate Mask memory reading | Complete;1024 updates/arm,6172 terminal rows; integrity PASS, quality FAIL | memory-native mIoU+0.00134pp, memory-start-0.03621pp; protected formal results unchanged |
| Raw object point appearance | Complete;1024 updates/arm,6172 endpoint; integrity PASS,quality FAIL | REC25/50 net versus native +2/+8, versus start -1/+31;formal results unchanged |
| Sr3D object input audit | Full1018 scans/34865 slots;old positive-size contract FAIL on2 real single-point objects | Minimal zero-axis fix:7 original-environment tests +actual152-slot CPU probe PASS;0 GPU/optimizer/Sr training |
| Sparse local-memory runtime | Independent original-Torch environment +synthetic GPU kernel checks PASS | Runtime-only PASS;actual memory preflight/learning tracked separately |
| ScanRefer teacher transfer | Complete512 fit rows;native491/461,teacher494/471,Hungarian511/499 | Zero updates;retain GT primary;rank and geometry separated;not student or unseen-scene evidence |
| ScanRefer joint readout | Complete2482 updates/arm and6887 terminal rows;integrity PASS, fixed REC screen FAIL | Joint-baseline -7/+20; joint-detached -5/-8; sealed with no formal promotion; full endpoint weights removed after audit per user |
| ScanRefer joint formal entry | Prepared but not launched | Fixed terminal REC screen failed; no9508 result from this trial |
| ScanRefer fixed endpoint audit | Actual weights,108 optimizer states/arm,fit traversal and6887 paired rows PASS | Metrics independently recomputed;106-room intervals diagnostic;before user-authorized deletion of failed full checkpoints |
| ScanRefer candidate local visual | Correct-mesh6887 baseline independently verified;originalPID42648 at64/2482 per arm23:02;new initial REC6684/6426 | Terminal estimate09-07 01:18-01:21;fixed budget continues;see20.98 |
| ScanRefer local visual official | Correct mesh v3 complete9508;independent audit PASS;local5549/4744 vs protected5570/4797 | REC promotion FAIL;ScanMask floors PASS;see20.97 |
| ScanRefer local visual endpoint audit | Old2482-step states audited before retirement;two failed weights deleted22:21 with logs/rows/proof retained | New mesh audit entry12CPUtests PASS;new trained endpoint pending;see20.97 |
| Weight storage cleanup | Complete14 explicit actions across two receipts;9.529GiB cumulative freed;10.10GiB free at22:21 | Ten duplicate copies share originals;four failed endpoints retired after audits;all logs/rows and six protected hashes retained |
| ScanRefer mesh post-training queue | Live waiting worker43358;CPU audit then fixed9508 formal launch after actual successful training exit | Firstcheck09-07 01:13:08 then240s;localobserver50584 at01:14:08;no Nr/Sr activation;see20.98 |
| Nr/Sr warm-start interface | CPU actual-weight strict loading and synthetic candidate filtering PASS | Weights-only Nr initialization needs fresh optimizer;butd_cls filtering differs from Scan metadata;no GPU/data/quality claim |
| Nr/Sr candidate-local native entry | Opt-in native factory/loader/optimizer added;55 original-environment checks PASS;real Nr weights load1154-state local model in both protocols | CPU integration only;fresh optimizer;zero native model updates/formal rows;Scan promotion and real-data GPU preflight still required |
| Nr/Sr native annotation/input audit | Actual CPU loaders:44909/77836 joint train rows;7899/17726 language val;B12 gives3742/6486 updates;32 fixed preflight inputs | No train/val physical-space overlap;0 point samples/GPU/updates/checkpoints;Scan formal and real GPU preflight remain pending |
| Nr/Sr native real-input CPU probe | Correct-mesh32 records/64 actual samples PASS20:57;all64 old/new pointSHA equal;input metadata boundaries recorded | GPU0/updates0/checkpoints0;v2 GPU entry prepared;Scan v3 REC failed so conditional GPU launch not executed;see20.97 |
| Sparse point memory | Complete6687 updates/arm;integrity PASS, fixed quality FAIL | Sparse-start Mask-20/-74,mIoU-0.4476098pp;no NrMask formal or continuation |

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


2026-09-06 03:38 CST authoritative update: C1 complete/controller0; independent row-level summary and actual artifact/optimizer checks PASS. Fixed quality FAIL. Both arms1024 updates, all6172 endpoint rows, REC/Text Mask/alpha frozen. No formal validation or protected result promotion. Object appearance native preflight still staged only. Master§20.64; goal active.


2026-09-06 03:45 CST authoritative update: object appearance native preflight complete/controller0. Zero-start exact, early Query/sampling/Text Mask/alpha invariant, last Query/Box/semantic/Query Mask paths connected and all5 addon REC gradients finite/nonzero.0 updates/formal rows; no quality claim. Matched object-attention training plan fixed, implementation pending. Master§20.65; goal active.


2026-09-06 04:06 CST authoritative update: object appearance paired v2 live after original-Py3.7 compile/7 tests/612 source/724 input checks. V1 preparation stopped on an arithmetic parameter-count error; actual six shared tensors333504 and appearance total374976 verified, no scope/budget change.04:03:39 PID24914,first-fit loss/shared gradients exactly equal, initial evaluation pending. Observer04:20:37 then240s. No terminal or formal result. Master§20.66; goal active.


2026-09-06 04:56 CST authoritative update: Sr protocol verified65846/17726 effective expressions and1018/255 scans;physical spaces490/123. Official train/test disjoint. Prospective scan-ID module grouping would overlap130 spaces;physical-space grouping0,actual Sr module split not created. Full training crop audit0 empty/2 zero-size objects;old FAIL retained. Zero-axis normalization fix passed7 original-server tests andactual2-scan152-slot probe with150 exact positive inputs and2 finite connected singleton features. Current Nr pair old module unchanged;baseline6172 identity PASS,both1024 updates finished04:53:15,terminal running. Master§20.67;goal active,formal results unchanged.


2026-09-06 05:19 CST authoritative update: O1 complete/controller0;independent complete6172 row summary and actual weights/AdamW verification PASS. Fixed quality FAIL. Same original data/output protocol;no Sr zero-axis code injected into running trial,no formal promotion. Master§20.68;goal active.


2026-09-06 05:28 CST authoritative update: sparse runtime v3 preparation PASS with unchanged original package inventory;v1 CA andv2 path-check failures retained. Pinned spconv2.3.6/cumm0.4.11 CPU imports and original-Torch GPU dense-reference/gradient/inverse-index checks PASS on80 synthetic voxels.0 dataset/native MCLN/optimizer,no actual sparse-memory training. O1 sealedFAIL and formal protected results unchanged. Master§20.69;goal active.


2026-09-06 05:59 CST authoritative update: actual sparse point memory native preflight PASS;3 CPU tests,16 fit scans/12 native forwards,zero-start exact andall17 tensors connected afterfixed perturbation.0 updates/holdout/formal. Same800000 sampled points preserved via2cm voxel inverse mapping;2.02GiB Torch allocator peak. Master§20.70;goal active.


2026-09-06 06:13 CST authoritative update: sparse point full-fit learning pair launched after8 tests/612 source/1026 data/actual parameter-count checks PASS. Manifestaa32b731f0e105a1dbd218a9f9edfb2c1919981340694577b9c12ed1edc99c05;native16 tensors1348960,sparse33 tensors1616896. Fixed1epoch6687 updates per arm,1e-5 shared/1e-4 new.06:11 PID27206 in data initialization;no first-fit/endpoint quality yet. Observer06:29 thenruntime estimate and240s nearcompletion. Master§20.71;goal active,formal protected results unchanged.


2026-09-06 06:42 CST authoritative update: V1 controller1 before any optimizer/holdout;native first-backward differences diagnosed from5 forwards and independent gradient tensors.3-forward native warmup regression PASS without relaxing exact GPU norms. V2 same scientific setup launched06:37:32 PID27756 after8 tests/612 source/1026 input checks;full-start preflight pending. Master§20.72;goal active.


2026-09-06 07:08 CST authoritative update: V2 controller1 still failed exact GPU gradient norms after warmup,0 updates/holdout. Full V2 initialization diagnostic launched06:54:01 PID28034,5 backwards/0 updates;next07:04. No new scientific endpoint or formal result. Master§20.73;goal active.


2026-09-06 07:19 CST authoritative update: full initialization5-backward diagnosis complete,0 updates;native self-repeat maxrelativeL2=1.939e-5. V3 usesrelativeL2<=1e-4,removes ineffective warmup;80 actual-gradient comparisonsPASS and16 one-percent perturbationsrejected. Same scientific setup launched07:17:29 PID28416 after8 tests;firstfullpreflightpending07:27. Master20.74;goalactive.


2026-09-06 07:29 CST authoritative update: V3 original full-start preflightPASS;07:27:31 PID28416 GPU10815MiB,baseline800/6172,0 updates. Actual maxsharedgradientrelativeL2=1.939e-5;old exactnormfalse retained. Exact source archive added after3 Git newline conversions. Master20.75;goalactive.


2026-09-06 07:58 CST authoritative update: V3 all6172 baseline rows exactly match protected reference and native/sparse;independent local replayPASS.07:53:40 both192/6687 updates;1.47644seconds per pair,initialterminalETA10:58,nextobserver10:53then240s. Formalresultsunchanged;master20.76;goalactive.


2026-09-06 08:18 CST: bounded official Sr3D checkpoint lookup found no usable URL; fixed README, live refs/releases and linked HF paper checked. No checkpoint restored or GPU action. V3 existing observer remains scheduled near10:53; last actual progress07:53:40 remains192/6687. Master20.77; goal active.


2026-09-06 08:39 CST: sparse conditional native formal entry implemented; protected/native/sparse on same7899 inputs only after recomputed terminal screen and artifact checks. OriginalPy3.7/Torch1.10 CPU19 tests +CLI PASS; synthetic only,no trained endpoint or formal/GPU run. Current training unchanged,nextobserver10:53. Master20.78;goalactive.


2026-09-06 09:27 CST: user scope changed: ScanRefer REC aim>59/>51, non-degradation floor5572/4797 on9508;Scan Mask baseline floors58.70/50.70/44.72. Nr/Sr only REC>=paper baseline;Mask excluded from gates. New training startsScan;advance toNr/Sr onceScanfloorpasses rather than waitingstretchgoals. FourScanweights CPU/hashcheckedPASS;noScantrainingyet. Master20.79.


2026-09-06 10:08 CST: ScanRefer pretrained readout online module544396params/42tensors implemented. CPU2 synthetic rows and checkpoint roundtripPASS,no native quality. Native16fit/B12+4 zero-update probe queued09:58 on GPU lock;session64873 observes11:03. Scoped same-capacity detached/joint one-fit-pass plan frozen before training;actual row manifest pending native probe. Master20.80;overallgoalactive.


2026-09-06 10:49 CST: Master20.81 records latest teacher-first feasibility preference. Frozen512fit teacher audit queued10:36,screen31455;native Scan selector disabled by authoritative CLI. Joint one-fit-pass trainer compiled10:45,not launched. Legacy NrMask receives no formal followup. Overall goal active.

2026-09-06 11:33 CST: Master20.82: old NrSparseMask sealed FAIL;Scan native V2 PASS,512teacher complete +3/+10 native but GT primary retained;paired Scan warm-start job launched11:29:54,2482 steps planned per arm. No new formal metrics. Goal active.

2026-09-06 12:23 CST: Master20.83: baseline6887 paired rows verified;12:15 both64/2482 on originalPID32521. Terminal estimated15:32,observer35333 scheduled15:27 then240s. Formal unified-checkpoint entry prepared/12CPUtests,not launched. Goal active.

2026-09-06 13:06 CST: Master20.84 records actual CPU Nr/Sr strict load and geometry-filter alignment on2 synthetic rows. No Nr/Sr training or current Scan changes;observer35333 remains scheduled15:27. Goal active.

2026-09-06 15:59 CST: Master20.86 seals the fixed Scan pair as quality FAIL after independent artifact/row audit. Master20.87 records requested weight cleanup;no new formal score or next training. Goal active.

2026-09-06 16:33 CST: Master20.88 records candidate-local visual implementation, CPU checks and live native preflight; no new formal score. Goal active.

2026-09-06 16:44 CST: Master20.89 records completed native preflight and verified ScanRefer local-visual job launch;first measured updates pending17:20 observation. Goal active.

2026-09-06 17:10 CST: Master20.90 records independent local-visual audit preparation only;actual training endpoint and9508 formal evaluation remain pending. Goal active.

2026-09-06 17:25 CST: Master20.91 records actual128-step Scan local-visual progress and independent6887-row initial parity. Terminal observer41352 scheduled19:24 then240s;no new formal metric. Goal active.

2026-09-06 17:52 CST: Master20.92 records native candidate-local loading/optimizer preparation;no Nr/Sr data forwards or training. Existing Scan job/schedule unchanged. Goal active.

2026-09-06 18:11 CST: Master20.93 records actual joint_det counts and32 fixed preflight input records;existing REC-only retention can bound future run weights;no new GPU result. Goal active.

2026-09-06 18:27 CST: Master20.94 records64 actual CPU point samples and prepared conditional native GPU preflight;Scan progress scheduled19:24, no new formal metrics. Goal active.

2026-09-06 19:46 CST: Master20.95 records actual negative development REC and audited endpoint;fixed9508 formal evaluation running. Protected formal results unchanged;goal active.

2026-09-06 20:42 CST: Master20.96 records actual fixed9508 Scan result and independent audit. Promotion FAIL;three-benchmark objective remains active.

2026-09-06 22:24 CST: Master20.97 records corrected-mesh9508 failure,actual GPU preflight PASS and same-setting Scan training repeat launch. Nr/Sr not launched;goal active.

2026-09-06 23:12 CST: Master20.98 records actual64-step Scan progress,independent6887 paired baseline PASS and verified live endpoint-audit/formal-launch queue. No new formal metric;goal active.
