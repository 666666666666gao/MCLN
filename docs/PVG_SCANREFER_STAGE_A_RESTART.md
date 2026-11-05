# ScanRefer Stage A Restart

## Current Continuation: 2026-11-05 22:28 UTC

The existing `scanrefer_qila_20261105` run is active in its original screen.
Epochs1-2 completed full9508 semantic validation at `5366/4423` and
`5398/4512`. The initial resumed callback exited after epoch2 because an
audit assertion incorrectly expected the initialization-only flag to
remain true; the epoch2 full-resume checkpoint correctly records false.
CPU audit passed own epoch2,1168 tensors,nonempty optimizer/scheduler,
LR groups2e-6/4e-6/2e-5,SHA
`fe2182575fc136004df5dee005058d19a627c817b1fbf48bc670db6c1804c5cf`.
The same run resumed from epoch2 with optimizer/scheduler restoration
confirmed, and began epoch3 without replaying completed epochs.

At22:28UTC it was active at epoch3 batch392/6081, GPU20707MiB/16%,
no recent errors. The epoch3 first optimizer witness was finite
(loss5.679346) with finite nonzero gradients/deltas and tensor updates
Reader10/10,Support10/10,Span14/14. Continue the full eight-epoch budget,
no early stopping and no duplicate trial.

This is still a legacy rank-loss run,not the V99 QILA objective. When it
finishes, audit all same-checkpoint full9508 semantic rows, then proceed
automatically to the plan's X0 control and to X1 only after its gates pass.
The user has authorized this progression without asking again.

## Live Monitoring: 2026-11-05 21:21 UTC

The same `pvg_scanrefer_qila_20261105` recovery remains active under
controller PID323406. At21:21:40UTC epoch2 reached4450/6081, GPU
22099MiB/18%, and no recent errors were reported. Its epoch2 first-update
witness is finite and shows Reader/Support/Span tensor updates10/10/14.
Full9508 epoch2 validation is still pending. Preserve the full eight-epoch
budget; do not stop, hot-change, or duplicate the run. This runner still
uses the legacy rank loss rather than the V99 QILA objective.

NR3D current full history remains best66=4255/3435 and latest97=4207/3369
over7899. SR3D remains best/latest38=11903/10012 over17726. CPU audits
passed for each retained best/latest checkpoint. Both official scratch
trainers are live and configurations are unchanged.

When this ScanRefer run completes naturally, audit its full semantic
history and proceed automatically through the plan's X0 control followed
by X1 only if the stated gates pass. Do not ask the user to authorize the
already requested continuation again.

## V99 Implementation Correction: 2026-11-05 21:06 UTC

The live `scanrefer_qila_20261105` process remains active and must not be
stopped, duplicated, or changed at runtime. At21:06:31UTC it was at epoch2
batch3368/6081 on GPU22095MiB/85%, with no recent errors. Static inspection
of the exact runner and imported sources found only the legacy
`bbf_quality_rank_loss_weight=0.3` path. The V99 QILA flags, IoU-utility
candidate distribution, same-class hard-negative term, IoU>0.5
unmatched-query no-object ignore, and 500-step warmup are not implemented
there. Treat this eight-epoch run as exploratory legacy rank-loss training,
not V99 QILA evidence. Let it finish its authorized budget and audit it;
do not overlap another ScanRefer run on the occupied GPU. Then implement
and remotely gate X0/X1 per the V99 plan before automatically launching the
next experiment. The user has authorized automatic continuation without
another approval request.

## QILA Resume Progress: 2026-11-05 20:55 UTC

The same user-authorized trial continues after its first runner completed
epoch1 validation and exited1 because it required a second-update witness
that was absent; its traceback confirms the missing
`pvground_second_update.json`. The epoch1 checkpoint passed CPU audit (own epoch,
1168 tensors, nonempty optimizer/scheduler), SHA
`27a367592e918b2defbd60d8dd0ce65f4a6e562a91d1b1944d2325f6986a7f87`.
Epoch1 full9508 last-semantic Top1 rates0.56437/0.46519 uniquely recover
5366/4423 hits; epoch2 validation is pending.

Recovery uses the same run and screen, resumed from epoch1 with
optimizer/scheduler restoration=true in the resume receipt. At20:55:01UTC it was alive at epoch2
batch2547/6081 on GPU0 (82% utilization). Its epoch2 batch6 witness records finite loss
3.204985 and Reader/Support/Span updates10/10/14 with finite nonzero
gradients/deltas. Recipe remains B2 initialized, batch8/accum6/seed0,
LR2e-6/4e-6/2e-5, QILA weight0.3, eight epochs/no early stopping. Do not
stop, duplicate, or alter the live recipe.

The generic checker had conflated attempt1 tracebacks with current errors.
It now reports historical error lines separately from errors in the latest
250 log lines and includes the resume receipt/recovery exit code. Current
check at20:40:30UTC showed no recent errors, active recovery workers/GPU,
and the original `exit_code=1` only; recovery exit receipt is not yet
written. This QILA trial is not claimed to be an exact document X1-vs-X0
comparison. Its recipe updates Reader/Support/Span, so it also does not
meet the plan's QILA-only-through-Reader isolation. Do not alter a live
run; after its natural completion, verify all same-checkpoint full9508
semantic rows and CPU audits, then proceed one-at-a-time only through
the plan's explicit preflight and promotion gates.

## QILA Follow-Up Launch: 2026-11-05 18:15 UTC

The completed BBF-quality adapter run is not extended. A separate
ScanRefer-only QILA-style IoU-aware BBF ranking trial is running from
immutable B2 epoch2 with model-only initialization and fresh
optimizer/scheduler. It uses 8 epochs, no early stopping, batch8,
accumulation6, seed0, learning rates `2e-6/4e-6/2e-5`, and ranking-loss
weight `0.3`. The full metric remains same-checkpoint
`last_semantic_alignment_top1/bbf` over 9508 samples.

Trial `scanrefer_qila_20261105` runs in screen
`pvg_scanrefer_qila_20261105`. The first attempt failed before training
with exit code2 because `torch.distributed.run` rejected `--use_env`; the
failure was retained and the same run restarted without that flag. At
18:23:11UTC the first real optimizer witness passed with finite loss
7.640077, Reader/Support/Span updates of10/10/14 tensors, and no error
lines. The run was at epoch1 batch385/6081; full validation is pending.

## BBF-Quality Completed: 2026-11-05 14:34 UTC

Run1793849982 completed all10 epochs without early stop or manual
cancellation. Direct run/complete.json records14:34:17.987751UTC,
reason full_budget_completed,complete_epochs=10,final_target_met=false;
exit_code0,screen/workers absent,no errors. No extension/replacement.

| Adapter Epoch | Hits025 | Hits050 | Acc025 (%) | Acc050 (%) |
| --- | ---: | ---: | ---: | ---: |
| 9 | 5281 | 4353 | 55.542701 | 45.782499 |
| 10 | 5276 | 4351 | 55.490114 | 45.761464 |

All10 full9508 semantic Top1 pairs agree with complete logs. Trialbest1
5383/4518 remains100/138hits below immutable globalbestB2=5483/4656.
No target achievement or causal improvement claim. Independent ownCPU
retained1/10 audits passed1169/optimizer/scheduler/config/SHA,CUDAoff.
Latest10 SHA89bfa7cac0e26083c967600be4ebcfaad053c1ea25cfa38b8ed61b637ad175b9.
Retention kept1/10,removed2-9;full CSV10 rows and audits retained.
Receipts:scanrefer-bbf-quality-{full-history,complete,progress-final}-
20261105T1437.json in results. Monitoring completion-path correction
does not change training code. No new run launched.

## Current BBF-Quality Update: 2026-11-05 12:28 UTC

Same trainer294910 live epoch9=326/6081 at12:27:24UTC,GPU59%,finite
first/second adapter updates1/1,frozen parameters/buffers unchanged,
no errors or completion. Preserve10 epochs/no early stop. Full9508
semantic history1-8:new7=5248/4319,8=5270/4336. Trialbest1/globalbest
B2 unchanged;8 is213/320hits belowB2. Own CPU retained1/8 audits
passed1169 tensors,optimizer/scheduler/config/SHA,CUDAoff. Epoch8 SHA
07f5c5347d9a1ce70cafee6f8238adae61640633fd03e90a90c33f3bd269af2b.
Existing retention removed2-7,history retained,no manual cleanup.
Receipt:scanrefer-bbf-quality-full-history-20261105T1227.json in results.
Estimated full9 around13:29UTC/full10 around14:34UTC;verify actual
complete evidence,do not stop based on an ETA or intermediate scores.

## Current BBF-Quality Update: 2026-11-05 10:35 UTC

Same trainer294910 live in epoch7=2031/6081 at10:32:39UTC,GPU75%,
finite first/second actual adapter updates,frozen parameters/buffers
unchanged,no errors. Continue authorized10 epochs without early stop.
New full9508 last semantic Top1:epoch5=5216/4284,6=5238/4307
(55.090450%/45.298696%). All6 pairs match complete semantic log lines.
Trialbest1/globalbest B2 unchanged;epoch6 remains245/349 hits belowB2.
Independent CPU own1/6 audits passed1169 tensors,nonempty optimizer/
scheduler,config/SHA,CUDAoff. Epoch6 SHA
c417a7efdaf475f0981b8f8de7ff3609fcf5f54d02a16710af4d9c8eacd6a4c3.
Existing retention kept1/6,removed2-5;historical metrics/audits remain.
No manual cleanup or running recipe change. Receipt:
docs/results/mcln_pvground_20261101/scanrefer-bbf-quality-full-history-20261105T1034.json.
Separate all-epoch CSV now6 rows. No causal claim follows from these
intermediate below-best results;do not use overall/position/mask.

## Current BBF-Quality Update: 2026-11-05 08:32 UTC

Same trainer294910 and screen remain live at08:29:56UTC,epoch5=2848/6081,
GPU57%,finite epoch5 first/second update losses and actual adapter1/1
updates,frozen parameters/buffers unchanged,no error lines. No stop,
restart,duplicate launch or recipe change. Keep the ten-epoch budget
with no early stopping as explicitly requested by the user.

Full9508 last semantic alignment Top1/BBF:

| Adapter Epoch | Hits025 | Hits050 | Acc025 (%) | Acc050 (%) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 5383 | 4518 | 56.615482 | 47.517880 |
| 2 | 5150 | 4203 | 54.164914 | 44.204880 |
| 3 | 5136 | 4196 | 54.017669 | 44.131258 |
| 4 | 5180 | 4254 | 54.480438 | 44.741271 |

All history pairs agree with complete semantic log lines to printed
precision. Epoch4 gained44/58 hits over3 but trails B2 by303/402;
best in this trial remains1 and global best remains B2=5483/4656.
The log's epoch4 overall50=0.5011569205 is not semantic Top1=0.44741;
do not use overall/position/mask to claim a semantic improvement.
This is an observed semantic regression,not a diagnosed root cause.

At08:31UTC independent CPU audits verified retained best1/latest4:
own epochs,1169 tensors,nonempty optimizer/scheduler,adapter-only config
and SHA matching history,CUDA uninitialized. Epoch4 SHA
00f0c6da0128b76b46344d8d4812a8de00181491e58010b03645362fea2b8593.
Epochs2/3 have been removed by existing authorized run retention; keep
their full metrics and runtime audit history. No manual cleanup.
Receipt:docs/results/mcln_pvground_20261101/scanrefer-bbf-quality-full-history-20261105T0831.json.
CSV:docs/results/mcln_pvground_20261101/scanrefer-bbf-quality-all-epochs.csv.
Collector:tools/collect_current_scanrefer_bbf_quality_history.sh,
executed remotely only through work/remote_pvground.ps1.

## Current BBF-Quality Fine-Tuning: 2026-11-05 06:20 UTC

Current-only check at06:19UTC confirmed the same trainer PID294910 and
screen live, epoch3 at2849/6081, GPU92%, finite update witnesses and no
recent error lines. Do not stop or relaunch. Training progress is not an
accuracy result; epoch2 remains latest complete audited validation.

The user explicitly requested that ScanRefer fine-tuning continue and not
be stopped. Do not cancel the active run or start a duplicate. The
isolated `scanrefer_bbf_quality_20261104` trial remains live in screen
`pvg_scanrefer_bbf_quality_20261104`, trainer PID294910. At06:03UTC it
was training epoch3 at523/6081; GPU compute was active, recent logged
losses finite, and no recent traceback/OOM was observed.

Epoch2 completed full9508 last semantic alignment Top1 validation:
5150/4203 hits025/050 =54.1649%/44.2049%. Compared with epoch1
5383/4518,this is-233/-315 hits; compared with immutable B2
5483/4656,this is-333/-453. The global best remains B2; within this
adapter trial epoch1 remains best. Epoch2 own CPU checkpoint audit
passed: own epoch2,1169 model tensors, nonempty optimizer/scheduler,
expected adapter-only configuration, SHA
`b6b4f2fd4d34fce820bad785094d8b92ae009858c1f82994d67539572859da10`,
CUDA uninitialized. Raw audit:
`docs/results/mcln_pvground_20261101/scanrefer-bbf-quality-epoch2-cpu-audit-20261105T0103.json`.

This is a ten-epoch run with no early stopping. Do not treat a lower
intermediate epoch as a stop condition; continue collecting complete
epochs, same-checkpoint metrics and CPU checkpoint audits. The target
5705/4945 hits is not met. Preserve immutable B2, training log, launch
failure receipt, update witnesses, all complete metric rows and audits.
Local machine performed no model import, test, diagnosis, forward,
evaluation or training; remote work used `work/remote_pvground.ps1`.

## Previous Update: 2026-11-05 05:10 UTC

The user explicitly requested that ScanRefer fine-tuning continue and not
be stopped. The isolated trial `scanrefer_bbf_quality_20261104` is active
in screen `pvg_scanrefer_bbf_quality_20261104`; do not stop it or launch a
duplicate. This direct request supersedes the earlier heartbeat's
prohibition on automatically opening a new Scan run for this registered
trial only.

Attempt1 exited1 before training because the legacy distributed launcher
prepended `--local-rank=0` to positional arguments. The runner therefore
failed its run-directory assertion before loading data or using the GPU.
The failure is retained in `launch_attempt_1.json` and
`exit_code_attempt_1`. Launcher now passes `--use_env`; attempt2 began at
03:39:36UTC using the same isolated run directory. B2 SHA
`46ecfb6a6831fca0704c55fe87fa96c9c92e749824f13173d67d5697da1b92f8`
is unchanged.

At05:10:03UTC PID294910 and four loader workers were active, GPU
4933MiB/100%, epoch2 2407/6081. Epoch1 completed full training and
validation. The latest log had finite training losses and no traceback/OOM;
first two actual optimizer-step receipts pass:
batch6 loss8.10970497, adapter gradient1.87043905/delta
0.00271401; batch12 loss8.73804569, gradient2.05502987/delta
0.00246970. Both report finite loss,1/1 adapter tensor updated and frozen
parameters/buffers unchanged. Full epoch validation is pending; training
progress counters are not accuracy metrics.

Run contract: ten epochs, B2 model-only initialization, fresh
optimizer/scheduler, seed0, batch8/accum6; train only the zero-initialized
adapter on native BBF semantic ranking (adapter LR2e-5, rank weight1.0),
native LR2e-6/backbone4e-6, remaining model frozen. No early stop.
Preserve B2 and the failed-launch receipt. Local machine did not execute
model code/tests; all remote activity used `work/remote_pvground.ps1`.
Current progress observation: `2026-11-05T05:10:03Z`, from the
current-run-only checker. Epoch1 was audited at05:00:56UTC using its own
CPU checkpoint: own epoch1,1169 tensors, nonempty optimizer/scheduler,
checkpoint SHA matched the metric row, CUDA uninitialized. The earlier
detailed progress receipt is
`docs/results/mcln_pvground_20261101/scanrefer-bbf-quality-progress-20261105T0355.json`.

Epoch1 full9508 last semantic alignment Top1/BBF is5383/4518 hits
(Acc025/Acc050=56.6155%/47.5179%). Compared with immutable B2
5483/4656,this is-100/-138 hits. This is the first of10 planned epochs;
the run has no early-stop policy and remains active. B2 remains global
best under Acc050-first selection. CPU audit receipt:
`scanrefer_bbf_quality_20261104/run/scanrefer/scanrefer_bbf_quality_20261104/1793849982/epoch1_cpu_audit.json`.

## Current Complete-Validation Bests: 2026-11-05 04:34 UTC

All counts are last semantic alignment Top1/BBF on the full split, with
both thresholds taken from the same selected checkpoint. ScanRefer B2
epoch2 is5483/4656 over9508 (57.6672%/48.9693%); this remains the global
best. Adapter epoch1=5383/4518 is a complete but lower first-round result.
NR3D epoch66 is
4255/3435 over7899 (53.8676%/43.4865%), latest epoch81 is4238/3397.
SR3D epoch15 is11743/9434 over17726 (66.2473%/53.2213%), latest epoch30
is11584/9434. Selection priority: Scan Acc050, NR/SR Acc025. Target
shortfalls are Scan222/289 hits, NR469 Acc025 hits and SR559 Acc025
hits; none is met. Current-only NR/SR collectors confirmed their best and
latest CPU checkpoints have matching own epochs,1134 tensors,nonempty
optimizer/scheduler and verified SHA; trainers remain live.

## Current Override: 2026-11-04 12:13 UTC

The registered Span-only pipeline completed both full9508 conditions and
exited0. Matched control weight0=5470/4568; treatment weight0.1=5483/4608
(57.667228%/48.465924%), a +13/+40-hit change versus control. Against
immutable B2=5483/4656, treatment is0/-48 hits; B2 remains global best
under Scan Acc@0.50-first selection. CPU own-epoch1/1168 tensor/
nonempty optimizer/scheduler/SHA audits passed for both conditions.
All1154 frozen non-Span tensors equal B2;14 Span tensors changed.
Treatment is single-seed/single-epoch exploratory evidence only, neither
a target achievement nor robust causal/core-innovation claim. SHA and raw
receipts are in `PVG_SPAN_ONLY_COMPARISON.md` and results archive.

NR current best64=4190/3394, latest65=4095/3254 over7899;
SR best15=11743/9434, latest24=11421/9200 over17726. Exact current-run
CPU checkpoint audits passed; trainers remained live. Original NR/SR
scratch configs and complete budgets remain unchanged. All three targets
remain unmet.

## Current Override: 2026-11-04 11:35 UTC

Treatment3419/6081 at11:34:40UTC,controller/launcher/GPU live,34 parsed
train losses finite,last6.9048/6.9039/6.9074,no errors. Actual optimizer
steps1/2 passed with Span14/14 updates and frozen state unchanged;full9508
score pending. No duplicate launch. Immutable B2 remains best5483/4656
(gaps222/289 to targets5705/4945).

Treatment actual11:09:23UTC step1/2 passed:finite loss6.557239/7.686158,
Span14/14 changed,finite gradients/deltas,frozen state unchanged.
At11:14:29UTC it reached1100/6081,train-only11 losses finite,last
7.0651/7.0369/7.0132,controller/launcher/GPU live. Roughly44min train
then9508 validation. See progress receipts;no duplicate run.

Span-only control full9508=5470/4568,completed10:58:21UTC. CPU own
checkpoint/optimizer/scheduler/SHA and1154 frozen tensors equal B2 passed.
Treatment0.1 automatically started10:58:35UTC under existing controller,
launcher268814/GPU268817live11:04:45,own GPU memory1168MiB. It completed
independent model-only B2 load with fresh optimizer/scheduler,but no first-
update witness yet;real optimization not verified at that time.
See PVG_SPAN_ONLY_COMPARISON.md. B2 best5483/4656 remains unchanged.
No duplicate launch,original NR/SR budgets unchanged.

## Previous Override: 2026-11-04 10:00 UTC

Historical Stage A/B and Reader trials below are complete. A new registered
Span-only matched comparison is live in screen pvg_scanrefer_span_only;
see PVG_SPAN_ONLY_COMPARISON.md. Controller265532 owns both conditions.
Control started09:49:09UTC and reached531/6081 at10:00:03. Its two actual
optimizer steps passed finite loss,14/14 Span updates and unchanged frozen
parameters/buffers. Full validation is pending;do not duplicate launch.
B2 remains best5483/4656. Treatment is independent B2 initialization with
projection weight0.1 versus matched control0;only existing Span is trained.
NR/SR continue original scratch budgets. No local ML execution.

## Previous Override: 2026-11-04 05:27 UTC

Stage A, Stage B, fixed-B2 diagnostics, and the registered Span pilot are
complete. The new isolated Reader competitor run `1793763075` also finished
one full augmented epoch and all9508 validation examples. Its same-checkpoint
last semantic alignment Top1 is5471/4608 hits (57.541018%/48.464451%).
Remote CPU audit passed own epoch1,1169 model tensors, nonempty optimizer
and scheduler, CUDA not initialized, SHA256
`67cef44cd6ec253806eced03b1ca79f115d13d4e681c9da69730ca767b0fda35`.

Reader is -6/-12 hits versus the matched control5477/4620 and -12/-48
versus immutable B25483/4656. This single-seed, single-epoch trial does not
support an accuracy benefit from the candidate-relative residual. It does
not identify per-example causes or establish the learned scale as the cause.
Do not promote it as best. Detailed experiment note, CSV and raw result
receipt are archived separately. No Scan trainer is active; do not start
another run based only on this result.

## Current Override: 2026-11-03 22:02 UTC

Stage B run1793674354 is complete and must not be restarted. The subsequent
fixed-B2 four-condition diagnostics and registered Span supervision pilot
are also complete; no Scan trainer is currently active. Do not launch a new
Scan run until the pilot is interpreted or the user explicitly chooses a
repeat.

Fixed-B2 full9508 conditions: none5483/4656, Reader bypass5504/4673,
Support bypass5475/4616, Span bypass5407/4262. These are inference
interventions, not independently trained module ablations. The separate
Span pilot control weight0=5477/4620 and treatment weight0.1=5484/4654;
versus control it gains7/34 hits, but versus strong B2 it is+1/-2.
The single-seed/epoch treatment is exploratory, not robust causal evidence.
See PVG_FORWARD_AUDIT.md and PVG_SPAN_GATE_COMPARISON.md for receipts.

The original NR/SR scratch runs continue independently and unchanged.
Their latest complete metrics, live status and best/latest SHA audits are
in the 2026-11-03 22:02 Current Override of the handoff and the current
full-history receipts. Do not run historical Scan collectors or inspect
stopped runs.

## Active Independent Pilot: 2026-11-03 17:38 UTC

CompletedB remains immutable,not restarted. New controlrun1793726936
under controller236004 has own finite first-update loss7.0552873611450195,
all34module tensors updated;actualGPUworker236606,220/6081.
FreshB2 model-only,weight0,max1;next condition independently reloadsB2
with weight0.1. Full9508results pending,control ETA19:10-19:25UTC.
See PVG_SPAN_GATE_COMPARISON.md for receipts and registered protocol.
Fixed-B2 all-four diagnostics completed and new paired subgroup analysis
is in PVG_FORWARD_AUDIT.md. NR/SR original scratch budgets unchanged.

## Next Minimal Comparison: 2026-11-03 17:16 UTC

StageB remains finished;B2 is an immutable common initialization.
Fixed-B2 four-condition diagnostics are finishing separately.
A minimal raw-Span gate auxiliary loss is prepared in a new source snapshot,
with six remote CPU tests passed. No local tests or new GPU training yet.
Protocol:one full48655-entry epoch per condition,full9508 validation,
common B2/fresh optimizer/scheduler/seed0/Brecipe,weights0 versus0.1.
First run a separate6-microbatch treatment preflight and discard its state.
See PVG_SPAN_GATE_COMPARISON.md;no Reader redesign/soft extent added.
NR/SR original scratch240/140 budgets and null initialization remain.

## Separate B2 Diagnostic: 2026-11-03 16:33 UTC

Stage B remains completed and must not be restarted. Its selected B2 is
now an immutable input to a separate eval-only diagnostic in
offline_audits/b2_fixed_audit_20261103_lrpositive.
Runner232612/evaluator232624 in pvg_scanrefer_b2_fixed_audit started16:30:52UTC;
360 real forward and semantic records verified16:32:47UTC. No full
diagnostic result yet. Initial zero-LR invocation failed before forward
and is archived separately; corrected command uses B's positive optimizer
group arguments with eval-only control flow. No source/trainer change.
See PVG_FORWARD_AUDIT.md for provenance and interpretation constraints.
NR/SR original scratch configurations and240/140 budgets remain unchanged.

## Stage B Finished: 2026-11-03 15:44 UTC

Stage B run1793674354 completed four full9508 validations and stopped
under its existing two-consecutive-Acc050-decreases policy.
The terminal receipt says reason=two_consecutive_acc050_decreases;
both stage_b_pipeline_exit_code and transition_exit_code are0.
Trainer207977 is absent,the controller screen has exited,and the Scan
GPU is free(0MiB/0% at15:40-15:44UTC). This is a verified normal early
stop,not a failure or merely a missing screen. Do not restart or keep
polling this finished run. NR/SR scratch training continues independently.

| Stage B Epoch | Examples | Acc025 Hits / Percent | Acc050 Hits / Percent |
| --- | ---: | ---: | ---: |
| 1 | 9508 | 5474 / 57.572570% | 4653 / 48.937737% |
| 2 selected | 9508 | 5483 / 57.667228% | 4656 / 48.969289% |
| 3 | 9508 | 5473 / 57.562053% | 4654 / 48.948254% |
| 4 latest | 9508 | 5485 / 57.688263% | 4623 / 48.622213% |

All pairs are same-checkpoint last semantic alignment Top1/bbf.
Acc050 sequence4656->4654->4623 gives strict-decrease streak2;
stop_requested=true,final_target_met=false.
B4 versusB3:+12/-31hits. Acc050-first selection keepsB2 despite
B4's two additional025hits;do not combine B4's025 withB2's050.
Best remains222/289hits short of5705/4945. No target or paper result
is claimed,and these results do not isolate any single module's effect.

Remote CPU final audit15:44:21UTC crosschecked all four full-log rows
against the evaluator history and checked actual B2/B4 own epochs,
1168model tensors,nonempty optimizer,scheduler and SHA256;CUDAoff.
B2 SHA46ecfb6a6831fca0704c55fe87fa96c9c92e749824f13173d67d5697da1b92f8.
B4 SHAe493862b4dbf1d1f97860276fe2a1871decb4367dcc2892279daa8cccfa9c431,
795607307bytes. Both are resumable saved checkpoints.
Retention15:07:20UTC independently audited2/4 and removed only olderB3;
free11485032448->12280643584bytes. B1/B3 CSV paths are marked deleted;
all11 restart rows,logs/configs/full audit history and official/A3
initialization inputs remain. No extra cleanup or training was started.
The existing retention watcher remains idle;it is not an active trainer.

Raw final receipt:
docs/results/mcln_pvground_20261101/scanrefer-stage-b-final-full-history-20261103T1544.json.
It includes both terminal exit codes and the completion reason.
One SSH collection attempt closed before a result;the later successful
CPU collection and downloaded final receipt resolved that interruption.
No local tests/model imports/forward/evaluation/training were run.
Prepared fixed-checkpoint bypass diagnostics remain unexecuted;the freed
GPU and completed run alone do not establish a cause for the decline.

## Stage B Epoch 3: 2026-11-03 13:50 UTC

B3 completed all9508 examples:5473/4654 hits,
Acc025=57.562053%,Acc050=48.948254%,same-checkpoint last semantic
alignment Top1/bbf. AgainstB2 it decreased10/2 hits.
The existing strict consecutive Acc050-decrease streak is1;
stop_requested=false. B4 continues the same recipe; if B4 is strictly
below4654 Acc050 hits,the existing two-decline policy requests a stop.
Ties or increases reset the streak. B1 remains the comparison baseline.

Selected best remainsB2=5483/4656,57.667228%/48.969289%;
targets5705/4945 still require222/289 hits. No target is met.
Remote CPU collection13:45:44UTC crosschecked all B1-3 full-log counters
against stage_b_semantic_history.json and audited actual B2/B3 weights:
own epochs,1168 tensors,nonempty optimizer,scheduler and SHA256.
B3:795607307bytes,SHA
f23c416f9019abc4765015dd2a0137c03c9c6c98dbc0f55593f2d28e9d3e0430.
Retention12:07:13UTC kept best2/latest3 without deleting another weight.
B1 was removed by the previous B2 retention;the restart CSV now marks
its deleted path. All10 restart rows and historical audit evidence remain.

At13:49:10UTC B trainer207977 was live at3696/6081 for epoch4;
latest logged total loss8.6958 finite,no error lines. Full B4 validation
ETA approximately15:05-15:15UTC,not a completed result.
No recipe or module change was made;no local tests or model operations ran.
The archived B4 first-update receipt identifies epoch4/batch6/step1,
finite loss8.481688499450684,Reader9/10,Support10/10,Span14/14 tensors
actually updated,with finite nonzero gradient and parameter-delta norms
for all three modules. Do not claim all34 updated;this observation alone
does not diagnose the unchanged Reader tensor as a defect.
Raw receipt:docs/results/mcln_pvground_20261101/scanrefer-stage-b-full-history-20261103T1345.json.
Collector:tools/collect_current_full_history.sh.
This aggregate epoch comparison does not isolate individual module gains.

## Stage B Epoch 2

StageB epoch2 completed full9508 at2026-11-03 08:59:14UTC:
last semantic alignment Top1/bbf=5483/4656hits,
Acc025=57.667228%,Acc050=48.969288%,same checkpoint.
VersusB1 this is+9/+3hits; the two-strict-consecutive-Acc050-decrease
streak remains0 and training continues. Target5705/4945 remains
222/289hits away; final_target_met=false.

At09:41:58UTC the remote CPU collector matched history counters against
the complete training log and actual checkpoint. Own epoch2/1168model
tensors/nonemptyoptimizer/scheduler and the three optimizer LR groups
were verified;CUDAoff. Weight795607307bytes,SHA256:
46ecfb6a6831fca0704c55fe87fa96c9c92e749824f13173d67d5697da1b92f8.
Best manifest selects2. Retention09:03:02UTC audited2/SHA/resume states,
removed only olderB1,and retained2asbest/latest.
Free bytes12284051456->13079662592.
Raw receipt:`docs/results/mcln_pvground_20261101/scanrefer-stage-b-epoch2-audited.json`.
Retention receipt:`docs/results/mcln_pvground_20261101/scanrefer-best-latest-retention-stage-b-epoch2.json`.
Remote collector:`tools/collect_scanrefer_stage_b_epoch2.sh`.
Restart table now9rows;B1/B2 raw results remain distinct.

At09:43:45UTC actual B trainer207977 was live on epoch3 at1555/6081;
its overwritten first-update receipt identifies epoch3,finite loss
6.407628059387207,with nonzero gradients and updates across all three modules.
Current config remains A3 model-only initialization/fresh optimizer and
scheduler/frozen text encoder,batch8/accum6 and original three LR groups.
This is training health,not a complete epoch3 validation.

## First Complete Stage B Validation

Stage B epoch1 completed all9508 examples at2026-11-03 06:01:50UTC.
Same-checkpoint last semantic alignment Top1/bbf:5474/4653hits,
Acc025=57.572570%,Acc050=48.937737%. Relative to matched selected A3
5461/4566,this is+13/+87hits (+0.136727/+0.915019 percentage points).
Relative to matched official5435/4327,it is+39/+326hits.
The original Stage A training validation4565hits050 remains a separate
measurement; it is not substituted for the matched4566 comparison.
This aggregate comparison does not isolate each module's contribution.

Targets5705/4945 remain231/292hits away; neither final target is met.
The first B validation establishes the early-stop baseline:
decrease_streak=0,stop_requested=false. Subsequent B epochs continue
the existing two-strict-consecutive050-decrease policy and10epoch budget.
No recipe/architecture/module change was made.

At06:04:19UTC a remote CPU collector crosschecked the training evaluator's
9508counters against complete semantic log rows and the actual checkpoint.
Own epoch1/1168model tensors/nonemptyoptimizer/scheduler were verified.
Saved optimizer groups are native2e-6/backbone4e-6/allthree modules2e-5;
batch8accum6 and model-onlyA3 initialization were checked.
Checkpoint795607307bytes,SHA256:
`cda46832c35f95fb70710da23768ea28eb27f8130bb6ae787de2665f0cf92f48`.
The semantic-best manifest selectsB1; its best/latest retention audit
at06:02:53UTC independently verified the same SHA and resume states.
No weight was removed. A3 remains the initialization input.

Raw receipt:`docs/results/mcln_pvground_20261101/scanrefer-stage-b-epoch1-audited.json`.
Collector:`tools/collect_scanrefer_stage_b_epoch1.sh`.
Restart table now8rows,including separate official/identity/matched/A/B phases.
CPU collection used no model import/construction/forward/evaluation,
CUDA remained uninitialized,and no local tests ran.
ActualBtrainer207977/retention209444 were live06:04:56UTC,epoch2=104/6081.
B2 own first update atbatch6 had finite loss7.124665260314941;
all34module tensors had nonzero finite gradients and actual updates.
The per-epoch first-update filename is overwritten; do not relabel that
epoch2 receipt as the earlier epoch1 first update.

## Complete Stage A Native Preservation

At2026-11-03 04:48:27UTC a remote CPU audit compared the selected
complete Stage A epoch3 checkpoint against official epoch54. Both
checkpoint SHA256 values were checked before loading. The1134 native
serialized tensors match exactly in shape,dtype and value; this includes
all204 native BatchNorm running means,variances and batch counters.
There are34 additional tensors,all within the three module prefixes.
No native tensor changed after the full three-epoch Stage A budget.

Official SHA:a9930065996fce1d0dd5ee9fe00a120bdb3a2c88d158b7a3666717d842ac113d.
Selected SHA:f4e98cce830eb474a94402df00dbd160c379b963e7d285c759b86c4a60b0a898.
Raw receipt:`docs/results/mcln_pvground_20261101/scanrefer-stage-a-native-preservation.json`.
Reproduction helper:`tools/audit_scanrefer_stage_a_native_preservation.sh`.
This strengthens the first-update/fixture freeze checks with actual final
checkpoint evidence. It does not verify Stage B freezing (B deliberately
unfreezes native parameters),attribute gains to an individual module,or
prove geometric data quality. No model was imported,constructed or run;
CUDA remained uninitialized and no local tests ran.

## Sustained Training Log Check

At2026-11-03 03:39:49UTC Btrainer207977 was live,1477/6081batches,
GPU21989MiB. The same PID had reached1315 at03:35:22UTC; the recheck
was scheduled240seconds later. Fourteen cumulative training total-loss
records are finite; the last three,atbatches1200/1300/1400,are
7.4475/7.4360/7.4088. No traceback/OOM/RuntimeError appeared in the
inspected log tail. Initialization/configuration remain A3model-only,
freshoptimizer/scheduler,batch8accum6/native2e-6/backbone4e-6/modules2e-5.
This is continuing-process/log evidence,not a complete Bvalidation or
proof that every individual batch/optimizer update was instrumented.

`tools/check_current_training_loss.sh` reads at most the last2MiB of each
exact current-run log,checks the actual known PID/config/GPU,and pairs
each total loss with a `Train:` header. The initial draft accidentally
associated `Eval:` losses with the last training position; the verified
version clears that association at `Eval:` and consumes one training-loss
line per `Train:` header. On real NRlogs,this removes a validation loss
24.4308 incorrectly attributed toepoch33/batch3500; its actual training
average is11.0797. This was a monitoring-label error,not a trainer defect.
NR/SR corrected remote snapshots at03:37:12UTC contain59/54finite
training records respectively; actual trainers96960/92354 remain live,
with unchanged null task/pp initialization and original240/140budgets.
No model code,training configuration,local test,model import or forward
was executed by this check.

Raw receipts are `scanrefer-stage-b-training-loss-health-20261103T0339.json`,
`nr3d-training-loss-health-20261103T0337.json` and
`sr3d-training-loss-health-20261103T0337.json` under the current results folder.

## Matched Validation Completed

At 2026-11-03 02:53 UTC the controller had completed both 9,508-row
evaluations and started Stage B run `1793674354` loading data.
Official identity: 5,435/4,327 hits. Selected A3: 5,461/4,566 hits,
57.435844%/48.022718%. Selected minus official: +26/+239 hits.
Matched official minus original E0: 0/0 hits. Both receipts agree on point
and identity hashes, seed, backend and innovations; selected checkpoint SHA
matches the audited A3. Original gate and E0 records were not changed.

A3 training-time validation was 5,461/4,565, one fewer Acc050 hit than this
reevaluation despite the same selected weight. Preserve both evaluations
as separate rows; do not claim bitwise repeatability or replace the old row.
The matched comparison establishes the aggregate effect of this trained
three-module system, not separate Reader/Support/Span causal contributions.
Final target remains unmet, short by 244/379 hits in this matched evaluation.

Stage B start/config receipts show A3 model-only initialization and fresh
optimizer/scheduler, batch8/accum6, native2e-6/backbone4e-6/modules2e-5.
By03:01UTC, actual trainer PID207977 had advanced to66/6081 batches,
GPU21951MiB. B's own first-update receipt verifies epoch1/batch6/size8/
optimizer-step1, finite loss7.021632194519043. Reader, Support and Span
gradient norms are0.113742/0.346667/0.134784; parameter delta L2 values
are0.006194/0.003311/0.003350, updating10/10,10/10,14/14tensors.
This verifies real B training; it is not a complete B validation result.
First full B validation is estimated around05:50UTC, subject to throughput.
At03:02UTC A1/A2 were deleted under the user's authorization; A3 remains.
The B retention watcher PID209444 is live, waiting for complete validation.
Raw `scanrefer-stage-b-first-update.json` and
`scanrefer-stage-a-weight-cleanup.json` retain the evidence.
Raw `scanrefer-matched-validation.json`; restart CSV now seven rows.

The Stage B monitor now queries the actual transition and retention screens,
the trainer/worker PIDs, and the retention Python process. The previous
`pvg_scanrefer_stage_b` screen query did not match this controller-owned
launch. Its start summary reports the selected epoch/SHA and fresh optimizer/
scheduler directly, instead of repeating the historical nested Stage A
decision containing `stage_b_started=false`. That nested value is a startup
snapshot, not live process state. The revised helper was executed remotely
at03:11:29UTC: trainer207977 and retention209444 live,463/6081 batches,
no complete B validation. No model code or training settings changed.

2026-11-03 02:29UTC:StageA completed02:23:49UTC/exit0.
Allthreefull9508epochs audited by controller's remoteCPUcollector:
A1=5446/4543,A2=5460/4563,A3=5461/4565.
A3=57.435844%/48.012200%,+1/+2hitsversusA2;selected050then025.
Owncheckpoint3/1168tensors/optimizer/scheduler verified,CUDAoff.
SelectedinitializationSHA
f4e98cce830eb474a94402df00dbd160c379b963e7d285c759b86c4a60b0a898.
Original5425/4328gatepassed;finaltargetdeficits244/380hits.
ActualmatchedevaluationPID205803live02:29:40UTCundercontroller204119.
Matchedpairnotcomplete;Bnotstarted. No modulecausalityclaim from
originalE0delta,which crosses the documentedbackendboundary.
Raw scanrefer-stage-a-complete-epochs-through3.json;restartCSV5rows.

Latest user scope: restart ScanRefer fine-tuning after reading the six
documents in `C:/Users/gb/Desktop/document/MCLN plan/`; continue NR3D and
SR3D original full reproduction runs unchanged. The documents' suggestions
to replace their scratch initialization are outside that scope.
All tests and model execution run remotely. No local Windows/WSL tests.

## Guarded Transition

At2026-11-03 01:39:12UTC,screen`pvg_scanrefer_restart_transition`
started the sequential controller;actualbashPID204119 and current
StageA trainer189152 were verified live at01:39:51UTC.
This is waiting-process evidence,not evidence of StageB training.
At01:44:26UTC the same controller/trainer were live and the log showed
the next check at01:43:12UTC. Remote and local controller SHA256 matched:
`9a8a89fb619623a0afe2fb278f1df80dbe8a3444d9874cfd7605a43834e19293`.

The controller checks A every240seconds. After pipelineexit0,it audits all
three complete A checkpoints,requires the existing5425/4328 eligibility
gate,selects050then025,and runs the existing full matched official/selected
validation. Only after successful matched validation and an empty GPU does
it execute the existing StageB runner. The runner independently checks the
matched checkpoint epoch/SHA/backend and preserves the declared B recipe.
Any failure exits without automatic retry or bypass;logs remain on disk.

Use`tools/check_scanrefer_guarded_transition.sh` and the stage's
`transition.log`,`transition_started.json`,`transition_exit_code`.
Matched results stay under`protocol_audit`;B logs and exit status stay in
`stage_b_pipeline.log`/`stage_b_pipeline_exit_code`. Both phases remain
under the transition screen. Actual B process,first-update evidence,
own checkpoints and complete semantic history still require inspection.
Do not manually launch matched validation or B while the controller is live.
The existing two-hour monitor now follows this rule,with its schedule and
notification policy preserved.
Startup receipt:`docs/results/mcln_pvground_20261101/scanrefer-guarded-transition-started.json`.
No model code,training configuration,NR/SR process or local test changed.

## Second Complete Stage A Result

Epoch2 completed2026-11-02 23:51:44UTC,full9508 semanticTop1:
5460/4563 hits =57.425326%/47.991165%. Own checkpoint epoch2/1168
tensors/optimizer/scheduler audited; preserved SHA256
`e22d6b7a4968fbafb6e4fcf97342ce88497c300e3fb8183b2da35c003b8f39e9`.
This supersedes epoch1 as the current semantic best. Three-epoch
transition remains pending; no Stage B launch or final target success.

| Condition | Unique025 /1419 | Multiple025 /8089 | Unique050 /1419 | Multiple050 /8089 | Overall025/050 /9508 |
| --- | ---: | ---: | ---: | ---: | --- |
| Stage A epoch1 | 1232 | 4214 | 1070 | 3473 | 5446/4543 |
| Stage A epoch2 | 1240 | 4220 | 1075 | 3488 | 5460/4563 |
| Observed hit difference | +8 | +6 | +5 | +15 | +14/+20 |

Epoch2 unique025/050=87.385483%/75.757576%;multiple=52.169613%/43.120287%.
Multiple misses3869/4601 dominate absolute residual errors,versus
unique179/344. These are aggregate counts,not paired fixed/broken examples
or evidence that any particular module caused the improvement.
The subgroup sums exactly match the complete9508 overall metrics.
This read-only audit parsed existing logs remotely without importing
Torch or evaluating a model. Raw receipt:
`docs/results/mcln_pvground_20261101/scanrefer-stage-a-epoch2-group-evidence.json`.
The original official E0 comparison still crosses the documented backend
boundary; wait for matched official/selected evaluation after all3epochs.

## First Complete Stage A Result

Epoch1 completed2026-11-02 21:19:22UTC on all9508 examples. Its own
checkpoint epoch1/1168 model tensors/optimizer/scheduler was audited on
the remote CPU and hardlinked. The semantic-best alias selects epoch1.
SHA256:`9fc7999242e766ec44ea569042eb750ca35662947713c8177bef51badf28e519`.
Raw receipt:`docs/results/mcln_pvground_20261101/scanrefer-stage-a-epoch1-identity.json`.

| Condition | Unique025 /1419 | Multiple025 /8089 | Unique050 /1419 | Multiple050 /8089 | Overall025/050 /9508 |
| --- | ---: | ---: | ---: | ---: | --- |
| Original official E0 | 1233 (86.8922%) | 4202 (51.9471%) | 1032 (72.7273%) | 3295 (40.7343%) | 5435/4327 |
| Stage A epoch1 | 1232 (86.8217%) | 4214 (52.0954%) | 1070 (75.4052%) | 3473 (42.9348%) | 5446/4543 |
| Observed hit difference | -1 | +12 | +38 | +178 | +11/+216 |

These first semantic-branch subgroup lines sum exactly to the official
overall counts. Multiple objects remain the larger absolute error source:
3875 misses at025 and4616 at050,versus187/349 unique misses. This
describes error allocation,not which module caused those errors or which
examples changed. Aggregate differences are not paired fix/break counts.
Raw log lines and exact subgroup reconstruction are retained in
`docs/results/mcln_pvground_20261101/scanrefer-stage-a-epoch1-group-evidence.json`.

Overall epoch1 is57.278082%/47.780816%,still259/402 hits below final
60/52 targets. It satisfies the declared5425/4328 Stage B metric gate,
but no transition is authorized until all three audited Stage A epochs
and successful pipeline exit. The original E0 and training-validation
backend boundary documented below still prevents causal attribution.
Perform the already prepared matched official/selected evaluation before
Stage B;do not change modules based on this unmatched aggregate comparison.
Only after the staged decision and GPU availability should paired
Reader/Support/Span diagnostics identify ranking versus geometry failures.
Existing conditional reference options are in
`docs/PVG_MULTIPLE_INSTANCE_REFERENCE_NOTES.md`; none establishes the cause
of these measured misses.

At21:20:34UTC Stage A epoch2 was actually training at42/6081. Its first
two updates again had finite losses6.6050129/8.2219801,nonzero gradients
and deltas in all three modules,and unchanged frozen parameters/buffers.
The per-epoch update receipts now refer to epoch2;the immutable startup
receipt retains epoch1's original proof. Keep the three-epoch recipe.

## Recipe

Use step2 of `05_逐步执行提示词.md`, rather than combining conflicting
learning rates and epoch counts from different plan sections.

| Field | Stage A |
| --- | --- |
| Initialization | Official MCLN ScanRefer epoch54, model only |
| Checkpoint SHA256 | a9930065996fce1d0dd5ee9fe00a120bdb3a2c88d158b7a3666717d842ac113d |
| Trainable prefixes | decoder.5.local_visual., pvground_support_corrector., pvground_span_mixer. |
| Native model | Frozen parameters, native children in eval mode including BatchNorm |
| Batch | 8 microbatch, accumulation6, full groups48 examples |
| Learning rate | Modules1e-4, native/backbone/text0 |
| Optimizer | Fresh AdamW, weight decay0.0005 |
| Budget | 3 complete epochs, full9508 validation every epoch |
| Scheduler | Existing step scheduler, decay100 outside Stage A budget |
| Seed | Python/NumPy/Torch, sampler and worker generator explicitly0 |
| Architecture | Existing Reader/Support/Span, identity initialization retained |

Derive actual counts from the run's loader/accumulation log.
With6081 microbatches,1014 optimizer
updates include a final partial3-microbatch group. Do not call every update
effective48 or claim48655 training examples were all consumed when the
training loader drops the last incomplete batch.

The previous source parsed rng_seed but never applied it to global model
initialization. The new source corrects this and replaces hardcoded loader
seed0 with the requested seed. This does not retroactively establish
reproducibility of earlier runs, and is not deployed to running NR/SR.

## Execution

Remote host: connect.bjb2.seetacloud.com:48073.
Root: `/root/autodl-tmp/mcln_pvground_ft_20261101`.
New isolated directory: `scanrefer_restart_20261102`.
Screen: `pvg_scanrefer_restart_pipeline`.
Secure control uses workspace `work/remote_pvground.ps1`.

Old Scan run1793579788 was interrupted18:06:56UTC November2 after
preserving best8/latestcomplete10 and its log/config/manifest. Epoch11
was incomplete. Old source/runs remain intact. No stopped clean-baseline
or older failed experiment was inspected or resumed.

The first E0 attempt failed before evaluation because the training loader
rejects model_only_initialization with eval. Preserve pipeline_attempt1.log.
The standalone E0 tool now validates and loads exactly1134 official native
tensors, allows only the three modules' new keys, ignores the known obsolete
position_ids buffer, and never restores optimizer/scheduler. It asserts
checkpoint epoch54. The production training loader restriction remains.

The second pipeline attempt started18:13:26UTC. Official model and
identity-initialized modules each evaluate9508 at seed0/batch8. The gate
requires identical semantic Top1 hits, sample identity hash, point hash,
seed and checkpoint hash. Both receipts and e0_gate.json must exist and
pass before training can start. Screen creation is not proof of training.

Official E0 completed all9508 examples,5435/4327 semantic Top1 hits=
57.162390%/45.509045%,pointSHA256
44cc887237c35c3266ed2384cea348372c6d85d1498b382750b3dbdcbbc77dc6.
Its initialization receipt confirms1134 native tensors,epoch54,no optimizer
restore. This is a measured protocol result,not an assertion that the
paper's exact scores were reproduced.

Identity evaluation then failed before forward because the general
optimizer builder requires positive native LR,even during evaluation.
The standalone E0 tool now skips optimizer construction entirely.
This changes no checkpoint tensor or forward computation. Preserve attempt2
logs and the completed official receipt; resume the identity condition only.
The continuation started18:26:28UTC. Stage A still uses its actual optimizer
and remains gated on the two complete E0 receipts.

Stage A own first-update receipt must prove finite loss, finite nonzero
gradients and actual updates in all three modules, exact unchanged frozen
parameters and buffers. Remote fixture tests are not this witness.
The current startup instrumentation saves both first and second actual
optimizer updates separately,after microbatches6 and12.
Keeper and preservation screens run separately every240 seconds.

Stage B requires a complete Stage A epoch with at least4328 Acc050 hits
and5425 Acc025 hits on9508 examples. If that gate fails, preserve
the results and diagnose; do not automatically unfreeze or claim success.
If it passes,continue the requested ScanRefer fine-tuning from the best
qualifying same-epoch checkpoint,ranked050then025: unfreeze native model
except RoBERTa,nativeLR2e-6/backbone4e-6/modules2e-5,batch8/accum6,
fresh optimizer/scheduler,max10epochs. Stop after two consecutive full-epoch
Acc050 decreases. Preserve the selected Stage A checkpoint and gate evidence
before the transition. NR/SR continue untouched throughout either Scan stage.
Proposed I1/I2 architecture and gate changes remain unimplemented.

## Evidence

### Validation Protocol Boundary

Read-only deployed-source audit at19:26UTC found:
`train_dist_mod.py:3796` sets cuDNN benchmark=True and deterministic=True;
the standalone matched E0 entry sets benchmark=False and deterministic=True.
Thus official-versus-identity E0 equality is proved within its own entry,
but the training validation entry does not have identical backend settings.
The effect on hits has not been measured. Do not call this a demonstrated
accuracy error or assume it explains any future gain/regression.
In particular,the one-hit Stage B transition margin is not evidence of
module benefit by itself. Preserve the declared gate and raw complete rows;
when attributing a small difference,re-evaluate the official initialization
and selected trained checkpoint under one explicitly matched entry/settings,
after the current Stage A GPU job permits it. Do not silently replace E0
numbers or alter the running recipe. Any measured protocol change and
resulting decision must be recorded separately.

The shared train/test loader generator advances during training,which by
itself does not establish random validation inputs. In the inspected deployed
dataset,__getitem__ restores scan.pc from scan.orig_pc;_get_pc augmentation
and detected-box corruption are restricted to the train split. No point
resampling was found in that path. This is source evidence,not a full
post-training input-hash comparison. Existing E0 input/order hashes cover
the two E0 conditions only.
Raw audit:
`docs/results/mcln_pvground_20261101/scanrefer-validation-protocol-source-audit.json`.
No dataset/model/test was executed for this audit;live trainers unchanged.

#### Prepared Matched Validation

A separate `protocol_audit/` entry is prepared remotely for the actual
post-Stage-A official-initialization vs selected-checkpoint comparison.
The current training source,original E0 receipts and declared Stage A gate
remain unchanged. The extended E0 tool accepts expected epoch and1134/1168
model tensors:official initialization retains identity modules;trained
weights must contain the complete1168-tensor model. It validates epoch,
keys,shape and dtype before loading;no optimizer is constructed/restored.
Both conditions explicitly use benchmark=True/deterministic=True,batch8,
seed0,the same source/model architecture and full9508 examples.
Pair completion requires identical point/order hashes,seed,backend flags,
sample count and innovations flag. The selected checkpoint SHA is checked
against the plan. Receipt `protocol_audit/matched_validation.json` reports
selected-minus-matched-official hits and matched-official-minus-original-E0
hits separately. The latter is an observed protocol/repeat difference,
not by itself causal attribution to cuDNN. Raw E0 is never overwritten.

Only run when all three Stage A epochs and pipeline completion are audited
and the GPU is free. For a ready transition,secure helper
`work/start_scanrefer_matched_validation.sh` selects the same gate-approved
checkpoint and starts screen `pvg_scanrefer_matched_validation`;inspect
`protocol_audit/pipeline.log` / `pipeline_exit_code` / the two raw receipts.
The underlying `tools/run_scanrefer_matched_validation.sh` takes an audited
checkpoint path and own epoch,so a failed Stage A gate can still be diagnosed
on an explicitly chosen completed checkpoint without changing that gate.
Stage B must not already have started while this diagnostic runs.

Six remote CPU loading tests passed,CUDA uninitialized:exact official
native/identity-module behavior,complete trained-state restoration,and
rejection before mutation of wrong epoch,missing module,wrong shape/dtype.
Bash syntax passed remotely. Receipt:
`docs/results/mcln_pvground_20261101/scanrefer-matched-preparation-remote-tests.json`.
These fixtures do not establish matched9508 metrics;the real pair has not
run and no protocol hit effect has been measured.

At20:26UTC the prepared Stage B entry was updated to enforce this ordering.
Previously it checked only the Stage A decision and initialization SHA,
so direct launch could skip the declared matched-validation prerequisite.
Before importing the GPU trainer,it now requires successful matched
pipeline exit,9508 examples,the selected epoch and initialization SHA,
1168 loaded tensors,and the expected cuDNN settings. Its startup receipt
records the matched result. Eleven remote CPU tests passed,including the
existing optimizer/stop fixtures and rejection of failed or mismatched
evaluation receipts. No matched GPU evaluation or Stage B run occurred.
The deployed entry SHA is dcb44691123301a9806ae700e51d038896647d6ca33cbda95b65c1a9fc3ce2a6.
Original `stage_b_prepared.json` remains historical;the later deployment
is recorded in `stage_b_matched_gate_prepared.json`.
Raw evidence:
`docs/results/mcln_pvground_20261101/scanrefer-stage-b-matched-gate-remote-tests.json`
and `scanrefer-stage-b-matched-gate-prepared.json`.
The live Stage A main_utils SHA remains a87d2524fd128b3586d81233636a8ffb47002f65ac69ac582de639b646058fa3.

### Final Acceptance And Next Decisions

Passing the Stage A transition gate is not the final target. On the fixed
validation splits, final success requires integer semantic Top1 hits:

| Dataset | Full samples | Required025 hits | Required050 hits |
| --- | ---: | ---: | ---: |
| ScanRefer | 9508 | 5705 | 4945 |
| NR3D | 7899 | 4724 | No separate target |
| SR3D | 17726 | 12302 | No separate target |

ScanRefer must meet both counts on one checkpoint; do not join independent
025-best and050-best rows. Its E0 has5435/4327 hits,so deficits270/618
refer to initialization,not a completed Stage A model. NR24 has3666
025hits(deficit1058); SR selected8 has11124(deficit1178).
These do not establish a module failure: current NR/SR have no innovations.
Complete their original240/140epoch runs before treating the reproduction
result as final or using their best checkpoints for a subsequent method run.

Keep the three main module roles and overall architecture consistent.
After a complete matched diagnosis,an approximately1percentage-point
dataset-specific gap can motivate internal module parameters or depth
changes while preserving these roles; do not deploy such a change on a
dataset that already meets its targets without revalidation.
For a gap around0.3percentage points,prioritize learning rate,batch and
seed experiments with the existing method rather than replacing modules.
Current results are not in that final tuning regime. New method changes
require measured failure attribution and relevant open-source references,
not partial training loss or a synthetic fixture. Stage A/B complete
results and the authorized original NR/SR runs remain the next evidence.

E0 gate PASSED18:38:40UTC November2: official and identity conditions
both9508,5435/4327 hits,identical point and sample-identity hashes.
Identity initialization loaded1134 native tensors and added34 module
tensors; no optimizer restored. The pipeline then started Stage A.
Run1793644724:
`scanrefer_restart_20261102/stage_a/scanrefer/scanrefer_stage_a_20261102/1793644724`.
At18:48:07UTC training was verified through two real optimizer updates:
epoch1 microbatches6/12,steps1/2,loss6.477235794067383/7.649016857147217.
Both receipts have finite nonzero gradients and actual deltas for all three
modules; native parameters and buffers are exactly unchanged twice.
Progress129/6081,GPU10125MiB; first full validation estimated21:10UTC
November2. No complete trained epoch yet. Immutable startup snapshot:
`docs/results/mcln_pvground_20261101/scanrefer-stage-a-startup-receipt.json`
and remote `scanrefer_restart_20261102/startup_receipt.json`.
Per-epoch first/second update files are overwritten at later epochs.
New E0/Stage A table:
`docs/results/mcln_pvground_20261101/scanrefer-restart-all-epochs.csv`.
E0 rows have no trained checkpoint: they evaluate native epoch54 with
fresh identity modules where enabled,not a saved epoch0 training result.

Four focused tests passed on the remote Scan server, Torch2.1.2+cu118,
CUDA uninitialized: seeded initialization, training worker/order
repeatability, evaluation order/worker repeatability and module-only
optimizer update/native BatchNorm preservation.
Remote receipt: `scanrefer_restart_20261102/remote_stage_a_tests.json`.
E0 entry syntax and pipeline Bash syntax also checked remotely.
The expanded two-update fixture and updated source syntax also pass4tests
remotely,CUDA uninitialized;receipt remote_stage_a_two_updates_tests.json.
Current main_utils.py SHA a87d2524fd128b3586d81233636a8ffb47002f65ac69ac582de639b646058fa3.

Stage A evidence collection is separate from the old Scan namespace:
`tools/collect_scanrefer_stage_a_epochs.py` reads complete semantic log
subgroups through the existing keeper parser,preserves numeric checkpoint
hardlinks,and verifies each weight's own epoch,1168 model tensors and
optimizer/scheduler. It reports transition eligibility and final targets
as separate fields; an unfinished epoch is not a row.
Secure workspace wrapper helper:`work/collect_scanrefer_stage_a_epochs.sh`.
Remote output:`scanrefer_restart_20261102/stage_a_complete_epochs.json`.
Initial execution against the real run returned an empty row list,CUDA
uninitialized,because no trained epoch was complete.
Two remote CPU fixtures passed:complete same-epoch transition without
final success and retention survival;wrong checkpoint epoch rejection.
Receipt:`docs/results/mcln_pvground_20261101/scanrefer-stage-a-collector-remote-tests.json`.
These fixtures validate collection logic,not a completed Stage A epoch.

The collector now also emits `stage_b_decision`. It remains pending until
the pipeline records `stage_a_complete`, exits successfully, and all three
own-checkpoint-audited epochs1/2/3 exist. A pipeline failure or missing epoch
cannot authorize Stage B. If no single epoch passes both5425/4328 hit gates,
the decision is `gate_failed`. Otherwise it selects among eligible epochs
by050then025,with the earlier epoch breaking exact ties,hardlinks the selected
initialization weight and records its SHA256. This is a transition receipt,
not a Stage B launch or a claim of final target success. The Stage A runner
still ends after Stage A; the separate Stage B entry is prepared below.
The validation protocol boundary above still applies.
All7 expanded collector tests passed on the remote Scan server,CUDA
uninitialized. They cover incomplete budget,pipeline failure,non-combinable
checkpoint hits,050priority/025tie-break/exact ties,selected hardlink/hash,
and the original collector integrity checks. Receipt:
`docs/results/mcln_pvground_20261101/scanrefer-stage-a-transition-remote-tests.json`.
The deployed collector returned empty rows and a pending decision on the
actual Stage A run. No Stage B checkpoint has been selected or launched.

### Prepared Stage B Entry

At19:48UTC a separate `stage_b_source` copy was prepared remotely; the live
Stage A main_utils.py hash remains
a87d2524fd128b3586d81233636a8ffb47002f65ac69ac582de639b646058fa3.
NR/SR source and training are unchanged.
`tools/run_scanrefer_stage_b.sh` refreshes the three-epoch audited collector
gate,requires `ready`,and rejects a previous Stage B launch. It starts
`tools/train_scanrefer_stage_b.py` only after that gate. No Stage B launched.
Secure launcher/check helpers:
`work/start_scanrefer_stage_b.sh` / `work/check_scanrefer_stage_b.sh`.

Read-only optimizer-source inspection found an actual Stage B recipe
mismatch:the existing `models/mcln_training_groups.py` special-rate group
matches only `decoder.5.local_visual.`; Support and Span default to decoder
LR. This does not affect the current Stage A optimizer,which explicitly
trains all three at1e-4. The separate Stage B entry explicitly unfreezes
native parameters except `text_encoder.`,assigns native2e-6/backbone4e-6,
and all three module prefixes2e-5. It loads only the selected Stage A full
model,uses fresh optimizer/scheduler,and does not reload the official
backbone over trained Stage A weights. Batch8/accum6,max10,seed0,constant LR
(decay100 outside budget),training cuDNN settings preserved.
Deployed-source read-only confirmation:
`source/models/mcln_training_groups.py:122-127`,SHA256
68c85958a4b0aab2b6f9ab5ee78c86febb7bc0cff17612e1880ce5da0a39d227.
Raw evidence:
`docs/results/mcln_pvground_20261101/scanrefer-stage-b-optimizer-source-audit.json`.
This is a proven configuration-routing issue for the proposed Stage B,
not a measured explanation of earlier accuracy results.

A default-false `should_stop_after_epoch` hook in the prepared source ends
the native loop after two strict consecutive full Stage B Acc050 decreases.
The first Stage B validation establishes its comparison baseline;equal or
improved hits reset the streak. An early stop skips the final budget-epoch
evaluation,preventing a misleading epoch10 label for an earlier checkpoint.
Each completed checkpoint is CPU-audited and hardlinked before another
epoch;semantic history and050then025 best are maintained separately from
the native position/mask retention schema. This does not claim the native
ordinary evaluator/retention issue is globally fixed.

Five remote CPU tests passed,CUDA uninitialized:actual native/three-module
parameter updates with frozen text,exact three-group LRs,semantic bbf rather
than bbs,partial-validation rejection,streak reset,model-only arguments
without backbone reload,and execution of the native validation-loop branch
confirming stop before epoch4 without an extra evaluation.
Bash syntax checked remotely. Raw tests/preparation receipts:
`docs/results/mcln_pvground_20261101/scanrefer-stage-b-preparation-remote-tests.json`
and `scanrefer-stage-b-prepared.json`. These tests are not a real Stage B
GPU update or validation. On actual launch verify the new run's own finite
loss,three-module gradient/parameter-update receipt,config and initialization.
The matched-entry protocol check remains required before attributing tiny
accuracy differences to the modules.

NR24 completed18:10:00UTC,7899 examples,semantic hits3666/2805,
Acc025/05046.410938%/35.510824%; own epoch24/1134 tensors and optimizer/
scheduler CPU verified, checkpoint hardlinked. New NR Acc025-first best24.
SR9 completed18:12:48UTC,17726 examples,hits11102/8632,
62.631163%/48.696830%; own epoch9/1134 tensors CPU verified and hardlinked.
SR best remains8,11124/8586; do not mix its025 with9's050.
Historical current-run table now43 complete rows:oldScan10/NR24/SR9.
Keep new E0/Stage A records separate from old Scan epoch numbers.
