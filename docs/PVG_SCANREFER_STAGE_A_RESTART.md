# ScanRefer Stage A Restart

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
