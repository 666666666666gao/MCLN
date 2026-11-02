# ScanRefer Stage A Restart

Latest user scope: restart ScanRefer fine-tuning after reading the six
documents in `C:/Users/gb/Desktop/document/MCLN plan/`; continue NR3D and
SR3D original full reproduction runs unchanged. The documents' suggestions
to replace their scratch initialization are outside that scope.
All tests and model execution run remotely. No local Windows/WSL tests.

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
