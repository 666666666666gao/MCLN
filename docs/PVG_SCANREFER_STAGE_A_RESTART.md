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

NR24 completed18:10:00UTC,7899 examples,semantic hits3666/2805,
Acc025/05046.410938%/35.510824%; own epoch24/1134 tensors and optimizer/
scheduler CPU verified, checkpoint hardlinked. New NR Acc025-first best24.
SR9 completed18:12:48UTC,17726 examples,hits11102/8632,
62.631163%/48.696830%; own epoch9/1134 tensors CPU verified and hardlinked.
SR best remains8,11124/8586; do not mix its025 with9's050.
Historical current-run table now43 complete rows:oldScan10/NR24/SR9.
Keep new E0/Stage A records separate from old Scan epoch numbers.
