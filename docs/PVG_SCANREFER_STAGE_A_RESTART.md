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

Stage A own first-update receipt must prove finite loss, finite nonzero
gradients and actual updates in all three modules, exact unchanged frozen
parameters and buffers. Remote fixture tests are not this witness.
Keeper and preservation screens run separately every240 seconds.

Stage B is conditional on some complete Stage A epoch exceeding E0
Acc050 while losing at most10 Acc025 hits. If that gate fails, preserve
the results and diagnose; do not automatically unfreeze or claim success.
Proposed I1/I2 architecture and gate changes remain unimplemented.

## Evidence

Four focused tests passed on the remote Scan server, Torch2.1.2+cu118,
CUDA uninitialized: seeded initialization, training worker/order
repeatability, evaluation order/worker repeatability and module-only
optimizer update/native BatchNorm preservation.
Remote receipt: `scanrefer_restart_20261102/remote_stage_a_tests.json`.
E0 entry syntax and pipeline Bash syntax also checked remotely.

NR24 completed18:10:00UTC,7899 examples,semantic hits3666/2805,
Acc025/05046.410938%/35.510824%; own epoch24/1134 tensors and optimizer/
scheduler CPU verified, checkpoint hardlinked. New NR Acc025-first best24.
SR9 completed18:12:48UTC,17726 examples,hits11102/8632,
62.631163%/48.696830%; own epoch9/1134 tensors CPU verified and hardlinked.
SR best remains8,11124/8586; do not mix its025 with9's050.
Historical current-run table now43 complete rows:oldScan10/NR24/SR9.
Keep new E0/Stage A records separate from old Scan epoch numbers.
