# Fixed-Checkpoint Forward Audit

Status: 15 focused CPU tests passed historically; the GPU evaluation entry point is unexecuted.
Per the user's latest instruction, all further tests and diagnostics must
run on remote servers. Do not run them on Windows or local WSL, and do not
continue the proposed local single-example probe or prepare its dependencies.
This tool does not attach to, change, stop, or resume an existing trainer.
Run it only when a remote GPU is available for isolated evaluation.

Observation update,2026-11-02 approximately16:43UTC:
the earlier tool measured Span but not Reader/Support activation effects.
The new hooks always observe Reader residual and input-query L2 norms,
Support mean/max absolute correction on observed native superpoint IDs,
and fused-mask foreground flips using the model's sigmoid>0.5 rule.
Ragged scenes stay separate; unobserved slot gaps are excluded.
The selected semantic query receives the same per-query measurements.
All correction/residual measurements explicitly say before_bypass;
reader_bypassed/support_bypassed identify whether that result was applied.
These statistics describe computation, not its usefulness or causal gain.

The updated helpers and nine focused CPU fixture tests passed on the Scan
server with Torch2.1.2+cu118/pytest9.2.0; CUDA remained uninitialized.
Tests cover each bypass, unchanged outputs/parameter keys without bypass,
per-scene ragged/native-slot statistics, selected-query association and
agreement with the tracked semantic evaluator/IoU functions on three seeds.
The trained model, full dataset and GPU forward were not executed.
Pytest was absent from the training environment; dependencies were installed
only under`offline_audits/observations_v2/test_dependencies`, without
installing into or modifying that environment.
Raw remote receipts:
`docs/results/mcln_pvground_20261101/scanrefer-offline-observations-v2-remote-tests.json`
and`scanrefer-offline-observations-v2-import-preflight.json`.
The latter confirms new helpers resolve from observations_v2 while trainer
and evaluator resolve from the active source directory.
Remote verification entry:`tools/verify_pvground_offline_observations.sh`.
Archive SHA256:
`bf7aba8910d14a57c76634fbf805aa9dbd4e00a29ae752e55e8a3d9cd0c4c568`.
No local tests were run and no running trainer/source was modified.

Scan server import preflight passed at approximately16:10UTC on2026-11-02.
Archive SHA25634a234ad96bbaea6b74a48550d0aeb868154d019118943961c921472f7cbca67
was checked before extraction into`$root/offline_audits/c8437a3`.
All three helpers are deployed there without changing active training source.
Current trainer/evaluator/IoU dependencies import; CUDA remained uninitialized.
No model construction,checkpoint load or forward occurred. Raw preflight:
`docs/results/mcln_pvground_20261101/scanrefer-offline-audit-import-preflight.json`.

Current scope,2026-11-02:the previous epoch8 command targeted a stopped
ScanRefer run and has been retired. Its historical checkpoint SHA256 was
`3572193fef0f32d1d6ca22b41cbbb02932a30de0a51b18deb3b0c40ee37bcf38`;
it is not the checkpoint for the current restart's diagnostics.

The active restart is under
`/root/autodl-tmp/mcln_pvground_ft_20261101/scanrefer_restart_20261102`.
Finish all three Stage A epochs and the matched official/selected complete
validation first. If diagnostics are then needed, take the fixed checkpoint,
its own epoch and SHA256 from `stage_a_complete_epochs.json` and
`protocol_audit/matched_validation.json`. The Stage A transition selects
050then025 among eligible checkpoints; a failed gate requires an explicit
audited checkpoint selection. Record that selection before evaluation.
Use the restart's `source` directory for trainer/evaluator/pointnet2 imports.
Recheck remote import resolution against that source before the first
diagnostic GPU run:the historical preflight above covered the old source,
not this restart. No current-restart bypass evaluation has run.

The current deployed staging directory contains`eval_pvground_audit.py`,
`pvground_forward_audit.py`,and`pvground_semantic_audit.py`under`tools/`.
The evaluator must load the full trained state, without
`model_only_initialization`, `checkpoint_start_epoch`, or initialization-only
`pp_checkpoint`. Remote imports passed; actual CUDA forward remains untested.
Do not run concurrent with the occupied training GPU.

The explicit`--local_rank 0`is required by the current parser:
`main_utils.py`defines only the underscore spelling,default1,and uses
parse_known_args. The launcher's hyphenated`--local-rank=0`is ignored by
that parser,leaving device1 selected if the explicit argument is omitted.
This was the same concrete single-GPU failure fixed in the training runner.
The retired offline invocation originally omitted it. Retain this explicit
argument when preparing the current-restart invocation.

Run `none`, then sequentially `reader`, `support`, and `span`, changing the audit
output, log directory, and experiment name for each condition. Keep all
other arguments, source, checkpoint bytes, preprocessing, seed, and batch
unchanged. Record the checkpoint epoch from its CPU audit, the full command,
source commit, and full semantic evaluator log beside each JSONL.

The bypasses are:

- Reader: replace only its residual with zeros.
- Support: keep its measured superpoint geometry, return original member
  mask logits, and let downstream reference/Span recompute from those logits.
- Span: return the original native center and size exactly.

These are interventions on a trained network. They do not replace separately
trained ablations and do not establish gains over official initialization.
The official initialization also needs a matched complete evaluation before
claiming improvement over MCLN.

Each JSONL begins with checkpoint SHA256 and invocation metadata. Sample
records contain candidate counts, reference-valid counts, raw gate bounds
and clipping counts, gate means, and mean absolute axis displacements.
Additional `semantic_sample` records contain scan/target/utterance identity,
input point SHA256, unique/multiple group, selected query and raw/clamped
gate, selected and all-candidate oracle IoU, first good candidate rank at
strict025/050 thresholds, top10 candidates/IoUs, and native/reference/final
selected boxes with signed per-axis center/size errors against the target.

The completion footer is emitted only after both forward and semantic
records cover exactly9508 examples and per-example025/050 hit counts agree
with the evaluator's own last/bbf Top1 counters. Position/mask metrics must
not be substituted. Failed/incomplete evaluations have no completion footer.
The standalone evaluator's normal full log remains an independent artifact.

The semantic helper follows the current root-only ScanRefer branch without
detector filtering. Tests execute the tracked evaluator's own semantic
method and box IoU functions (isolated from CUDA-only imports), compare
selected boxes/IoUs/counters on three seeds, reject a deliberately altered
counter, and check strict-threshold rank behavior. This establishes CPU
agreement on those fixtures, not remote GPU validation.

All-candidate gate statistics can differ from the selected query distribution;
selected gates are recorded separately. All-candidate oracle coverage uses
the current predicted boxes, not ground-truth candidate boxes. Displacement
is distinct from the recorded GT errors. A module bypass can change rankings
and boxes together; paired identities and full metrics are needed to interpret
differences. These diagnostics do not themselves prove a causal failure.
No real gate-occupancy or bypass-accuracy result is available yet.
