# Fixed-Checkpoint Forward Audit

## All Four Complete: 2026-11-03 17:21 UTC

The pipeline exited0;all four conditions have9508 forward records,9508
semantic records and evaluator-verified footers. The summary recomputes
strict-threshold hits and verifies identical ordered identities,pointSHA
and GT against none. Same immutable B2,batch8,seed0,source/evaluator.

| B2 Inference Condition | Hits025 | Hits050 | Bypass Gained/Lost050 | Query Changes |
| --- | ---: | ---: | ---: | ---: |
| None | 5483 | 4656 | -- | -- |
| Reader bypass | 5504 | 4673 | 83 / 66 | 349 |
| Support bypass,reference recomputed | 5475 | 4616 | 39 / 79 | 0 |
| Span bypass,native box | 5407 | 4262 | 114 / 508 | 0 |

Relative to none,the bypass nets are+21/+17,-8/-40,-76/-394 at025/050.
Support bypass recomputes the Mask reference and reruns Span;it is not
merely suppressing the final correction. Span bypass retains the selected
semantic query and uses its native box. Reader changes the representation
and can change query selection. These are conditional inference effects
of a trained network,not independently trained ablations or additive gains.

Current Span repairs508 strict050 failures while breaking114 successful
native predictions. Its394-hit net benefit rules out claiming it is simply
harmful;the114 breaks and54-hit gap to selected reference motivate checking
whether explicit gate supervision improves decisions. The separately
registered matched training pilot tests this without changing the forward
gate/extent/architecture. Reader's net17 bypass benefit is small compared
with its opposing83/66changes;contextual verification remains to be tested.

Compact paired summary and all four full evaluator logs are archived locally
and under docs/results/mcln_pvground_20261101;raw JSONL remains remotely at
offline_audits/b2_fixed_audit_20261103_lrpositive.
No training CSV row is added for these repeated inference conditions.
No target has been reached. See PVG_SPAN_GATE_COMPARISON.md for next run.

## Paired Reader Result: 2026-11-03 17:07 UTC

Completed none and Reader-bypass each have9508 forward/semantic records
and evaluator-verified footer. Ordered identity,pointSHA and GT match exactly.
None5483/4656;Reader5504/4673. Selected query changes349.
At025 bypass gains47/loses26,net21;at050 gains83/loses66,net17.
This is a trained-checkpoint inference intervention,not a trained ablation.
It suggests competing beneficial/harmful effects,not automatic deletion.
Receipt:scanrefer-b2-fixed-audit-reader-paired-summary.json.
At17:08UTC Support9200/9508,Span queued;do not infer incomplete metrics.

## Complete B2 Baseline: 2026-11-03 16:45 UTC

The none condition finished16:43:27UTC with9508 forward records,
9508 semantic records and evaluator-verified footer:5483/4656hits,
57.667228%/48.969289%. These counts equal the archived B2 validation;
this single repeat is not a claim of universal bitwise determinism.
Raw JSONL SHA256:
b09637f1ed5a34d1c3ee7c9e1597df0b33873e3258a07c9190ac5095e02cd5d3.
Reader bypass began16:43:34UTC; Support/Span remain sequentially queued.
No paired bypass effect is established until that condition completes and
the exact input identities/point hashes/GT are checked.

Measured candidate coverage in this same baseline:

| Strict Threshold | Selected Top1 | Top10 Contains Valid Box | All-Candidate Oracle |
| --- | ---: | ---: | ---: |
| IoU>0.25 | 5483 | 6701 | 9082 |
| IoU>0.50 | 4656 | 6021 | 8260 |

At050,3604 examples contain a successful predicted candidate but the
semantic Top1 does not select it. This is a measured selection gap, not
achievable accuracy or a promised module gain; oracle uses GT only for
diagnosis. With the baseline's same selected queries, native/reference/
final boxes score4262/4710/4656 at050 and5407/5448/5483 at025.
Final mixing is394hits above native at050 and54below reference; these
within-query measurements do not establish a separately trained variant.

Support correction magnitudes have extreme selected-query outliers:
mean absolute correction median15.7797, maximum9914876.0. A targeted
remote record/source inspection finds exactly7selected native boxes with
a nonpositive size axis; exactly these7have mean correction>1000,
and all7miss both thresholds. Deployed source normalizes relative
positions by size.detach().clamp_min(1e-6), lines68/70; SHA256
b4f6f8417f37c9450e3f4e1830601009eedd5da72b3da1be6625575ba40ed112.
This is an observed association consistent with amplified geometry inputs,
not a demonstrated sole cause. Seven cases cannot explain the289hit
Acc050 target deficit; do not redesign Support on the aggregate mean.
Selected gate lower/upper clipping counts by x/y/z are158/340,
137/313,128/272; these are per-axis counts, not unique-example totals.

Raw baseline summary, full evaluator log and outlier receipt are stored
under docs/results/mcln_pvground_20261101 with prefix
scanrefer-b2-fixed-audit-. Full JSONL remains in the remote audit directory.
No training metrics CSV row is added for this repeated checkpoint eval.
No architecture change, causal module attribution or target attainment
is claimed. NR/SR original scratch runs continue.

## B2 Actual GPU Launch: 2026-11-03 16:33 UTC

The isolated diagnostic now runs against immutable selectedB2, not the
completed B trainer. Screen pvg_scanrefer_b2_fixed_audit, runner232612,
evaluator232624, start16:30:52UTC. Directory:
`offline_audits/b2_fixed_audit_20261103_lrpositive`.
At16:32:47UTC actual GPU5407MiB,360forward/360semantic records, no errors.
No complete metric or paired intervention result is available yet.
Baseline ETA~16:44UTC, four conditions approximately17:25UTC; update from
actual speed. The sequence is none/reader/support/span, batch8/seed0.

First attempt `b2_fixed_audit_20261103` failed before any forward, exit1:
`ValueError: decoder_lr must be finite and positive`.
The existing optimizer-group builder executes even for eval. The corrected
command uses native2e-6/backbone4e-6/module2e-5 with `--eval`, which returns
after evaluation. It does not request optimizer steps or training.
Failure log, metadata-only JSONL, original runner and receipt are preserved.
No fallback or modification to model/trainer source was needed.

CPU preparation rechecked B2 epoch2,1168 tensors, nonempty optimizer/
scheduler and SHA46ecfb6a6831fca0704c55fe87fa96c9c92e749824f13173d67d5697da1b92f8.
Exact deployed source hashes are recorded in the preparation receipt.
Its recording_repository_commit fcc1e7e is a provenance reference, not a
claim that deployed main_utils.py matches the current local file.
Require all9508 samples and evaluator-verified footer, then exact paired
ordered identity/pointSHA/GT agreement before interpreting interventions.
They are not independently trained ablations or per-module gain proof.
NR/SR scratch trainers/configurations remain unchanged. No local tests ran.

`tools/summarize_pvground_fixed_audit.py` requires full completion before
emitting accuracy or paired comparisons. It rechecks footer hits against
individual strict-threshold IoUs, records JSONL SHA, candidate coverage,
selected native/reference/final errors, gate distributions and module
observations, then compares exact ordered pointSHA/identity/GT records.
Five remote stdlib parser tests derived from eight real forward records
passed: valid input, missing footer, evaluator-count mismatch, changed
pointSHA and strict-threshold gained/lost flips. These are parser checks,
not additional full evaluations. Raw receipt:
`docs/results/mcln_pvground_20261101/scanrefer-b2-fixed-audit-summary-remote-tests.json`.
Use `tools/summarize_scanrefer_b2_baseline.sh` after none completes and
`tools/summarize_scanrefer_b2_all_conditions.sh` only after all four exit0.

The following unexecuted-status paragraphs describe earlier snapshots.

2026-11-03 15:44UTC:Scan Stage B finished four epochs with normal
two-consecutive-Acc050-decreases early stop,both pipeline/controller
exit codes0. Scan GPU was free at15:40-15:44UTC. Best isB2,latestB4;
their own CPU identities and full9508 semantic counters are archived in
scanrefer-stage-b-final-full-history-20261103T1544.json.
The prepared GPU forward/bypass entry remains unexecuted. No B2 bypass
accuracy or individual-module causal evidence is available;do not treat
two validation declines as a diagnosis of Reader,Support or Span.
No diagnostic or new run was launched by the completion monitor.

Status: 15 focused CPU tests passed historically; the GPU evaluation entry point is unexecuted.
Current-restart import check passed2026-11-03 01:31:59UTC.
The deployed audit entry was loaded without invoking main, then trainer,
evaluator,IoU and box-converter imports were resolved and asserted inside
`scanrefer_restart_20261102/source`. Both observation helpers resolved
inside`offline_audits/observations_v2`. CUDA remained uninitialized.
No model construction,checkpoint load,forward,evaluation or local test ran.
Receipt:
`docs/results/mcln_pvground_20261101/scanrefer-restart-audit-import-preflight.json`.
Reproduction helper:`tools/preflight_scanrefer_restart_audit.sh`.
This closes the restart import-resolution prerequisite only; complete
Stage A/matched validation and a free remote GPU remain required before
any diagnostic forward. No current-restart bypass accuracy is available.
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
The2026-11-03 preflight at the top of this document verifies import resolution
against that restart source. The older historical preflights cover the old
source only. No current-restart bypass evaluation has run.

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
