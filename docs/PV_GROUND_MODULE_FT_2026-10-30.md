# PV-Ground Module Fine-Tuning: ScanRefer, NR3D, and SR3D

Current verified state: 2026-10-31 01:00 America/New_York.
ScanRefer run1793345477 has full metrics through epoch17 and is training
epoch18. Its trained Acc050-best is epoch17,
59.2659%/50.4102%, preserved as best_trained_epoch17.pth.
NR3D run1793419199 and SR3D run1793419200 have full initialization
metrics and passing own-run first-update audits; their first trained
evaluations are pending. Both reproduced author baselines are complete.
The historical stopped NR/SR runs are labeled separately below.

Experiment start: 2026-10-30 (America/New_York).

## Targets and Protocol

| Dataset | Target |
| --- | ---: |
| ScanRefer Acc@0.25 | 60.0% |
| ScanRefer Acc@0.50 | 52.0% |
| NR3D Acc@0.25 | 59.8% |
| SR3D Acc@0.25 | 69.4% |

All values below are native semantic-alignment `last/bbf` top-1 accuracies. A completed row is accepted only when the evaluation covers the full validation set and reports `same_complete_model=true`. Epoch 0 is the untrained initialization and is shown for reference, never as a fine-tuning result.

The runs use batch size 10 and seed 111, with `lr=1e-5`, `lr_modules=1e-5`, `lr_backbone=2e-5`, and LR decay epochs `[10,30]`. The pretrained core is frozen; the three PV-Ground modules are trained. NR3D and SR3D use their respective paper-repository dataset flags, including `--butd_cls` and `--self_attend`. The active runs are monitored independently; their metric definitions are not mixed.

## ScanRefer

Validation size: 9,508.

| Epoch | Hits@0.25 | Hits@0.50 | Acc@0.25 | Acc@0.50 |
| ---: | ---: | ---: | ---: | ---: |
| 0 (init) | 5606 | 4843 | 58.961% | 50.936% |
| 1 | 5559 | 4542 | 58.467% | 47.770% |
| 2 | 5552 | 4538 | 58.393% | 47.728% |
| 3 | 5619 | 4641 | 59.098% | 48.812% |
| 4 | 5580 | 4581 | 58.687% | 48.180% |
| 5 | 5598 | 4600 | 58.877% | 48.380% |
| 6 | 5597 | 4629 | 58.866% | 48.685% |
| 7 | 5623 | 4716 | 59.140% | 49.600% |
| 8 | 5613 | 4720 | 59.034% | 49.642% |
| 9 | 5608 | 4739 | 58.982% | 49.842% |
| 10 | 5617 | 4776 | 59.077% | 50.231% |
| 11 | 5644 | 4779 | 59.361% | 50.263% |
| 12 | 5632 | 4781 | 59.234% | 50.284% |
| 13 | 5626 | 4773 | 59.171% | 50.200% |
| 14 | 5601 | 4769 | 58.908% | 50.158% |
| 15 | 5645 | 4778 | 59.371% | 50.252% |
| 16 | 5612 | 4756 | 59.024% | 50.042% |
| 17 | 5635 | 4793 | 59.266% | 50.410% |

Best trained evaluation by Acc050 is epoch17, improving on epoch12 by
3/12 hits (+0.0316/+0.1262pp). Epoch15 remains the Acc025 high at5645
hits. Relative to baseline59.518%/46.992%, epoch17 is
-0.2521/+3.4182pp. Target minima are5705/4945 of9508;
epoch17 is short by70/152 hits (0.7341/1.5898pp). Do not combine the
epoch15 Acc025 and epoch17 Acc050 scores as one checkpoint.

Epoch17 latest checkpoint was CPU-matched to its complete metric row
and model/optimizer/scheduler/RNG. Retained it as
best_trained_epoch17.pth, 622927846 bytes, link_count2.
Built-in best.pth remains untrained epoch0. Epoch17 three-round means
59.2203%/50.2279% rose; decline streak resets to0 and no early stop
occurred. Epoch18 started; its first-update confirms frozen_core=true
and optimizer_step_complete=true. Epoch0 is excluded from early stopping.

## NR3D

Validation size: 7,899.

Current run1793419199, from the reproduced author checkpoint:

| Epoch | Hits@0.25 | Hits@0.50 | Acc@0.25 | Acc@0.50 |
| ---: | ---: | ---: | ---: | ---: |
| 0 (init) | 4739 | 4132 | 59.995% | 52.310% |

Epoch1 is training. No trained result or trained best exists yet for this
run. Own-run epoch1/batch1 confirms frozen_core=true and
optimizer_step_complete=true. Select trained checkpoints by Acc025 first,
Acc050 second, using tools/audit_preserve_nr_sr_best.sh after each new
complete validation. Built-in best.pth remains Acc050-first.

Historical stopped run1793382384 (not the current run):

| Epoch | Hits@0.25 | Hits@0.50 | Acc@0.25 | Acc@0.50 |
| ---: | ---: | ---: | ---: | ---: |
| 0 (init) | 4739 | 4132 | 59.995% | 52.310% |
| 1 | 4786 | 3918 | 60.590% | 49.601% |
| 2 | 4780 | 3909 | 60.514% | 49.487% |
| 3 | 4787 | 3902 | 60.603% | 49.399% |
| 4 | 4774 | 3918 | 60.438% | 49.601% |
| 5 | 4787 | 3902 | 60.603% | 49.399% |
| 6 | 4784 | 3910 | 60.565% | 49.500% |
| 7 | 4772 | 3900 | 60.394% | 49.373% |
| 8 | 4768 | 3925 | 60.362% | 49.690% |

This historical run stopped during epoch9; its last complete result is
epoch8. Historical Acc025-first best is epoch3/5 tied at4787 hits
(60.603%), 63 hits above target, but those checkpoints were not
hard-linked before overwrite. These metrics do not establish a trained
result for the current run1793419199. Interrupted run1793417914 also
has no completed trained validation and is excluded.


## SR3D

Validation size: 17,726.

Current run1793419200, from the reproduced author checkpoint:

| Epoch | Hits@0.25 | Hits@0.50 | Acc@0.25 | Acc@0.50 |
| ---: | ---: | ---: | ---: | ---: |
| 0 (init) | 11620 | 10023 | 65.553% | 56.544% |

Epoch1 is training. Own-run epoch1/batch1 confirms frozen_core=true and
optimizer_step_complete=true. There is no trained result yet. Use the
same Acc025-first checkpoint-retention procedure as NR3D.

Historical stopped run1793380711 (not the current run):

| Epoch | Hits@0.25 | Hits@0.50 | Acc@0.25 | Acc@0.50 |
| ---: | ---: | ---: | ---: | ---: |
| 0 (init) | 11620 | 10023 | 65.553% | 56.544% |
| 1 | 11944 | 10166 | 67.381% | 57.351% |
| 2 | 11934 | 10206 | 67.325% | 57.576% |
| 3 | 11947 | 10167 | 67.398% | 57.356% |

This historical run stopped during epoch4; its last complete result and
Acc025-first best is epoch3, 11947/10167 hits
(67.3982%/57.3564%), short of the Acc025 target by355 hits.
Interrupted run1793418888 has no completed trained validation and is
excluded. Do not reuse historical first-update audits for the current run.

## Baseline Reproduction Request

The requested plain author-checkpoint evaluations completed after the
user-directed NR/SR training interruption. Successful eval-only runs use
the original /tmp/PV-Ground/train_dist_mod.py, model class PVGround,
full validation, batch10, butd_cls, self_attend and last/bbf Top1.

| Dataset | Baseline run | Checkpoint epoch | Hits025 / Hits050 | Acc025 / Acc050 | Seed |
| --- | --- | ---: | --- | --- | ---: |
| NR3D | 1793417143 | 25 | 4888 / 3945 of7899 | 61.8813% / 49.9430% | 0 |
| SR3D | 1793417141 | 31 | 12152 / 10302 of17726 | 68.5547% / 58.1180% | 111 |

Counts are reconstructed from the rounded official rates and full split
sizes. NR baseline seed0 differs from module seed111; SR matches.
This is a provenance difference, not a measured cause of the module-init
gap. See the geometry replay note for control requirements. Current fresh
module runs were started after both baselines met reproduction tolerance.
The reproducible commands are tools/run_author_baseline_nr3d.sh and
tools/run_author_baseline_sr3d.sh; earlier model=MCLN attempts failed
before evaluation and produced no valid baseline result.

Paper reference: Table 2 of the [CVPR 2026 PV-Ground paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Shang_PV-Ground_Text-Guided_Point-Voxel_Interaction_for_3D_Visual_Grounding_CVPR_2026_paper.pdf)
reports two-stage MCLN at 59.8% NR3D / 68.4% SR3D and PV-Ground at
62.1% / 68.9%, all at Acc@0.25. The user's NR3D target matches reported
MCLN; the SR3D target 69.4% is 1.0 pp above MCLN and 0.5 pp above the
paper's PV-Ground result. Measured author-checkpoint Acc025 differs
from the matching PV-Ground row by -0.219pp NR3D and -0.345pp SR3D.

## Verified Operational Issue

The ScanRefer checkpoint audit showed that `best.pth` retention includes the epoch-0 initialization. This left ScanRefer's selected `best.pth` at epoch 0 even after trained rounds had become the best fine-tuned evaluations. `latest.pth` is overwritten as training advances, and in the audited ScanRefer run its retained-metric field described the historical retained best rather than the latest checkpoint's own evaluation.

For ScanRefer epochs 8 and 9 and NR3D epoch 1, the full checkpoint epoch was matched to `native_metrics.jsonl` and the model, optimizer, scheduler, and RNG payload was checked before a same-filesystem hard link was created to preserve the trained checkpoint. This is an operational preservation measure; it does not modify the training process or weights. Future result reporting must select among trained epochs separately from initialization and join each checkpoint to the native metric row for its own epoch.

## Three Module Roles and Testable Hypotheses

CPU state comparison of initialization epoch0 and retained epoch17 found
all1234 frozen state tensors unchanged in shape, dtype and SHA256,
including running statistics. All61 trainable parameter tensors changed:

| Module | Changed / total tensors | Changed / total scalar elements | Delta L2 |
| --- | ---: | ---: | ---: |
| Task-observation query reader | 37 / 37 | 771309 / 923616 | 5.540323 |
| Mask-support corrector | 10 / 10 | 27265 / 27841 | 4.286732 |
| Extremal span mixer | 14 / 14 | 29793 / 29793 | 4.249388 |

This supports actual frozen-core preservation through epoch17, beyond
the first-update marker. Parameter deltas include optimizer effects;
their magnitude does not measure module importance, gradient flow,
gate saturation or causal accuracy gains. The span output bias changed
from0 to0.0003249672, but its feature-dependent output must still be
measured on validation data before making saturation claims.

Reproducible CPU audit: tools/pvground_checkpoint_update_audit.py.
Raw result: docs/results/ScanRefer-epoch17-checkpoint-cpu-audit.json,
SHA2562fb9cd525bc5175945867d3b674e64dafe55bda2546ee1145626074105943b48.
One CPU negative-control test passes: detects changed frozen running
statistics, reports an unchanged module, and selects epoch17's own metric
instead of retained epoch0 best. Training remains live and unchanged;
the next empirical decision still needs same-forward GPU geometry replay.

The deployed model contains three modules whose intended roles are:

1. **Task-observation query reader:** separate semantic and geometry query transforms read shared observed memories, then use the decoder's shared attention parameters.
2. **Candidate mask-support corrector:** adds candidate-specific residuals to superpoint mask logits from query/support features and current mask state.
3. **Extremal span mixer:** aggregates per-axis face evidence from predicted mask support and mixes mask-derived and native box centers/sizes with bounded axis-wise coefficients.

These are implementation roles, not novelty claims. The current metric pattern motivates, but does not yet prove, the following tests:

- **ScanRefer:** separate Unique/Multiple performance and inspect whether semantic candidate selection is correct while final box IoU misses 0.50. Compare predicted-mask and native-box IoU and inspect per-axis span-mixer gate distributions before changing the corrector or mixer.
- **NR3D:** track whether the first-epoch Acc@0.25 gain and Acc@0.50 drop persist; distinguish candidate selection errors from box-extent calibration errors before dataset-specific internal-parameter tuning.
- **SR3D:** two trained rounds improve over initialization but leave a 2.075-point Acc@0.25 gap at epoch 2. Inspect relational subgroup accuracy and candidate selection before attributing the gap to a module.
- Across all datasets, use controlled ablations or internal-parameter sweeps while keeping the overall network and three-module boundaries fixed. Prefer a dataset-specific internal setting only when the corresponding dataset diagnostics support it; do not infer cross-dataset benefit from one dataset.

Potential ideas to study, without copying implementations or comparing incompatible headline metrics:

- [ConcreteNet](https://github.com/ouenal/concretenet): inter-instance attentive fusion and contrastive grounding; its Unique/Multiple split motivates subgroup diagnostics. The repository has a non-commercial-use license, so code reuse requires a separate licensing decision.
- [3D-SPS](https://github.com/fjhzhixi/3D-SPS): language-guided coarse selection followed by target-oriented progressive mining; a conceptual reference for testing staged evidence use in the query-reader path. MIT licensed.
- [3DRefTR](https://github.com/Leon1207/3DRefTR): a unified REC/RES model with superpoint mask prediction, relevant to mask-support diagnostics. Its reported REC metric is not interchangeable with this experiment's native Acc@IoU.

No architectural or module change is justified by the current aggregate metrics alone. First complete and audit the next full validation rows, then use component-level and subset-level evidence to choose a narrow experiment.

## Checkpoint and Monitoring References

- ScanRefer run root: `/root/autodl-tmp/pvground_scanrefer_module_ft_20261030`; native metrics: `logs_semantic/scanrefer/scanrefer_module_ft_semantic/1793345477/native_metrics.jsonl`.
- NR3D run root: `/root/autodl-tmp/pvground_nr3d_module_training_20261030`; native metrics: `logs/nr3d/nr3d_module_ft/1793382384/native_metrics.jsonl`.
- SR3D run root: `/root/autodl-tmp/pvground_sr3d_module_training_20261030`; native metrics: `logs/sr3d/sr3d_module_ft/1793380711/native_metrics.jsonl`.

This record is a snapshot, not a final result. Append later full evaluations with exact row counts, metric provenance, checkpoint epoch, and target deltas.

## Follow-up status: 15:35 America/New_York

No new complete native validation rows were present. ScanRefer epoch 10 was at 2,470/4,865 training batches, NR3D epoch 2 at 1,985/4,490, and SR3D epoch 1 at 6,308/7,783. All three screens and GPUs were active and the checked log tails showed no errors. The latest complete evaluations remain ScanRefer epoch 9, NR3D epoch 1, and SR3D initialization epoch 0. No run parameters or processes were changed.

## Follow-up status: 15:40 America/New_York

No new complete native validation rows were present. ScanRefer epoch 10 was at 2,906/4,865 training batches, NR3D epoch 2 at 2,425/4,490, and SR3D epoch 1 at 6,757/7,783. All three screens and RTX 4090 GPUs remained active; checked logs contained no errors. The latest full-set `last/bbf`, `same_complete_model=true` evaluations remain ScanRefer epoch 9 (5,608/9,508 at IoU 0.25; 4,739/9,508 at IoU 0.50), NR3D epoch 1 (4,786/7,899; 3,918/7,899), and SR3D initialization epoch 0 (11,620/17,726; 10,023/17,726). Commands retain `lr=1e-5`, `lr_modules=1e-5`, `lr_backbone=2e-5`, and decay epochs `[10,30]`. No run settings or processes were changed.

## Follow-up status: 15:44 America/New_York

No new full evaluation row had appeared. ScanRefer epoch 10 was at 3,197/4,865 batches, NR3D epoch 2 at 2,725/4,490, and SR3D epoch 1 at 7,047/7,783. All three screens and GPUs remained active; checked log tails showed no errors. Metric histories and checkpoint state remain unchanged from the 15:40 audit.

## Follow-up status: 15:55 America/New_York

SR3D epoch 1 training had completed and its native validation was underway at 122/1,773 batches. ScanRefer epoch 10 was at 4,018/4,865 training batches and NR3D epoch 2 at 3,570/4,490. All screens and GPUs were active; checked logs showed no errors. No new full evaluation row is available yet. The SR3D epoch-0 initialization remains its only completed evaluation; after epoch-1 validation, match the native metric row to the checkpoint epoch and preserve the full trained state before it can be overwritten.

## Follow-up status: 16:04 America/New_York

SR3D epoch 1 native validation progressed to 884/1,773 batches. ScanRefer epoch 10 reached 4,649/4,865 training batches and NR3D epoch 2 reached 4,224/4,490. All three jobs and GPUs remained active; checked log tails showed no errors. There were no new complete native metric rows or checkpoint updates at this poll.

## Follow-up status: 16:14-16:20 America/New_York

SR3D epoch 1 completed its full native validation: hits@0.25/0.50=11944/10166 of 17726, Acc=67.381%/57.351%, `primary_score=last/bbf`, `same_complete_model=true`. Compared with its untrained epoch-0 initialization (65.553%/56.544%), this is +1.829/+0.807 percentage points; Acc@0.25 remains 2.019 points below the 69.4% target. The epoch-1 checkpoint epoch and full model/optimizer/scheduler/RNG payload were verified, then retained as `best_trained_epoch1.pth` using a same-filesystem hard link.

ScanRefer epoch 10 completed full validation: hits@0.25/0.50=5617/4776 of 9508, Acc=59.077%/50.231%, `last/bbf`, same complete model. Relative to baseline 59.518%/46.992%, the changes are -0.441/+3.239 points; gaps to targets 60.0%/52.0% are 0.923/1.769 points. Epoch 10 is the best trained evaluation by Acc@0.50. Its complete checkpoint was matched to the native row and retained as `best_trained_epoch10.pth`.

NR3D epoch 2 completed full validation: hits@0.25/0.50=4780/3909 of 7899, Acc=60.514%/49.487%, `last/bbf`, same complete model. Versus untrained epoch 0 (59.995%/52.310%), the changes are +0.519/-2.823 points. Acc@0.25 is 0.714 points above its 59.8% target. Epoch 1 remains the best trained checkpoint under the Acc@0.50-first selection rule.

At approximately 16:20, the next rounds were active: ScanRefer epoch 11 at 210/4865 batches, NR3D epoch 3 at 279/4490, and SR3D epoch 2 at 488/7783. All screens and GPUs were active and checked log tails were clean. The verified commands still use `lr=1e-5`, `lr_modules=1e-5`, `lr_backbone=2e-5`, and LR decay epochs `[10,30]`; no process or setting was changed.

At 16:30 America/New_York, no new validation rows had appeared. ScanRefer epoch 11 reached 965/4865 batches, NR3D epoch 3 reached 1056/4490, and SR3D epoch 2 reached 1258/7783. All three screens and GPUs remained active, checked logs contained no errors, and requested learning-rate/decay settings were unchanged.

Arithmetic correction at 16:32 America/New_York: raw NR3D hit counts and checkpoint selection are unchanged. Exact hits/7899 conversions are epoch 1 60.5899%/49.6012% and epoch 2 60.5140%/49.4873%; previous conversions for epoch-1 Acc@0.25 and epoch-2 Acc@0.50 were incorrect and are superseded.

## Follow-up status: 16:48 America/New_York

No new complete validation rows were available. ScanRefer epoch 11 was at 2298/4865 training batches, NR3D epoch 3 at 2425/4490, and SR3D epoch 2 at 2613/7783. All three screens and RTX 4090 GPUs were active; checked log tails showed no errors. Live arguments remain `lr=1e-5`, `lr_modules=1e-5`, `lr_backbone=2e-5`, and LR decay epochs `[10,30]`; no settings or processes were changed.

Latest completed native `last/bbf` semantic-alignment results remain ScanRefer epoch 10: hits@0.25/0.50=5617/4776 of 9508 (59.077%/50.231%); NR3D epoch 2: 4780/3909 of 7899 (60.514%/49.487%); SR3D epoch 1: 11944/10166 of 17726 (67.381%/57.351%). All rows have `same_complete_model=true`. The exact NR3D percentage correction above is reflected in the current status and this branch.

## Follow-up status: 17:00 America/New_York

All three runs remained active; no new full native validation rows or checked log errors. Progress: ScanRefer epoch 11 at 3167/4865 batches, NR3D epoch 3 at 3320/4490, and SR3D epoch 2 at 3485/7783. The live configurations still use `lr=1e-5`, `lr_modules=1e-5`, `lr_backbone=2e-5`, and decay epochs `[10,30]`; no settings or processes were changed.

Latest full `last/bbf`, `same_complete_model=true` results remain ScanRefer epoch 10 5617/4776 of 9508 (59.077%/50.231%), NR3D epoch 2 4780/3909 of 7899 (60.514%/49.487%), and SR3D epoch 1 11944/10166 of 17726 (67.381%/57.351%). Based on observed batch rates, NR3D full validation is expected around 17:25-17:30 and ScanRefer around 17:35; SR3D is later. The next check should be near the validation window.

## Follow-up status: 17:25 America/New_York

NR3D epoch 3 completed full native validation: hits@0.25/0.50=4787/3902 of 7899, Acc=60.603%/49.399%, `primary_score=last/bbf`, `same_complete_model=true`. Its 59.8% Acc@0.25 target is exceeded by 0.803 percentage points (63 hits). CPU audit confirmed `latest.pth` epoch 3 matches this native row and includes model, optimizer, scheduler, and RNG. Epoch 1 remains best trained by the Acc@0.50-first rule (3918 hits@0.50); epoch 4 has started at 36/4490 batches.

ScanRefer epoch 11 validation was at 223/951 batches; SR3D epoch 2 training was at 5393/7783. All three screens and GPUs remain active and checked logs are clean. Their verified learning rates and decay schedule remain `lr=1e-5`, `lr_modules=1e-5`, `lr_backbone=2e-5`, `[10,30]`; no settings or processes were changed. Latest complete results are ScanRefer epoch 10 5617/4776 of 9508 (59.077%/50.231%), NR3D epoch 3 4787/3902 of 7899 (60.603%/49.399%), and SR3D epoch 1 11944/10166 of 17726 (67.381%/57.351%).

## Follow-up status: 17:34 America/New_York

ScanRefer epoch 11 completed full native validation: hits@0.25/0.50=5644/4779 of 9508, Acc=59.361%/50.263%, `primary_score=last/bbf`, `same_complete_model=true`. Both thresholds increased versus epoch 10 by +0.284/+0.032 percentage points. It remains 0.639/1.737 points below the 60.0%/52.0% targets (61/166 hits short). Versus baseline 59.518%/46.992%, the change is -0.157/+3.271 points. CPU audit matched `latest.pth` epoch 11 with the full row and confirmed model/optimizer/scheduler/RNG; preserved as `best_trained_epoch11.pth` via hard link. It is now the best trained round by Acc@0.50.

NR3D epoch 3 completed full native validation at 4787/3902 of 7899 (60.603%/49.399%); this exceeds its 59.8% Acc@0.25 target by 0.803 points (63 hits). Its checkpoint epoch and full state were verified; epoch 1 remains best trained by Acc@0.50-first selection. NR3D epoch 4 was at 688/4490 batches.

SR3D epoch 2 completed full native validation at 11934/10206 hits of 17726 (67.325%/57.576%), `primary_score=last/bbf`, `same_complete_model=true`. Relative to epoch 1 (11944/10166; 67.381%/57.351%), Acc@0.25 changed by -0.056 pp and Acc@0.50 by +0.226 pp. Relative to epoch-0 initialization (11620/10023; 65.553%/56.544%), changes are +1.772/+1.032 pp. Acc@0.25 remains 2.075 pp below the 69.4% target; at least 12302/17726 hits are needed, leaving 368 hits. CPU audit matched `latest.pth` epoch 2 to this full row and confirmed model, optimizer, scheduler, and RNG state. Preserved as `best_trained_epoch2.pth` using a same-filesystem hard link (same inode, link count 2); epoch 2 is now best trained by Acc@0.50-first selection. SR3D epoch 3 was at 397/7783 training batches at 18:22.

NR3D epoch 4 completed full native validation at 4774/3918 of 7899 (60.438%/49.601%), `primary_score=last/bbf`, `same_complete_model=true`. This is +0.638 pp over its 59.8% Acc@0.25 target (50 hits above the 4724-hit minimum). Versus epoch 3, Acc@0.25 decreased 0.165 pp and Acc@0.50 increased 0.202 pp. CPU audit matched `latest.pth` epoch 4 to this row and confirmed model, optimizer, scheduler, and RNG. Epoch 1 and epoch 4 tie at 3918 Acc@0.50 hits; epoch 1 remains selected by the Acc@0.25 tie-break (4786 vs 4774 hits). NR3D epoch 5 had just started (6/4490 batches) at 18:35.

At 18:35 America/New_York, ScanRefer epoch 12 was at 4690/4865 batches and SR3D epoch 3 at 1408/7783. All three screens and GPUs remained active and checked logs were clean. The initial learning rates remain one tenth of original (`lr=1e-5`, `lr_modules=1e-5`, `lr_backbone=2e-5`), with decay epochs `[10,30]`; ScanRefer has passed decay epoch 10. No settings or processes were changed.

ScanRefer epoch 12 completed full native validation at hits@0.25/0.50=5632/4781 of 9508, Acc=59.2343%/50.2840%, `primary_score=last/bbf`, `same_complete_model=true`. Versus epoch 11, Acc@0.25 decreased 0.1262 pp and Acc@0.50 increased 0.0210 pp. It is now the best trained evaluation by Acc@0.50-first selection. Compared with baseline 59.518%/46.992%, changes are -0.2837/+3.2920 pp. The 60.0%/52.0% targets require 5705/4945 hits, leaving 73/164 hits (0.7657/1.7160 pp). CPU audit matched `latest.pth` epoch 12 to this exact native row and confirmed model/optimizer/scheduler/RNG; preserved as `best_trained_epoch12.pth` via hard link.

At 18:48 America/New_York, ScanRefer epoch 13 was at 17/4865 batches, NR3D epoch 5 at 167/4490, and SR3D epoch 3 at 1408/7783. All screens/GPUs were active, checked logs clean, and first-update audits confirm frozen core and completed optimizer steps. Requested initial rates (`lr=1e-5`, `lr_modules=1e-5`, `lr_backbone=2e-5`) and decay `[10,30]` remain unchanged. ScanRefer has passed the epoch-10 decay. Its verified automatic early-stop decline streak is 0, not due.

## Validated Results and Status: 18:35-19:07 America/New_York

The following subsequent full validations extend the tables above. Each was accepted from the native full validation row with `primary_score=last/bbf` and `same_complete_model=true`; checkpoint epoch was matched to that row and the complete model/optimizer/scheduler/RNG payload was checked before preserving the selected trained checkpoint:

- NR3D epoch 3: 4787/7899 hits@0.25 (60.603%) and 3902/7899 hits@0.50 (49.399%). The Acc@0.25 target is exceeded by 0.803 pp (63 hits); epoch 1 remains selected best by Acc@0.50-first.
- ScanRefer epoch 11: 5644/9508 (59.361%) and 4779/9508 (50.263%). Versus epoch 10, +0.284/+0.032 pp; target gaps were 0.639/1.737 pp.
- SR3D epoch 2: 11934/17726 (67.325%) and 10206/17726 (57.576%). Versus epoch 1, -0.056/+0.226 pp; 368 hits short of the 69.4% Acc@0.25 target. Epoch 2 is the current best trained checkpoint by the selection rule.
- NR3D epoch 4: 4774/7899 (60.438%) and 3918/7899 (49.601%). The Acc@0.25 target is exceeded by 0.638 pp (50 hits). Epochs 1 and 4 tie on Acc@0.50; epoch 1 remains selected by its higher Acc@0.25.
- ScanRefer epoch 12: 5632/9508 (59.234%) and 4781/9508 (50.284%). Versus epoch 11, -0.126/+0.021 pp. This is the best trained ScanRefer round by Acc@0.50. Against baseline 59.518%/46.992%, changes are -0.284/+3.292 pp; target gaps are 73/164 hits (0.766/1.716 pp). Its checkpoint is preserved as `best_trained_epoch12.pth`.

At approximately 19:07, all three jobs remained active without checked log errors. ScanRefer epoch 13 was at 1474/4865 batches, NR3D epoch 5 at 2588/4490, and SR3D epoch 3 at 3452/7783. Current first-update records confirm frozen core and completed optimizer steps. Live NR3D/SR3D commands still use `lr=1e-5`, `lr_modules=1e-5`, `lr_backbone=2e-5`, and decay `[10,30]`; ScanRefer's logged post-epoch-10 rates are `1e-6`, `1e-6`, and `2e-6`. ScanRefer's automatic early-stop decline streak is 0, not due.

## Current Diagnosis and Next Evidence

### Initialization Is Not Function Preserving

Live NR3D/SR3D source inspection at approximately20:34 America/New_York
found zero output layers in the reader and support corrector, including
zero task-query residuals. Strict core checkpoint loading is present.
The verified nonidentity is downstream geometry:
`native_model_initialization.py` sets `use_predicted_mask_reference=True`;
`native_mask_geometry.py` replaces `last_center`/`last_pred_size` with
thresholded fused-mask support extents. `ExtremalSpanMixer` initializes its
output weight and bias to zero, so its clamped gate starts at0 and
`final=(1-gate)*mask+gate*native` selects mask geometry. Original author
forward in `/tmp/PV-Ground/models/pv_ground.py` returns without this added
geometry replacement. The detected-box filter depends on geometry, so the
switch can change both candidate filtering and IoU even with unchanged
semantic scores.

This proves a nonidentity insertion, not its measured share of the accuracy
gap. NR3D/SR3D epoch0 are modified-model evaluations (59.995/52.310% and
65.553/56.544%), not same-environment plain-model controls. No matching
plain-model full validation was established in this inspection. The
59.8/69.4% targets must not be relabeled as measured plain-model baselines.
An epoch0 difference cannot be caused by optimizer learning rate.

`tools/pvground_span_initialization_cpu.py` checks the actual deployed
mixer on synthetic CPU inputs. Zero weight and biases0/0.5/1 produce
exact mask/midpoint/native geometry. Under torch2.1.2, the output bias
gradient of the synthetic sum objective is nonzero (-2304) at all three,
including the clamp endpoints. This is not a real-loss or end-to-end
accuracy test. Source SHA256:
`5ab81a56ffcd635eb61ff245f6771c40d20027b7708656b09e8d92bf3f5ac087`.
Run from the deployed code directory with
`PYTHONPATH=. python /path/to/pvground_span_initialization_cpu.py`.

The official [LoRA Linear implementation](https://github.com/microsoft/LoRA/blob/main/loralib/layers.py)
zeros lora_B, and [ControlNet](https://github.com/lllyasviel/ControlNet/blob/main/cldm/cldm.py)
uses zero convolutions. Both source files were read directly with HTTP200.
The applicable principle is an initially zero incremental path preserving
the pretrained function, not adopting either architecture or claiming
their results transfer. No external code was copied.

Next compare original model, module-equipped native-box path, and current
mask-box path on identical inputs/configuration. Check reader/mask
equivalence and separate fixed-candidate geometry from full reranking.
Prepare a native-endpoint initialization only after this comparison;
gate1 selects native boxes under the current equation. No running weights,
configuration, process, or module architecture changed.

At20:37, all screens/GPUs were active: ScanRefer e14=2684/4865; NR3D
e6=4270/4490; SR3D e3 validation=835/1773. No new full validation row.

Cross-domain reference checked on 2026-10-30: the [MMDetection GFL implementation](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/losses/gfocal_loss.py) supervises positive classification outputs with localization-quality (IoU) targets and modulates by prediction/target discrepancy. Its [GFL description](https://github.com/open-mmlab/mmdetection/blob/main/configs/gfl/README.md) explains the joint quality/class representation. Both raw GitHub source requests returned HTTP 200 and their contents were inspected. This motivates checking score/box-quality consistency in the existing query reader, not adding a fourth quality head or changing the deployed ranker. A training-only experiment would require proving the existing semantic-score gradient reaches the reader and defining an appropriate objective: our aggregated token score is not a class probability and cannot simply be inserted into binary cross entropy. First measure matched candidate IDs and quality inversions; do not deploy a new objective from Top-10 aggregate gaps alone. No external code was copied.

The log parser has three passing standard-library tests: source separation (position/mask values ignored, incomplete epoch omitted), rejection of native-hit mismatch, and rejection of incomplete dataset rows. Command: `python -m unittest discover -s tools -p test_pvground_log_diagnostics.py -v`. Actual-log validation remains separate evidence for all three datasets.

The standalone standard-library tool `tools/pvground_log_diagnostics.py` extracts each completed epoch's semantic box Top-1/5/10 and subgroup rates from its original log and checks Top-1 against `native_metrics.jsonl` within the log's five-decimal rounding. It was run successfully on all three active logs (ScanRefer epochs 0-13, NR3D 0-5, SR3D 0-2). Acc@0.50 Top-1/Top-10 is 50.200%/66.649% for ScanRefer epoch 13, 49.399%/68.338% for NR3D epoch 5, and 57.576%/74.743% for SR3D epoch 2. The gaps show that some failed top-1 cases have a qualifying box within the first ten ranked candidates. They motivate ranking diagnostics but are neither deployable accuracies nor guaranteed recoverable gains. NR3D/SR3D log Unique=0 is not interpretable as failure on a populated Unique subgroup without subgroup denominators.

Source-audited epoch-13 subgroup evidence (2026-10-30 approximately 20:12 America/New_York): the existing semantic box evaluator reports Unique Acc@0.25/0.50=88.372093%/78.576462% and Multiple=54.048708%/45.221906%. Integer reconstruction from both rates, total rows, and global hit pairs yields Unique 1419 rows (1254/1115 hits) and Multiple 8089 rows (4372/3658 hits), exactly summing to 5626/4773. These subgroup counts are inferred from log ratios rather than exported counters. Most failures are in Multiple, but ranking versus localization remains unresolved. The log's mask overall25/50 values (61.379891%/53.712663%) are mask IoU and must not substitute for box targets.

Active source already exposes mask/native/final boxes and span gate/source-fraction tensors. An offline replay should first reproduce the retained checkpoint's exact metric pair, then compare box sources at a fixed formal candidate ID and separately with original filtering/ranking. `--butd_cls` uses geometry-dependent evaluator filtering, so changing box source can change selection. Capture candidate coverage, top-1 success, repairs/damages, Unique/Multiple splits, and raw/clamped gate boundary mass before choosing internal module parameters. Hard gate clipping is present in source; saturation has not yet been measured. No live training source or process was modified and no additional GPU evaluation was launched alongside the occupied training jobs.

- **ScanRefer:** Acc@0.50 has risen over the initial checkpoint, but the current trained best is still short of both goals, especially the 1.716 pp Acc@0.50 gap. Epoch 11-to-12 moved the two thresholds in opposite directions. A localization/extent-precision bottleneck is a working hypothesis only. Before changing module-internal parameters, inspect Unique/Multiple results, candidate selection correctness, mask/native-box IoU, and extremal-span gate distributions. A simple threshold trade-off is not enough evidence to identify the responsible module.
- **NR3D:** Acc@0.25 is above goal in multiple trained rounds, while Acc@0.50 remains below its initialization value and fluctuates across rounds. This suggests checking candidate ranking separately from final-box precision. Preserve the existing best-round rule; do not treat the epoch-4 Acc@0.50 tie as a new best.
- **SR3D:** Two trained evaluations show gains over initialization, but Acc@0.25 is still 2.075 pp below target. Acc@0.50 improved at epoch 2 while Acc@0.25 was essentially flat. The next full rounds and component-level localization evidence are needed before choosing an internal-depth or gate-parameter sweep.

The three module boundaries and overall architecture remain unchanged. Public repositories already identified for mechanism study are ConcreteNet (Unique/Multiple relational behavior; non-commercial license), 3D-SPS (coarse-to-fine language-guided selection; MIT), and 3DRefTR (superpoint mask prediction); their reported headline metrics are not interchangeable with this native protocol. No method change is claimed or deployed from aggregate evidence alone. Current rates remain one tenth of original with decay `[10,30]`; no training process or live setting was modified in this update.

## NR3D Epoch 5 and Live Status: 19:43 America/New_York

NR3D epoch 5 completed full native validation: 4787/7899 hits@0.25 (60.603%) and 3902/7899 hits@0.50 (49.399%), `primary_score=last/bbf`, `same_complete_model=true`. The 59.8% Acc@0.25 target is exceeded by 0.803 pp (63 hits). Its `latest.pth` epoch 5 was matched to the native row; model, optimizer, scheduler, and RNG payloads are present. The result exactly repeats epoch 3, so it does not alter the best-round selection: epoch 1 remains best trained by Acc@0.50-first (3918 hits@0.50 versus 3902 at epochs 3 and 5). Epoch 6 is training.

At approximately 19:43, ScanRefer epoch 13 was at 4269/4865 batches, NR3D epoch 6 at 186/4490, and SR3D epoch 3 at 5519/7783. All three screens and GPUs remained active with checked logs clean. ScanRefer still has no validation row beyond epoch 12 and its automatic early-stop decline streak remains 0. The one-tenth initial LRs and `[10,30]` decay schedule remain unchanged; no training process or setting was modified.

## ScanRefer Epoch 15 and Live Status: 22:32 America/New_York

ScanRefer epoch15 completed full native validation with 5645/9508 hits@0.25
(59.3711%) and 4778/9508 hits@0.50 (50.2524%); the row is
`primary_score=last/bbf` and `same_complete_model=true`. CPU audit matched
`latest.pth` epoch15 to the exact native row and confirmed model, optimizer,
scheduler, and RNG payloads. Epoch15 is the new Acc@0.25 high by one hit
over epoch11; epoch12 remains best by Acc@0.50. Relative to baseline
59.518%/46.992%, epoch15 is -0.147/+3.260 percentage points. It remains
60/167 hits short of the 60.0%/52.0% targets. The three-round means are
59.150189%/50.203338%; automatic early-stop decline streak is 0. Epoch16
training began and reached 177/4865 batches at the latest check.

NR3D epoch8 was training at 2800/4490, with no new full validation beyond
epoch7 (4772/3900 of 7899 = 60.3940%/49.3733%). Its Acc@0.25-first best
remains epoch3/5 tied at 4787 hits (60.603%). SR3D epoch4 was at 5761/7783,
with epoch3 still the latest full row and Acc@0.25-best: 11947/10167 of
17726 = 67.3982%/57.3564%. NMB GPUs remain occupied; prepared plain
author-checkpoint evaluations have not been launched. No training code,
settings, processes, or checkpoints were modified in this monitoring check.

## NR3D Epoch 8 and Current Progress: 23:06 America/New_York

NR3D epoch 8 completed full native validation: 4768/7899 hits@0.25
(60.3621%) and 3925/7899 hits@0.50 (49.6898%). The row has
`primary_score=last/bbf` and `same_complete_model=true`. CPU audit matched
`latest.pth` epoch 8 to this exact row and confirmed model, optimizer,
scheduler, and RNG state. This is not an Acc@0.25 best; epochs 3 and 5
remain tied at 4787/7899 (60.603%). The prior epoch-3/5 checkpoint
preservation gap remains; do not claim those checkpoint files exist.

At the same check, ScanRefer epoch 16 was at 2736/4865 training batches,
NR3D epoch 9 at 30/4490, and SR3D epoch 4 at 7562/7783. All three screens
and GPU processes remained active; no errors or early-stop event were
observed. The author-checkpoint baselines remain pending because neither
NR3D nor SR3D has released its GPU. SR3D briefly read 0% utilization on
one poll, but its process and 14,108 MiB allocation remained active;
utilization alone is not a GPU release signal. No run was interrupted.

## User-Directed Baseline Reproduction Pause

At 23:10 America/New_York, the user instructed stopping NR3D and SR3D
module fine-tuning until the plain paper-repository checkpoint baseline is
reproduced close to the paper, then restarting fine-tuning from that
baseline. Sent Ctrl-C to both named screen sessions. Both exited with
torch.distributed `SignalException` (signal 2); verification found no
training screen/process and GPU memory at 0 MiB (NR3D) and 1 MiB (SR3D).
ScanRefer remains active and was not interrupted.

The last complete module-fine-tuning validations remain NR3D epoch 8:
4768/7899 (60.3621%/49.6898%), and SR3D epoch 3: 11947/17726
(67.3982%/57.3564%). In-flight NR3D epoch 9 (30/4490) and SR3D epoch 4
(7562/7783) were incomplete at stop and are not metric rows. Previous
complete trained scores and checkpoints remain on the hosts.

Started pure author-checkpoint `--eval` jobs, batch size 10. NR3D screen
`pvg_nr3d_author_baseline`, run ID `1793416414`, checkpoint
`PV-Ground_NR3D.pth`; SR3D screen `pvg_sr3d_author_baseline`, same run ID,
checkpoint `PV-Ground_SR3D.pth`. Initial saved configs confirm `eval=true`,
correct per-dataset validation split, and official `butd_cls`,
`self_attend`, and `joint_det` switches. At launch both jobs were loading
validation data; baseline metrics are pending.

Compare these checkpoints to the paper's PV-Ground results, not the MCLN
row: NR3D 62.1%, SR3D 68.9% Acc@0.25. MCLN is a different model (59.8% and
68.4%). Treat within approximately 1.0 pp as comparable; otherwise audit
checkpoint/config/split/protocol before restarting the corresponding
fine-tune. Keep measured baseline, paper figure, module-init epoch0, and
fine-tuned scores separate. NR3D/SR3D selection remains Acc@0.25 first,
then Acc@0.50.

## Author Baseline Evaluation Update: 23:42 America/New_York

The first prepared baseline launch used `--model MCLN` and failed before
evaluation with `NameError`; the actual author checkout's
`models/__init__.py` exports `PVGround`, and `train_dist_mod.py` imports
only that class. Corrected both eval-only commands to `--model PVGround`.
Both subsequent runs report that their corresponding PV-Ground checkpoint
loaded successfully.

NR3D author-checkpoint evaluation completed all 790 validation batches
using `PV-Ground_NR3D.pth` (stored checkpoint epoch 25). Official log
`last_ semantic alignment` Top-1 is 61.8813% Acc@0.25 and 49.9430%
Acc@0.50, equivalent to 4888/7899 and 3945/7899. Against PV-Ground's own
paper NR3D Acc@0.25=62.1%, delta is -0.219 pp. This is within the
approximately 1 pp reproduction tolerance. The old module fine-tune had
been stopped by user request; restarted the configured NR3D three-module
fine-tune from its author-checkpoint initialization at run
`1793417914`. Saved config confirms lr=1e-5, backbone lr=2e-5, batch10,
seed111, decay `[10,30]`, eval=false. First update audit is pending.

SR3D author-checkpoint eval remains active at 962/1773 validation batches
on the latest observation. It loaded `PV-Ground_SR3D.pth`. No complete
baseline score is available. Compare to PV-Ground paper Acc@0.25=68.9%;
keep MCLN's 68.4% and the user's 69.4% target identified separately.
Do not resume SR3D fine-tuning until the baseline is complete and within
about 1 pp of the matching PV-Ground paper score.

## ScanRefer Epoch 16 and Follow-up: 23:47 America/New_York

ScanRefer epoch16 completed full validation: 5612/9508 hits@0.25 (59.024%)
and 4756/9508 hits@0.50 (50.042%), `primary_score=last/bbf`,
`same_complete_model=true`. CPU audit matched `latest.pth` epoch16 and
confirmed model/optimizer/scheduler. It is not a new best at either
threshold: epoch15 remains the Acc@0.25 high (5645 hits), epoch12 the
Acc@0.50 high (4781 hits).

NR3D restarted fine-tune run `1793417914` is active; latest poll showed
initial validation at 280/790. This is a fresh run from the author checkpoint
through the existing native-init spec, not continuation from stopped epoch9.
The run's own first-update audit must be checked before asserting its
frozen-core/optimizer-step witness.

SR3D author baseline is at 1720/1773 validation batches; its screen remains
active. No full metric yet and SR3D fine-tuning remains stopped. The new
baseline scripts and earlier documentation correction were pushed in commits
`3ad73fa` and `683f301` on `codex/pvground-scanrefer-nr3d-sr3d-20261030`.

## Baselines Complete and Fresh Runs: 2026-10-31

Both author-checkpoint evaluations completed on the full validation splits.
These measurements are from the unmodified author model's `last_ semantic
alignment` Top-1 logs, separate from module-initialized epoch0:

| Dataset | Author checkpoint epoch | Acc@0.25 | Acc@0.50 | PV-Ground paper Acc@0.25 | Difference (pp) |
| --- | --- | --- | --- | --- | --- |
| NR3D | 25 | 61.8813% (4888/7899) | 49.9430% (3945/7899) | 62.1% | -0.219 |
| SR3D | 31 | 68.555% (12152/17726) | 58.118% (10302/17726) | 68.9% | -0.345 |

Hit counts are reconstructed from the logged rounded rates and full split
sizes. MCLN's separate paper scores (59.8% / 68.4%) are not the comparator
for these PV-Ground checkpoints; the user's SR3D 69.4% goal is also distinct.

The user requested stopping NR3D/SR3D until baseline reproduction was close
to the paper, then fine-tuning from the reproduced baseline. The running
module jobs were interrupted and process/screen exit and idle GPUs verified.
Because both measured baselines were within 1 pp of their matching paper
results, fresh fine-tunes were launched from the corresponding author
checkpoint specs at approximately 2026-10-31 00:00 America/New_York:
NR3D run `1793419199` and SR3D run `1793419200`.

Both configs retain lr/module lr `1e-5`, backbone lr `2e-5`, batch10,
seed111, decay `[10,30]`, joint detection, `butd_cls`, and `self_attend`.
The newly added modules are freshly initialized. The new runs' own
first-update witnesses are pending; older runs' audits and metrics cannot
validate these runs. Passing author-model reproduction does not establish
function-preserving module insertion: the existing mask-geometry
replacement remains a separate initialization difference.

ScanRefer continues unchanged. Monitoring retains NR3D/SR3D Acc@0.25-first
selection, exact full-split metric checks, and 180-300 second polling.

## New-run First Update and Geometry Diagnostic: 2026-10-31

NR3D run1793419199 completed its own initialization on all7899 rows:
4739/4132 hits =59.9949%/52.3104%, last/bbf, same_complete_model=true.
Its epoch1/batch1 audit confirms frozen_core=true and
optimizer_step_complete=true; training was159/4490 at00:16 New York.
The initial changes versus the reproduced author model are
-1.8863/+2.3674pp. New SR3D run1793419200 initialization was548/1773
batches at00:20; do not substitute the stopped run's metric or audit.

Added `tools/pvground_geometry_replay.py`, reusing the deployed semantic
evaluator on the same forward outputs for final/native/mask geometry.
Official geometry-dependent filtering and unfiltered fixed-query
diagnostics are reported separately, including gate endpoint/saturation
mass. A selected trained checkpoint must reproduce its own epoch's
recorded full metric pair; retained_metrics is not its own score.
Four CPU contract checks passed on the deployed evaluator with GPU
disabled. No full GPU replay or live model change has been performed.
The idle-GPU execution procedure, mechanism evidence, baseline table,
limits, and cross-domain sources are in
`docs/PV_GROUND_GEOMETRY_REPLAY_2026-10-31.md`.

## Current-run Audit and Acc025 Retention: 2026-10-31 00:40 New York

SR3D run1793419200 completed its own epoch0: 11620/17726 and
10023/17726 =65.553%/56.544%, last/bbf, same_complete_model=true.
Its own epoch1/batch1 witness confirms frozen_core=true and
optimizer_step_complete=true. This replaces the earlier pending status;
the initialization changes versus author baseline remain
-3.0012/-1.5740 pp. NR3D's current-run initialization remains
4739/4132 of7899 =59.995%/52.310%, also with its own valid first-update.

At00:40, all three screens/GPUs were live: ScanRefer epoch17
4214/4865, NR3D epoch1 1944/4490, SR3D epoch1 75/7783.
No new trained validation has completed. ScanRefer's retained trained
Acc050-best remains epoch12, 5632/4781 of9508.

User reiterated NR3D/SR3D selection by Acc025 first, then Acc050.
Source inspection found the live trainer still saves built-in best.pth
by Acc050 first. Added tools/audit_preserve_nr_sr_best.sh, executed
successfully on both hosts against the current run IDs. It validates
full split sizes/provenance, CPU-loads checkpoint metadata and matches
its own epoch to native_metrics.jsonl, excludes initialization from
trained selection, and hard-links latest.pth when it is the trained
Acc025-first best. The writer's temporary-file/os.replace contract
preserves that inode on future saves. Both runs currently have only
best.pth epoch0 and no latest.pth, so no trained alias was created.
Execute after new trained validations before the next save; monitoring
must report a missing selected alias rather than claim historical
weights are retained. No live training code/config/process was modified.
