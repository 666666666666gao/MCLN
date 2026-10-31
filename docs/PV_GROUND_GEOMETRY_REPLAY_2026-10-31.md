# PV-Ground Geometry Replay

## Observed Problem

The author-checkpoint baselines reproduce the matching PV-Ground paper
Acc@0.25 values closely: NR3D 61.8813% versus 62.1%, SR3D 68.555% versus
68.9%. These measurements establish the plain model's baseline, not
function-preserving insertion of our three modules.

| Dataset / source | Rows | Hits025 | Hits050 | Acc025 | Acc050 |
| --- | ---: | ---: | ---: | ---: | ---: |
| NR3D author baseline | 7899 | 4888 | 3945 | 61.8813% | 49.9430% |
| NR3D new run1793419199 epoch0 | 7899 | 4739 | 4132 | 59.9949% | 52.3104% |
| SR3D author baseline | 17726 | 12152 | 10302 | 68.5547% | 58.1180% |
| SR3D new run1793419200 epoch0 | 17726 | 11620 | 10023 | 65.5534% | 56.5441% |

Author-baseline hit counts are uniquely reconstructed from the rounded
official semantic Top-1 logs and full split sizes. Module-equipped counts
come from native_metrics.jsonl. NR3D initialization changes are
-1.8863/+2.3674 pp; SR3D initialization changes are
-3.0012/-1.5740 pp. The new SR3D run1793419200 epoch0 and its own
epoch1/batch1 frozen_core/optimizer_step_complete witnesses were directly
verified at00:40 America/New_York; these are current-run measurements.

Read-only checks of actual author-baseline configs also found NR3D used
rng_seed=0, whereas its module run uses111. SR3D baseline and module run
both use111. The author dataset source sets augment only for split=train;
point/color augmentation and box jitter are train-gated. This inspection
did not establish random validation sampling or a measurable seed effect.
Keep the NR3D seed difference in provenance; do not assign any portion of
the accuracy gap to it without a matched evaluation. A plain author model
with the module run's seed is still a pending control after GPU release.
Even matching the seed alone does not prove paired inputs across different
entrypoints. The same-forward geometry replay avoids this confound for
its internal geometry comparison, but does not test plain-model identity.

NR3D run1793419199 first_update.json confirms epoch1/batch1,
frozen_core=true and optimizer_step_complete=true. It is training from
the author checkpoint with fresh modules; it is not a resumed old epoch9.

## Mechanism Confirmed in Deployed Source

The reader output and support-corrector residual start at zero. However,
native_mask_geometry replaces last_center/last_pred_size with the extent of
the fused predicted mask. The span mixer's output weight and bias start at
zero, giving gate0 in:

`final = (1 - gate) * mask + gate * native`

Thus zero-initialized extra modules do not preserve the original
regression-box function. The formal evaluator's detected-box filter also
depends on candidate geometry, so a box switch may change both IoU and
which query is selected. This source evidence explains why identity is not
guaranteed; the measured causal contribution still requires paired replay.
LR cannot explain epoch0 before any optimizer update.

## New Diagnostic Tool

`tools/pvground_geometry_replay.py` wraps an eval-only invocation of the
actual deployed train_dist_mod.py. It does one model forward per batch and
reuses the deployed GroundingEvaluator.evaluate_bbox_by_sem_align for:

| Geometry | Official geometry-dependent filter | No geometry filter |
| --- | --- | --- |
| Final model output | Reproduces the formal result | Fixed query ordering from unchanged semantic scores |
| Native regression | Counterfactual, filter recalculated | Same unfiltered query ordering |
| Predicted-mask extent | Counterfactual, filter recalculated | Same unfiltered query ordering |

Read-only deployed configuration audit found ScanRefer uses butd=true,
butd_cls=false, augment_det=true; NR3D/SR3D use butd_cls=true.
GroundingEvaluator's filter_non_gt_boxes follows butd_cls, so ScanRefer's
formal filter is disabled. The two protocol columns then coincide.
The tool now preserves the original evaluator's setting without requiring
butd_cls for all datasets. This fixes a diagnostic-only assertion that
would have rejected ScanRefer after evaluation; it does not change the
training configuration or any formal metric.

The unfiltered protocol is diagnostic only and is not a replacement paper
metric. The native-box replay uses the modified model's scores and features;
it is not an independent proof of identity with the plain author model.
No GT is used to generate or select a replacement box.

The tool reports complete row/hit counts, all-query-axis gate mean, exact
clamped endpoint mass, raw out-of-range mass, init spec, checkpoint path,
and deployed source hashes. Every replay requires --reference-metrics.
The reference is selected by the checkpoint's own epoch (epoch0 for fresh
module initialization), never its retained_metrics historical-best field. Final official
hits must exactly match the recorded row before a replay result is saved.
Partial splits and disagreements fail the check.

Run from the deployed code directory with one torchrun worker, using the
dataset's verified arguments and adding:

```text
--eval
--replay-output /new/results/geometry_replay.json
--reference-metrics /original/run/native_metrics.jsonl
```

Replace the train_dist_mod.py entrypoint with the replay tool. For a trained
model also pass the selected complete checkpoint through --checkpoint_path;
for the author-initialized module model, omit --checkpoint_path and keep
its checked native-init spec, matching reference epoch0. Use a new eval log
directory. The tool rejects training mode and an existing result path.

The tool was uploaded only to /tmp for CPU tests on NR3D and ScanRefer.
tools/run_scanrefer_geometry_replay.sh provides the exact verified
ScanRefer epoch12 command, including butd (not butd_cls), augment_det,
batch10, seed111, four workers, checked native-init path, retained full
epoch12 checkpoint, and the original run's reference metrics. Give it a
new absolute result filename. It rejects an occupied GPU using nvidia-smi
compute-process evidence and must only be launched after the training
process terminates. New diagnostic logs use logs_geometry_replay.
No full GPU replay has
been run, and the eval-only entrypoint has not yet been validated end to
end. Wait for the appropriate training process to terminate or a verified
idle GPU. Do not stop training or modify a live model to run this tool.

Five CPU tests passed against both actual deployed evaluators with
CUDA_VISIBLE_DEVICES empty: (1) a controlled example distinguishing
geometry-dependent filtering from fixed-query box quality, while
preserving every input tensor; (2) rejection of incomplete row counts or
disagreement with the original evaluator; (3) matching the checkpoint's
own epoch instead of its retained best, and rejecting a mismatched pair.
The fourth check rejects a training-mode invocation before creating its
output. The fifth verifies that a disabled official filter matches the
unfiltered protocol across all three geometries. These tests are
synthetic protocol checks, not dataset performance.

## Next Decision

First reproduce the selected checkpoint's formal pair and compare the
geometry protocols. A native-endpoint initialization (output bias1 under
the existing equation) is a minimal within-module candidate only if paired
results support it; it has not been applied. Do not add another module or
claim that aggregate Top-10 headroom guarantees a gain.

Both cross-domain reference implementations were re-read on 2026-10-31
with HTTP200 responses:

- Microsoft LoRA, loralib/layers.py Linear.reset_parameters zeros lora_B:
  https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
- ControlNet, cldm/cldm.py make_zero_conv wraps its convolution in
  zero_module:
  https://github.com/lllyasviel/ControlNet/blob/main/cldm/cldm.py

The transferable idea is preserving the pretrained function at insertion;
these references do not validate our geometry choice or establish novelty.
The three module boundaries and current training parameters remain intact.
