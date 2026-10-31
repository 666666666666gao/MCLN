# PV-Ground Supervision Audit

Observed 2026-10-31 07:24 America/New_York on current SR3D run1793419200.
This was a read-only source inspection, not a gradient or performance test.
No training configuration, weights, architecture or module was changed.

## Confirmed Paths

| Component | Observed training path | Limit of the evidence |
| --- | --- | --- |
| Task observation reader | Separate semantic/geometric query transforms with shared attention/output/tail parameters | Source wiring does not measure each branch's real gradient |
| Support corrector | Corrected mask logits enter native mask losses and the additional selected-query mask objective | A hard extent conversion blocks the direct mask-to-box gradient |
| Span mixer | Final center/size enter native Hungarian box losses; gate is raw output clamped to [0,1] | Actual gate saturation and selected-query gradient are unmeasured |

models/losses.py lines959-970 add both semantic_assignment_correction
and selected_query_mask_loss to total loss when native_joint_training
and their respective flags are true. Current init specs enable both.
The main loss already includes native semantic-alignment and mask terms.
Therefore neither the detach in geometry nor this audit supports a claim
that the reader or support module receives no training signal.

native_mask_geometry.extent_reference detaches mask logits and mixing
alpha before thresholding foreground membership. The final box loss
cannot differentiate through that discrete extent path back into masks.
Other differentiable mask-state features in the mixer can still provide
indirect paths. Do not claim that every box-to-mask gradient is absent.
The clamp can suppress gradients outside its interval, but endpoint
gradients were previously demonstrated on a synthetic CPU objective.
No real-batch saturation diagnosis has been completed.

## Selection and Target Differences

selected_query_mask_objective.py selects its one extra supervised query
using native_root_bbs(last_sem_cls_scores.detach(), batch). That helper
aggregates softmax probabilities from the classifier head. It does not
apply the formal geometry-dependent detected-box filter.

The formal last/bbf evaluator instead uses projected query/token
similarities, temperature0.07, token aggregation, and optionally the
detected-box filter. Current NR3D/SR3D enable that filter; ScanRefer does
not. Thus the extra-mask selector and reported-metric selector differ in
both score source and, for NR/SR, candidate filtering. Different source
functions do not establish a measured disagreement frequency. This is
an auxiliary-objective alignment question, not evidence that reported
metrics used bbs or that standard matched-mask supervision is missing.

pvground_semantic_assignment.qualified_unmatched uses detached final boxes
and IoU>0.5 to qualify extra unmatched queries for token-label replacement.
It preserves native matching and regression losses. NR3D/SR3D checkpoint
selection prioritizes Acc025, so the extra correction does not directly
cover the entire IoU0.25-0.5 region. The normal losses still operate;
this is not proof of an accuracy defect or a reason to lower the
threshold without a controlled comparison.

## Relevant Open-Source Reference

Read the official OpenMMLab MMDetection implementation, HTTP200:
https://github.com/open-mmlab/mmdetection/blob/main/projects/AlignDETR/align_detr/align_detr_head.py

Source text SHA256 observed:
0a392a8b4fccc535493ad0f23dfbf968fbaa13eec41b230b0cac2435c753704e.
This records the fetched text, not an immutable Git commit.

Its positive classification targets combine predicted class confidence
with matched-box IoU, detach the resulting target, and apply rank
weights among queries matched to the same object. The relevant principle
is coordinating classification supervision with localization quality.
It does not solve our bbs/bbf mismatch automatically: our token
aggregation is not its independent sigmoid class probability, and
our formal filter/ranking must remain unchanged in any comparison.
No Align-DETR code, BCE objective, new head or ranker was installed.

## Next Discriminating Measurements

1. Reproduce a retained checkpoint's full formal pair before interpreting
   any diagnostic output.
2. Measure classifier-bbs versus unfiltered-bbf top-query agreement on
   identical forward outputs, then distinguish geometry-filter changes.
   Reuse the deployed evaluator ordering, including ties.
3. On actual training expressions, record selected query roles
   (matched root, matched other, unmatched), extra-supervised row counts,
   and the bbf-selected query's coverage. Validation alone cannot prove
   training supervision coverage.
4. Measure extra G-qualified queries by final-box IoU bands, including
   (0.25,0.5], before considering a threshold change.
5. Only after evidence supports one explanation, compare one existing
   module/objective parameter at a time on a new run. Keep the three
   module boundaries, author checkpoint and formal metric fixed. Do not
   modify the current live run. No improvement is claimed by this audit.

## Reproducibility

Raw source excerpts, line ranges, paths and complete-file hashes:
docs/results/SR3D-supervision-source-audit-2026-10-31.json.
Download SHA256 matched the remote export:
308c809bca1301675743a8b12f392bcb65d0347a4d60203692875ce0bf1040a4.
Exporter: tools/export_sr3d_supervision_source.sh. The snapshot covers
the current SR3D deployment; generalizing to another host requires
matching its relevant source hashes first.

## Measured Separate Score-Path Totals: 2026-10-31 07:30 New York

Exported current-run position-alignment and semantic-alignment log totals
for NR3D epochs0-5 and SR3D epochs0-2. Top1 integer hits were uniquely
reconstructed from five-decimal log fractions and the full split size;
semantic hits were checked against every own-epoch native metric row.
These are separate diagnostic totals, not paired query identities and
not replacements for the formal semantic last/bbf metric.

| Current selected model | Position hits025/050 | Formal semantic hits025/050 | Semantic minus position pp025/050 |
| --- | ---: | ---: | ---: |
| NR3D epoch5 | 4779/3880 of7899 | 4791/3894 of7899 | +0.151918/+0.177238 |
| SR3D epoch2 | 11920/10150 of17726 | 11944/10187 of17726 | +0.135394/+0.208733 |

Across trained epochs, NR3D differences range from-13 to+12 hits at025
and-1 to+44 at050; SR3D ranges-6 to+24 and-11 to+37. Thus neither
path uniformly dominates across epochs. Similar total scores also cannot
prove that the two paths select the same query or fail on the same sample.
The selected models' score-path differences alone do not quantify the
SR3D target deficit of358hits025. Do not change reporting protocol,
combine thresholds from different paths, or infer recoverable improvements
from this table. The per-sample query/role audit remains necessary.

The position evaluator may include its official geometry filter, while
the training helper does not. These logged totals therefore are not a
measurement of that unfiltered helper's training selection.

Raw exports retain all Top1/5/10 log fractions and epoch identities:

- docs/results/NR3D-current-score-paths-2026-10-31.json,
  SHA256547319b123ecdded54a4d950d243d861b00640ec7414e056888ae8b71bcc1241.
- docs/results/SR3D-current-score-paths-2026-10-31.json,
  SHA25601e00dea8258f0124de9e4654a9a33ab5020feb031adaa7ae1baf5898c498060.

Both remote/local download hashes matched. Exporter:
tools/export_current_score_paths.sh. No GPU evaluation, live training
change or new completed epoch was produced by this log audit.
