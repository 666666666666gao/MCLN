# Fixed appearance endpoint: conditional ScanRefer validation

This continues the fixed ScanRefer object-appearance pair, not a new training
experiment. The original paired source, manifest, settings and checkpoint
selection remain unchanged. Only the2482-update appearance endpoint is eligible.

## Prerequisites and scope

1. Existing paired training finishes normally and passes its CPU integrity audit.
2. The independent native REC follow-up completes for initial/control/appearance
   on the same6887 module-holdout rows. Its result is reported, not a new gate.
3. Full V99 module REC passes the predeclared nonregression screen at both
   thresholds versus both the initial system and trained control. Otherwise
   the queue writes `module_screen_rejected` and does not run validation
   preprocessing or model inference.

The module-holdout scenes were seen during backbone pretraining. These results
are not formal generalization evidence. No loss/threshold/epoch search occurs.

## Frozen validation preprocessing

After eligibility, acquire the existing GPU lock and generate OpenShape features
for the141 ScanRefer validation scenes. The scene list SHA256 is
`3d342c3eb72476536554073bfc148f3db96d31bd2453d994954063778fe4fca4`.
All141 predicted GroupFree box files were found. The existing mesh protocol
still checks312 ScanNet validation superpoint files; these are different counts.

Reuse the reviewed OpenShape inference prefix and checkpoint. The validation
script is the training cache recipe with only its scene count, val point-pickle,
predicted detector directory and split label changed. Inputs remain50,000 XYZ/RGB
points, predicted object boxes, deterministic scene/slot sampling, G14 Z-axis
orientation and the same384-point availability minimum. No GT mask, target ID,
expression or optimization is used for feature extraction. Binary caches stay
remote. This is inference preprocessing, not training on the validation set.

## Formal model and evaluator

Use the historical formal evaluator as the control implementation, source SHA
`1e5e1e30e8efba15a2696e612d677c4f18e4c587c9076d59b3f0be3bc96cc935`.
Its local copy and archived actual remote executable were verified identical.
Retain the native9508-row loader, two workers, batch12, worker seeding, same-batch
protected/appearance comparison and the recorded formal runtime settings.

The protected arm strictly loads E71. The appearance arm instantiates the new
appearance projection and strictly loads the fixed saved model. Both use the
preserved Parent/Geometry/V99 artifacts and original decision rules. Inputs
receive val features only after exact detector-box/count and point-SHA checks;
the protected arm ignores those extra fields. A matching SHA is required for
training, audit, native follow-up, checkpoint, feature cache and evaluator code.

From each actual forward, record native evaluator Top1 and full V99 REC plus
semantic Mask IoUs. Recount thresholds against actual evaluator counters and
verify model states, artifact files and mesh inputs. A separate CPU auditor
recounts metrics, repair/break transitions, paired identities and cache binding.
No optimizer or checkpoint writing occurs in formal evaluation.

## Promotion

Candidate REC must reach historical5572/4797 of9508 and not regress against the
same-run protected arm at either threshold. Scan Mask floors are58.70/50.70/44.72
percent. Passing permits Nr3D/Sr3D REC training without waiting for59/51; their
Mask metrics are not promotion gates. Four CPU boundary tests verify these
conditions. They do not establish model quality or complete forward correctness.

## Current execution evidence

At01:22:42CST the conditional queue PID71123 was confirmed live at
`/root/autodl-tmp/mcln_scanrefer_object_appearance_formal_preparation_20260908_v1`.
It waits240seconds between checks of the existing native queue. All four new
entrypoints compiled and their actual bdetr `--help` invocations passed; the
four promotion tests also passed remotely. No validation cache or formal model
forward has started. The first real end-to-end runtime remains unverified.

Errors retain the original log/exit code and are not automatically retried.
The final decision records formal qualification; Nr/Sr launch still requires
their dataset-specific input protocol and model-loading checks. This queue does
not claim those trainings have started or the three-dataset goal is complete.
