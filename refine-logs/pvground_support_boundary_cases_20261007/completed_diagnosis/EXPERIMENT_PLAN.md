# Retained-best support and boundary diagnosis — planned, not executed

Purpose: identify the actual support errors behind the retained Mask reference's loss of Acc@0.25 before designing another learned refiner. No new training, checkpoint, score, teacher, candidate pruning, or inference rule is part of this diagnosis.

The authoritative goal remains one complete ScanRefer model with at least5620/4764 hits on9508 expressions, three effective contributions, then independent Sr3D/Nr3D training from corresponding author weights. Seed2027 only.

## Existing evidence and scope

- Protected state:5598/4848; both completed face-support arms5593/4832. Their hitsets agree and each loses5/16 hits. Do not rerun that pair.
- Already executed cached-box analysis: `pvground_reference_keep_20261006/cached_reference_errors/SUMMARY.json`. It requires no new recount.
- Its191 prior-correct/reference-wrong @0.25 expressions contain128 with >=95% GT box coverage and reference volume >4 times GT volume, and63 with <95% GT coverage. These are descriptive geometry groups, not deployment gates.
- High Mask IoU can coexist with either an oversized or incomplete reference. Raw predicted support, raw point membership and GT point labels were not included in the cached box arrays. Those arrays alone cannot establish erroneous extreme points or mixed-superpoint failure.
- This targeted development-set investigation is descriptive. It is not an unbiased new accuracy estimate or a new training split. No development GT may become a training target.

## One read-only pass

Use the existing retained-best factory and warm PV runtime, with the declared official PV/originalG/protected delta dependencies. Do not use either deleted terminal head. Keep every model candidate; only archive the selected diagnostic slices.

Take exactly the191 row IDs in the existing `0.25_damaged.jsonl`; retain the128/63 grouping. Use the original formal data interface, no augmentation, and existing seed/protocol. For each expression save:

1. Actual row/scene/target identity, input-point identity, deployed selected Query, historical selected Query, original prior and fused-reference bounds.
2. Raw input XYZ, point-to-superpoint membership and target point labels for offline diagnosis.
3. Text, own-Query and fused prediction logits plus actual fusion weight; take the corresponding point-level values through the real member map.
4. For each of the six extremal reference coordinates: supporting member indices, predicted support and GT target membership. Record whether the supporting superpoint is pure target, pure background or mixed, using actual member labels.
5. Per direction distinguish excessive extent from missing GT member extent. Report the actual observed counts instead of assuming every extreme point is noise.

Keep the historical and deployed Query identities explicit: earlier reviews observed three historical query changes. Do not relabel a newly selected Query's support as the old Query's support. This is an observed provenance issue, not a new fallback.

## Execution and output

The collector, offline evidence helper and launch/observation scripts are implemented and syntax-checked. No SOURCE approval, launch, GPU result, point-support finding, or cleanup is claimed at this preparation stage.

The input manifest contains161 original B8 contexts (1288rows); only191 diagnostic slices are saved. The formal dataset uses fixed `scan.orig_pc`, and the eval voxel processor has no point shuffle. Retaining complete original batches also preserves the original per-batch slots. Each target's XYZRGB identity and GT box must match its historical source. Query and reference differences remain separately observable.

The existing protected evaluation has39 selected invalid Mask extents, and the completed paired audit observed12 nonselected validity flips. Report an invalid current diagnostic extent as such, with its actual support counts and no invented extremal faces. Preserve the existing model's native invalid-extent behavior; introduce no new prediction fallback.

Before execution, implement the smallest collector by reusing the existing factory/formal dataset and verify its source call chain. An applicable run-experiment/experiment-bridge review is required before remote NN execution. Check the actual save reservation; create no weights. Estimate from the completed formal phase's measured rows/time and make the first observation a few minutes before expected closure, with180–300-second later intervals only when needed.

Archive bounded raw slices, per-row six-face evidence, and an offline summary; collect and verify local files before deleting remote temporary arrays under standing authorization. Preserve the protected weights, all256 candidates, original/native score and same-Query Box/Mask path.

## Decision supported by this diagnosis

Use actual source-of-boundary findings to decide whether to study predicted-support cleanup, member-level boundary correction, or missing-support completion. A new mechanism must train on training data and have a direct same-start/same-budget control. No GT-volume threshold, GT support label, dataset ID, or validation-derived oracle selection may enter deployment.

The diagnosis itself is not one of the required three effective contributions. Do not resume failed ordinary readback, quality ranking, fixed-quantile clipping or the same face-support head merely to produce another run.

## Completed-run limitation and additional CPU face-coordinate check

The actual fresh audit found native eval Gumbel sampling and no advancement through skipped old full-validation batches. The191 fresh records contain167 selected Query changes; a historical numeric slot is not a recovered old Query. The actual row128 recovery assertion failed identically twice (coarse difference0.027122974m). Keep this limitation; do not replay the model or replace old records.

After complete local intake and the once-only fresh-slice CPU summary, run an additional once-only CPU analysis of the OLD cached reference-box face coordinates against IDENTICAL raw XYZ and dataset GT member labels. Current predicted masks and Query features are not used. All coordinate-compatible points are possible old extremal members. A face is background-certain only if every such point is background; pure-background-superpoint certainty similarly requires every possible source to be pure background. Shared-coordinate target/background possibilities remain ambiguous. Possible mixed sources are not proof which source the old mask selected. Record observed target members beyond missing old faces and separately retain GT box/member range differences. This checks cached geometry provenance; it does not recover an old foreground mask or historical Query representation.

Implementation: historical_face_provenance.py, analyze_historical_faces.py. Three synthetic certainty checks passed in check_historical_face_provenance.py; they are not validation evidence. Full actual191 execution remains pending intake/source acceptance. Fresh auditor will independently verify actual coordinate candidates and the new scope. All GT is offline; no deployment rule, candidate removal, training, weight or scoring change occurs. Existing best5598/4848 and full goal5620/4764 plus3effective contributions remain unchanged.

## Completed archive and storage retirement

Both actual191 CPU analyses and their independent recomputations are complete. All205 archived files match bytes/SHA, including191 NPZ. Remote-copy cleanup requires the closed task, complete local archive and exact remote/local file identities; it does not depend on completion of the scientific report wording. All local evidence remains available to the original fresh auditor. The publisher still requires the final actual-evidence verdict and its scope limitations before publishing claims. No scientific acceptance gate is removed.
