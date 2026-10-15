# Matched endpoint supervision: bounded normal-training comparison

Status: implementation preparation only. Current lower_span_lr E2 continues unchanged. No GPU admission, deployment, queue or new result is implied by this plan.

The persistent end state remains one complete ScanRefer9508 model at >=5705/4945 hits (60%/52%), three independently effective substantive mechanisms, then corresponding-author Nr3D/Sr3D full evaluation and independent same-method training above matching-protocol paper figures. This auxiliary loss alone is not a third proven innovation.

## Evidence and question

The 09:00 due observation reports E0 5677/4920 and E1 5652/4732, paired strict repairs301/breaks489. E1's same-selected-Query native endpoint hits4093, mask endpoint4791 and final4732. The changes include upstream and Query selection changes; they do not prove a pure regression-head failure.

The bound current source constructs six distinct prediction heads. It preserves last-layer coarse center/size, then replaces last_center/last_pred_size with the support/B output. Native losses supervise this final output; the other five auxiliary heads do not directly supervise this particular coarse output. Gradients through B still exist, including coefficient/content paths. This is a training-role question, not a certified bug, missing gradient or decline cause.

Question: does direct matched-GT supervision of this still-used coarse endpoint help normal training, beyond merely adding another final-box loss?

## Two arms, one controlled difference

1. `final_duplicate`: reuse the original last Hungarian indices/targets, add another final-box L1/GIoU term.
2. `coarse_endpoint`: use the same indices/targets and coefficients, but supervise saved native_coarse_center/size.

The two arms share model code/parameters, all original native/G/C objectives, optimizer policy, dataset/input/seed/budget and deployed last/bbs. Only the explicitly selected auxiliary endpoint differs. The duplicate-final arm controls added localization weight; comparing only with an old no-auxiliary endpoint is insufficient. Loss magnitudes and gradient directions can still differ and must be reported.

Use the actual native `SetCriterion.loss_boxes`: center L1 plus0.5*size L1, and GIoU. Normalize with the original last matching count. Single A100; additional term `(10*L_bbox + 2*L_giou)/(num_decoder_layers+1)`, multiplier1. Both retain every original matching target, including other instances. No rematching, unmatched-positive expansion, root rewriting, semantic/contrastive target change, teacher or new inference score.

## Run order and admission

1. Local CPU tensor-component checks and exact source insertion checks. These do not construct the PV model or provide accuracy evidence.
2. Fresh same-family provisional Astra/max code review required by experiment-bridge. Fix genuine blockers before deployment.
3. At the original14:40 observation or a later real terminal node, consume existing native60359. If the current normal run continues, this candidate remains unqueued; do not preempt it. Once it is actually terminal, run fresh real-factory two-step sanity for both arms on the existing warm runtime, serially. Verify common initialization, normal core/G/A/B trainability, native matched targets, coarse/final actual gradients, optimizer/scheduler restore, memory/throughput. No preflight state may enter normal runs.
4. Only after actual sanity passes, launch both normal arms from the same official/G/A/B initialization and fresh optimizer, serially. No learning-rate search or multi-seed.

Fixed normal recipe: seed2027; batch/effective batch8, accumulation1; core/backbone1e-6,A1e-5,B1e-6; weight decay5e-4; clip0.1; full36665 training expressions with native drop-last36664,4583 updates/epoch; max8 epochs; stop after two consecutive epoch-to-epoch Acc@0.5 decreases including E0; equality/rise resets. Both use all9508 validation expressions and all256 candidates, original last/bbs, same Query Box/Mask. Dynamic actual budgets may differ under this common stopping rule; compare shared completed epochs and disclose total updates/best selection, rather than calling unequal terminals fixed-budget.

Estimated normal maximum about45 GPU-hours per arm, not a promise; two-decline runs historically stop near11 GPU-hours. No extraGPU, runtime rebuild, temperature/power query or current-source edit. Protect strongest/current/dependencies; clean only completed unused verified duplicates under existing user authorization.

## Acceptance and claims

Report official parent/G/retained5677/4920, both arms at common completed epochs, best/terminal actual budgets, strict/wide repairs and breaks, fixed-Query endpoints, candidate coverage and deployed selection. Auxiliary-target losses must not be labeled deployment quality. A trained gain requires beating the common control and the retained starting capability. Three mechanisms and Nr3D/Sr3D remain separate unfulfilled obligations.

Keep current/duplicate/coarse dataset/source provenance and model history transparent. No dev GT in inference or teacher fitting. If the coarse endpoint does not improve actual geometry/selection, do not turn this auxiliary term into a claimed module or extend the unchanged recipe indefinitely.

AUTO_DEPLOY remains conditional on source review and actual GPU availability under the user's single-card serial rule. No permission question is needed for the already authorized experiment; the current14:40 observer is the sole training-progress reader until its node.
