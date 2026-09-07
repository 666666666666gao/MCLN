# Fixed ScanRefer native object-appearance training comparison

This plan is fixed before paired fitting. It tests pretrained object evidence
in native object memory; it does not restart O1 or the candidate-local reader.

## Shared conditions

- Protected E71 checkpoint and fixed Parent/Geometry/V99 readouts, exact hashes
  in the input manifest; correct `DATA_ROOT_mcln_meshsp` and audited superpoints.
- Existing scene split: 29778 fit expressions / 6887 module-holdout expressions,
  456 / 106 physical spaces. The backbone previously saw these training scenes.
- One complete fit traversal, batch12, 2482 updates per arm, identical shuffled
  batch order and actual point hashes, seed0, no augmentation in this comparison.
- Native GT loss only. Existing cross_encoder, all Decoder layers and all
  prediction heads train in both arms at LR1e-6. Other original parameters and
  buffers remain fixed. Eval mode preserves the pretrained BatchNorm/dropout
  behavior used in earlier controlled adaptations.
- AdamW, weight decay0.0005, clip norm0.1; no auxiliary source-selection loss,
  learning-rate scan, best-epoch selection or post-result threshold adjustment.

## Difference between arms

Control uses native box/class object memory. Appearance adds the frozen
OpenShape scene/slot feature through the new204800-parameter zero-initialized
1280-to-160 projection, LR1e-4. The position128 channels remain unchanged.
This measures the added appearance path under the same broadly trainable
multimodal core, not merely a last-layer output adjustment. Parameter count
and preprocessing overhead are explicitly additional in the appearance arm.

Both consume the same dataloader batches; the control ignores the additional
feature fields. Each feature is bound to its original detector boxes and
native point hash. This first test has no augmented slot corruption. The
frozen cache covers562 training scenes,16759 slots,12392 available; missing
4367 slots contribute no appearance and keep their native box/class memory.

## Engineering gate and outputs

The actual16-row native appearance probe passed initial REC/Mask/V99 parity,
native gradients and two disposable projection updates while preserving all
1144 original state tensors. These weights were discarded. The paired runner
also performs a batch12 forward/backward for both complete chosen parameter
groups, without any update, before baseline evaluation. If that capacity
check fails, the process stops with its original error; no silent batch change.

Baseline and final module-holdout evaluation record the actual native and full
V99 REC decisions, Mask IoUs and point hashes. The final model/optimizer
checkpoints are saved before final evaluation to protect against evaluation
interruption. Only the fixed2482-update endpoints are compared.

Native and full-system REC@0.25/@0.50, repair/break counts versus initial V99
and the trained control, and all Scan Mask metrics must be reported. Module
holdout is a screening result, never a new-scene generalization claim. A failed
candidate is not renamed or rescued by selecting a different epoch. A passing
screen leads to the fixed formal9508-row comparison; formal REC must preserve
the same V99 pair58.6033/50.4523, with Scan Mask floors58.70/50.70/44.72.
Nr3D/Sr3D training follows formal Scan qualification; their Mask is waived.

## Runtime and remaining deliverables

The native runner reuses the unchanged bdetr environment and the verified
625-file appearance source. OpenShape itself is not imported into training:
its frozen preprocessing was completed in its separately reviewed prefix.
No native environment rebuild or dependency replacement is needed.

The paired run has a persistent screen, GPU lock, log and terminal exit file.
Its first observed timing determines a later poll near completion; do not
duplicate a job because a transport call timed out. A full endpoint integrity
audit and formal evaluator integration remain required before promotion.
