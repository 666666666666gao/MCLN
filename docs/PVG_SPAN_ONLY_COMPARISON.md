# Fixed Upstream Span Training Comparison

## Evidence And Hypothesis

2026-11-04:the Reader re-evaluation's152 lost strict050 examples include
52 with a valid selected reference box before fusion. A separate geometry
audit reconstructs all9508 final boxes within5e-7 from the recorded axis
gates. At fixed B2 selected queries,GT projection gates yield5103 strict050
hits versus learned4656;125GT-informed gate combinations yield5152.
The same figures for the Reader checkpoint are5090/4609/5136.
These are GT diagnostics,not inference accuracy or achievable gains.

The projection proxy is therefore not dismissed as generally ineffective:
it loses only4 B2 strict050 successes while gaining451 in this diagnostic.
Its exact optimum differs from IoU,but replacing it with an IoU search
target addresses a smaller measured gap than the learned/proxy gap.
The prior jointly trained gate pilot remains4654vsB24656 and matched
control4620;it does not establish robust improvement.

Hypothesis:keeping upstream representations and geometry fixed makes
existing matched-object gate supervision more learnable and prevents
upstream drift from obscuring a fusion improvement. This is a training
scope intervention,not a new architecture or established cause.

## Registered Protocol

Separate directory:scanrefer_span_only_20261104. Both conditions
independently initialize the complete B2 model (epoch2,1168 tensors,
SHA46ecfb6a6831fca0704c55fe87fa96c9c92e749824f13173d67d5697da1b92f8).
Use its original Reader,not the unsuccessful competitor variant.
Model-only initialization,fresh optimizer/scheduler,seed0,batch8,
accumulation6,module LR2e-5,weight decay0.0005,one full augmented48655-entry
epoch per condition,full9508 validation. Existing native losses remain.

Only the existing14 Span parameter tensors are trainable. All other
parameters and buffers remain frozen. Put the whole model in evaluation
mode except the root training flag and Span;BN/dropout modules upstream
remain in evaluation mode. Compare:

- Control:Span-only training,projection auxiliary weight0.
- Treatment:identical Span-only training,projection auxiliary weight0.1.

This changes train-time mode relative to the earlier joint pilot,so the
new Span-only control is required;the old joint control is context only.
The matcher and inference BBF protocol remain unchanged. Projection
targets use matched train GT only;validation GT is diagnostic/scoring
only. Neither grid nor proxy oracle is used at inference.

## Execution Gates

Before full runs,remote12-batch GPU preflight must verify two real
optimizer steps,finite loss and nonzero Span gradient/delta,nonzero raw
auxiliary gradient,and unchanged frozen parameters/buffers. The common
trainer now accepts --pvground_span_train_only and limits its existing
first-update witness to the one intentionally trainable module.
At full epoch save,compare all1154 non-Span model tensors to B2 exactly
and require at least one Span tensor changed. Audit own CPU epoch1,
1168 tensors,nonempty optimizer/scheduler and SHA. Stop on failed gates;
no fallback. NR/SR budgets/configurations remain untouched.

Report each full same-checkpoint BBF025/050 against its matched control
and immutable B2. A single-seed one-epoch positive result remains
exploratory;do not claim a core innovation or target attainment without
full evidence. The current targets remain5705/4945hits. This trial cannot
fix examples missing an adequate candidate,or alter semantic ranking
through Reader;its purpose is to measure existing Span learnability.

Helpers:prepare_scanrefer_span_only.sh,run_scanrefer_span_only.sh,
start_scanrefer_span_only.sh,check_scanrefer_span_only.sh;existing
train_scanrefer_span_gate_comparison.py accepts --span-only.
No local model imports,tests,forward or training are permitted.

## Deployment: 2026-11-04 09:46 UTC

First patch check stopped before any model run because the snapshot uses
CRLF. Downloaded source text and git diff ignoring EOL confirmed only the
intended23insertions/5deletions;reapplied the same patch with whitespace
context handling on the isolated trial copy. No unrelated source changed.
The explicit finish preparation path records this actual failed-preparation
recovery;it is not an automatic training retry.

Remote CPU preparation09:41:15 passed:B2 own epoch2/1168 tensors,
optimizer/scheduler,SHA,source compilation,no competitor Reader parameter,
CUDAuninitialized. Raw preparation receipt is archived. A remote-only
test executed the actual optimizer/mode methods extracted from this
source on a small model:only Span optimized,upstream BN/dropout eval,
upstream parameters/buffers unchanged after update. One test passed.
This is a focused unit check,not proof of full-model frozen-state integrity.

Controller265532 in screen pvg_scanrefer_span_only started09:42:21UTC.
At09:45:56 its preflight was loading48655-entry training data;no GPU
process or first-update witness yet. Do not claim actual optimization
or full-training transition until the12-batch/two-step gates pass.
Prepared main_utils SHA:
cc88b8cd5753a4c5a11633cdafe3dec0336dd18eaa5e64182fe0d5565cc2457d.
Trainer SHA:
0a480d29e68999e0a03456da818c6003e222e09538ddd5c69d5fbee66c3d0eba.
The reused internal experiment directory labels still say20261103;
the enclosing rootscanrefer_span_only_20261104 distinguishes this trial.

### Full-Model Preflight: 2026-11-04 09:48 UTC

The actual preflight completed two optimizer steps at batches6/12:
loss6.550971508026123/7.685822010040283,both finite. Span14/14 tensors
updated at both steps;gradient norms0.26116848/0.21298252 and parameter
delta norms0.00344559/0.00295755 are finite and nonzero. Both witnesses
confirm frozen parameters and buffers unchanged. The actual auxiliary
gradient check has48matched axes,loss0.0899460,raw gate gradientL10.336378,
including4out-of-clamp entries with auxiliary gradients. It did not access
formal validation.

At09:48:53 preflight GPU265542/controller265532 were live. The controller
will verify both witness files before transitioning;full-control first
update is a separate requirement. No complete validation or accuracy
claim follows from this check. Matching can still change with Span boxes
even when upstream weights are fixed;this does not freeze Hungarian
assignments or eliminate train data augmentation.

### Formal Control: 2026-11-04 10:00 UTC

The pipeline passed its preflight gates and started full control09:49:09UTC.
Controller265532/launcher266273 and GPU worker266278 were live10:00:03UTC.
Own batch6/12 steps have finite loss6.5439133644104/7.6695380210876465,
Span14/14 updates,gradient norms0.13783537/0.09377335 and parameter delta
norms0.00344798/0.00286784. Both confirm frozen parameters/buffers unchanged.
Control reached531/6081 at~1.92batches/s;rough full validation ETA
10:55-11:05UTC. No full metric or final frozen-state checkpoint audit yet.
The same controller owns treatment transition;do not start a second copy.
Raw progress receipt:scanrefer-span-only-progress-20261104T1000.json.
