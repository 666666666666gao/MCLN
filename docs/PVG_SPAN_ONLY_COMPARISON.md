# Fixed Upstream Span Training Comparison

## NR3D Follow-Up: 2026-11-04 12:51 UTC

While the fixed-upstream Scan Span pilot remains completed and unchanged,
the original NR3D scratch run advanced to epoch66:full7899=4255/3435
(53.867578%/43.486517%),new Acc@0.25 best,up65/41 hits versus64 and469
hits short of the4724 target. Own CPU epoch66/1134 tensors/nonempty
optimizer/scheduler/SHA passed;SHA
ca645bf8f191a0609b603b77e829b61e492b6459805125ae012c164b783fcb5b.
Existing retention kept66 and removed preserved64/65;full historical rows
remain. Epoch67 training started. A recent finite24.1624 loss among
finite values with no error lines is not sufficient to diagnose a fault.
No NR recipe or architecture change follows.

## Completed Matched Pilot: 2026-11-04 12:13 UTC

The registered pipeline exited0 after one full augmented training epoch
and full9508 validation in both conditions. On the exact matched protocol,
control weight0 scored5470/4568 and Span auxiliary weight0.1 scored
5483/4608 hits (Acc@0.25/0.50=57.667228%/48.465924%). Treatment is
therefore+13/+40 hits versus control, but versus immutable B2 5483/4656
it is0/-48. This exploratory single-seed, single-epoch result does not
improve the global best under the Scan Acc@0.50-first selection policy,
does not meet either target, and is not evidence of robust causal benefit
or a core innovation claim.

Both own CPU checkpoints independently passed epoch1,1168-model-tensor,
nonempty optimizer/scheduler and SHA audits. Control SHA is
`f7082ff3b1fd16572eed9b9dc13433321e2b2ce79951e8683519ac4bb3ce755d`;
treatment SHA is
`6acbcf58351b9ddae8499a6e65392e52c73647e537b0f50a02f5aff4aed54be9`.
The final frozen-state receipts verify all1154 non-Span tensors equal the
B2 initialization and the14 Span tensors changed in each condition.
Treatment's own first two updates were finite and changed14/14 Span
tensors with finite nonzero gradients/deltas; frozen parameters/buffers
were unchanged. Training losses remained finite. Large finite Eval losses
are recorded separately and were not treated as training divergence.

Interpretation: isolating existing Span from upstream updates improved
both thresholds over its matched weight0 control, especially Acc@0.50,
but the treatment only matched B2 at Acc@0.25 and remained48 hits below
B2 at Acc@0.50. The registered experiment supports continuing diagnosis
of gate supervision/Span learnability, not promoting weight0.1 as a
general improvement. Preserve B2 as Scan best and this treatment only as
its own exploratory checkpoint. Raw results and independent CPU audit
are archived in `docs/results/mcln_pvground_20261101/`.

At12:13UTC NR latest65=4095/3254, best64=4190/3394; SR latest24=
11421/9200, best15=11743/9434. Both current trainers remained live and
their best/latest CPU checkpoint audits passed. Original scratch recipes
and full budgets remain unchanged.

## Follow-Up: 2026-11-04 11:34 UTC

Treatment3419/6081,controller265532/launcher268814/GPU268817 live.
Train-only34 parsed losses finite,last6.9048/6.9039/6.9074.
NR65=3211/3742,70 recent finite losses,validation ETA about7min.
SR25=1362/6486,64 finite losses. NR/SR scratch configurations unchanged.
No new full metrics. See three progress receipts observed11:34:40-47UTC.

## Follow-Up: 2026-11-04 11:25 UTC

Treatment2369/6081,controller265532/launcher268814/GPU268817 live.
Train-only23 parsed losses finite,last6.9396/6.9367/6.9335,no error lines.
NR65=2545/3742 and SR25=937/6486;both current trainers/keepers/retainers
live and recent loss lists finite,with original scratch configs unchanged.
No full results. Current bests remain Scan B25483/4656,NR64 4190/3394,
SR15 11743/9434. See dated progress receipts in results directory.

## Follow-Up: 2026-11-04 11:18 UTC

Treatment1599/6081,controller265532/launcher268814/GPU268817 live.
15 recent train losses finite,last6.9688. Actual first two optimizer
steps passed,Span14/14 changed and frozen parameters/buffers unchanged.
No full validation. NR65 simultaneously2073/3742 and SR25 637/6486;
current-only receipts show finite recent losses and original configs.
Current bests unchanged:Scan B25483/4656,NR64 4190/3394,SR15 11743/9434.
See dated progress receipts in results directory. All targets unmet.

## Treatment Update Gate: 2026-11-04 11:09 UTC

Treatment weight0.1 passed its actual first two optimizer-step witnesses:
finite loss6.557239/7.686158,existing Span14/14 tensors updated at both
steps,finite nonzero gradients/deltas,and frozen parameters/buffers unchanged.
At11:09:23UTC it reached520/6081 (~1.89batch/s),five recent train losses
finite,last7.2928/7.2559/7.1997. Estimated remaining train time~49min,
then complete9508validation. No score exists yet;control and B2 comparisons
remain pending until its own checkpoint and full-validation receipt.
Raw receipt:scanrefer-span-only-progress-20261104T1109.json.

Follow-up11:14:29UTC reached1100/6081 with controller265532/launcher268814/
GPU268817 live. Train-only11 parsed losses all finite,last7.0651/7.0369/
7.0132;estimated~44min training remains then9508 validation. No score yet.
Raw receipt:scanrefer-span-only-progress-20261104T1114.json.

## Full Control Result: 2026-11-04 11:01 UTC

Control completed10:58:21UTC,full9508=5470/4568
(57.530501%/48.043753%). Versus immutable B25483/4656,this is-13/-88hits.
CPU own epoch1/1168 tensors/nonempty optimizer/scheduler/SHA passed:
f7082ff3b1fd16572eed9b9dc13433321e2b2ce79951e8683519ac4bb3ce755d.
All1154non-Span tensors exactly equal B2;Span changed. This verifies the
registered frozen-state intervention,not success of the method. Retain B2
as global best/input and this condition's sole own best/latest checkpoint.

Existing controller265532 automatically started treatment weight0.1 at
10:58:35UTC,independently from B2,fresh optimizer/scheduler. At11:01:03
actual launcher268814 live,loading data,no GPU or optimizer-step witness
yet. Do not duplicate launch or describe loading as verified training.
Treatment full result must be compared with this5470/4568 control and B2;
beating the control alone does not establish improvement over B2.
Raw receipt:scanrefer-span-only-progress-20261104T1101.json.

## Loss Observation Correction: 2026-11-04 10:58 UTC

The original progress collector mixed Train and Eval loss lines into one
recent-value list. Remote exact log context confirms521.9311 belongs to
Eval100/1189,not a training spike. The scoped collector now additionally
groups values by explicit Train/Eval log headers. At10:57:35UTC it found
60train values,all finite,last6.8842/6.8820/6.8798,and11eval values,all
finite,last459.1912/419.6783/383.0044. Large Eval losses are real records;
their numerical scale alone does not diagnose training divergence or
establish their cause. No training-source,recipe or loss change follows.
Remote reader only:no model import,new forward or test.
Raw context:scanrefer-span-control-loss-context-20261104.json.
Control actual controller265532/GPU266278 live,validation1102/1189;
final full result and frozen-state audit still pending at that timestamp.

## Capacity Update: 2026-11-04 10:27 UTC

The new continuous-gate offline audit in PVG_SPAN_FIXED_QUERY_CAPACITY.md
finds B2 fixed selected-query optimum5633/5158 over9508. Its0255633 is
72hits below the5705target,even with per-sample GT gates. This is a
fixed archived box-family limitation,not a model result or global MCLN
ceiling. Continue the matched experiment to measure learnability;the
joint60/52goal also requires selected-candidate or upstream box improvement.
No current recipe changed;all GT diagnostics remain separate from scores.

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

## Clamp Diagnostic: 2026-11-04 10:11 UTC

Remote stdlib-only postprocessing of the same archived9508-record B2 and
Reader evaluations added selected-gate saturation counts. Input SHA and
reconstructed final boxes agree with the earlier proxy audit;no model
was imported,no forward or new validation was run. Original proxy receipt
is retained;new receipt is scanrefer-span-proxy-saturation-summary-20261104.json.

| Selected-Query Statistic | B2 | Reader |
| --- | ---: | ---: |
| Total gate axes | 28524 | 28524 |
| Raw gate below0 / above1 | 423 / 925 | 393 / 1220 |
| Samples with any raw gate outside[0,1] | 857 | 1002 |
| Outside axes whose clamped gate differs from GT projection by>1e-6 | 609 | 724 |
| GT-projection050 gains | 451 | 486 |
| These gains with any raw gate outside[0,1] | 35 | 39 |
| These gains with all three raw gates outside[0,1] | 4 | 2 |
| These gains with an outside axis differing from GT projection | 33 | 32 |

B2 has416of451 projection-gain samples with no selected gate outside
the clamp;Reader has447of486. Thus selected-gate saturation does not
describe most of this measured projection gap. It does not establish
why the remaining interior gates were learned incorrectly. Counts refer
to selected validation queries,not training Hungarian eligibility or
total model gradients. Outside-clamp derivatives are zero through the
clamp alone;shared network parameters may still receive gradients from
other axes,examples and auxiliary objectives.

Do not replace clamp with sigmoid based on this diagnostic alone.
Continue the registered matched weight0/0.1 training without source or
hyperparameter changes. Its existing raw-gate auxiliary objective can
also supervise interior gates. GT projection gains remain diagnostic,
not achievable gains,method results,or causal proof. The125-point grid
is a sampled oracle,not a certified continuous upper bound.

## Progress Monitoring: 2026-11-04 10:19 UTC

The short8KB log tail was entirely tqdm progress,so it could not show
recent periodic losses. The scoped collector now reuses the existing
NR/SR2MB tail and loss parser,reporting count,finite status and last three
logged values separately from first-update receipts. An absent loss list
remains unobserved,null rather than finite. No training source changed.

Remote-only collection10:19:32UTC verified controller265532/GPU266278
live,control2769/6081,27parsed losses all finite,last6.9144/6.9086/6.9102,
no error lines. Full validation and final checkpoint freeze audit remain
pending. Receipt:scanrefer-span-only-progress-20261104T1019.json.
