# Span Gate Supervision Comparison

## Status And Question

17:38:48UTC:formal controlrun1793726936 has real training updates,
220/6081microbatches;GPU worker236606 uses21647MiB total GPU memory.
Own first-update receipt:epoch1,batch6,size8,step1,finite loss
7.0552873611450195;Reader10/10,Support10/10,Span14/14actuallyupdated,
all three gradient/delta norms finite/nonzero. Config verifies weight0,
max1,max_train_batches0,model-onlyB2,nullPP,fresh optimizer/scheduler.
Receipt:scanrefer-span-gate-progress-20261103T1738.json.
At~1.2microbatches/s,full control validation ETA19:10-19:25UTC;
recalculate from actual progress near19:05. Treatment has not started.
Control/preflight first losses differ;do not claim identical stochastic
training microbatches or bitwise determinism. No full pilot accuracy yet.

17:32UTC:real GPU preflight passed17:28:44UTC. Atbatch1:auxloss
0.05352035164833069,48validmatchedaxes,rawgradientL1
0.24082067608833313,and3outsideclampentriesreceiveauxiliarygradient.
Atbatch6/size8/optimizerstep1:finite total7.0388007164001465,
Reader10/10,Support10/10,Span14/14tensorsupdated with all module
gradient/deltanormsfinite/nonzero. The preflight state is discarded.
Controller transitioned17:28:52UTC to controlrun1793726936.
Actual17:29:33UTC loadingtrain/noerrors,no controlupdate yet.
Formal pilot validation results are pending. Code/evidence published234d38a.

17:26UTC:all-four B2 diagnostics finished17:21:26UTC,pipelineexit0,
paired identities/pointSHA/GT passed. Pilot controller started17:21:50UTC
in pvg_scanrefer_span_gate_comparison,runner236004. At17:25:10UTC the
6batch preflight was loading training/text decoupling,noerrors,GPU3MiB;
no actual forward/update or pilot full validation yet.
The five critical patched file hashes match local source exactly.

Prepared on2026-11-03 17:16UTC in a separate remote source snapshot.
Six remote CPU tests and compilation passed;no local tests or model imports.
The following preparation snapshot preceded the controller launch above.

Question:does explicit matched-object gate supervision improve full9508
semantic Top1 box accuracy relative to otherwise identical continuation?
This is an exploratory one-seed,one-full-epoch pilot,not a final main result.
No improvement or target attainment is assumed.

## Minimal Change

Keep all three modules,the existing hard foreground Mask extent,the zero
initialization and raw_gate.clamp(0,1) prediction path. No new parameters,
attention,soft extent or Reader verifier. The model exposes raw gates in
end_points;the existing final Hungarian assignment supplies matched real GT.
Do not apply the root GT to all queries or to unmatched no-object queries.

For each matched object and axis,let d_c=c_mask-c_native and
d_s=s_mask-s_native. Native size is clamped exactly as in Span forward.
The detached target is:

    g_target = clamp(((c_gt-c_native)*d_c + (s_gt-s_native)*d_s)
                     / (d_c^2+d_s^2), 0, 1)

This uses equal center/size squared-error weights in physical coordinates.
It minimizes a per-axis squared geometry proxy along the segment,NOT IoU.
Require valid Mask reference,positive GT size and squared direction>1e-8;
the denominator uses the same1e-8 numerical threshold.
SmoothL1(raw_gate,g_target) is averaged over valid matched axes.
Target/endpoints are detached;original L1/GIoU/mask/semantic losses remain.
Supervising raw gates permits auxiliary gradients outside clamp saturation.
Default weight0 leaves the normal loss path unchanged.

## Registered Pilot

Common immutable B2 SHA:
46ecfb6a6831fca0704c55fe87fa96c9c92e749824f13173d67d5697da1b92f8.
Both conditions independently load all1168 tensors model-only,fresh
optimizer/scheduler,frozen text encoder,seed0,batch8/accum6,
native2e-6/backbone4e-6/Reader+Support+Span2e-5,weight_decay0.0005,
constant schedule(decay100),one full48655-entry augmented training epoch.
The unchanged native loader uses drop_last:6081 microbatches consume48648
entries from the48655-entry dataset per epoch. Both conditions use this
same order/batching and partial final accumulation-group handling.
Every condition evaluates all9508 examples using unchanged evaluator.
Control weight0;method weight0.1. No seed/LR search or unequal early stop.
Compare both to each other and to archived strong B2=5483/4656.
Select/check both thresholds from the SAME checkpoint,Acc050 first.
One epoch/seed cannot establish convergence,robustness or final causal gain.

Before either full run,a separate6-microbatch treatment preflight verifies
real matched-axis gradients,finite loss and actual module updates.
It computes no validation metrics and is discarded:both formal pilots
reload B2. Standard trainer setup still constructs its validation dataset;
the bounded training path returns before any validation evaluation.
Controller order:preflight,control,span;stop on any nonzero exit.
No additional cleanup/watchers are required for one saved epoch per condition.
Existing NR/SR trainers and their240/140 budgets are untouched.

Remote directory:
/root/autodl-tmp/mcln_pvground_ft_20261101/scanrefer_span_gate_comparison_20261103.
Parent is the actual completed stage_b_source,not an assumed local clone.
git apply --check preceded patching;preparation.json records exact deployed
file hashes and B2 own CPU epoch/1168/nonemptyoptimizer/scheduler.
Code:models/pvground_span_supervision.py and surgical mcln/losses/main_utils
changes. Entry:tools/train_scanrefer_span_gate_comparison.py.

## Primary Source Checks

Only source text/API metadata was read locally;no reference code was executed.
These are design references,not benchmarks against our data or evidence of
our proposed gate loss working.

- Official MCLN,qzp2018/MCLN commit
  9744a4ed219062d448ed0dba587eeb864491f158,
  src/grounding_evaluator.py:
  https://github.com/qzp2018/MCLN/blob/9744a4ed219062d448ed0dba587eeb864491f158/src/grounding_evaluator.py
  BBF uses projected query-token dot products,temperature0.07,token softmax,
  then main+modifier+pronoun+relation-other maps. BBS uses classification
  logits instead. Any future Reader verifier must attach to actual BBF,
  not reuse a BBS score helper under a renamed label. Current diagnostics
  already verify their BBF counts against evaluator counters.
- RefineRank,linzhe001/RefineRank commit
  90f5c18378985679df7f668a3e176954b1790451,
  refinerank/proposal.py::proposal_adapter_loss:
  https://github.com/linzhe001/RefineRank/blob/90f5c18378985679df7f668a3e176954b1790451/refinerank/proposal.py
  The implementation detaches refined IoUs for quality targets and separately
  supervises bounded coordinate deltas with SmoothL1 and GIoU. It supports
  distinguishing quality selection from geometry supervision;it does NOT
  implement our per-axis convex Span projection or prove a transferable gain.
  Its surgical2D/video setting differs from end-to-end3D predicted queries.
- EG-3DVG,Gwan9Wook/EG3DVG commit
  174e34894aea6513442da6b5dfa9b3e2bf8a1efa,models/eg.py:
  https://github.com/Gwan9Wook/EG3DVG/blob/174e34894aea6513442da6b5dfa9b3e2bf8a1efa/models/eg.py
  Actual source thresholds sigmoid masks below0.5 for attention masks and
  detaches them,and detaches proposal/base geometry between decoder steps.
  This does not justify claiming that replacing our hard extent with a soft
  extent is necessary or is an equivalent differentiable implementation.

Contextual Reader/instance verification remains a later experiment after
the measured selection gap and this minimal comparison;no fourth main
module or teacher transfer is included in this pilot.
