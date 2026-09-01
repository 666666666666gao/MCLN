# Result-to-Claim Review Prompt

Intended claim: A-V4 Counterfactual-Parent supervision can improve Nr3D unseen-scene safe Top-K switching enough to justify formal validation or long training while preserving the V99 parent.

Experiment: unique pre-registered Nr3D scene-disjoint Fold-4 audit; protected E57 to E58; B16 by A1; all `27004` fit rows exactly once; `1688` optimizer steps; `5915` held-out train-scene rows; no formal validation and no weight output.

Results: `805` switches. At REC@0.25, parent/selected=`5661/5641`, fix/break=`38/58`, net=`-20`. At REC@0.50, parent/selected=`5011/4860`, fix/break=`108/259`, net=`-151`. Actual/CF gradient L1=`0.0208565/0.0220884`; nonfinite=`0/0`; gate failed.

Hard constraints: no tuning on the consumed fold, no formal validation to rescue a failed gate, no baseline fair reproduction, no rejected Section 7/8 or E0-E7, and no automatic downstream experiment.

Requested fields: claim support, supported/unsupported claims, missing evidence, claim revision, orthogonal next experiments, and confidence.
