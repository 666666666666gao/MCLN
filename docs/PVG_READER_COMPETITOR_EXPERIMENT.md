# Reader Competitor-Context Experiment

## Hypothesis And Provenance

The immutable B2 ScanRefer checkpoint scores 5483/9508 at IoU0.25 and
4656/9508 at IoU0.50. Its fixed-checkpoint audit found 6021 examples
with a valid box among the top ten candidates at IoU0.50, while the
semantic Top1 selects 4656. This is a candidate-selection opportunity,
not recoverable accuracy.

Reader bypass changes349 selected queries and nets only+17 IoU0.50 hits.
The baseline already has language-conditioned pairwise spatial attention.
Thus the experiment adds candidate-relative local visual evidence inside
Reader instead of duplicating generic spatial attention or adding a
candidate scalar before BBF's token softmax.

The new Reader branch averages the local summaries of the four nearest
predicted candidate centers and adds a learned zero-initialized residual
`scale * (candidate_summary - competitor_mean)` before the existing output
projection. It uses predicted boxes, point features and existing Reader
representations only. It does not use GT boxes, object labels, oracle
identity, or human anchor IDs. Support and Span retain their existing
responsibilities and code.

The design is informed by the candidate/distractor framing in the pinned
CoT3DRef source review and by assignment-aware supervision principles in
DETR/3DETR. These references motivate the question only; no external loss,
matcher, candidate protocol or reported result is copied. CoT3DRef's
segmented anchor protocol is not equivalent to MCLN's full-scene queries.

## Registered Trial

Run `1793763075`, separate from the completed Stage B and previous Span
pilot. It initializes model weights only from immutable B2 epoch2
SHA256 `46ecfb6a6831fca0704c55fe87fa96c9c92e749824f13173d67d5697da1b92f8`;
optimizer and scheduler are fresh; text encoder is frozen. Seed0,
batch8, accumulation6, native LR2e-6, backbone LR4e-6, module LR2e-5,
one full augmented training epoch (48655 entries), and full9508 validation.
The protocol-matched control and immutable B2 remain separate comparisons.
This is a one-epoch exploratory test, not a final method claim.

## Completed Result: 2026-11-04 05:10 UTC

Run `1793763075` completed its one augmented training epoch and full
ScanRefer validation. Same-checkpoint `last_semantic_alignment_top1` over
9508 examples is 5471/4608 hits at IoU0.25/0.50
(57.541018%/48.464451%). The run checkpoint's own CPU audit passed:
epoch1, 1169 model tensors, nonempty optimizer and scheduler, CUDA not
initialized, SHA256
`67cef44cd6ec253806eced03b1ca79f115d13d4e681c9da69730ca767b0fda35`.
The first optimizer update was finite (loss7.0552873611450195) and
updated Reader11/11, Support10/10 and Span14/14 tensors.

Against the protocol-matched control (5477/4620), this is -6/-12 hits.
Against immutable B2 (5483/4656), it is -12/-48 hits. The trial therefore
does not support an accuracy benefit from this candidate-relative Reader
residual in the tested one-seed, one-epoch setup. It does not establish
why individual examples changed or prove that the residual scale caused
the regression; the observed learned `competitor_scale` is not causal
evidence. Do not promote this checkpoint as a new best or claim a robust
module result.

## Registered Paired Diagnosis: 2026-11-04 09:08 UTC

The completed trial's aggregate regression does not identify whether lost
examples lack a valid predicted box or have a valid box ranked below Top1.
The existing B2 sample audit cannot answer that question for the newly
trained checkpoint. No additional training recipe is selected from the
negative aggregate result alone.

A read-only full9508 evaluation was launched in screen
`pvg_scanrefer_reader_result_audit`, controller263822, from the completed
Reader checkpoint SHA67cef44cd6ec253806eced03b1ca79f115d13d4e681c9da69730ca767b0fda35.
Remote CPU preparation independently verified own epoch1,1169 tensors,
nonempty optimizer/scheduler and SHA; CUDA was not initialized there.
The evaluator uses the trial's existing source,seed0,batch8 and existing
forward/semantic audit hooks with bypass=none. It writes under
`offline_audits/reader_result_20261104`; no parameter update is performed.
At09:08:52UTC controller263822 was live and loading validation data;
there were no semantic records or completion footer yet.
At09:12:05UTC GPU worker263830 was live;the audit had2280 forward and
2280 semantic records of9508,no completion footer,and roughly8minutes
remaining. This verifies actual forward progress,not just screen launch.

After completion, require9508 forward and semantic records, agreement
with the real evaluator counters, and exact ordered identity/point/GT
hash pairing with the existing B2 JSONL. Count gained/lost examples at
both thresholds,unique/multiple groups,and disjoint failure categories:
no valid candidate;valid candidate outside Top10;valid candidate inside
Top10 but not selected. These are measured candidate coverage/ranking
categories,not isolated module causes. Different trained checkpoints can
change both features and geometry. This comparison is against immutable
B2,not the matched one-epoch trained control; do not attribute all
differences to the competitor residual.

Helpers:tools/run_scanrefer_reader_result_audit.sh,
tools/check_scanrefer_reader_result_audit.sh,and
tools/summarize_scanrefer_reader_result_audit.sh. The reducer reuses the
existing complete-JSONL verifier and refuses unpaired or incomplete data.
Run it remotely after completion; no local ML operation or local test.
Preparation receipt:results/mcln_pvground_20261101/
scanrefer-reader-result-audit-preparation-20261104.json.

### Historical Progress: 2026-11-04 04:46 UTC

The registered run remains active at5118/6081 augmented training batches
(84%); GPU worker254229 was active. The latest first-update witness remains
finite and verified: loss7.0552873611450195, with actual updates in Reader
11/11, Support10/10, and Span14/14 parameter tensors. A status SSH attempt
timed out during banner exchange; retry succeeded and confirmed progress.
There is still no full9508 validation or own-checkpoint audit, so no
accuracy change is established. NR3D and SR3D continue their original
scratch runs unchanged.

## Full Validation Progress: 2026-11-04 05:01 UTC

The training log reached6068/6081 augmented batches at04:59UTC and then
entered full validation. At05:01UTC validation had reached168/1189 batches;
the registered GPU worker remained active. The full9508 result and
own-checkpoint CPU audit are pending. No accuracy gain is established.

Remote CPU check passed zero-scale identity and finite nonzero scale
gradient. The six-batch GPU preflight completed. At03:38:47UTC the full
trial had a finite first loss7.05528736114502 and completed optimizer step1.
Reader local visual updated11/11 parameter tensors, Support10/10, and
Span14/14 with finite nonzero gradient and parameter deltas. Full epoch
validation and own-checkpoint audit are pending; no accuracy gain is
claimed. At03:38:07UTC training was at88/6081 batches, about1.2 batches/s.

Results must be reported from the same checkpoint's last semantic
alignment Top1/BBF metrics over9508 samples. Compare both thresholds to
the protocol-matched control5477/4620 and immutable B25483/4656. The
targets are5705/4945. Keep the result exploratory even if positive until
replicated; preserve the best and latest only after completed validation
and CPU own-epoch/model/optimizer/scheduler/SHA audit.
