# M2 result: most conditional Mask failures already exist in both raw branches

M2 completed one native protected-model pass on7899 Nr3D validation expressions,
with zero optimizer steps and zero checkpoint writes. On the fixed828 rows
whose majority GT superpoint label already exceeds.5 but every fused prediction
fails,767 rows also have no raw text or raw Query Mask above.5. Only61 rows have
a passing raw branch; requiring the same Query to have a legal Box IoU>.5
reduces that number to40. This supports investigating Mask learning and its
connection to the Box Query before another fusion-weight modification.

The result does not isolate RSA, backbone, query projection, sampling, or the
loss as the unique cause. These are diagnostic oracle counts, not a new model's
accuracy. Both raw branches failing at the current threshold also does not
prove that every possible mixture would fail; no mixture sweep was performed.

## Fixed execution and native-output comparison

The protected averaged checkpoint, frozen612-file model tree,627 data files,
original native CLI/config, B16, four workers, candidate legality and original
logit>0 thresholds were preserved. All parameters/buffers and source/data/
checkpoint hashes pass before/after checks. Reconstructed fused point Masks
equal the native evaluator Masks on every row; point-count float64 IoUs match
the native evaluator. Dataset identity/order and all prior cohort memberships
are checked. Native evaluation takes717.419 seconds.

Native results exactly match the sealed P1 aggregate receipt:
REC4478/3763; Mask4194/3482; Mask IoU sum2958.9085414748506,
or37.45928018% mIoU. As in P1, these current-source values differ from the
historical protected4475/3759 and37.43374912%. Historical core source remains
unrecovered. No new formal best or method gain is claimed.

Exact equality of every old prediction field does **not** hold: the final batch
of11 rows, IDs7888–7898, differs in some Box values and raw-size counts. All five
recorded Query identities and all recorded Mask IoUs on these11 rows remain
equal; the maximum recorded Box-IoU absolute difference is0.0013219714. The cause
is not isolated. The difference is retained in`p1_last_batch_comparison.json`,
not suppressed by a tolerance or used to select another run.

## Fixed failure cohorts, all256 Query Masks

Every row below has an existing legal Box IoU>.5 and no fused Query Mask IoU>.5
in the sealed P1 audit. The membership and154/123/828 partition are unchanged.
“Raw Query passes” means at least one of its256 Masks passes; it need not be the
native selected Query or have a good corresponding Box.

| Fixed cohort | Rows | Text passes.5 | Raw Query passes.5 | Either raw branch passes.5 | Both raw branches fail.5 |
|---|---:|---:|---:|---:|---:|
| All good-Box/bad-fused-Mask cases | 1105 | 12 | 53 | 62 | 1043 |
| Optimal SP union itself<=.5 | 154 | 0 | 0 | 0 | 154 |
| SP union>.5, majority GT label<=.5 | 123 | 0 | 1 | 1 | 122 |
| Majority GT label>.5 | 828 | 12 | 52 | 61 | 767 |

In the828 cohort,49 rows have only a passing Query branch,9 only text, and3
both;767/828=92.63% have neither. In the complete1105 cohort,1043/1105=94.39%
have neither. Thus fusion damage exists, but it does not explain most of this
particular conditional failure cohort.

When an oracle is restricted to legal Queries whose Box IoU>.5, the1105 cohort
has12 text passes and30 raw Query passes, with1 overlap:41 rows can pass in
either raw branch. The828 cohort has12 text and29 Query passes, with1 overlap:
40 rows. Its remaining788 rows have neither branch passing on a good-Box Query.
Do not call a good Mask on a poor-overlap Box Query joint REC/Mask success.

For the828 cohort, raw Query all-Query oracle mIoU is39.027652%, while the
fused all-Query oracle is36.879834%. Text mIoU is17.851183%. These numbers
describe missing predicted shape quality, not an attainable deployed oracle.

## Deleting the text contribution is not an established gain

Keep the actual native Mask Query fixed and only observe its existing three
branches; all7899 rows use the same Query identity across these columns.

| Existing branch at native Mask Query | Mask hits.25 | Mask hits.50 | Mask mIoU |
|---|---:|---:|---:|
| Text | 4127 | 3172 | 35.58273509% |
| Raw Query | 4198 | 3478 | 37.46368037% |
| Actual native fusion | 4194 | 3482 | 37.45928018% |

Replacing the current fusion with the raw Query branch at that fixed selection
would repair10 and break6 at.25, but repair5 and break9 at.50. Its mIoU
difference is only+0.00440019 percentage points. This counterfactual was not
adopted as a new evaluator path.

Over all256 Query Masks, raw Query oracle gives7180/6016 hits and65.175013%
mIoU; native fused oracle gives7126/5973 and64.642192%. The oracle's43 extra
strict hits do not translate into a deployment gain at the current selection.

The actual scalar text mixing weight alpha averages0.0504500 overall
(median0.0453756), and0.0481796 in the828 cohort. It is shared across Query
Masks within each expression. The active code obtains alpha from the selected
SWA text token's prediction head, and already supervises fused Masks with
focal/Dice terms. A sigmoid-bounded mixing weight is not a calibrated hit
probability; no alpha values were selected using these validation outcomes.

Native REC and native Mask Query identities remain separate in the artifact.
The native REC Query exists on7898 rows and its fused Mask gives4241/3509 hits;
the native Mask path has7899 rows and4194/3482. Keep these denominators explicit.

## Consequence and bounded next step

The completed P1 and M2 evidence contains both a selection gap and a conditional
shape-quality gap. Improving ranking remains relevant, while R1's specific pair
readout has failed and is sealed. M2 supports tracing the native matched-Query
Mask supervision and the feature projection/superpoint representation on
fixed training examples before designing a new RSA or Mask head. The existing
code selects Mask logits using the Hungarian Query index, so a gross index
substitution bug is not established by the code review.

The next bounded check should compare the actual matched Query, good-Box
Queries and native selected Query, including their Mask supervision and
gradient connection, using training data only. It should precede any full
training plan. Do not restart a failed selector/fusion-gate route, relax the
R1 screens, or launch another official validation sweep from this report.

No three-benchmark improvement has been achieved by these diagnostics. Nr3D's
protected formal result remains4475/3759; ScanRefer and Sr3D are untouched.

## Evidence

`refine-logs/mask_branch_diagnostic_20260905_v1/` contains the pinned inputs,
CPU tests, progress and process receipts, complete native receipt and analysis,
losslessly compressed7899-row output, and the explicit P1 comparison. Two CPU
tests pass in the actual Py3.7/Torch1.10.2 environment; full native assertions
then pass. The original M2 controller and CPU analysis both exit0. At19:01:06,
neither R1 nor M2 has a remaining process; GPU use is1MiB/0%.

- Input manifest SHA: `fd44d2abe72824eed0bad7426daa946836b42551446fc1d316a31366438bf113`.
- Receipt SHA: `9fb8a345f8823f21e474df41667622cf6d7b3d6a07ebd5aff9ffb5209c57a681`.
- Analysis SHA: `7c27ba0a463dc9eb0b7867ed650a4ae2d644ec4b4b2076bb242201e24204648a`.
- Raw rows:36565615bytes, SHA `80b80b64b4e1d0310a7e7db0a37388f54377647fe22962beecd80d60ded47267`.
- Gzip rows:2335582bytes, SHA `5a35056094e73b30f1d0060fb30dbd967a7487fc45fae1840f6d2edf6ef68f92`.

The active source path and existing fusion supervision are separately recorded
in`active_fusion_source_note.json`. The literal source tree used by the run
remains immutable on the server.
