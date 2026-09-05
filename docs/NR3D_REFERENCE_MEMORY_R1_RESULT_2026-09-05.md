# R1 result: object-reference pair readout failed the fixed screens

R1 completed one full fit epoch and one full holdout pass. Independent terminal
verification passes. The preregistered primary `object_pair` fails all three
advancement screens; no head advances to Decoder integration or formal
validation, and no control replaces the primary after seeing results.

These are 6172 expressions from 98 training scenes held out only from the new
heads. The frozen backbone previously saw these scenes. They are not the
7899-row official Nr3D validation, nor whole-system unseen-scene results.
Protected formal Nr3D remains 4475/3759 REC hits and Mask mIoU37.433749%.
ScanRefer and Sr3D protected results are unchanged.

## Complete matched comparison

All four heads receive the same frozen backbone outputs, legal Default Top32,
text sequence, five-dimensional center geometry, data order, seed, optimizer,
and 6687 updates. Training covers26747 rows, of which26483 have a qualifying
ranking target; no batch is skipped. Fit takes2482.013 seconds.

Query versus object memory is parameter matched within each readout. Global
and pair have347953 and931729 parameters respectively, so their comparison is
not a capacity-matched text ablation. Object slots contain existing box-position
and predicted-class features, not independent local appearance evidence.

| Mode | REC hits .25 / .50 | Mask hits .25 / .50 | Mask mIoU |
|---|---:|---:|---:|
| protected | 6005 / 5312 | 5767 / 5056 | 68.8765531% |
| default | 6005 / 5312 | 5754 / 5046 | 68.7141129% |
| query_global | 6003 / 5514 | 5752 / 5044 | 68.6682155% |
| query_pair | 6002 / 5430 | 5751 / 5038 | 68.7241976% |
| object_global | 5999 / 5665 | 5754 / 5047 | 68.7323140% |
| object_pair | 6011 / 5336 | 5760 / 5048 | 68.7307705% |

Protected Mask retains its actual native Query selection. Each new head uses
its own selected Query for Mask evaluation; these paths are not substituted.
All selections use native mask-before-sort and float64 point-count Mask IoU.
The Query control totals also equal the sealed P2 controls in this run; this
was observed after completion, not assumed from architecture equivalence.

## Fixed-primary decision and repairs versus damage

| object_pair versus | .25 fixes / breaks / net | .50 fixes / breaks / net |
|---|---:|---:|
| query_pair | 23 / 14 / +9 | 140 / 234 / -94 |
| object_global | 18 / 6 / +12 | 52 / 381 / -329 |
| protected | 19 / 13 / +6 | 217 / 193 / +24 |

Against query_pair, long-sentence .25 net is+2 and hard-distractor net+1, but
.50 decreases94 hits: **memory screen FAIL**. Against object_global, long .25
net+5 and hard net+6, but .50 decreases329: **readout screen FAIL**. Against
protected, Mask mIoU decreases0.1457826 percentage points, with Mask hits-7/-8:
**practical screen FAIL**. Long .25 net versus protected is0; hard net is-1.

The object-global control improves .50 by151 hits over query-global, while
object-pair loses94 against query-pair. Wider existing object memory therefore
does not validate the proposed pair readout. This result rejects this trained
variant under the fixed contract; it does not establish that all relation
modeling or all object memories are ineffective. The better strict-threshold
control is retained as diagnostic evidence, not promoted after the fact.

## Scene-cluster uncertainty

Paired whole-scene bootstrap:2000 resamples, seed20260905, expression-weighted
ratios and95% percentile intervals, all98 scenes retained as sampling units.
These exploratory intervals do not change the registered screens.

| Effect | REC .25 pp [95% interval] | REC .50 pp [95% interval] | Mask mIoU pp [95% interval] |
|---|---:|---:|---:|
| interaction | +0.2106 [-0.0499, +0.5308] | -3.9695 [-5.1787, -2.8178] | -0.0575 [-0.2405, +0.1306] |
| memory_with_global | -0.0648 [-0.3113, +0.1317] | +2.4465 [+1.6089, +3.3491] | +0.0641 [-0.0844, +0.2043] |
| memory_with_pair | +0.1458 [-0.0619, +0.3500] | -1.5230 [-2.4685, -0.7145] | +0.0066 [-0.1678, +0.1638] |
| primary_minus_protected | +0.0972 [-0.0813, +0.2916] | +0.3889 [-0.7050, +1.4037] | -0.1458 [-0.3799, +0.0491] |
| readout_with_object | +0.1944 [+0.0000, +0.4292] | -5.3305 [-6.7879, -3.9680] | -0.0015 [-0.1144, +0.1192] |
| readout_with_query | -0.0162 [-0.1688, +0.1312] | -1.3610 [-2.1283, -0.6404] | +0.0560 [-0.0681, +0.1779] |

## Integrity, evidence, and execution correction

All6172 CSV identities, order and raw token lengths match. All four update
counts, final head hashes/finiteness, protected parameters/buffers, absent
backbone gradients, source/addon manifests and evaluator row parity pass.
Fit-order SHA is285ea28b72d7a88a26251a1d92471b50aa726aa6eae725c8822fe0a26271ca7b.
No formal validation dataset was constructed for R1.

Original training/evaluation controller exited0. The scheduled CPU analysis
completed, then its initial verifier exited1 because the source helper reads
`data/meta_data/nr3d_train_scans.txt` relative to cwd. A separate v2 verifier
sets cwd to the already pinned source directory and passes against exactly
the same sealed training receipt and analysis. No training, evaluation,
threshold, gate, result, or source-tree modification was made for this fix.
Original error and controller exit are retained in`verification_v2/`.

Evidence in this PR: `refine-logs/reference_memory_v1_20260905/terminal/`;
raw rows are losslessly gzip-compressed, not filtered. The four final addon
weights remain on the server under
`/root/autodl-tmp/mcln_reference_memory_train_20260905_v1/results/`.

- Receipt SHA: `02cac2912e3bdc8a5aac5bb915209cc7a2d15947f40cfc59d38a004d0c97ce24`.
- Raw6172-row SHA: `3fddc3cf84b07d77049d9d35cd25339c3b90bd7a961feb95034e4030e3acc407`.
- Analysis SHA: `b211faf763042c7ce6736fa49dc0ce71a3c8ea098bbc51bf88700837cefe213c`.
- Verification SHA: `60f9ea7c032407b65d2e1602645670fd943b7836c64fe0525ea3a44bdfb12247`.

The next already scoped diagnostic separates the protected text, Query and
fused Mask branches, using fixed P1 cohorts and original thresholds. R1 remains
sealed without a same-route margin/capacity/seed sweep or Decoder replacement.
The overall three-benchmark objective remains incomplete.
