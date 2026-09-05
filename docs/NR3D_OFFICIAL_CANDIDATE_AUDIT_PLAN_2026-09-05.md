# Protected Nr3D official-path candidate audit

Fixed on 2026-09-05, before this run's validation outputs are read.

This is a read-only P1 reproduction and diagnosis of the protected averaged
checkpoint. P2 v1 failed both fixed screens and is sealed; neither P2 head is
used here. This audit does not reopen the failed G0-to-G1 performance route.

## Recovered evidence and the remaining source boundary

The original E57/E69 one-shot command, config, metric receipt, decision, and
checkpoint are present. The protected checkpoint SHA-256 is
`76aa6cd49ca20a34e78509465f1185b1b9040e60807ad327d6b0876aeb6edba1`.
The original receipt contains 7899 rows, REC 4475/3759, Mask 4192/3479,
and Mask IoU sum 2956.891842937146 (mIoU 0.3743374911934607).

The same historical receipt's `position.learned_selector` and `fixed_default`
are both 4271/3705. Source-choice diagnostics use unfiltered argmax, whereas
the primary REC path applies detector-overlap filtering before ranking. Thus
these counters are not interchangeable even within one evaluator invocation.
This is direct evidence for the protocol audit; it does not by itself explain
every difference from the separately produced 4275-row cache result.

Exact historical core model/training/loss file bytes have not been recovered
from the bounded runtime/backup/Git search. This run uses the existing frozen
`inputs_v3/fixed_source` tree, with all 612 source files checked against its
manifest. Its evaluator and selector match the historical hashes after LF
normalization, while several core files differ. The data augmentation change
is inactive in validation. Call this a current-source reproduction check;
do not claim historical code identity even if aggregate metrics match.

## Fixed execution

- One full native `TrainTester.main(... --eval)` pass on the normal Nr3D
  validation dataset, 7899 rows, B16, four workers, prefetch 2, one GPU/rank.
- Use the recovered historical CLI and compare every saved config field.
  Only checkpoint alias, output log directory, and experiment name may differ.
- Preserve native sampler order, worker seeding, loss computation, prediction
  size clamp, legal-candidate filter, source choice, and REC/Mask evaluator.
  cuDNN enabled/benchmark/deterministic all remain true as in the native entry.
- Compare every loaded parameter/buffer with the protected checkpoint before
  and after evaluation. No optimizer steps or checkpoint saves. No candidate,
  threshold, LR, epoch, padding, or structure sweep. Protected outputs stay intact.
- Add forward hooks only to observe raw sizes and the four existing FPS index
  outputs. The observer returns no replacement tensors. Native inference and
  native evaluator source are not edited.

## Recorded diagnostics and required validation

Each row records identity, scene, target class, sentence length, distractors,
input target points, pre/post-filter Top16/32/64/256 oracles for Default and
protected scores, final REC Query and native Mask Query, selected/box-oracle/
Mask-oracle quality, and detector-object availability. Candidate ranking uses
the native mask-before-sort convention; Mask IoUs use float64 integer-count
division to match the native NumPy metric.

Four FPS layers, fp2 seeds, and KPS Queries report the number of sampled centers
inside the root target. These counts do not measure receptive-field coverage.
Query seed membership is also not a ground-truth object identity assignment.
Nr3D's standard annotation loader supplies no text-to-anchor identity labels;
detector-object coverage remains a potential-anchor availability proxy.

The observer's REC hit counts must equal the primary position subgroup totals,
and its native Mask hit counts and IoU sum must equal the native metric receipt
(absolute sum tolerance 1e-8). Full row count/order and final state/source hashes
must match. Report historical metric reproduction separately from this observer
parity. A reproduction mismatch is recorded as a mismatch, not as a new model
gain/loss, and does not permit selecting another run by its score.

After completion, split errors into legal-TopK reselectable, larger-set only,
filtered-out, and Full256 missing. Analyze target sparsity and conditional Mask
quality using the full fixed outputs. Object identity versus boundary error
remains a separate attribution question unless direct evidence supports it.

The immediate deliverable is a closed official-path diagnosis. No failed P2
variant advances to Decoder integration, and this audit cannot promote weights.
