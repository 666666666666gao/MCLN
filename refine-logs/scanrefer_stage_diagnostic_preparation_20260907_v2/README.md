# Isolated stage diagnostic entry

`scripts/diagnose_scanrefer_readout_stages.py` is a diagnostic fork of the existing
formal evaluator, not an update to the evaluator queued for the current experiment.
`build_runner.py` records the exact base SHA and bounded source changes.

The entry keeps the original input loader, seeding, native selection, model
forward, V99 deployment policy, and frozen-state checks. It records six stages
from each completed forward, then computes IoU against the root GT. It also
records actual normalized Parent/Geometry input moments on valid candidates.
The positions in those vectors follow each bound artifact's feature schema.
Different arms may have different valid populations, so a moment difference is
not itself a causal explanation.

Run prerequisites are an actually completed formal reference and its independent
audit. A later run manifest must have schema
`mcln-scanrefer-stage-diagnostic-input-v1`, carry `training_directory`,
`training_receipt_sha256`, `data_root`, and `val_superpoint_files` unchanged from
that reference, and bind these seven reference files by SHA-256:

- `input_manifest.json`
- `controller.exit`
- `result/receipt.json`
- `result/rows.json`
- `result/native_rows.json`
- `result/protocol.json`
- `result/independent_audit.json`

The `files` map must bind the isolated run's script copies, including the original
formal helpers and three new diagnostic scripts. Runtime imports use only that
run's scripts and the existing frozen model snapshot. The actual manifest and
GPU launch remain pending the current formal result; this preparation is not a
queued or running diagnostic job. If ScanRefer passes, Nr3D/Sr3D takes priority.

Outputs go to `diagnostic_result/`, with `formal_rows=0`, `diagnostic_rows=9508`,
and `used_for_promotion=false`. `stage_summary.json` reports stage transitions and
paired effects; `reference_agreement` records prediction differences from the
bound formal run rather than silently replacing its metrics. Point identities
must match. Query slot changes do not prove instance changes.

Preparation evidence: 13 CPU tests, direct-file CLI loading, and 10 runtime
module imports passed in the original environment with CUDA hidden. There were
zero actual scene traces, GPU forwards, optimizer updates, or weight writes.
V1 only checked module-style CLI loading. V2 pins script search paths explicitly
for the direct-file invocation used by the existing job controllers.
