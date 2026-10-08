# Storage source revision review: CPU RNG replay

**Verdict: PASS for the bounded RNG correction; no blocking findings.** Initial finding B1 is resolved in `postrun/inspect_closed_terminals_rng_replay_authorized.py`, SHA-256 `1d17e095bfdca9a5caec2defbd31627bdec6c5b26168d6c8e9cce5db8005ae0c`. The original inspector and its initial FAIL reports remain unchanged. This is a `SOURCE_ONLY` revision review, not an actual reconstruction or trained-result audit.

Reviewer: `/root/pvg_compressed_support_storage_source_audit`. This is the same reviewer that began the initial review with fresh context: `fresh_context_at_initial_spawn=true`, `revision_same_reviewer=true`; no new fresh spawn occurred for this revision. Review independence is `same-family`, acceptance is `provisional`. Requested model/effort are `gpt-6-astra` / `max`; actual backend/model/effort remain **UNATTESTED**.

Relative references below use `C:/Users/gb/.codex/tmp/pvground_compressed_geometry_support_20261008`. Readback helpers use `C:/Users/gb/.codex/tmp/pvground_final_quality_20261005/runtime_bundle`.

## B1 resolution

The revised embedded program calls `torch.manual_seed(spec['seed'])` and captures `parent_rng_state=torch.get_rng_state()` immediately before constructing the baseline parent (`postrun/inspect_closed_terminals_rng_replay_authorized.py:59–61`). The unchanged spec seed is 2027. For each arm, it restores that exact saved CPU state immediately before calling `build_support_model` again (lines 81–82). The checkpoint checks and loads occur before this reset, so they cannot shift the starting RNG state of that construction.

This directly addresses the actual source of B1. `selected_mask_reference_factory.py:52` installs the readback module; its constructor initializes hidden Linear/attention/face-embedding tensors using PyTorch RNG (`runtime_bundle/pvground_boundary_evidence_readback.py:18–27`). Those tensors are not supplied by the selected ten geometry tensors or the ten support tensors. Replaying the starting state reproduces the same construction draws. The support head is installed after the shared parent construction and loaded strictly from its own payload (`mask_support_model_factory.py:9–24`; `mask_support_corrector.py:77–81`), so additional support-head initialization does not alter the already constructed parent tensors or the starting state of the next arm.

The full comparison is preserved: the baseline still has 1,304 state tensors; expected terminal state remains the complete baseline plus the terminal delta; key-set equality, 1,314-tensor count, and exact equality of every tensor remain required (revised lines 62–63 and 83–85). One deployed support head, zero geometry output weight/bias, CPU device residency, and uninitialized CUDA are still asserted (lines 86–89).

Deterministic AST comparison verified that the outer program is identical except for the embedded source literal. Inside the embedded program, the only changes are three RNG statements and four additional receipt fields. Removing precisely those additions makes the embedded AST equal to the original. **All 28 original embedded assertions are unchanged.** No equality condition was removed, narrowed to a subset, or relaxed to a tolerance.

## Receipt schema and downstream consumers

The revised success record retains its original status, `weights`, `required_dependencies`, `full_cpu_state_tensors`, and `neural_forwards` (revised lines 90–99). These are exactly the five inspection fields read by the archival and cleanup sources.

| Consumer | Relevant fields and source evidence | Result |
|---|---|---|
| `postrun/archive_retained_content_authorized.py` | Requires the same inspection status, 1,314 CPU tensors and zero forwards, then selects the content weight (lines 17–20). It consumes each weight's path/bytes/SHA-256 and retains dependency records (lines 27–36). | Compatible; field names, meanings within CPU scope, and nested weight/dependency schemas are preserved. |
| `postrun/retire_closed_nonbest_authorized.py` | Requires the same status, selects the box-conditioned weight, and reads dependency paths/hashes (lines 17–18, 27, 40, 107). | Compatible; no changes to its deletion targets or prerequisite checks. |

The original archive and deletion source hashes are unchanged. Their selected-checkpoint, complete local archive, prior warm archive, exact deletion bounds, protected dependency, closure and actual-audit gates remain intact. The revised inspector retains the outer `ACTUAL_CLOSED_TRAINED_PAIR`, fresh-audit, no-blocker and exact-reviewed-input-hash requirements (lines 14–20). This revision report cannot satisfy those actual-result gates.

The corrected inspector still writes `postrun/checkpoint_inspection.json` only after the remote program exits successfully and returns the expected status (lines 104–112). The source correction receipt truthfully says `SOURCE_ONLY_CPU_RNG_REPLAY_CORRECTION_NOT_EXECUTED`; its old and new paths/hashes match the actual files (`postrun/CPU_RNG_REPLAY_SOURCE_CORRECTION.json:2–12`). It is not a runtime inspection receipt.

## Claim scope

The four added receipt fields correctly qualify the intended result (revised lines 96–98):

- `cpu_parent_rng_seed=spec['seed']`;
- `cpu_parent_rng_replayed_per_build=True`;
- `uncheckpointed_zero_R_hidden_states_matched_between_cpu_builds_only=True`;
- `original_formal_gpu_all_state_identity_verified=False`.

They are emitted only after both complete comparisons succeed. `full_gpu_cold_reconstruction=False` remains explicit (line 99).

Thus a future successful execution can establish consistency between these cold CPU builds and exact application of the supplied checkpoint tensors. It cannot establish that unsaved, zero-output readback hidden tensors equal every hidden tensor of the historical formal GPU instance. The per-weight `full_state_exact=True` and downstream `full_cpu_strict_restore=True` must be read within this CPU construction scope. Neither storage script asserts historical formal GPU all-state identity; the archive already retains `full_gpu_cold_reconstruction=False` (archive line 37).

The initial A–F assessment is not rerun as an actual experiment audit. Its source conclusions remain qualified: dataset-derived GT and ordinary metric denominators; `real_gt` classification; reachable metric paths; pending complete-array/result verification; one seed/two arms/one validation split; no established geometry advantage or completed goal. This RNG revision changes no metrics, selection rule, training history, neural model source, or authorization. The initial result-existence and evaluated-scope cautions remain applicable and are not converted into result-level PASS claims.

## Preservation and validation

The initial 31 evidence inputs are byte-identical to their recorded hashes. This includes the original failed inspector, model factories, readback constructor, reviewed NN sources, analyzer, specification and storage scripts. The corrected program is a separate postrun inspector; no existing runner source was changed in these reviewed inputs.

The following original artifacts are preserved byte-for-byte:

- `STORAGE_SOURCE_REVIEW.md`: `86b95b6c7d6132c931e838755d0919f37c8d623ca689b8d7e84b0b421efb9de9`;
- `STORAGE_SOURCE_REVIEW.json`: `b8da314c1b3cb6e967a500e1c2bfc314bc95e35a6b3152b1ed1065ebd2c39dfa`;
- all four files in the run05 trace, individually hashed in the revision JSON.

The revision JSON records exact SHA-256 values for all 40 read inputs, distinguishing direct semantic review from unchanged-evidence hash checks and trace preservation checks. Standard-library parsing passes for 24 Python files and four embedded programs, including both complete inspector programs and the cleanup program. Twelve JSON inputs parse. The exact corrected embedded-source SHA-256 is `7944dddc61ae4ffe85ba2f2d58a16db7d724781ad0c461ef9e5bb229ef73b702`.

The review's first AST-difference helper selected the dependency loop instead of the arm loop and stopped with an assertion before completing that comparison. Selecting the explicit `arm` loop corrected the helper; the full deterministic comparison then passed. No reviewed program was imported or executed by that helper.

Only these distinct revision reports and run06 response/metadata are written. No SSH connection, credential/AUTH-file read, checkpoint load, neural/optimizer execution, deletion, source edit, archival/retention execution, further agent, or actual-result audit occurred.

The source correction is accepted within this bounded review. Complete-array intake, CPU recount, an actual closed-pair audit, CPU inspection, and the storage actions still require their real prerequisites and successful execution. This review establishes none of those runtime outcomes.
