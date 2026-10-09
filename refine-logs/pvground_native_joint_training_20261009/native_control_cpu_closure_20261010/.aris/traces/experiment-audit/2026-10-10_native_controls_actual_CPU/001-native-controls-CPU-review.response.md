# Experiment audit: closed native-control CPU module execution

Date: 2026-10-10. **Overall verdict: WARN. Blocking issues within the closed CPU engineering scope: 0.**

The supplied closed record supports one synthetic CPU module check and its specific assertions. It does not establish full PV construction, native loss equivalence, optimizer responsibility, complete recovery, real input behavior, or REC/Mask accuracy. No GPU or control-training admission is granted by this review.

Execution scope: **CLOSED_CPU_SYNTHETIC_MODULE_ENGINEERING_NOT_PV_OR_ACCURACY**.

Requested reviewer route: gpt-6-astra / max. Actual model and effort: **UNATTESTED**. Review independence: **same-family**; acceptance status: **provisional**. Requested routing is not actual identity attestation.

## Evidence and reproducibility

All 30 requested files were read directly. The common-behavior initialization manifest was additionally read to compare the two reported payload digests, yielding 31 semantically reviewed inputs. All their SHA256 values appear in EXPERIMENT_AUDIT.json. The old source audit's 49 input hashes were independently rechecked; 37 of those files were hash-only continuity checks, not a new semantic review. The old seal's two report digests, 15 original bundle hashes and 27 prepared bundle hashes all agree with local bytes.

Only local text reads, hashes, JSON/base64 comparisons and AST parsing were performed by this reviewer. Ten supplied Python files and the transport's embedded remote program parse successfully. No reviewed program, Torch import, checkpoint load, neural operation, SSH request, GPU workload or current-training status query was executed.

The deterministic verifier first made an incorrect LF-only assumption about the local witness serialization. The witness actually contains 84 CRLF line endings from Windows text output. Parsed JSON equality and raw stdout decoding had already passed. The initial verifier/error are preserved; the corrected verifier explicitly checks the observed CRLF bytes and passes. No experiment was rerun and no reviewed input was changed.

The workspace bootstrap instruction caused SOUL.md, USER.md and two daily-note files to be read; the daily-note tool output was truncated. This additional context was disclosed to the parent, which directed continuation with disclosure. It was not used as experiment evidence and its contents/snapshots are not copied into the audit. The context is therefore not claimed to be pristine paths-only. MEMORY.md, AUTH, credential, askpass and known-host files were not read.

Citation roots:

- **R/**: D:/Program Files/UserCache/gb/codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/source/
- **C/**: D:/Program Files/UserCache/gb/codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/native_direct_controls_20261010/
- **P/**: D:/Program Files/UserCache/gb/codex/tmp/pvground_native_joint_training_20261009/native_joint_v2/
- **M**: D:/Program Files/UserCache/gb/codex/tmp/pvground_extremal_span_evidence_20261009/runner_v1/source_review/snapshots/034_whole_mask_range.py

The deterministic evidence is saved in the specified trace directory as DETERMINISTIC_CHECKS.json.

## A. GT/reference provenance — PASS for explicitly synthetic engineering evidence

The checker creates random query/support embeddings, points, native boxes and mask logits, with eight equally populated synthetic superpoints. It reads no dataset scenes or GT. The original A/B modules provide consistency references; fixed-half fusion supplies an algebraic reference. These are explicitly labeled synthetic module engineering checks, not benchmark accuracy (C/check_direct_control_modules_cpu.py:73-88,94-121,140-159).

The mask reference is prediction-derived: logits are detached, fused and thresholded, while invalid extents retain the native prior (R/native_mask_geometry.py:13-23). This is an inference construction, not independent GT. Member statistics count sampled point rows per superpoint (M:12-37); they do not establish deduplicated visibility truth.

The closed checker hashes and loads the A and B payloads and asserts their role/mode and entry counts (C/check_direct_control_modules_cpu.py:44-53). The reported d06adb8e... support and f9898c5d... span digests match the directly read common manifest (C/common_behavior_check/init.json:8-11). The reviewer did not open checkpoint bytes or independently certify the parents' training history. The full initializer's span-to-support parent-identity assertion is not in the executed checker (C/source/native_model_initialization.py:57-61).

## B. Score normalization — PASS within this engineering scope

No accuracy or performance score is normalized by the model's own maximum, mean or other prediction statistic. The checker uses exact-value assertions and records the raw maximum absolute content/full mask difference, **0.10565423965454102** (C/check_direct_control_modules_cpu.py:104-138,152; C/cpu_transport_attempt2/CPU_MODULE_WITNESS.json:67-75). That value is an output diagnostic, not a benefit, quality score or accuracy metric; it was not neurally recomputed by the reviewer.

Count scaling, support-weight normalization, geometric scaling and clamped gates are module feature/parameter computations (C/source/mask_support_corrector.py:43-53; C/source/extremal_span_mixer.py:72-106). They are not self-normalized reported performance.

## C. Result existence and numbers — PASS for the current closed CPU record

Both the attempt-2 transport and module exit codes are 0. Module and outer transport stderr are empty. The recorded subprocess interval is **05:50:34.790431–05:50:36.915157 +08:00**, a difference of **2.124726 seconds**, using the receipt's timestamps. The record reports Torch **1.10.2+cu111**, empty CUDA visibility and CPU devices for the two recorded tensors (C/cpu_transport_attempt2/CPU_MODULE_EXECUTION.json:2-14; C/cpu_transport_attempt2/CPU_MODULE_WITNESS.json:2-16). The CUDA-enabled build string does not make this a GPU result.

The base64 stdout/stderr in CPU_REMOTE_RAW_STDOUT.json decode byte-for-byte to the local module files. The parsed stdout equals CPU_MODULE_WITNESS.json; the latter also matches the exact Windows CRLF pretty serialization. The local witness is therefore a copy of the same result, **not a second independent witness or run**. The execution JSON equals the remote fields plus the transport's documented local additions (P/run_direct_control_modules_cpu_attempt2_authorized.py:72-85).

The execution's source-audit digest matches the current old audit JSON, and the old seal's report hashes match. The original/prepared maps in the module output equal CPU_BUNDLE.json and locally rehashed files. The successful source path asserts 15 remote original hashes and 27 transported-file hashes before module operations (C/check_direct_control_modules_cpu.py:34-37). This is stronger than a bare success string but does not supply a post-execution remote rehash.

The two earlier root-level exit-255 receipts and their error text remain separate (C/CPU_TRANSPORT_EXIT.json:1; C/CPU_STATIC_ROOT_READ_EXIT.json:1; corresponding stderr files). They are not rewritten as successes. The static witness records the CPU root absent and no CPU receipts; attempt 2 independently asserts root absence before creating it (P/read_only_static_diagnostic_20261010_attempt1/REMOTE_STATIC_WITNESS.json:9,15; P/run_direct_control_modules_cpu_attempt2_authorized.py:43). The static witness's remote timestamp is later than the local transport finish timestamp, so exact cross-host timing/order is not inferred without clock synchronization.

Preparation documents and the old source audit are retained historical snapshots. Their earlier “not executed” states are not the current CPU result. Historical 5677/4920 accuracy and the “39 invalid” source comment are outside this result verification; 5658/4850 remain target literals, not new observed outcomes (C/DIRECT_CONTROLS.md:35; R/native_mask_geometry.py:8).

## D. Actual called code — WARN for unexecuted integration paths

The successful checker actually reaches the standalone old/common/content-only/bypass A forwards, native mask-reference construction, old/common/fixed B forwards and the two selected backward checks before writing the report (C/check_direct_control_modules_cpu.py:38-138,160-161). Thus these are supported execution-record assertions, not merely dormant test definitions.

The checker creates seven standalone module objects: four A variants and three B variants. It loads **10 A state entries / 27,841 registered parameters** and **14 B entries / 29,793 parameters** (C/check_direct_control_modules_cpu.py:52-71). It does not call configure_native_model, construct PVGround or execute the native training controllers. Freezing is applied manually in the checker (lines 63,71), so actual factory integration and optimizer exclusion are not demonstrated.

The snapshot's member_statistics is imported/called; mask_range_evidence is not called in this path (C/source/mask_support_corrector.py:9,66-67; M:12-37,40-67). Hashing or transporting initialization, loss, training or controller source does not establish that those paths ran.

## E. Scope and gradients — WARN

This is **one synthetic fixture, seed 2027, batch 1, 256 candidates, 8 superpoints and 50,000 points**. It is not four completed full-model control experiments. The positive native sizes are generated as rand + 0.5. Real scenes, official evaluation, formal control training and completed training seeds are absent from this execution (C/check_direct_control_modules_cpu.py:73-88,145,156-159).

Supported assertions are:

- Common A mask values agree with old A; their member-statistics arrays agree.
- Common B centers, sizes and evidence values agree with old B for the materialized common-A reference.
- A bypass returns the original query-mask tensor object.
- Content-only A input columns 64:70 are exactly zero.
- Fixed B reports 0.5 raw/effective gates and satisfies the half-fusion formula.
- Bypass A and fixed B have their parameters manually frozen.
- Common A's query input and output weight have nonzero gradients under mean squared module output.
- Fixed B's summed-output derivatives to independent native-box leaves are 0.5, and its frozen parameters have no gradients.

These map to C/check_direct_control_modules_cpu.py:91-138. The report's “bitwise equal” fields are backed by torch.equal and np.array_equal; there is no explicit tensor storage-byte or payload-dtype comparison. Strict standalone loads, state-key checks and parameter counts should not be promoted into dtype guarantees or full 1295-state validation (lines 59-70,104-117; C/DIRECT_CONTROLS.md:28-30).

**The exact-half gradient has a crucial dependency limit.** The mask reference in p is computed first from ordinary tensors. The checker subsequently creates new gradient_center/gradient_size leaves and replaces only native_coarse_center/size, leaving the reference already materialized and independent (C/check_direct_control_modules_cpu.py:109-110,131-137). It checks the partial derivative of summed module outputs in the positive-size region, not a native-loss or end-to-end derivative.

In a full forward, an invalid reference can retain the same native prior (R/native_mask_geometry.py:19-21). For center, half times reference plus half times native then has derivative 1 when both refer to that same input; the positive-size analogue also holds. An assertion that all full-chain native-box gradients are always 0.5 would therefore be unsupported and false for such dependencies. No rerun or speculative implementation fix is required to preserve the narrower valid observation.

The content-only intervention zeros the residual predictor's five mask-state and one count columns; it still adds the residual to the native mask, and downstream B retains mask-derived information (C/source/mask_support_corrector.py:43-57). Frozen branches keep registered states but change effective trainable capacity. The shared trained-parent history and these scope limits are disclosed in C/DIRECT_CONTROLS.md:10-23. The fixture does not show A/B/C effectiveness, extremal-source-specific value, never-A/B histories, or coverage of every invalid/degenerate/tied-support configuration.

## F. Evaluation type — PASS for explicit classification

Actual type: **synthetic_module_engineering_fixture_not_accuracy_evaluation**. Within the skill's taxonomy, the original-module comparison is **synthetic_proxy**, combined with algebraic and selected local autograd checks. It is not real_gt evidence.

The witness explicitly sets full PV/1295-state, native criterion and optimizer/scheduler/recovery checks false, formal_accuracy null, and full_goal_complete false (C/cpu_transport_attempt2/CPU_MODULE_WITNESS.json:76-83). Those limits agree with the called code.

## Claim impact and remaining evidence

The bounded CPU engineering assertions are supported with the qualifications above. The old source audit is not retroactively a GPU approval, and this actual CPU review also gives none.

Still untested by this execution: full factory and official/G/1295-state integration; explicit payload dtype identity; same-input complete common/native outputs and native losses; all chosen-arm task gradients and optimizer groups; optimizer/scheduler behavior; complete save/cold recovery; real inputs/data/evaluator provenance; REC/Mask accuracy; effective contributions or target attainment; Nr3D/Sr3D. Current normal-training state was not queried or inferred.

The zero GPU/status counters and original_source_unchanged flags are literal fields in the scripts, not independent telemetry (C/check_direct_control_modules_cpu.py:159; P/run_direct_control_modules_cpu_attempt2_authorized.py:56-58,79-81). The visible CPU-only path, -B invocation and narrow output writes support the stated workload scope, but no broader remote-state or actual-model identity attestation is claimed.

Retain the closed receipts and this qualified review. Any later broader claim needs its own already-specified full-model or real-data evidence. The seal is a SHA256 consistency snapshot, not a signature, execution authorization or scientific-result approval.
