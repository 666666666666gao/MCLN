# PV-Ground candidate-consistency follow-up code review

Verdict: **WARN — both first-review blockers are resolved; no blocking source findings remain.**

Review time: 2026-10-03 05:35 CST. Attribution: delegated Codex reviewer, **same-family / provisional**. This is the one requested follow-up by the same reviewer, comparing the actual minimal corrections with the preserved first-round source and FAIL report. It is not another independent external review or an executed model audit. The exact underlying model/version is not independently exposed in this review context.

The corrected implementation satisfies the source-review gate for the already planned CPU fixture followed by real GPU preflight. **CPU and GPU sanity remain NOT_RUN. The full pair remains NOT_STARTED and must not launch until the actual output-directory storage requirement is met.** This review does not authorize deletion.

Task-relative paths refer to `C:/Users/gb/.codex/tmp/pvground_candidate_consistency_20261003`. `P` denotes the sealed parent `C:/Users/gb/.codex/tmp/pvground_g_p2_20261002/complete/source`.

## First-review findings and their disposition

### B1 — CPU/CUDA concatenation: resolved in the actual GPU path

`pvground_candidate_consistency.py:21` now constructs:

```python
additional = selected[bid].nonzero(as_tuple=False).flatten().to(queries.device)
```

The actual native matcher returns both query and target indices as CPU int64 tensors (`P/ported/models/losses.py:378-390`), and the runner still clones those original pairs (`run.py:285-288`). Therefore the additional query indices and their `zeros_like` root-target indices are now on the same device as both original tensors before the two concatenations (`pvground_candidate_consistency.py:22-23`). Native loss already supports those CPU indices against CUDA prediction/target tensors; that is its original matching path.

This is the direct fix for the reported failure. It only moves detached loss-index metadata; it does not change qualification values, query ordering, target identities, gradients, actual matcher results, or regression/Mask responsibility. The fix is present in the common-root module that the launcher uploads and imports. All three updated specs match that module's actual bytes. GPU execution has not yet verified it; this is a source-established resolution.

### B2 — disconnected provenance checks: resolved

The runner now checks the existing expected module pins against `sys.modules[name].__file__` for the semantic-assignment and consistency modules (`run.py:43-46`) and for the observation/task/source modules (`run.py:66-69`). These are the modules actually loaded by the common-root execution path, rather than unused per-arm copies. All five resolved module paths and their digests are included in `imports.json` (`run.py:120-124`).

The early `import pvground_task_observation_query` (`run.py:47`) makes its observation/source dependencies available before their checks. I inspected this chain: task imports observation, observation imports source and the existing OpenPCDet pointnet2 utilities. The inspected module-level code does not instantiate the model or invoke a model forward, and it does not preload the top-level `models` namespace whose identity is checked later. Model construction and RNG reset remain in their existing locations. Actual target-runtime imports remain NOT_RUN.

`run.py:77` now compares `sha(spec['source_port'])` with the already supplied `spec['source_port_sha256']` **before** reading and trusting the manifest's file list (`run.py:78-79`). That closes the unused expected-pin finding. No new manifest format, compatibility mechanism, or hash framework was added.

The corrections are also present in `build_runner.py:76-94`. I executed only its read-only string-generation prefix, stopping before every write/copy/compile action; its result exactly equals the current `run.py` text. Regeneration will preserve these fixes.

### N1 — qualified-fixture receipt wording: resolved

`cpu_test.py:84` now reports `expanded_native_loss_and_gradient_allclose=True`, matching its tolerance comparisons at lines 76-79. The no-qualified fixture still asserts zero correction and exact equality of query-projection gradients (`cpu_test.py:66-70`). Only the receipt label changed; no test assertion was weakened. Neither fixture has been run here.

## Semantic and experimental contract remains correct

The full round-one/current diff shows only the index-device transfer, the receipt label, the connected provenance checks/import recording, and the regenerated existing module pin. `pair.py`, `run_preflight_authorized.py`, and `prepare_pair.py` are byte-identical to round one. All four reused G/D/C modules are byte-identical to the sealed parent. No architecture, loss formula, dataset, score rule, optimizer schedule, fallback, or new experiment was introduced by this patch.

- **Qualification:** the unchanged `qualified_unmatched()` detaches prediction/root geometry, requires strict root IoU `> 0.5`, and excludes all Hungarian matches (`pvground_semantic_assignment.py:9-23`). A query matched to another GT remains protected even if it overlaps root. High IoU is geometric qualification, not verified physical-instance identity.
- **Loss-only correspondence:** every original pair is retained; selected unmatched queries are appended with root target zero only for `loss_sem_align()` (`pvground_candidate_consistency.py:18-31`). The actual matcher output is not modified or sent back into native box or Mask matching. All 256 queries remain. Low-IoU and matched queries are not deleted or indiscriminately made root positives.
- **Replacement:** the native semantic method overwrites unmatched not-mentioned positives at the expanded indices (`P/ported/models/losses.py:660-671`). The runner adds `(new - old) * (.5 / 7)` to the already computed native objective, alongside the existing G CE correction (`pvground_candidate_consistency.py:27-33`, `run.py:292-304`). It replaces the final semantic term rather than adding contradictory target supervision.
- **Native recipe:** the method retains the native temperature, weights, other-entity term and bidirectional formula. Both calls use the ORIGINAL matched-count denominator. ScanRefer's 0.5 weight and seven-head average match the sealed source (`P/ported/models/losses.py:831-835`, `P/ported/models/losses.py:944-954`). The native prefix order is proposal, last, then heads 0-4, so `matching[1]` still selects the final layer. Intermediate targets are unchanged.
- **No-qualified case:** empty additions retain exactly the original correspondence values. The prepared fixture still checks zero correction and exact query-projection gradient equality, protects the other-GT match, preserves the original pair tensors, and compares the qualified case against explicit expanded native indices. These are inspected assertions, not completed test results. The loss may propagate through shared parameters and affect other queries' later predictions; unchanged matching does not imply all global gradients remain unchanged.
- **GT boundary:** qualification is called only by the training step on fit-batch GT. The evaluation path does not call the new correction. Model inputs remain points/voxels, text, detector proposals/classes/masks, and superpoints; GT is supplied to the criterion after prediction. Last/bbs scoring, parsed-language token maps, offline GT metrics, and diagnostic oracles are unchanged. No validation geometry enters training qualification or an inference gate.
- **Fixed comparison:** both arms retain original G plus its author parent checkpoint, fresh AdamW, seed 2027, nominal batch 8, learning rates `1e-5`, weight decay `5e-4`, clip norm 0.1, and one pass over 29,778 fit rows / 3,723 updates. The final two-row batch is retained. The 6,887-row module holdout and separate full 9,508-row formal validation are unchanged; author exposure to holdout scenes remains disclosed. P2 is disabled; no P3, fused tail, teacher, or quality head is installed. Efficacy remains unproven.

## Deployment sequence, SSH, and storage

The deployment script still requires a PASS/WARN review with no blocking findings (`run_preflight_authorized.py:10-11`). Its ordered command list runs the CPU fixture first, then `run.py --mode preflight` under the existing GPU lock (`run_preflight_authorized.py:35-40`). A nonzero exit stops the sequence (`run_preflight_authorized.py:49-53`). It does not invoke `pair.py` or start either full arm.

The preflight still performs a backward-capacity witness with no optimizer update, followed by exactly two optimizer steps, then in-memory delta/optimizer serialization and restore checks (`run.py:323-349`). The capacity forward can update training buffers and consume RNG, but it belongs to the disposable preflight process; full arms reload original G independently. No CPU/GPU success, measured memory use, serialization size, or disk-save result is claimed by this review.

SSH still loads known-host keys and leaves Paramiko's default rejecting policy in force; no accept-unknown-host policy was introduced. Credentials are read from the environment and remote arguments/environment entries are quoted using `shlex.join` (`run_preflight_authorized.py:13-16`, `run_preflight_authorized.py:32-42`). I found no host-verification bypass or shell-injection defect in the reviewed path. No network connection was made by this review.

**The full pair is still on a storage hold.** The supplied actual-directory inventory reports **437,751,808 free bytes**; the larger free-space value observed through parent-checkpoint paths does not establish capacity for the output directory. The new preflight serialization size is still unmeasured.

`pair.py:18-21` requires a passing real two-step preflight receipt. Before launching children, `pair.py:28` requires actual `disk_usage(root).free >= 3 * serialization_bytes + 256 MiB`; it also checks `2 * serialization_bytes + 128 MiB` before each training arm (`pair.py:38-42`). These checks remain in the unchanged deployment path and cover retained endpoints plus atomic checkpoint replacement. The launcher does not bypass them. No launch, deletion, cleanup, or storage relocation was performed or authorized by this review. Pending deletion permission is not treated as approval; full training must remain stopped until adequate actual capacity is established.

## Performed checks and limits

The first-round source and both FAIL-report files remain preserved under `review_round1`; their original recorded runner/module pins match. I read the current module, CPU fixture, generator, runner changes and their surrounding execution path, all three specs, pair, launcher, and unchanged qualification/import dependencies. I compared every relevant current file to round one, confirmed reused modules against the sealed parent, checked all existing module pins in all three specs, parsed task Python source locally, and reconstructed the generated runner in memory without writes. The unchanged semantic/data/optimizer/metric conclusions above retain their native-source basis from the first review.

Current existing identities:

- `run.py`: `919fc36d61b8f21a7285f55c153717f05baa4e969ca43ccbc25171e165bc7392`
- `pvground_candidate_consistency.py`: `dc25727853c178f312db05f5f66c49fb56ecc0b72d390e91aa28dcd6f9085186`

Local static parsing/generator reconstruction used Python 3.13. The pinned execution environment remains Python 3.7.11 / Torch 1.10.2+cu111. This review does not turn local parsing into a target-runtime result or require a speculative compatibility layer.

**NOT_RUN:** CPU loss/gradient fixtures; target-runtime imports; GPU forward/backward; two optimizer steps; CUDA execution of the corrected indices; GPU memory measurement; model/optimizer serialization witness; disk checkpoint save/reload; SSH deployment; fresh remote capacity measurement; either full training arm; holdout/formal evaluation; efficacy or novelty assessment. Local CPU/preflight/pair receipts are absent, consistent with the stated status. No implementation file was edited, no GPU job was started, and no file was deleted by this review.

There are **no remaining blocking or nonblocking code findings** from this follow-up. WARN records the unrun sanity stage and continuing full-pair storage hold. The source fixes are complete; the next existing workflow step is CPU sanity followed by real GPU sanity, with full training still subject to the storage gate.
