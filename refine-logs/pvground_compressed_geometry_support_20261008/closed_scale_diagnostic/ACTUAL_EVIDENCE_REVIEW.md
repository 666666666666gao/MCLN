# Actual geometry-scale diagnostic evidence review

**Verdict: WARN. No blocking findings.** Execution scope: **ACTUAL_CLOSED_DIAGNOSTIC**.

This fresh-context review is **same-family / provisional**. Requested route: `gpt-6-astra`, reasoning effort `max`, backend `codex`. Actual backend, model and effort are **UNATTESTED**. Requested routing is not backend attestation. No outside reviewer was called.

All 25 supplied artifacts were read and bound to byte SHA256 in `ACTUAL_EVIDENCE_REVIEW.json`. The reviewer used only artifact/source inspection and PowerShell/.NET arithmetic; no personal memory, SSH, GPU/neural execution, optimizer operation, installation, source modification or new experiment. The required experiment-audit instruction file was also read. References below are relative to `C:/Users/gb/.codex/tmp/pvground_support_geometry_scale_20261008/`, except the explicitly named historical log.

The actual artifacts support a completed and closed 58-row, eight-batch, zero-update diagnostic. They support a strong local gradient response to geometric-input compression in one fresh batch. They do not support trained efficacy or historical-spike recreation.

## A. Ground-truth provenance — PASS

The active target path uses dataset annotation fields, valid GT boxes and majority-superpoint GT masks (`matched_mask_objective.py:6`, `:20`, `:40`). The wrapper obtains boxes from scan annotations (`run_mask_support_pair.py:159`) and passes points, text, superpoints and detector proposals/classes as inference inputs (`:232`). Predicted-Mask extents are an inference reference, not replacement supervision (`mask_reference.py:10`).

The result has 58 actual assignments, one per expression, all to valid target ID 0; all arm loss records have matched-query and valid-GT counts equal to their batch size. The raw dataset and external native implementations are outside the supplied paths, so those bytes were not independently inspected.

## B. Normalization — PASS

The active helper divides by the actual valid-GT count and uses coefficients **5/1/10/2** (`matched_mask_objective.py:35`, `:51`). The 24 recorded scalar losses reconstruct from their four components to maximum absolute error **7.54e-8**, consistent with float accumulation. Each batch's three arms have exactly equal components and total losses.

There is no reported accuracy score or self-normalized result. Division by clamped predicted size is the input construction being measured (`mask_support_corrector.py:32`); count normalization uses fixed 50,000 points (`:43`). The diagnostic records unclipped parameter and total L2 gradient norms (`diagnostic_scale_run.py:69`). All totals reconcile from their ten parameter norms to maximum relative error **5.25e-8**.

## C. Actual artifacts and completion — PASS

`actual/diagnostic.exit:1` is 0. The eight progress lines match the diagnostic's batch labels, geometry counts and gradients exactly, followed by `INPUT_SCALE_DIAGNOSTIC_COMPLETE` (`actual/diagnostic.log:7`, `:15`). The complete parsed diagnostic equals the copy in `observation_002.json:1`; `observer_terminal.terminal` equals that observation.

Original neural PID **901298** was live on the first observation and absent with exit 0 at the terminal observation (`observation_001.json:1`; `observer_terminal.json:4`). The observer receipt has two observations and no neural/optimizer restart. All 13 supplied files shared with the prior source-review manifest retain their hashes. All six supplied runner files listed in the spec match its hashes. The actual import/load receipts match the spec and declared native evaluator digest (`actual/load.json:15`; `actual/imports.json:14`).

The preserved earlier `SOURCE_REVIEW` remains correctly labeled SOURCE_ONLY / NOT_EXECUTED for that earlier stage. This separate review supplies actual completion evidence.

## D. Called and inactive paths — WARN

The entry binds the diagnostic override (`run_mask_support_pair.py:111`, `:255`). Its active path calls eight frozen-parent forwards, GT matching and three geometry-input/loss/backward probes per batch (`diagnostic_scale_run.py:33`, `:60`, `:68`). Each batch reports one final semantic-head call.

The base constructor creates two AdamW objects (`paired_support_loop.py:42`), but no optimizer step occurs. The inherited clipping/update, checkpoint save/restore/integration, original run and evaluation routes are inactive (`:109`, `:139`, `:244`, `:339`). The log head has no optimizer.

The **native loss equality witness is inactive** (`paired_support_loop.py:64`). Official evaluator methods, `mask_reference.reference_bounds_witness` and the local box-IoU routine are also not invoked by this diagnostic. Their presence or import cannot be described as a new numeric native-criterion reconciliation or accuracy evaluation. The unsupplied forward helper implementation was not reread here.

## E. Scope and historical panel — PASS

Observed scope is **8 batches, 58 distinct training expression rows, 58 distinct recorded point hashes, 14,848 candidate inputs, 58 matched queries, 24 probe records, one seed (2027)**. Batch sizes are `[8,8,8,8,8,8,8,2]`; every expression retains 256 queries.

The panel matches the historical row order and both archived arm norms at steps **1, 2, 522, 523, 524, 3211, 3722 and 3723**. Independent parsing of all 3,723 lines in `C:/Users/gb/.codex/tmp/pvground_mask_support_correction_20261008_v2/complete_fit/train.jsonl` gives 29,778 distinct rows, correct batch sizes and matching arm/GT counts. Both headline gradient/loss summary tables reconcile independently with `TRAINING_SCALE_ANALYSIS.json`.

These are expression rows, not 58 established physical scenes. Fresh augmentation and RNG reset are active (`diagnostic_scale_run.py:26`); heads have fresh identical zero-output initialization (`mask_support_corrector.py:20`; `paired_support_loop.py:35`; `diagnostic_scale_run.py:23`). Historical augmentation, Gumbel RNG and learned support-head state are not restored.

## F. Evaluation classification — PASS

**`real_gt`: training-panel supervised loss and zero-output local derivative diagnostic.** Feature/floor measurements are tensor diagnostics. This is not a trained ablation, historical-spike replay, holdout test or formal benchmark evaluation. Compression remains an input probe.

## Independently reconciled measurements

“Historical batch label” identifies the chosen row group; it is not a replayed training step. Floor counts mean at least one coarse-size dimension `<=1e-6`. Gradients are measured total parameter L2 norms. The loss shown is exactly equal across all three arms.

| Historical batch label | Rows | All floored candidates | Matched floored candidates | Content gradient | Raw geometry gradient | Signed-log gradient | Common scalar loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 434 | 0 | 0.0083706025 | 0.0084391534 | 0.0084123882 | 0.33477736 |
| 2 | 8 | 437 | 0 | 0.016990922 | 0.016966680 | 0.016969806 | 0.56529289 |
| 522 | 8 | 463 | 0 | 0.035824750 | 0.035826661 | 0.035826020 | 0.40517753 |
| 523 | 8 | 468 | 0 | 0.012916850 | 0.012961212 | 0.012944290 | 0.42308256 |
| 524 | 8 | 450 | 0 | 0.018363893 | 0.018594306 | 0.018483188 | 0.80771446 |
| 3211 | 8 | 308 | 1 | 0.10348465 | 45358.313 | 0.46410975 | 0.90543294 |
| 3722 | 8 | 361 | 0 | 0.034958858 | 0.035155252 | 0.035113666 | 0.54451561 |
| 3723 | 2 | 122 | 0 | 0.023061389 | 0.023378987 | 0.023292240 | 1.0399179 |

The runtime floor aggregates sum to **3,043 / 14,848 candidates (20.49%)** and **8,911 / 44,544 dimensions**. Among 58 matched queries, there is **one floored candidate with three floored dimensions**, in the step-3211-labeled batch (`actual/diagnostic.json:76557`).

All per-query feature maxima were inspected arithmetically. Raw maximum amplitude is **11,154,019**; after signed-log transformation it is **16.22731018**. There are 3,102 raw query maxima above 100 and 3,044 above 10,000. Only one matched-query maximum exceeds either threshold: **row 34293, query 175**, with raw maximum **8,175,157.5** and compressed maximum **15.91661072** (`actual/diagnostic.json:63846`, `:64974`, `:73412`, `:76583`). Matched maxima exactly index the recorded query vectors; all 120 stored per-arm/batch quantiles recompute exactly. The log-amplitude relation differs from double-precision `log(1+x)` by at most **9.99e-7**.

The row-level floor attribution is **conditional**. The sole matched floor event and sole large matched amplitude occur in the same batch, but per-query coarse sizes/floor indicators were not retained. This audit cannot independently prove that the floor event belongs to that particular row/query. Likewise, per-member threshold counts and floor counts are runtime aggregates, not recomputations from complete feature/size tensors.

## Gradient interpretation and state evidence

In the step-3211-labeled fresh batch, the raw gradient is **45,358.3125**, versus **0.4641097486** after signed-log compression and **0.1034846455** for content-only input. The raw/log ratio is approximately **97,732**. All three losses remain **0.9054329395** (`actual/diagnostic.json:63897`, `:68116`, `:72335`, `:63903`). The large derivative is measured by backward; it is not inferred from loss magnitude.

All **192 non-output parameter norm records are zero**. Only the output weight/bias derivatives are nonzero; the output-bias norm is identical across arms. This matches the fresh zero-output architecture: gradient differences measure output-layer sensitivity to the hidden activations at this initialization. The intervention supports that local derivative conclusion; it does not establish trained stability or accuracy.

Every arm records exact zero-output Masks. Equal-loss, unchanged-head, no-parent-gradient and final unchanged-parent assertions lie on the executed route before completion (`diagnostic_scale_run.py:66`, `:71`, `:74`, `:82`, `:91`; parent comparison implementation `paired_support_loop.py:58`). Completion records **zero optimizer updates, zero weight files, unchanged parent/head states and no accuracy result**. No clipping or checkpoint-save path is active.

The evidence level is **successful active runtime assertions plus receipts**. Raw Masks, state snapshots and gradient tensors are not retained in the supplied files, so the reviewer did not independently repeat tensor equality or autograd. No before/after remote checkpoint inventory is supplied.

## Timing and resource evidence

Launch record: **11:26:43.685536 CST**. The first target is exactly launch-record time plus **420 seconds**. Actual observations are **11:33:46.738620** and **11:37:47.379936 CST**: observed gaps are **423.053084 seconds** and **240.641316 seconds**. These match the intended 420/240 schedule with observation overhead; they are not exact observed 420/240 gaps (`launch.json:3`, `:22`; `observation_001.json:1`; `observer_terminal.json:4`).

The original neural PID closed without restart. `observer_started.json:9` preserves the local linked-process registration issue and explicitly records no observer restart/additional neural query.

The recorded diagnostic elapsed time is **474.754 seconds**, measured from runner startup, including setup/data loading. This is not pure kernel timing. Recorded CUDA peak allocation/reservation are **6.248 / 7.330 GiB**. Launch records one A100 40 GB GPU and the existing warm environment; the launcher uses the existing venv and resource lock without an installation path. These are retained receipts, not new reviewer GPU/environment measurements.

## Claim disposition

- **Supported:** actual completed/closed diagnostic; 58 distinct fit expressions; recorded scale/feature measurements; exact between-arm losses; measured zero-output derivative reduction on the current step-3211-labeled batch; zero updates/new trained weights on the reviewed route.
- **Conditional:** exact attribution of the floored matched query to row 34293/query 175, and a specifically isolated floor-to-gradient mechanism. Raw per-query coarse sizes and per-example gradients are absent.
- **Unsupported:** recreation or causal explanation of historical step 523. Its archived raw gradient is **1,206,195.5**; the current fresh probe is **0.01296121**. The old trained state, augmentation and RNG were not restored.
- **Unsupported:** trained compression efficacy, deployment selection, robustness across seeds/scenes, formal 9,508-row accuracy, newly executed native-criterion equality, or completed research gates. The declared gates remain **ACTIVE_UNMET**.
- **Unsupported here:** independent before/after byte preservation of remote historical checkpoints or revalidation of historical accuracy counts and the 9,508/39 comment.

No blocking correction, additional experiment or remote action is requested. Retain the bounded claim language and the same-family/provisional, actual-UNATTESTED review attribution.

